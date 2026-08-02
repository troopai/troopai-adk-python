"""``Executable`` — the composition seam every multi-agent primitive plugs into.

Why an ABC and not a duck-typed protocol? Two reasons:

1. **Discoverability.** A named ABC lets adapter authors inherit
   ``Executable[TContext]`` and get a pyright-verified signature for
   ``invoke`` / ``stream_async``. With a bare ``Protocol`` the
   signature mismatch only shows up at the call site.

2. **Default streaming.** The ABC can ship a non-abstract
   :meth:`stream_async` whose default implementation calls
   :meth:`invoke` and yields a single terminal event. Cheap primitives
   (a callable node) get streaming for free without each adapter
   reimplementing an async iterator. Strands' ``MultiAgentBase`` uses
   the same trick (see ``src/strands/multiagent/base.py``).

The only abstract method is :meth:`invoke`. Subclasses that want
first-class streaming (e.g. an ``AgentExecutable`` that forwards
per-token events up to a ``Graph``) override :meth:`stream_async`
as well.

Uniform input type. Each :class:`Executable` receives an
:class:`ExecutableInput` — a ``list[LLMInputContentItem]`` plus a
small envelope of routing metadata. Layer 1 for the payload keeps the
adapters provider-agnostic; the envelope carries graph-level context
(incoming edge label, upstream node id) that an adapter may or may
not use.

Uniform output type. :class:`NodeResult` wraps the terminal value,
the Layer 3 items produced, per-run usage, and a metadata dict. This
matches Strands' ``NodeResult`` in spirit while staying under
project rules (``@dataclass`` for framework types; no provider-specific
content leakage).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from troopai.adk.types.tokens.llm_usage import LLMUsage

if TYPE_CHECKING:
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.items.items import RunItem


logger = logging.getLogger(__name__)


TContext = TypeVar("TContext")


@dataclass
class ExecutableInput:
    """Uniform input envelope passed to every :meth:`Executable.invoke`.

    Keeps the payload as Layer 1 items so adapters stay
    provider-agnostic. The envelope fields carry graph-level routing
    metadata that an adapter MAY use to shape its behaviour (e.g. the
    ``AgentExecutable`` can embed ``from_node`` in its injected
    system prompt; the ``CallableExecutable`` ignores it).

    Attributes:
        content: Layer 1 items the node should consume. Usually the
            merged outputs of upstream nodes; on the entry node it is
            the original user prompt normalized via
            :meth:`ItemHelpers.input_to_new_input_list`.
        from_node: Id of the upstream node whose output triggered this
            invocation. ``None`` for the entry node or when multiple
            upstream nodes contributed (AND-join fan-in).
        from_nodes: All upstream node ids that contributed to this
            invocation (including ``from_node``). Empty tuple on the
            entry node. Populated for AND-join multi-input nodes so
            adapters can do attribution.
        edge_label: Optional :attr:`GraphEdge.label` of the edge that
            fired. Useful for "route via intent" patterns where the
            downstream adapter wants to know which branch fired.
        metadata: Free-form dict for adapter-to-adapter signalling.
            Carries the human reply on resume (read via
            :meth:`resume_reply`); otherwise typically empty.
    """

    content: list[LLMInputContentItem]
    """Layer 1 items to feed the executable. Never ``None``."""

    from_node: str | None = None
    """Upstream node id (``None`` for entry or multi-input fan-in)."""

    from_nodes: tuple[str, ...] = field(default_factory=tuple)
    """All upstream node ids that contributed (empty for entry)."""

    edge_label: str | None = None
    """Label of the triggering :class:`GraphEdge`, if any."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Adapter-to-adapter side-channel; carries the resume reply (see ``resume_reply()``)."""

    def resume_reply(self) -> Any | None:
        """Return the human-supplied reply injected by the BSP loop on resume.

        On the first invocation of a node the loop has not injected a reply,
        so this returns ``None``. When a run is resumed after a human-in-the-loop
        pause, the loop re-invokes the interrupted node with the human-supplied
        value stored under a reserved metadata key; this method surfaces that
        value so node code never touches raw metadata keys directly.

        Returns:
            The injected reply value, or ``None`` when the node is running for
            the first time or no resume context is present.
        """
        return self.metadata.get("__resume_reply__")


@dataclass
class NodeResult[TContext]:
    """Uniform return envelope every :meth:`Executable.invoke` produces.

    ``NodeResult`` is the composition currency. Upstream values are
    merged into a downstream node's :class:`ExecutableInput` via the
    receiving node's ``merge_fn`` (see
    :mod:`troopai.adk.graphs.merge`). Conditional edges read the
    ``output`` + ``metadata`` to decide whether to fire.

    Attributes:
        output: The terminal value of this node's execution. Shape
            depends on the adapter: ``str`` or the parsed Pydantic
            model for an ``AgentExecutable``;
            :class:`~troopai.adk.swarms.result.SwarmRunResult` for a
            ``SwarmExecutable``; arbitrary return value for a
            ``CallableExecutable``; :class:`GraphRunResult` for a
            nested-graph node. Loops, type guards, and merge
            strategies MUST introspect ``output`` defensively.
        new_items: Layer 3 items produced while the node was running.
            For agent-backed adapters this is the turn's produced
            items; for callables it's either empty or whatever items
            the callable synthesised. The graph loop concatenates these
            into :attr:`GraphState.all_items` in node-completion order.
        usage: Cumulative :class:`LLMUsage` for this node's run. Used
            by :class:`GraphRunResult.per_node_usage` for cost
            attribution — the feature that neither LangGraph nor
            Strands exposes ergonomically on a Graph result.
        final_text: Normalized plain-text view of ``output`` when
            ``output`` is a string or carries text. ``None`` otherwise.
            Provided for adapters that want to chain textual context
            without doing their own extraction. The default
            :mod:`~troopai.adk.graphs.merge` strategies read this.
        metadata: Adapter-specific free-form dict for downstream edges
            and merge strategies to read. Keep this dict small — it is
            emitted on every :class:`GraphStreamEvent`.
    """

    output: Any
    """Terminal value of the node's execution. Shape depends on adapter."""

    new_items: list[RunItem] = field(default_factory=list)
    """Layer 3 items produced during this node's run."""

    usage: LLMUsage = field(default_factory=LLMUsage)
    """Cumulative LLM usage for this node. Zero for pure callables."""

    final_text: str | None = None
    """Normalized text view of ``output`` when text-extractable."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Free-form adapter metadata. Kept small."""


class Executable[TContext](ABC):
    """Abstract base class every graph-composable primitive implements.

    Three concrete primitives live in the ADK: :class:`Agent`,
    :class:`Swarm`, :class:`Graph`. None of them inherit from
    ``Executable`` directly — those classes stay config-only, never
    gaining execution methods. Instead, thin adapters in
    :mod:`troopai.adk.graphs.adapters` (``AgentExecutable``,
    ``SwarmExecutable``, ``CallableExecutable``) wrap them. A
    :class:`~troopai.adk.graphs.graph.Graph` itself *does* inherit
    from ``Executable`` directly so graphs nest uniformly (a graph
    node can hold another graph without an intermediate adapter).

    Type Parameters:
        TContext: The user-provided context type threaded through
            :class:`~troopai.adk.run.context.RunContext`. Adapters
            propagate this onto the embedded Agent / Swarm / Graph.
    """

    @abstractmethod
    async def invoke(
        self,
        input: ExecutableInput,
        context: RunContext[TContext],
        config: RunConfig,
    ) -> NodeResult[TContext]:
        """Run the executable once and return its terminal result.

        Called by the graph loop during a superstep. MUST NOT raise
        for normal termination; exceptional termination (tool failure,
        guardrail trip, policy error) MAY raise the appropriate
        exception which the graph loop surfaces on the enclosing
        ``GraphRunResult`` and ``GraphHooks.on_node_error``.

        Args:
            input: Uniform input envelope — Layer 1 items plus
                routing metadata.
            context: Shared :class:`RunContext` — usage accumulator
                and user context. Threaded verbatim so nested
                executables contribute their usage to the graph total.
            config: :class:`RunConfig` for the run. Adapters forward
                this to the underlying primitive
                (``Runner.arun(..., run_config=config)`` etc.).

        Returns:
            A :class:`NodeResult` wrapping the terminal output,
            generated items, usage delta, and metadata.
        """

    async def stream_async(
        self,
        input: ExecutableInput,
        context: RunContext[TContext],
        config: RunConfig,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream execution events, yielding a terminal ``{"result": ...}`` event.

        Default implementation calls :meth:`invoke` and yields a
        single ``{"result": node_result}`` event. Cheap primitives
        (a ``CallableExecutable``) get streaming for free. Adapters
        that want per-token forwarding (``AgentExecutable``) override
        this to forward the underlying streamed events first and emit
        the terminal event last.

        Events are plain ``dict`` — the concrete
        :class:`~troopai.adk.graphs.events.GraphStreamEvent` subclasses
        also subclass ``dict`` for zero-conversion forwarding. When
        embedded in a ``Graph``, the graph loop rewraps each yielded
        event with a ``graph_path`` tag so nested execution produces
        readable traces.

        Args:
            input: Uniform input envelope.
            context: Shared :class:`RunContext`.
            config: :class:`RunConfig` for the run.

        Yields:
            Dict events. The final event is always
            ``{"type": "result", "result": NodeResult}``.
        """
        result = await self.invoke(input, context, config)
        yield {"type": "result", "result": result}


__all__ = [
    "Executable",
    "ExecutableInput",
    "NodeResult",
]
