"""``A2AExecutableAdapter`` — wrap :class:`A2AAgent` as a graph node.

Lets a remote A2A agent slot directly into an
:class:`troopai.adk.graphs.graph.Graph` alongside local :class:`Agent`,
:class:`Swarm`, and callable nodes. The adapter is a thin shim: it
extracts the prompt text from the upstream :class:`ExecutableInput`,
dispatches via :meth:`A2ARunner.arun` (non-streaming), and projects
the resulting :class:`A2ARunResult` into a framework
:class:`NodeResult`.

Usage::

    from troopai.adk.a2a import A2AAgent
    from troopai.adk.a2a.adapters import A2AExecutableAdapter
    from troopai.adk.graphs import GraphBuilder

    remote_research = A2AAgent(name="Researcher", url="https://research.example.com")

    builder = GraphBuilder.new("research_pipeline")
    builder.node("remote_research", remote_research)  # auto-wrapped via to_executable()
    # or explicitly:
    builder.node("remote_research", A2AExecutableAdapter(agent=remote_research))

The adapter is also auto-registered with the graph's ``to_executable()``
factory: the dispatcher recognises :class:`A2AAgent` and wraps it
without the developer importing :class:`A2AExecutableAdapter` by hand.

Why an adapter and not making :class:`A2AAgent` extend
:class:`Executable` directly? Same rationale as
:class:`AgentExecutable`: keeping :class:`A2AAgent` config-only (free
of execution methods). The adapter is the
single place where the orchestration contract meets the network
boundary.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any, override

from troopai.adk.a2a.a2a_agent import A2AAgent
from troopai.adk.a2a.a2a_runner import A2ARunner
from troopai.adk.orchestration.executable import (
    Executable,
    ExecutableInput,
    NodeResult,
)

if TYPE_CHECKING:
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext

logger = logging.getLogger(__name__)


def _content_to_prompt_text(content: list[Any]) -> str:
    """Extract a plain prompt string from an :class:`ExecutableInput.content` list.

    A2A peers expect a string prompt (the protocol's ``Message`` is
    text-first; richer multi-modal parts can be added to the
    server-side converters when a real call site needs them). The
    graph passes content in one of two shapes:

    * Layer-1 *easy-message* dicts: ``{"role": "user", "content": "text"}``
      or ``{"role": "user", "content": [{"type": "input_text", "text": ...}]}``
      (the entry-node normalised user prompt).
    * Layer-1 input items with a top-level ``text`` field
      (``{"type": "input_text", "text": "..."}``) — the case fed by
      :class:`AgentExecutable` outputs that have been merged via the
      ``concat_text`` strategy.

    The extraction walks both shapes. Non-text items (images, tool
    calls, reasoning blocks) are skipped with a debug log so a graph
    that wires a tool-emitting node into an A2A node doesn't leak
    provider structure across the network boundary.

    Args:
        content: The :attr:`ExecutableInput.content` list passed by
            the graph orchestrator.

    Returns:
        All extracted text chunks joined into a single string. Empty
        string when no text content is found.
    """
    chunks: list[str] = []
    for item in content:
        if isinstance(item, dict):
            # Top-level "text" field (Layer-1 input_text / easy item).
            text = item.get("text")
            if isinstance(text, str) and len(text) > 0:
                chunks.append(text)
                continue
            # Easy-message shape: ``{"role": ..., "content": str | list}``.
            inner = item.get("content")
            if isinstance(inner, str):
                if len(inner) > 0:
                    chunks.append(inner)
                continue
            if isinstance(inner, list):
                for part in inner:
                    if isinstance(part, dict):
                        part_text = part.get("text")
                        if isinstance(part_text, str) and len(part_text) > 0:
                            chunks.append(part_text)
                continue
            logger.debug(
                "A2AExecutableAdapter: skipping non-text dict item with keys=%s",
                list(item.keys()),
            )
            continue
        text_attr = getattr(item, "text", None)
        if isinstance(text_attr, str) and len(text_attr) > 0:
            chunks.append(text_attr)
            continue
        logger.debug(
            "A2AExecutableAdapter: skipping non-text input item of type %s",
            type(item).__name__,
        )
    return "".join(chunks)


@dataclasses.dataclass
class A2AExecutableAdapter(Executable[Any]):
    """Wrap an :class:`A2AAgent` so it can sit inside a :class:`Graph` node.

    Calling :meth:`invoke` extracts the upstream prompt text,
    dispatches via :meth:`A2ARunner.arun` (blocking, non-streaming),
    and packages the result. The remote agent's own ``task_id`` and
    ``context_id`` are surfaced on :attr:`NodeResult.metadata` so
    downstream nodes / edge predicates can route on them.

    **Fresh remote context per invocation.** Each :meth:`invoke`
    call dispatches without a ``continuation_token`` or
    ``context_id``, so the remote endpoint allocates a fresh
    conversation context for every node firing. This is the
    intentional default — graph supersteps in this codebase
    represent independent units of work, and propagating remote
    state across them would require a stateful adapter that
    persists ``A2AContinuationToken`` instances per
    ``(graph_run_id, node_id)`` pair. That scaffolding is not
    yet shipped; if a use case needs cross-superstep continuity,
    open an issue describing the routing semantics you need.

    The adapter does NOT propagate ``RunConfig`` to the remote agent
    — the remote endpoint has its own runner with its own config.
    Per-remote tuning belongs on :class:`A2AAgent` itself
    (``timeout``, ``max_stream_chunks``, etc.).

    Attributes:
        agent: The :class:`A2AAgent` to invoke.
    """

    agent: A2AAgent
    """The wrapped remote A2A peer."""

    @override
    async def invoke(
        self,
        input: ExecutableInput,
        context: RunContext[Any],
        config: RunConfig,
    ) -> NodeResult[Any]:
        """Call the remote A2A agent once and package its response.

        Threads ``input.content`` text into the prompt; ignores
        ``context`` and ``config`` because the remote endpoint owns
        its own runtime. The remote's ``task_id`` and ``context_id``
        surface on :attr:`NodeResult.metadata` for graph-level
        observability.

        ``NodeResult.usage`` is left at zero — the remote agent's
        token usage is opaque to this side of the boundary. Cost
        accounting for the remote happens server-side; the local
        graph only sees the network call as a single opaque step.

        Args:
            input: The upstream :class:`ExecutableInput` carrying the
                content list to extract the prompt from.
            context: The graph run context — not forwarded to the
                remote agent (the remote endpoint owns its own runtime).
            config: The run configuration — not forwarded to the
                remote agent.

        Returns:
            A :class:`NodeResult` whose ``output`` and ``final_text``
            are the remote agent's response text and whose ``metadata``
            carries ``"task_id"``, ``"context_id"``, ``"agent_name"``,
            ``"remote_url"``, and ``"adapter": "a2a"``.

        Raises:
            A2ATaskError: Remote task reached a terminal failure state.
            A2ATransportError: Network failure contacting the remote
                endpoint.
            A2AProtocolError: Protocol-level failure from the remote
                endpoint.
        """
        del context, config  # Remote owns its own runtime; not propagated.

        prompt = _content_to_prompt_text(input.content)
        if len(prompt) == 0:
            raise ValueError(
                f"A2AExecutableAdapter.invoke: empty prompt extracted from input content "
                f"(node='{self.agent.name}', upstream='{input.from_node}'). "
                f"Empty prompts are a graph wiring bug; refusing to call remote with empty string."
            )

        logger.debug(
            "A2AExecutableAdapter.invoke: agent=%s url=%s from_node=%s",
            self.agent.name,
            self.agent.url,
            input.from_node,
        )
        result = await A2ARunner.arun(self.agent, prompt)
        return NodeResult(
            output=result.text,
            new_items=[],
            final_text=result.text,
            metadata={
                "adapter": "a2a",
                "agent_name": self.agent.name,
                "remote_url": self.agent.url,
                "task_id": result.task_id,
                "context_id": result.context_id,
            },
        )


__all__ = ["A2AExecutableAdapter"]
