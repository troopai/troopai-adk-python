"""Span implementations and factory functions.

Defines :class:`Span` as a generic protocol over a :class:`SpanData`
payload, the :class:`NoOpSpan` default, and the public
``*_span()`` factory functions that runner and ADK code call to
create spans. Factories always route through
:func:`get_tracer`, so swapping the tracer at runtime is a
single-call affair.

Span objects are context managers::

    with custom_span("checkout", data=CustomSpanData(name="checkout", data={"sku": 42})) as span:
        do_work()

``span.error`` and ``span.finish()`` let callers record errors and
close the span explicitly when the context-manager form does not fit.

Parent tracking
---------------

:class:`Span` uses a module-level :class:`contextvars.ContextVar` to
maintain an implicit parent chain across nested ``with`` blocks and
``await`` boundaries. On :meth:`Span.start` the current span is captured
as ``self._parent`` and the span installs itself as the new current
span; :meth:`Span.finish` restores the previous value. The chain is
exposed via the :attr:`Span.parent_id` property so non-OTel tracers can
attribute children to the correct parent.

:class:`NoOpSpan` overrides these hooks to be truly empty — no
``ContextVar`` reads or writes — so the hot path remains zero-cost when
tracing is disabled.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar, Token
from types import TracebackType
from typing import Any, TypeVar, override

from troopai.adk.tracing.tracer import get_tracer
from troopai.adk.types.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GraphNodeSpanData,
    GraphSpanData,
    GraphSuperstepSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    SandboxSpanData,
    SpanData,
    SwarmSpanData,
    SwarmTurnSpanData,
)

logger = logging.getLogger(__name__)

TData = TypeVar("TData", bound=SpanData)


_current_span: ContextVar[Span[Any] | None] = ContextVar("troopai_current_span", default=None)
"""Process-wide current-span stack, scoped by :mod:`contextvars`.

Each :class:`Span` subclass that wants implicit parent tracking reads
this on :meth:`start` and writes itself as the new value; :meth:`finish`
restores the previous value via the captured :class:`~contextvars.Token`.
:class:`NoOpSpan` intentionally never touches this variable so the
disabled path stays zero-cost.
"""


class Span[TData: SpanData]:
    """Generic span wrapping a typed data payload.

    Subclasses specialise the ``TData`` type parameter for a given
    span kind. :class:`NoOpSpan` is the default implementation.
    """

    data: TData
    span_id: str | None
    error: dict[str, Any] | None

    def __init__(self, data: TData, span_id: str | None = None) -> None:
        """Construct a span wrapping the given typed payload.

        Args:
            data: The typed span-data payload. Set once, then mutated
                in-place as the span body runs.
            span_id: Optional caller-assigned span identifier.
        """
        self.data = data
        self.span_id = span_id
        self.error = None
        self._finished = False
        self._parent: Span[Any] | None = None
        self._token: Token[Span[Any] | None] | None = None

    @property
    def parent_id(self) -> str | None:
        """Span identifier of the enclosing span, if any."""
        if self._parent is None:
            return None
        return self._parent.span_id

    def set_error(self, message: str, *, data: dict[str, Any] | None = None) -> None:
        """Record an error on this span.

        Args:
            message: Short human-readable error message.
            data: Optional structured error context.
        """
        self.error = {"message": message, "data": data or {}}

    def start(self) -> None:
        """Install this span as the current span on the context stack.

        Captures the previous current-span value so :meth:`finish` can
        restore it. Subclasses that manage their own parent tracking
        (e.g. the OTel bridge) MUST override this hook.
        """
        self._parent = _current_span.get()
        self._token = _current_span.set(self)

    def finish(self) -> None:
        """Pop this span from the current-span stack."""
        if self._token is not None:
            _current_span.reset(self._token)
            self._token = None
        self._finished = True

    def __enter__(self) -> Span[TData]:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        del exc_type, exc_tb
        if exc_val is not None:
            self.set_error(str(exc_val), data={"type": type(exc_val).__name__})
        self.finish()


class NoOpSpan(Span[TData]):
    """Span that records nothing.

    Returned by :class:`~troopai.adk.tracing.tracer.NoOpTracer`. Hooks
    are truly empty — no :class:`~contextvars.ContextVar` reads or
    writes — so the disabled tracing path stays zero-cost.
    """

    @override
    def start(self) -> None:
        # Intentionally empty: do not touch the current-span ContextVar.
        pass

    @override
    def finish(self) -> None:
        self._finished = True


def current_span() -> Span[Any] | None:
    """Return the current span on the context stack, or ``None``.

    Public helper for tracer implementations (e.g. the OTel bridge, a
    :class:`~troopai.adk.tracing.multi_tracer.MultiTracer` composite) that
    need to attribute children to the active parent without importing
    the private :data:`_current_span` directly.
    """
    return _current_span.get()


def agent_span(
    *,
    name: str,
    handoffs: list[str] | None = None,
    tools: list[str] | None = None,
    output_type: str | None = None,
    metadata: dict[str, Any] | None = None,
    tenant_id: str | None = None,
    disabled: bool = False,
) -> Span[AgentSpanData]:
    """Create an agent-turn span via the current tracer.

    Args:
        name: Agent name.
        handoffs: Downstream agent names.
        tools: Tool names available this turn.
        output_type: Name of the agent's output type.
        metadata: Arbitrary JSON-safe metadata attached to the span
            (typically ``RunConfig.tracing_metadata``).
        tenant_id: Opaque tenant identifier, surfaced as the
            ``troopai.tenant.id`` span attribute.
        disabled: When ``True``, bypass the tracer and return a
            :class:`NoOpSpan` regardless of the installed tracer.
    """
    data = AgentSpanData(
        name=name,
        handoffs=handoffs,
        tools=tools,
        output_type=output_type,
        metadata=dict(metadata) if metadata is not None else {},
        tenant_id=tenant_id,
    )
    if disabled:
        return NoOpSpan(data)
    return get_tracer().agent_span(data)


def function_span(
    *,
    name: str,
    input: str | None = None,
    output: Any | None = None,
    mcp_data: dict[str, Any] | None = None,
    a2a_data: dict[str, Any] | None = None,
    disabled: bool = False,
) -> Span[FunctionSpanData]:
    """Create a function-tool span via the current tracer.

    Args:
        name: Tool name (e.g. ``"lookup"`` or ``"list_files"``).
        input: Raw tool arguments as a JSON string. **Recorded verbatim**
            — upstream code MUST redact secrets before invocation.
        output: Tool result. **Recorded verbatim.**
        mcp_data: Populated when the tool originated from an MCP server
            (``{"server_name": ..., "tool_name": ...}``). The OTel bridge
            uses this to switch the span name prefix from ``tool.`` to
            ``mcp.``.
        a2a_data: Populated when the call crossed an Agent-to-Agent
            boundary (``{"task_id": ..., "context_id": ..., "remote_url":
            ...}``). The OTel bridge uses this to switch the span name
            prefix from ``tool.`` (or ``mcp.``) to ``a2a.``. Takes
            precedence over ``mcp_data`` when both are set.
        disabled: When ``True``, bypass the tracer and return a
            :class:`NoOpSpan` regardless of the installed tracer.
    """
    data = FunctionSpanData(
        name=name,
        input=input,
        output=output,
        mcp_data=mcp_data,
        a2a_data=a2a_data,
    )
    if disabled:
        return NoOpSpan(data)
    return get_tracer().function_span(data)


def generation_span(
    *,
    input: list[dict[str, Any]] | None = None,
    output: list[dict[str, Any]] | None = None,
    model: str | None = None,
    model_config: dict[str, Any] | None = None,
    usage: dict[str, Any] | None = None,
    tenant_id: str | None = None,
    disabled: bool = False,
) -> Span[GenerationSpanData]:
    """Create an LLM generation span via the current tracer.

    Args:
        input: Prompt messages sent to the model.
        output: Model response messages.
        model: Model identifier (e.g. ``"gpt-4o"``).
        model_config: Provider-specific generation parameters.
        usage: Token-usage counters (input, output, total).
        tenant_id: Opaque tenant identifier, surfaced as the
            ``troopai.tenant.id`` span attribute.
        disabled: When ``True``, bypass the tracer and return a
            :class:`NoOpSpan` regardless of the installed tracer.
    """
    data = GenerationSpanData(
        input=input,
        output=output,
        model=model,
        model_config=model_config,
        usage=usage,
        tenant_id=tenant_id,
    )
    if disabled:
        return NoOpSpan(data)
    return get_tracer().generation_span(data)


def response_span(
    *,
    response_id: str | None = None,
    input: list[dict[str, Any]] | None = None,
    disabled: bool = False,
) -> Span[ResponseSpanData]:
    """Create a provider-level response span via the current tracer.

    Args:
        response_id: Provider-assigned response identifier.
        input: Prompt messages sent to the provider.
        disabled: When ``True``, bypass the tracer and return a
            :class:`NoOpSpan` regardless of the installed tracer.
    """
    data = ResponseSpanData(response_id=response_id, input=input)
    if disabled:
        return NoOpSpan(data)
    return get_tracer().response_span(data)


def handoff_span(
    *,
    from_agent: str | None = None,
    to_agent: str | None = None,
    disabled: bool = False,
) -> Span[HandoffSpanData]:
    """Create an agent-handoff span via the current tracer.

    Args:
        from_agent: Name of the agent initiating the handoff.
        to_agent: Name of the agent receiving the handoff.
        disabled: When ``True``, bypass the tracer and return a
            :class:`NoOpSpan` regardless of the installed tracer.
    """
    data = HandoffSpanData(from_agent=from_agent, to_agent=to_agent)
    if disabled:
        return NoOpSpan(data)
    return get_tracer().handoff_span(data)


def guardrail_span(
    *,
    name: str,
    triggered: bool = False,
    disabled: bool = False,
) -> Span[GuardrailSpanData]:
    """Create a guardrail-evaluation span via the current tracer.

    Args:
        name: Guardrail name (e.g. ``"pii_filter"``).
        triggered: ``True`` when the guardrail fired and blocked the
            request or response.
        disabled: When ``True``, bypass the tracer and return a
            :class:`NoOpSpan` regardless of the installed tracer.
    """
    data = GuardrailSpanData(name=name, triggered=triggered)
    if disabled:
        return NoOpSpan(data)
    return get_tracer().guardrail_span(data)


def custom_span(
    name: str,
    *,
    data: dict[str, Any] | None = None,
    span_id: str | None = None,
    disabled: bool = False,
) -> Span[CustomSpanData]:
    """Create a developer-authored custom span.

    The only tracing factory the ADK exposes for application code —
    use it to instrument business logic that sits outside the
    framework's built-in span kinds.

    Args:
        name: Short human-readable span name.
        data: Arbitrary JSON-safe payload.
        span_id: Optional caller-assigned span identifier.
        disabled: When ``True``, return a NoOp span regardless of the
            installed tracer.

    Example::

        with custom_span("rank_search_results", data={"n": len(results)}):
            ranked = rank(results)
    """
    payload = CustomSpanData(name=name, data=data or {})
    if disabled:
        span: Span[CustomSpanData] = NoOpSpan(payload, span_id=span_id)
        return span
    tracer_span = get_tracer().custom_span(payload)
    if span_id is not None:
        tracer_span.span_id = span_id
    return tracer_span


def sandbox_span(
    *,
    backend_id: str,
    command: str | None = None,
    exit_code: int | None = None,
    duration_ms: int | None = None,
    manifest_hash: str | None = None,
    resource_usage: dict[str, int] | None = None,
    snapshot_id: str | None = None,
    disabled: bool = False,
) -> Span[SandboxSpanData]:
    """Create a sandbox command-execution span via the current tracer.

    The minimal sandbox span: backend_id is required, every other
    field is optional so backends with partial telemetry still emit
    useful records. Routed through ``custom_span`` so tracers that
    implement only the core ``Tracer`` protocol still record sandbox
    spans without a protocol extension.

    Args:
        backend_id: Backend identifier (``"unix_local"``, ``"docker"``,
            hosted-provider name, …).
        command: Truncated/redacted command string.
        exit_code: Process exit code (None when killed before report).
        duration_ms: Wall-clock duration in milliseconds.
        manifest_hash: Optional content hash of the manifest.
        resource_usage: Per-command resource record (cpu_ms,
            memory_peak_mb, bytes_read, bytes_written) as a JSON-safe dict.
        snapshot_id: Optional snapshot address persisted during this
            command's lifecycle.
        disabled: When ``True``, bypass the tracer and return a
            :class:`NoOpSpan` regardless of the installed tracer.
    """
    data = SandboxSpanData(
        backend_id=backend_id,
        command=command,
        exit_code=exit_code,
        duration_ms=duration_ms,
        manifest_hash=manifest_hash,
        resource_usage=resource_usage,
        snapshot_id=snapshot_id,
    )
    if disabled:
        return NoOpSpan(data)
    # Wrap via custom_span: tracers without a dedicated sandbox factory
    # still record the span (with SandboxSpanData.export() shape on
    # span.data).
    custom = get_tracer().custom_span(
        CustomSpanData(name=f"sandbox.{backend_id}", data=data.export()),
    )
    # Return a Span typed against SandboxSpanData so callers' type
    # checks pass. The underlying tracer carries CustomSpanData;
    # this wrapper just relabels the type for callers.
    return custom  # type: ignore[return-value]


def graph_span(
    *,
    graph_id: str,
    entry: str | None = None,
    status: str | None = None,
    supersteps_total: int | None = None,
    disabled: bool = False,
) -> Span[GraphSpanData]:
    """Create a root graph-execution span via the current tracer.

    Routed through ``custom_span`` so user-installed tracers receive a
    uniform :class:`CustomSpanData` payload; the inner
    ``data["type"] == "graph"`` discriminator lets the OTel bridge
    apply graph-specific attribute conventions (the ``troopai.graph.*``
    namespace).

    Args:
        graph_id: Stable graph identifier.
        entry: Entry node id, when set on the compiled graph.
        status: Terminal status; set by the closing caller.
        supersteps_total: Total supersteps executed; set by the
            closing caller.
        disabled: When ``True``, bypass the tracer and return a
            :class:`NoOpSpan` regardless of the installed tracer.
    """
    data = GraphSpanData(
        graph_id=graph_id,
        entry=entry,
        status=status,
        supersteps_total=supersteps_total,
    )
    if disabled:
        return NoOpSpan(data)
    custom = get_tracer().custom_span(
        CustomSpanData(name=f"graph.{graph_id}", data=data.export()),
    )
    # Custom-span routing: underlying span carries CustomSpanData; the
    # factory's return type is Span[GraphSpanData] so callers see the
    # graph-typed surface. The OTel bridge re-derives the kind from
    # data["type"].
    return custom  # type: ignore[return-value]


def graph_superstep_span(
    *,
    graph_id: str,
    index: int,
    ready_nodes: tuple[str, ...] | None = None,
    fired_nodes: tuple[str, ...] | None = None,
    disabled: bool = False,
) -> Span[GraphSuperstepSpanData]:
    """Create a per-superstep span nested under the active graph span.

    Args:
        graph_id: Parent graph identifier.
        index: Zero-based superstep index.
        ready_nodes: Node ids that were ready at superstep start.
        fired_nodes: Node ids that fired; set by the closing caller.
        disabled: When ``True``, bypass the tracer and return a
            :class:`NoOpSpan` regardless of the installed tracer.
    """
    data = GraphSuperstepSpanData(
        graph_id=graph_id,
        index=index,
        ready_nodes=ready_nodes,
        fired_nodes=fired_nodes,
    )
    if disabled:
        return NoOpSpan(data)
    custom = get_tracer().custom_span(
        CustomSpanData(name=f"graph.superstep.{index}", data=data.export()),
    )
    # See :func:`graph_span` for the custom-span routing rationale.
    return custom  # type: ignore[return-value]


def graph_node_span(
    *,
    graph_id: str,
    node_name: str,
    attempts: int | None = None,
    status: str | None = None,
    duration_ms: int | None = None,
    resume_attempt: int | None = None,
    disabled: bool = False,
) -> Span[GraphNodeSpanData]:
    """Create a per-node-attempt span nested under the active superstep span.

    Retries are aggregated into ``attempts`` rather than spawning
    per-attempt child spans, to avoid span-tree explosion on flaky
    nodes. Resumes after a suspend open a NEW span with
    ``resume_attempt`` set rather than re-using the original (OTel
    best-practice: spans should end within a single request).

    Args:
        graph_id: Parent graph identifier.
        node_name: Node id.
        attempts: Total attempts; set by the closing caller.
        status: Terminal status; set by the closing caller.
        duration_ms: Wall-clock duration; set by the closing caller.
        resume_attempt: Set when this span covers a resumed node.
        disabled: When ``True``, bypass the tracer and return a
            :class:`NoOpSpan` regardless of the installed tracer.
    """
    data = GraphNodeSpanData(
        graph_id=graph_id,
        node_name=node_name,
        attempts=attempts,
        status=status,
        duration_ms=duration_ms,
        resume_attempt=resume_attempt,
    )
    if disabled:
        return NoOpSpan(data)
    custom = get_tracer().custom_span(
        CustomSpanData(name=f"graph.node.{node_name}", data=data.export()),
    )
    # See :func:`graph_span` for the custom-span routing rationale.
    return custom  # type: ignore[return-value]


def swarm_span(
    *,
    swarm_id: str,
    entry: str | None = None,
    status: str | None = None,
    turns_total: int | None = None,
    disabled: bool = False,
) -> Span[SwarmSpanData]:
    """Create a root swarm-execution span via the current tracer.

    Routed through ``custom_span`` so user-installed tracers receive a
    uniform :class:`CustomSpanData` payload; the inner
    ``data["type"] == "swarm"`` discriminator lets the OTel bridge
    apply swarm-specific attribute conventions.

    Args:
        swarm_id: Stable swarm-run identifier. Typically a UUID
            generated at the runner entry point.
        entry: Entry-member display name (``swarm.entry.name``).
        status: Terminal status; set by the closing caller.
        turns_total: Total turns executed; set by the closing
            caller.
        disabled: When ``True``, bypass the tracer and return a
            :class:`NoOpSpan` regardless of the installed tracer.
    """
    data = SwarmSpanData(
        swarm_id=swarm_id,
        entry=entry,
        status=status,
        turns_total=turns_total,
    )
    if disabled:
        return NoOpSpan(data)
    custom = get_tracer().custom_span(
        CustomSpanData(name=f"swarm.{swarm_id}", data=data.export()),
    )
    # See :func:`graph_span` for the custom-span routing rationale.
    return custom  # type: ignore[return-value]


def swarm_turn_span(
    *,
    swarm_id: str,
    index: int,
    member: str,
    status: str | None = None,
    duration_ms: int | None = None,
    resume_attempt: int | None = None,
    disabled: bool = False,
) -> Span[SwarmTurnSpanData]:
    """Create a per-turn span nested under the active swarm span.

    Args:
        swarm_id: Parent swarm-run identifier.
        index: One-based turn index (matches
            ``state.total_turns``).
        member: Active member's display name.
        status: Terminal status; set by the closing caller.
        duration_ms: Wall-clock duration; set by the closing
            caller.
        resume_attempt: Resume sequence number when this turn
            opened the deep-resume splice; ``None`` on fresh
            turns.
        disabled: When ``True``, bypass the tracer and return a
            :class:`NoOpSpan` regardless of the installed tracer.
    """
    data = SwarmTurnSpanData(
        swarm_id=swarm_id,
        index=index,
        member=member,
        status=status,
        duration_ms=duration_ms,
        resume_attempt=resume_attempt,
    )
    if disabled:
        return NoOpSpan(data)
    custom = get_tracer().custom_span(
        CustomSpanData(name=f"swarm.turn.{index}", data=data.export()),
    )
    # See :func:`graph_span` for the custom-span routing rationale.
    return custom  # type: ignore[return-value]
