"""Typed span-data dataclasses for framework-level tracing.

Each span kind emitted by the runner has a dedicated
``@dataclass(frozen=True)`` that extends :class:`SpanData`. The base
class defines the contract: a ``type`` discriminator and an
``export()`` method that returns a JSON-safe ``dict[str, Any]`` for
downstream exporters.

The dataclasses are intentionally plain and frozen — no validation, no
side effects, zero runtime overhead when tracing is disabled. They hold
the minimum information a tracing backend needs to understand the
shape of an agent run without having to import any framework internals.

Example::

    from troopai.adk.types.tracing import AgentSpanData

    data = AgentSpanData(
        name="triage_agent",
        handoffs=["billing_agent"],
        tools=["lookup_order", "refund"],
        output_type="str",
    )
    exported = data.export()
    # {"type": "agent", "name": "triage_agent", ...}
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

if TYPE_CHECKING:
    from troopai.adk.types.guardrails.action import GuardrailAction


class SpanData(ABC):
    """Abstract base class for typed span data.

    Subclasses MUST set :attr:`type` to a stable string literal and
    implement :meth:`export` to produce a JSON-compatible dict.
    """

    type: str
    """Discriminator string identifying the span kind."""

    @abstractmethod
    def export(self) -> dict[str, Any]:
        """Serialize this span data to a JSON-safe dict.

        Returns:
            A dict containing the ``type`` discriminator plus all
            span-specific fields. Every value must be JSON-serialisable
            (``str``, ``int``, ``float``, ``bool``, ``None``, ``list``,
            ``dict``).
        """


@dataclass(frozen=True)
class AgentSpanData(SpanData):
    """Span data captured when an agent turn starts.

    Attributes:
        name: The agent's name.
        handoffs: Names of downstream agents reachable via handoff.
        tools: Names of tools available to the agent.
        output_type: Name of the agent's output type, if set.
        metadata: Arbitrary JSON-safe metadata attached at run time.
            Backends may surface this as custom span attributes.
        tenant_id: Opaque tenant identifier, surfaced as the
            ``troopai.tenant.id`` span attribute.
        type: Discriminator. Always ``"agent"``.
    """

    name: str
    """The agent's name."""

    handoffs: list[str] | None = None
    """Names of downstream agents reachable via handoff."""

    tools: list[str] | None = None
    """Names of tools available to the agent this turn."""

    output_type: str | None = None
    """Name of the agent's output type, or ``None`` for text output."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Arbitrary JSON-safe metadata attached to the root agent span."""

    tenant_id: str | None = None
    """Opaque tenant identifier, surfaced as the ``troopai.tenant.id`` span attribute."""

    type: str = "agent"

    @override
    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "handoffs": self.handoffs,
            "tools": self.tools,
            "output_type": self.output_type,
            "metadata": self.metadata,
            "tenant_id": self.tenant_id,
        }


@dataclass(frozen=True)
class FunctionSpanData(SpanData):
    """Span data captured for a single function-tool execution.

    Attributes:
        name: The tool's name.
        input: JSON-encoded tool arguments as received from the LLM.
        output: The tool's result (any JSON-safe value), or ``None``.
        mcp_data: MCP-specific metadata when the tool came from an MCP server.
        a2a_data: A2A-specific metadata when the call crossed an Agent-to-Agent
            boundary. Keys: ``task_id`` (str), ``context_id`` (str), and
            optionally ``remote_url`` (str) for client-side spans.
        type: Discriminator. Always ``"function"``.
    """

    name: str
    """The tool's name."""

    input: str | None = None
    """JSON-encoded tool arguments as received from the LLM."""

    output: Any | None = None
    """Tool result, or ``None`` if unavailable."""

    mcp_data: dict[str, Any] | None = None
    """MCP-specific metadata when the tool came from an MCP server."""

    a2a_data: dict[str, Any] | None = None
    """A2A-specific metadata when the call crossed an A2A boundary."""

    type: str = "function"

    @override
    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "input": self.input,
            "output": (str(self.output) if self.output is not None else None),
            "mcp_data": self.mcp_data,
            "a2a_data": self.a2a_data,
        }


@dataclass(frozen=True)
class GenerationSpanData(SpanData):
    """Span data captured for a single LLM request/response turn.

    Attributes:
        input: Provider-agnostic input items sent to the LLM.
        output: Response items produced by the LLM.
        model: Model identifier used for the call.
        model_config: Subset of ``LLMConfig`` fields actually sent to the provider.
        usage: Token-usage breakdown from the LLM response.
        cost_usd: Best-effort USD cost of this call, or ``None``.
        tenant_id: Opaque tenant identifier, surfaced as the
            ``troopai.tenant.id`` span attribute.
        type: Discriminator. Always ``"generation"``.
    """

    input: list[dict[str, Any]] | None = None
    """Provider-agnostic input items sent to the LLM."""

    output: list[dict[str, Any]] | None = None
    """Response items produced by the LLM."""

    model: str | None = None
    """Model identifier used for the call."""

    model_config: dict[str, Any] | None = None
    """Subset of ``LLMConfig`` fields actually sent to the provider."""

    usage: dict[str, Any] | None = None
    """Token-usage breakdown from the LLM response."""

    cost_usd: float | None = None
    """Best-effort USD cost of this call (set by the agent loop), or ``None``."""

    tenant_id: str | None = None
    """Opaque tenant identifier, surfaced as the ``troopai.tenant.id`` span attribute."""

    type: str = "generation"

    @override
    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "input": self.input,
            "output": self.output,
            "model": self.model,
            "model_config": self.model_config,
            "usage": self.usage,
            "cost_usd": self.cost_usd,
            "tenant_id": self.tenant_id,
        }


@dataclass(frozen=True)
class ResponseSpanData(SpanData):
    """Span data for a completed LLM response as a whole.

    Unlike :class:`GenerationSpanData`, this wraps the single
    provider-level response identifier so that traces can be correlated
    with downstream provider-side observability.

    Attributes:
        response_id: Provider-assigned response identifier.
        input: Provider-agnostic input items that produced the response.
        type: Discriminator. Always ``"response"``.
    """

    response_id: str | None = None
    """Provider-assigned response identifier."""

    input: list[dict[str, Any]] | None = None
    """Provider-agnostic input items that produced the response."""

    type: str = "response"

    @override
    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "response_id": self.response_id,
            "input": self.input,
        }


@dataclass(frozen=True)
class HandoffSpanData(SpanData):
    """Span data captured when one agent hands off to another.

    Attributes:
        from_agent: Name of the source agent.
        to_agent: Name of the destination agent.
        type: Discriminator. Always ``"handoff"``.
    """

    from_agent: str | None = None
    """Name of the source agent."""

    to_agent: str | None = None
    """Name of the destination agent."""

    type: str = "handoff"

    @override
    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
        }


@dataclass(frozen=True)
class GuardrailSpanData(SpanData):
    """Span data captured for a guardrail evaluation.

    Attributes:
        name: The guardrail's name.
        triggered: Whether the guardrail's tripwire fired.
        type: Discriminator. Always ``"guardrail"``.
    """

    name: str
    """The guardrail's name."""

    triggered: bool = False
    """Whether the guardrail's tripwire fired."""

    action: GuardrailAction | None = None
    """The resolved guardrail action (pass/raise/transform), when recorded."""

    type: str = "guardrail"

    @override
    def export(self) -> dict[str, Any]:
        exported: dict[str, Any] = {
            "type": self.type,
            "name": self.name,
            "triggered": self.triggered,
        }
        if self.action is not None:
            exported["action"] = self.action
        return exported


@dataclass(frozen=True)
class CustomSpanData(SpanData):
    """Span data for a developer-authored custom span.

    Attributes:
        name: Human-readable span name.
        data: Arbitrary JSON-safe payload provided by the developer.
        type: Discriminator. Always ``"custom"``.
    """

    name: str
    """Human-readable span name."""

    data: dict[str, Any] = field(default_factory=dict)
    """Arbitrary JSON-safe payload."""

    type: str = "custom"

    @override
    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "data": self.data,
        }


@dataclass(frozen=True)
class SandboxSpanData(SpanData):
    """Span data captured for a single sandbox command invocation.

    Attributes:
        backend_id: Backend that produced the span (e.g. ``"unix_local"``,
            ``"docker"``, ``"k8s_pod"``, or a hosted-provider name).
        command: The command string. Backends MUST truncate or redact
            sensitive arguments before constructing the span.
        exit_code: Process exit code, or ``None`` when the command was
            killed before reporting.
        duration_ms: Wall-clock duration in milliseconds.
        manifest_hash: Optional content hash of the manifest used to
            provision the session. Helps trace replays to inputs.
        resource_usage: Optional per-command resource record
            (cpu_ms, memory_peak_mb, bytes_read, bytes_written) as a
            JSON-safe dict.
        snapshot_id: Optional address of a snapshot persisted during
            this command's lifecycle.
        type: Discriminator. Always ``"sandbox"``.
    """

    backend_id: str
    """Backend that produced this span."""

    command: str | None = None
    """The command string (truncated by backend)."""

    exit_code: int | None = None
    """Process exit code, or ``None`` if the command was killed."""

    duration_ms: int | None = None
    """Wall-clock duration in milliseconds."""

    manifest_hash: str | None = None
    """Content hash of the manifest used to provision the session."""

    resource_usage: dict[str, int] | None = None
    """Per-command resource record (cpu_ms, memory_peak_mb, …)."""

    snapshot_id: str | None = None
    """Address of a snapshot persisted during this command."""

    type: str = "sandbox"

    @override
    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "backend_id": self.backend_id,
            "command": self.command,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "manifest_hash": self.manifest_hash,
            "resource_usage": self.resource_usage,
            "snapshot_id": self.snapshot_id,
        }


@dataclass(frozen=True)
class GraphSpanData(SpanData):
    """Span data captured for a whole graph run.

    The root span of a graph-execution span tree. Children are
    superstep spans (:class:`GraphSuperstepSpanData`); grandchildren
    are per-node spans (:class:`GraphNodeSpanData`).

    Attributes:
        graph_id: Stable graph identifier.
        entry: Entry node id, when set on the compiled graph.
        status: Terminal status (``"completed"``, ``"failed"``,
            ``"interrupted"``). Set when the span closes.
        supersteps_total: Total supersteps executed by the run;
            set when the span closes.
        type: Discriminator. Always ``"graph"``.
    """

    graph_id: str
    """Stable graph identifier."""

    entry: str | None = None
    """Entry node id, when set on the compiled graph."""

    status: str | None = None
    """Terminal status; set when the span closes."""

    supersteps_total: int | None = None
    """Total supersteps executed by the run; set when the span closes."""

    type: str = "graph"

    @override
    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "graph_id": self.graph_id,
            "entry": self.entry,
            "status": self.status,
            "supersteps_total": self.supersteps_total,
        }


@dataclass(frozen=True)
class GraphSuperstepSpanData(SpanData):
    """Span data captured for one BSP superstep boundary.

    Sits between :class:`GraphSpanData` (parent) and
    :class:`GraphNodeSpanData` (children). Surfaces the BSP structure
    in traces so operators can see where parallelism happened and
    where the loop blocked.

    Attributes:
        graph_id: Parent graph identifier.
        index: Zero-based superstep index.
        ready_nodes: Node ids that were ready to fire at the start of
            the superstep.
        fired_nodes: Node ids that actually fired (subset of ready);
            set when the span closes.
        type: Discriminator. Always ``"graph_superstep"``.
    """

    graph_id: str
    """Parent graph identifier."""

    index: int
    """Zero-based superstep index."""

    ready_nodes: tuple[str, ...] | None = None
    """Node ids that were ready to fire at the start of the superstep."""

    fired_nodes: tuple[str, ...] | None = None
    """Node ids that actually fired; set when the span closes."""

    type: str = "graph_superstep"

    @override
    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "graph_id": self.graph_id,
            "index": self.index,
            "ready_nodes": list(self.ready_nodes) if self.ready_nodes is not None else None,
            "fired_nodes": list(self.fired_nodes) if self.fired_nodes is not None else None,
        }


@dataclass(frozen=True)
class GraphNodeSpanData(SpanData):
    """Span data captured for one node attempt inside a superstep.

    Leaf of the graph-execution span tree. Retries are aggregated into
    ``attempts`` rather than spawning per-attempt child spans, to avoid
    span-tree explosion on flaky nodes.

    Attributes:
        graph_id: Parent graph identifier.
        node_name: Node id.
        attempts: Total invocation attempts including retries (≥1);
            set when the span closes.
        status: Terminal status (``"success"``, ``"failed"``,
            ``"interrupted"``); set when the span closes.
        duration_ms: Wall-clock duration in milliseconds; set when
            the span closes.
        resume_attempt: Resume sequence number when the node resumed
            from a prior interrupt; ``None`` for the original attempt.
        type: Discriminator. Always ``"graph_node"``.
    """

    graph_id: str
    """Parent graph identifier."""

    node_name: str
    """Node id."""

    attempts: int | None = None
    """Total invocation attempts including retries; set when the span closes."""

    status: str | None = None
    """Terminal status; set when the span closes."""

    duration_ms: int | None = None
    """Wall-clock duration in milliseconds; set when the span closes."""

    resume_attempt: int | None = None
    """Resume sequence number for resumed nodes; ``None`` for the original attempt."""

    type: str = "graph_node"

    @override
    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "graph_id": self.graph_id,
            "node_name": self.node_name,
            "attempts": self.attempts,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "resume_attempt": self.resume_attempt,
        }


@dataclass(frozen=True)
class SwarmSpanData(SpanData):
    """Span data captured for a whole swarm run.

    Root of the swarm-execution span tree. Children are per-iteration
    spans (:class:`SwarmTurnSpanData`). The ``swarm_id`` is a UUID
    generated at runner entry and persisted on the swarm state so a
    resumed run reuses the same id.

    Attributes:
        swarm_id: Stable swarm-run identifier.
        entry: Entry-member display name; set on construction.
        status: Terminal status; set when the span closes.
        turns_total: Total turns executed; set when the span closes.
        type: Discriminator. Always ``"swarm"``.
    """

    swarm_id: str
    """Stable swarm-run identifier."""

    entry: str | None = None
    """Entry-member display name."""

    status: str | None = None
    """Terminal status; set when the span closes."""

    turns_total: int | None = None
    """Total turns executed; set when the span closes."""

    type: str = "swarm"

    @override
    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "swarm_id": self.swarm_id,
            "entry": self.entry,
            "status": self.status,
            "turns_total": self.turns_total,
        }


@dataclass(frozen=True)
class SwarmTurnSpanData(SpanData):
    """Span data captured for one swarm turn.

    Leaf of the swarm-execution span tree. One span per loop iteration
    that actually runs a member turn — termination/guard exits before
    the turn body open no turn span.

    Attributes:
        swarm_id: Parent swarm-run identifier, repeated so a single
            attribute query can correlate turn spans back to their run.
        index: One-based turn index.
        member: Active member's display name.
        status: ``"success"`` / ``"interrupted"`` / ``"failed"`` /
            ``"handed_off"``; set when the span closes.
        duration_ms: Wall-clock duration of the turn body; set when
            the span closes.
        resume_attempt: Resume sequence number when this turn opened
            a splice; ``None`` on fresh turns.
        type: Discriminator. Always ``"swarm_turn"``.
    """

    swarm_id: str
    """Parent swarm-run identifier."""

    index: int
    """One-based turn index."""

    member: str
    """Active member's display name."""

    status: str | None = None
    """Terminal status; set when the span closes."""

    duration_ms: int | None = None
    """Wall-clock duration in ms; set when the span closes."""

    resume_attempt: int | None = None
    """Resume sequence number; ``None`` on fresh turns."""

    type: str = "swarm_turn"

    @override
    def export(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "swarm_id": self.swarm_id,
            "index": self.index,
            "member": self.member,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "resume_attempt": self.resume_attempt,
        }


AnySpanData = (
    AgentSpanData
    | FunctionSpanData
    | GenerationSpanData
    | ResponseSpanData
    | HandoffSpanData
    | GuardrailSpanData
    | CustomSpanData
    | SandboxSpanData
    | GraphSpanData
    | GraphSuperstepSpanData
    | GraphNodeSpanData
    | SwarmSpanData
    | SwarmTurnSpanData
)
"""Union of every built-in span-data kind."""
