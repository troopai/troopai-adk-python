"""Interrupt primitives for graph human-in-the-loop pauses.

These typed primitives define the interrupt/resume contract so the
public import graph stays stable. Raising :class:`InterruptException`
inside a node signals that the run must pause for human input; the BSP
loop in ``run/graph_loop.py`` captures the carried :class:`Interrupt`
onto ``GraphState.pending_interrupts`` and exits with
status=INTERRUPTED so the caller can resume the run via the checkpoint
API.

- :class:`InterruptException` — raised by a hook or a node to
  request a pause; carries the :class:`Interrupt` payload describing
  what the human approver must decide.
- :class:`Interrupt` — the structured request: node id, a question,
  and optional kind-specific metadata.
- :class:`GraphResume` — the caller's reply when resuming a paused
  run: human-supplied values keyed by ``node_id`` (``replies``) and
  model-visible decline messages (``rejected``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from troopai.adk.exceptions import TroopAIError
from troopai.adk.orchestration.executable import ExecutableInput

if TYPE_CHECKING:
    from troopai.adk.exceptions import AgentToolDeferral

logger = logging.getLogger(__name__)


NESTED_AGENT_TOOL_APPROVAL_KIND: str = "nested_agent_tool_approval"
"""Discriminator value identifying a ``NestedAgentInterrupt`` in a
serialised ``GraphState.pending_interrupts`` payload. Both the
``NestedAgentInterrupt.kind`` default and the ``GraphState.from_dict``
rehydration branch reference this constant so renaming requires changing
exactly one site."""


NESTED_GRAPH_INTERRUPT_KIND: str = "nested_graph_interrupt"
"""Discriminator value identifying a ``NestedGraphInterrupt`` in a
serialised ``GraphState.pending_interrupts`` payload. Distinct from
``NESTED_AGENT_TOOL_APPROVAL_KIND`` so ``GraphState.from_dict`` rehydrates a
lifted *plain* inner ``Interrupt`` WITHOUT the non-empty-``agent_name`` guard
that a nested-agent tool approval requires."""


@dataclass(frozen=True)
class Interrupt:
    """Structured request for a human decision.

    Attributes:
        node_id: Id of the node that requested the interrupt.
        question: Short human-readable question explaining what
            decision is needed.
        kind: Discriminator for the interrupt kind (``"tool_approval"``,
            ``"route_choice"``, ``"input_request"``). A plain string
            discriminator.
        metadata: Free-form dict with kind-specific payload
            (``"tool_call_id"`` for tool approvals,
            ``"options": [...]`` for route choices, etc.).
    """

    node_id: str
    """Id of the requesting node."""

    question: str
    """Human-readable question."""

    kind: str = "generic"
    """Kind discriminator; defaults to ``"generic"``."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Kind-specific payload."""


class InterruptException(TroopAIError):
    """Raised by a hook or a node to signal that the run must pause for human input.

    The BSP loop in ``run/graph_loop.py`` captures the carried
    :class:`Interrupt` onto :attr:`GraphState.pending_interrupts` and
    exits with status=INTERRUPTED so the caller can resume the run via
    :class:`GraphResume` and the checkpoint API.

    Attributes:
        interrupt: The :class:`Interrupt` describing what decision is
            needed.
    """

    def __init__(self, interrupt: Interrupt) -> None:
        super().__init__(f"graph interrupt requested by node {interrupt.node_id!r}: {interrupt.question}")
        self.interrupt = interrupt


@dataclass(frozen=True)
class GraphResume:
    """Human replies for resuming a paused graph run.

    Pass an instance to the resume entry-point so the graph loop can
    unblock nodes that raised :class:`InterruptException`.

    Attributes:
        replies: Human-supplied approval values keyed by ``node_id``.
            Analogous to calling ``state.approve(value=...)`` for each
            pending interrupt.
        rejected: Model-visible decline messages keyed by ``node_id``.
            Analogous to calling ``state.reject(message=...)`` for each
            declined interrupt.
    """

    replies: dict[str, Any] = field(default_factory=dict)
    """Approval values keyed by ``node_id``."""

    rejected: dict[str, str] = field(default_factory=dict)
    """Decline messages keyed by ``node_id``."""


def request_human_input(
    input: ExecutableInput,
    question: str,
    *,
    kind: str = "generic",
    **metadata: Any,
) -> Any:
    """Return the human-supplied reply, or raise InterruptException to pause.

    Called from inside a node's executable body. When the BSP loop has
    not injected a reply for this node, raises :class:`InterruptException`
    so the loop can suspend the run (status=INTERRUPTED, checkpointed).
    On a resumed invocation the loop sets
    ``ExecutableInput.metadata["__resume_reply__"]`` to the human-supplied
    value (which MAY be ``None`` — e.g. an "abstain" answer) and this
    function returns it. Presence of the reserved key — not the value's
    truthiness — is the authoritative signal that a reply was supplied.

    The reserved metadata keys ``__resume_reply__`` and
    ``__interrupt_node_id__`` are the loop-injected channel used to pass
    the node id in and the human reply back. Node code should never access
    these keys directly — use this helper as the public API.

    Args:
        input: The ``ExecutableInput`` received by the node.
        question: Short human-readable question explaining what decision is
            needed.
        kind: Discriminator for the interrupt kind (e.g. ``"tool_approval"``,
            ``"route_choice"``, ``"input_request"``). Defaults to
            ``"generic"``.
        **metadata: Kind-specific payload forwarded verbatim onto
            ``Interrupt.metadata``.

    Returns:
        The human-supplied reply value when the node is being re-invoked
        after a resume.

    Raises:
        InterruptException: When no reply has been injected — the node is
            running for the first time and the BSP loop should capture the
            interrupt and suspend the run.
    """
    if "__resume_reply__" in input.metadata:
        return input.metadata["__resume_reply__"]
    node_id = str(input.metadata.get("__interrupt_node_id__", ""))
    raise InterruptException(
        Interrupt(
            node_id=node_id,
            question=question,
            kind=kind,
            metadata=dict(metadata),
        )
    )


@dataclass(frozen=True)
class NestedAgentApproval:
    """Approve one deferred tool call inside a nested agent.

    Attributes:
        tool_call_id: ``DeferredToolCall.tool_call_id`` the approval
            targets.
        approver_id: Opaque id of the human or service granting the
            approval. Forwarded to ``RunState.approve``'s ``approver_id``.
        reason: Free-form rationale. Forwarded to
            ``RunState.approve``'s ``reason``.
    """

    tool_call_id: str
    """``DeferredToolCall.tool_call_id`` the approval targets."""

    approver_id: str | None = None
    """Opaque id of the human or service granting the approval."""

    reason: str | None = None
    """Free-form rationale for the approval."""


@dataclass(frozen=True)
class NestedAgentRejection:
    """Reject one deferred tool call inside a nested agent.

    Attributes:
        tool_call_id: ``DeferredToolCall.tool_call_id`` the rejection
            targets.
        message: Model-visible decline message. Forwarded to
            ``RunState.reject``'s ``message``.
        approver_id: Opaque id of the human or service that rejected
            the call.
        reason: Internal rationale (not shown to the model).
    """

    tool_call_id: str
    """``DeferredToolCall.tool_call_id`` the rejection targets."""

    message: str | None = None
    """Model-visible decline message."""

    approver_id: str | None = None
    """Opaque id of the human or service that rejected the call."""

    reason: str | None = None
    """Internal rationale (not shown to the model)."""


NestedAgentDecision = NestedAgentApproval | NestedAgentRejection
"""Discriminated decision for one deferred tool call."""


@dataclass(frozen=True)
class NestedAgentReply:
    """Decisions to apply when resuming a :class:`NestedAgentInterrupt`.

    Pass via ``GraphResume.replies[node_id]`` for a node whose interrupt
    is a :class:`NestedAgentInterrupt`. Each decision targets one
    deferred tool call by ``tool_call_id``. Decisions whose
    ``tool_call_id`` is not in the snapshot's
    ``deferred_tool_requests.approvals`` raise
    :class:`NestedAgentResumeError` at resume time.

    Attributes:
        decisions: Tuple of approvals/rejections to apply. Empty tuple
            is permitted — useful when the caller decides to let every
            pending call re-defer.
    """

    decisions: tuple[NestedAgentDecision, ...] = ()
    """Tuple of approvals/rejections to apply."""


@dataclass(frozen=True, kw_only=True)
class NestedAgentInterrupt(Interrupt):
    """Interrupt raised when a tool inside a graph-node Agent defers.

    Subtypes :class:`Interrupt` so existing consumers (BSP loop's
    suspension path, ``GraphRunResult.interrupts``, telemetry) keep
    working. The full mid-run ``RunState`` lives in
    ``GraphState.nested_agent_snapshots[node_id]`` — this object only
    carries the metadata a human reviewer needs to decide.

    Attributes:
        agent_name: Display name of the sub-agent that deferred.
        tool_call_ids: ``DeferredToolCall.tool_call_id`` values awaiting
            decision. The reviewer constructs a
            :class:`NestedAgentReply` indexed by these ids.
    """

    agent_name: str
    """Display name of the sub-agent that deferred."""

    tool_call_ids: tuple[str, ...]
    """``DeferredToolCall.tool_call_id`` values awaiting decision."""

    kind: str = NESTED_AGENT_TOOL_APPROVAL_KIND
    """Kind discriminator pinned to ``"nested_agent_tool_approval"``."""

    @classmethod
    def from_deferral(
        cls,
        *,
        node_id: str,
        deferral: AgentToolDeferral,
    ) -> NestedAgentInterrupt:
        """Build the interrupt payload from a caught ``AgentToolDeferral``.

        Args:
            node_id: Id of the graph node whose Agent deferred.
            deferral: The :class:`AgentToolDeferral` raised by
                ``Runner.arun`` for the sub-agent.

        Returns:
            A :class:`NestedAgentInterrupt` carrying the sub-agent name
            and the deferred tool-call ids the reviewer must decide on.

        Raises:
            ValueError: When ``deferral.deferred_requests.approvals`` is
                empty — an interrupt with no decisions to make would
                deadlock the resume path.
        """
        if len(deferral.deferred_requests.approvals) == 0:
            raise ValueError(
                f"NestedAgentInterrupt.from_deferral: deferral for agent "
                f"{deferral.agent_name!r} carries 0 deferred approvals — "
                f"refusing to build an empty interrupt."
            )
        return cls(
            node_id=node_id,
            question=(
                f"agent {deferral.agent_name!r} requires approval for "
                f"{len(deferral.deferred_requests.approvals)} tool call(s)"
            ),
            agent_name=deferral.agent_name,
            tool_call_ids=tuple(c.tool_call_id for c in deferral.deferred_requests.approvals),
        )


@dataclass(frozen=True, kw_only=True)
class NestedGraphInterrupt(Interrupt):
    """Interrupt raised when a graph-node's inner ``Graph`` suspends on a PLAIN ``Interrupt``.

    Lifted by :meth:`Graph.invoke` (and the resume re-lift in
    ``run/graph_loop.py``) when the inner graph's lexicographically-first
    pending interrupt is a plain :class:`Interrupt` — NOT a sub-agent tool
    approval (:class:`NestedAgentInterrupt`).

    Distinct from :class:`NestedAgentInterrupt` precisely so it carries no
    ``agent_name`` / ``tool_call_ids``: the human reply is a plain value
    forwarded verbatim into the inner graph's :class:`GraphResume`, not a
    tool-approval decision. Using a distinct kind also lets
    :meth:`GraphState.from_dict` rehydrate it WITHOUT the non-empty
    ``agent_name`` guard a nested-agent interrupt requires — the guard that
    previously made an outer graph permanently non-resumable after a
    lifted-plain-Interrupt checkpoint.

    ``metadata`` carries ``inner_graph_id`` + ``inner_node_id`` so the
    resume path can target the inner node.
    """

    kind: str = NESTED_GRAPH_INTERRUPT_KIND
    """Kind discriminator pinned to ``"nested_graph_interrupt"``."""


class NestedAgentResumeError(TroopAIError):
    """Raised when a :class:`NestedAgentReply` is inconsistent with the snapshot.

    Triggered when a decision's ``tool_call_id`` is not in the
    snapshot's ``deferred_tool_requests.approvals``, or when the same
    id appears twice in ``reply.decisions``.

    Attributes:
        node_id: Graph node id the reply targeted.
        detail: Human-readable explanation of the mismatch.
    """

    def __init__(self, node_id: str, detail: str) -> None:
        super().__init__(f"nested-agent resume failed for node {node_id!r}: {detail}")
        self.node_id = node_id
        self.detail = detail


class GraphResumeError(TroopAIError):
    """Raised by the BSP loop on a resume-payload / state mismatch.

    Surfaced when ``GraphResume.replies[node_id]`` does not match the
    shape of the interrupt parked under that node_id, or when the
    persisted ``GraphState`` is inconsistent (e.g., a
    ``NestedAgentInterrupt`` without its matching snapshot). The
    bridge raises before invoking the resumed executable so the
    caller can fix the payload and retry against the same checkpoint.

    Attributes:
        detail: Human-readable explanation of the mismatch.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(f"graph resume failed: {detail}")
        self.detail = detail


class NestedAgentSerializationError(TroopAIError):
    """Raised when a nested-agent ``RunState`` snapshot cannot be serialised at deferral time.

    Surfaced by ``AgentExecutable`` when the snapshot carries a
    non-JSON-serialisable field (e.g. a closure inside a tool's
    metadata). The bridge MUST NOT silently fall back — losing the
    snapshot would defeat the entire HITL contract by erasing the
    user's pending decision.

    Attributes:
        node_id: Graph node id of the agent that deferred.
        field_name: ``RunState`` field that failed serialisation.
    """

    def __init__(self, node_id: str, field_name: str) -> None:
        super().__init__(
            f"nested-agent snapshot serialisation failed for node {node_id!r}: "
            f"field {field_name!r} is not JSON-serialisable"
        )
        self.node_id = node_id
        self.field_name = field_name


__all__ = [
    "NESTED_AGENT_TOOL_APPROVAL_KIND",
    "NESTED_GRAPH_INTERRUPT_KIND",
    "GraphResume",
    "GraphResumeError",
    "Interrupt",
    "InterruptException",
    "NestedAgentApproval",
    "NestedAgentDecision",
    "NestedAgentInterrupt",
    "NestedAgentRejection",
    "NestedAgentReply",
    "NestedAgentResumeError",
    "NestedAgentSerializationError",
    "NestedGraphInterrupt",
    "request_human_input",
]
