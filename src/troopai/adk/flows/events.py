"""Stream event types for :class:`FlowRunResultStreaming`.

Flow events are step-granularity, NOT token-granularity. They fire when a
step starts, when a step ends, when a router resolves to a label, when
the flow finishes. Inner token-level streams from sub-agent calls inside
a step body are NOT forwarded through this stream by default — that
would require the framework to subscribe step bodies to a hidden event
sink, an implicit subscription outside the developer's declared opt-in.
Developers wanting token-level streaming inside a step call
:meth:`Runner.arun_streamed` themselves and consume its events directly.

Frozen dataclasses (rather than ``dict`` subclasses) are appropriate
here because flow events fire at second-scale intervals; the per-event
serialization overhead is negligible compared to token-streaming hot
paths. ``graphs/events.py`` uses ``dict`` subclasses because those events
fire per-token. Different cost profiles, different representations.

Every event carries the ``flow_id`` so a consumer multiplexing multiple
concurrent flow runs can demultiplex.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from troopai.adk.flows.triggers import FlowTriggerEvent
from troopai.adk.types.tokens.llm_usage import LLMUsage

if TYPE_CHECKING:
    from troopai.adk.flows.result import FlowRunStatus


@dataclass(frozen=True)
class FlowStartEvent:
    """Emitted once at the beginning of a Flow run.

    Attributes:
        flow_id: Stable identifier from the :class:`Flow` instance.
        start_steps: Tuple of ``@flow_start`` method names that are about to
            fire in parallel.
        type: Discriminator string; pinned to ``"flow.start"``.
    """

    flow_id: str
    """Stable identifier of the Flow instance being run."""

    start_steps: tuple[str, ...]
    """Names of the ``@flow_start`` methods that will fire first."""

    type: Literal["flow.start"] = "flow.start"
    """Event discriminator."""


@dataclass(frozen=True)
class FlowStepStartEvent:
    """Emitted just before a step body is invoked.

    Attributes:
        flow_id: Stable identifier of the Flow instance.
        step_name: Method name of the step about to run.
        step_count: Number of step invocations so far in this run
            (including this one). Useful for progress tracking.
        type: Discriminator string; pinned to ``"flow.step_start"``.
    """

    flow_id: str
    """Stable identifier of the Flow instance."""

    step_name: str
    """Method name of the step about to run."""

    step_count: int
    """Cumulative step invocation count after this step starts."""

    type: Literal["flow.step_start"] = "flow.step_start"
    """Event discriminator."""


@dataclass(frozen=True)
class FlowStepEndEvent:
    """Emitted after a step body returns (or raises).

    Attributes:
        flow_id: Stable identifier of the Flow instance.
        step_name: Method name of the step that just finished.
        next_steps: Always an empty tuple in the current implementation;
            reserved for a future where successor names are resolved
            before the step completes.
        usage: LLM usage delta produced by this step's body (zero when
            the step did not call into an LLM). Sum across all steps
            equals :attr:`FlowRunResult.cumulative_usage`.
        type: Discriminator string; pinned to ``"flow.step_end"``.
    """

    flow_id: str
    """Stable identifier of the Flow instance."""

    step_name: str
    """Method name of the step that just finished."""

    next_steps: tuple[str, ...]
    """Always an empty tuple in the current implementation; reserved for a
    future where successor names are resolved before the step completes."""

    usage: LLMUsage
    """LLM usage delta from this step's body."""

    type: Literal["flow.step_end"] = "flow.step_end"
    """Event discriminator."""


@dataclass(frozen=True)
class FlowRouteEvaluatedEvent:
    """Emitted after a ``@flow_router`` method returns a route label.

    Attributes:
        flow_id: Stable identifier of the Flow instance.
        router_step: Method name of the router that returned the label.
        route_label: The string the router returned. Used as the dispatch
            key for downstream ``@flow_listen("<label>")`` methods.
        triggered_steps: Tuple of step names that fired in response to
            this label. Empty when no listener subscribes to the label.
        type: Discriminator string; pinned to ``"flow.route_evaluated"``.
    """

    flow_id: str
    """Stable identifier of the Flow instance."""

    router_step: str
    """Method name of the router that emitted this route."""

    route_label: str
    """The label the router returned."""

    triggered_steps: tuple[str, ...]
    """Step names that fired in response to this label."""

    type: Literal["flow.route_evaluated"] = "flow.route_evaluated"
    """Event discriminator."""


@dataclass(frozen=True)
class FlowStepErrorEvent:
    """Emitted when a step body raises an exception.

    Both ``"halt"`` and ``"route_to_error_handler"`` error policies emit
    this event before the executor decides whether to halt or recover.

    Attributes:
        flow_id: Stable identifier of the Flow instance.
        step_name: Method name of the step whose body raised.
        error_type: Exception class name (e.g. ``"ValueError"``).
        error_message: ``str(exc)`` for the raised exception.
        type: Discriminator string; pinned to ``"flow.step_error"``.
    """

    flow_id: str
    """Stable identifier of the Flow instance."""

    step_name: str
    """Method name of the step that raised."""

    error_type: str
    """Class name of the raised exception."""

    error_message: str
    """``str(exception)`` for the raised exception."""

    type: Literal["flow.step_error"] = "flow.step_error"
    """Event discriminator."""


@dataclass(frozen=True)
class FlowEndEvent:
    """Emitted once at the end of a Flow run.

    Attributes:
        flow_id: Stable identifier of the Flow instance.
        status: Final status — see :data:`FlowRunStatus`.
        completed_steps: Tuple of every step method name that ran (in
            completion order). Steps that raised but were recovered via
            ``"route_to_error_handler"`` are included.
        cumulative_usage: Sum of every step's LLM usage delta.
        type: Discriminator string; pinned to ``"flow.end"``.
    """

    flow_id: str
    """Stable identifier of the Flow instance."""

    status: FlowRunStatus
    """Final flow status."""

    completed_steps: tuple[str, ...]
    """Step names that ran, in completion order."""

    cumulative_usage: LLMUsage
    """Sum of all step LLM usage."""

    type: Literal["flow.end"] = "flow.end"
    """Event discriminator."""


@dataclass(frozen=True)
class FlowStepSkippedEvent:
    """Emitted when a step is skipped because its ``enabled`` gate returned ``False``.

    The step body never runs. No successor dispatch (the gate is the
    framework's way to silently disable a path without rewiring the
    DAG).

    Attributes:
        flow_id: Stable identifier of the Flow instance.
        step_name: Method name of the skipped step.
        triggers: Triggers that would have scheduled the step. Same
            shape as :attr:`FlowStepContext.triggers` — empty for
            ``@flow_start``, single-element for direct / OR, multi
            for AND.
        type: Discriminator string; pinned to ``"flow.step_skipped"``.
    """

    flow_id: str
    """Stable identifier of the Flow instance."""

    step_name: str
    """Method name of the skipped step."""

    triggers: tuple[FlowTriggerEvent, ...]
    """Triggers that would have scheduled the step."""

    type: Literal["flow.step_skipped"] = "flow.step_skipped"
    """Event discriminator."""


@dataclass(frozen=True)
class FlowStepDeferredEvent:
    """Emitted when a step's ``requires_approval`` gate fires.

    The step is captured into :attr:`FlowCheckpoint.deferred_steps`
    and the run halts with ``status="deferred"``. Consumers of the
    streaming API see this event just before the stream ends; there
    is no live-inject channel — decisions are recorded on the
    returned :class:`FlowCheckpoint` via :meth:`FlowCheckpoint.approve`
    / :meth:`FlowCheckpoint.reject` and the run resumes through
    :meth:`Runner.arun_flow_from_checkpoint`. Same contract as the
    tool layer's :meth:`RunState.approve` / :meth:`RunState.reject`
    → :meth:`Runner.arun(agent, state)`.

    Attributes:
        flow_id: Stable identifier of the Flow instance.
        step_name: Method name of the deferred step.
        triggers: Triggers that scheduled the step.
        completed_steps: Step names that completed before the
            deferral, in completion order.
        type: Discriminator string; pinned to ``"flow.step_deferred"``.
    """

    flow_id: str
    """Stable identifier of the Flow instance."""

    step_name: str
    """Method name of the deferred step."""

    triggers: tuple[FlowTriggerEvent, ...]
    """Triggers that scheduled the step."""

    completed_steps: tuple[str, ...]
    """Step names that completed before the deferral."""

    type: Literal["flow.step_deferred"] = "flow.step_deferred"
    """Event discriminator."""


@dataclass(frozen=True)
class FlowStepRejectedEvent:
    """Emitted when a step is rejected by an approval decision (``approved=False``).

    The rejection is routed through :attr:`FlowConfig.error_policy` —
    ``@flow_listen("__error__")`` when configured, otherwise the run
    halts with ``status="failed"``. Mirrors the tool-layer surface:
    the model-visible / handler-visible message lives in
    :attr:`message`; audit metadata (approver_id, audit-reason) lives
    on :class:`FlowApprovalDecision` and never makes it into this
    event.

    Attributes:
        flow_id: Stable identifier of the Flow instance.
        step_name: Method name of the rejected step.
        message: Routed rejection explanation taken from
            :attr:`FlowApprovalDecision.message`. ``None`` when no
            explanation was supplied.
        type: Discriminator string; pinned to ``"flow.step_rejected"``.
    """

    flow_id: str
    """Stable identifier of the Flow instance."""

    step_name: str
    """Method name of the rejected step."""

    message: str | None
    """Routed rejection explanation (from :attr:`FlowApprovalDecision.message`)."""

    type: Literal["flow.step_rejected"] = "flow.step_rejected"
    """Event discriminator."""


FlowEvent = (
    FlowStartEvent
    | FlowStepStartEvent
    | FlowStepEndEvent
    | FlowRouteEvaluatedEvent
    | FlowStepErrorEvent
    | FlowStepSkippedEvent
    | FlowStepDeferredEvent
    | FlowStepRejectedEvent
    | FlowEndEvent
)
"""Discriminated union of all events emitted by a streaming Flow run.

Consumers can ``match event.type:`` to dispatch; each member carries a
distinct ``type`` literal so pyright narrows the union per-arm.
"""
