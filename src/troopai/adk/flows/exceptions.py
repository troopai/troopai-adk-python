"""Flow-specific exception types.

All inherit from :class:`troopai.adk.exceptions.TroopAIError` so the
framework-wide exception hierarchy is preserved. Module-scoped rather than
in the central :mod:`troopai.adk.exceptions` file because they are tightly
coupled to Flow internals — same precedent as
:class:`troopai.adk.graphs.interrupt.InterruptException`.
"""

from __future__ import annotations

from troopai.adk.exceptions.exceptions import TroopAIError, UserError


class FlowDefinitionError(UserError):
    """Raised at Flow class-definition time when the decorator wiring is invalid.

    Subclass of :class:`UserError` because the cause is always a
    misconfiguration by the developer authoring the Flow class — never a
    framework internal failure. Typical triggers:

    - No ``@flow_start`` method declared on the class.
    - A method decorated with both ``@flow_listen`` and ``@flow_router`` (rejected
      by the decorator itself, but defense-in-depth checked in
      :class:`FlowMeta`).
    - A step method that is not ``async def`` or takes parameters
      besides ``self``.
    - A ``@flow_listen(trigger=...)`` referencing an unknown form.

    Args:
        message: Human-readable description of the misconfiguration.
            Surfaces directly to the developer at class-load time, so it
            should name the offending method and the rule that was violated.
    """


class FlowMaxStepsExceeded(TroopAIError):
    """Raised by :class:`FlowExecutor` when the configured ``max_steps`` cap is hit.

    Distinct from :class:`troopai.adk.exceptions.MaxTurnsExceeded` (which
    counts LLM turns inside a single agent loop). ``max_steps`` counts
    the number of Flow step invocations across the entire run; the cap
    exists to prevent unbounded fan-out in mis-wired flows.

    Args:
        message: Description including the cap value and the steps that
            ran. The default message is sufficient; pass a custom one
            only to add business context (tenant id, request id).
    """


class FlowStepError(TroopAIError):
    """Raised by :class:`FlowExecutor` when a step raises and ``error_policy = "halt"``.

    Wraps the underlying exception so the caller of :meth:`Runner.arun_flow`
    sees a structured error rather than the raw step exception. The
    underlying exception is available via the standard ``__cause__``
    attribute (set by ``raise ... from`` inside the executor).

    Args:
        step_name: The name of the step method whose body raised.
        message: Pre-formatted human-readable description; defaults to
            including the step name. The original exception is chained
            via ``__cause__``.
    """

    step_name: str
    """The step method that raised."""

    def __init__(self, step_name: str, message: str | None = None) -> None:
        self.step_name = step_name
        super().__init__(message or f"Flow step {step_name!r} raised an exception.")


class FlowStepDeferred(TroopAIError):
    """Internal signal raised by the executor when a ``requires_approval`` gate trips.

    NOT part of the public error surface — caught inside
    :meth:`FlowExecutor.run` to capture the step into the
    :class:`FlowCheckpoint` and halt the run with
    ``status="deferred"``. Developers never see this exception; they
    see a :class:`FlowRunResult` with ``deferred_steps`` populated.

    Args:
        step_name: Method name of the deferred step.
    """

    step_name: str
    """The step method whose ``requires_approval`` gate fired."""

    def __init__(self, step_name: str) -> None:
        self.step_name = step_name
        super().__init__(f"Flow step {step_name!r} requires approval (HITL).")


class FlowStepRateLimitExceeded(TroopAIError):
    """Raised when a step's :class:`FlowStepRateLimit` window is saturated.

    Surfaces when ``FlowStepRateLimit.behavior == "error"`` or when
    a ``"wait"`` configuration would exceed
    :attr:`FlowStepRateLimit.max_wait_seconds`. Routes through
    :attr:`FlowConfig.error_policy` exactly like any other step
    exception.

    Args:
        step_name: Method name of the step whose window saturated.
        rpm: The configured cap.
    """

    step_name: str
    """The step method whose window saturated."""

    rpm: int
    """The configured cap."""

    def __init__(self, step_name: str, rpm: int) -> None:
        self.step_name = step_name
        self.rpm = rpm
        super().__init__(
            f"Flow step {step_name!r} rate-limit window saturated (rpm={rpm}).",
        )


class FlowStepGovernanceError(TroopAIError):
    """Wraps an exception raised by a step's governance hook.

    Surfaces when a developer-supplied callable inside a step's
    :class:`FlowStepGuardrails` / :class:`FlowStepCachePolicy` /
    :class:`FlowStepRateLimit` machinery raises an exception that is
    NOT a typed verdict / rate-limit signal. Carries breadcrumb
    metadata (``hook``, ``phase``) so operators can tell the source
    of the failure apart from a step-body raise — the
    :class:`FlowStepErrorEvent` then reports
    ``FlowStepGovernanceError`` rather than the underlying
    exception's bare type.

    The original exception is chained via ``__cause__`` (``raise ...
    from exc``) so the full traceback survives.

    Args:
        step_name: Method name of the step whose hook raised.
        hook: ``"guardrail"`` / ``"cache_key_fn"`` / ``"cache_snapshot"``
            — identifies which governance surface produced the
            failure.
        phase: ``"pre"`` / ``"post"`` for guardrails; ``""`` for
            cache hooks (unscoped).
    """

    step_name: str
    """The step method whose governance hook raised."""

    hook: str
    """Which governance hook raised (guardrail / cache_key_fn / cache_snapshot)."""

    phase: str
    """``"pre"`` / ``"post"`` for guardrails; ``""`` for cache hooks."""

    def __init__(self, step_name: str, hook: str, phase: str = "") -> None:
        self.step_name = step_name
        self.hook = hook
        self.phase = phase
        phase_suffix = f" ({phase})" if len(phase) > 0 else ""
        super().__init__(
            f"Flow step {step_name!r} governance hook {hook!r}{phase_suffix} raised.",
        )


class FlowStepGuardrailTripped(TroopAIError):
    """Raised when a :class:`FlowStepGuardrails` member returns a non-allow verdict.

    Wraps the routed rejection ``message`` so the verdict's
    explanation surfaces through error handlers. Routes through
    :attr:`FlowConfig.error_policy`.

    Args:
        step_name: Method name of the step whose guardrail tripped.
        phase: ``"pre"`` or ``"post"`` — which evaluation phase.
        message: Routed explanation from the verdict.
    """

    step_name: str
    """The step method whose guardrail tripped."""

    phase: str
    """``"pre"`` or ``"post"``."""

    verdict_message: str | None
    """Routed verdict message (renamed to avoid LSP clash with :attr:`TroopAIError.message`)."""

    def __init__(self, step_name: str, phase: str, message: str | None) -> None:
        self.step_name = step_name
        self.phase = phase
        self.verdict_message = message
        suffix = f": {message}" if message is not None else ""
        super().__init__(
            f"Flow step {step_name!r} {phase}-guardrail tripped{suffix}.",
        )


class FlowAgentDeferred(TroopAIError):
    """Internal signal raised by :func:`arun_flow_agent` when an inner agent run defers.

    A step body that calls :func:`arun_flow_agent` may surface an
    agent-level HITL deferral (a tool with ``requires_approval=True``
    inside the agent run). This exception carries the agent's
    serialised :class:`RunState` JSON so the executor can stash it
    into :attr:`FlowDeferredStep.agent_run_state`, halt the flow, and
    resume the agent through the same checkpoint round-trip used for
    step-level approvals.

    NOT part of the public error surface — caught inside
    :meth:`FlowExecutor._process_batch_results`. Developers never see
    this exception; they see a :class:`FlowRunResult` with
    ``deferred_steps`` populated.

    Args:
        step_name: Method name of the flow step whose body raised.
        defer_key: Stable key identifying this agent invocation
            within the step. Used as the resume map key so a single
            step that runs multiple agents can target each
            independently. Defaults to ``step_name`` when the
            developer doesn't override.
        run_state_data: JSON-encoded :class:`RunState` snapshot
            produced by :meth:`RunState.to_dict` ``+`` ``json.dumps``.
            Stored on :attr:`FlowDeferredStep.agent_run_state`.
    """

    step_name: str
    """Method name of the deferred flow step."""

    defer_key: str
    """Stable key identifying the inner agent invocation."""

    run_state_data: str
    """Serialised :class:`RunState` payload."""

    def __init__(self, step_name: str, defer_key: str, run_state_data: str) -> None:
        self.step_name = step_name
        self.defer_key = defer_key
        self.run_state_data = run_state_data
        super().__init__(
            f"Flow step {step_name!r} agent run {defer_key!r} requires approval (agent HITL).",
        )


class FlowStepSkipped(TroopAIError):
    """Internal signal raised by the executor when an ``enabled`` gate returns ``False``.

    NOT part of the public error surface — caught inside the executor
    to suppress successor dispatch for the step. Developers never see
    this exception; they see a :class:`FlowStepSkippedEvent` on the
    streaming path and the absence of the step from
    :attr:`FlowRunResult.completed_steps`.

    Args:
        step_name: Method name of the skipped step.
    """

    step_name: str
    """The step method whose ``enabled`` gate returned ``False``."""

    def __init__(self, step_name: str) -> None:
        self.step_name = step_name
        super().__init__(f"Flow step {step_name!r} skipped (enabled=False).")


class FlowCheckpointNotFoundError(UserError):
    """Raised by :meth:`Runner.arun_flow_from_id` when no checkpoint exists for the given id.

    Subclass of :class:`UserError` because the cause is a developer
    mistake — supplying an id that was never persisted, or targeting the
    wrong backend — not a framework internal failure.

    Args:
        checkpoint_id: The id that was not found.
    """

    checkpoint_id: str
    """The id that could not be resolved to a stored checkpoint."""

    def __init__(self, checkpoint_id: str) -> None:
        self.checkpoint_id = checkpoint_id
        super().__init__(
            f"Runner.arun_flow_from_id: no checkpoint found for id {checkpoint_id!r}. "
            "Ensure the checkpoint was persisted to the backend before resuming.",
        )


class FlowStepRejected(TroopAIError):
    """Internal signal raised when a resumed step's approval decision is ``approved=False``.

    NOT part of the public error surface — caught inside the executor
    to route the rejection through ``FlowConfig.error_policy``. Carries
    the routed rejection :attr:`decision_message` (from
    :attr:`FlowApprovalDecision.message`) so the error handler / final
    result can surface it. Audit metadata stays on the originating
    :class:`FlowApprovalDecision` and never travels via this exception.

    Args:
        step_name: Method name of the rejected step.
        message: Routed rejection explanation from
            :attr:`FlowApprovalDecision.message`.
    """

    step_name: str
    """The step method whose approval decision was ``False``."""

    decision_message: str | None
    """Routed rejection explanation (from :attr:`FlowApprovalDecision.message`).

    Renamed from ``message`` to avoid the LSP clash with the
    :attr:`TroopAIError.message` attribute (which is typed
    ``str``, never ``None``).
    """

    def __init__(self, step_name: str, message: str | None = None) -> None:
        self.step_name = step_name
        self.decision_message = message
        text = f"Flow step {step_name!r} was rejected"
        if message is not None:
            text = f"{text}: {message}"
        super().__init__(text)
