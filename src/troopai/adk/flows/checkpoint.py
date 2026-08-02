"""FlowCheckpoint — serializable resume state for partially-executed Flows.

A :class:`FlowCheckpoint` records a Flow's between-step state so the run
can pause and resume across processes. It captures:

- The completed step set (so the executor doesn't re-fire what's done).
- The pending queue at checkpoint time (so the resume knows what's next).
- Partial AND-gate arrivals (so a fan-in that's halfway met isn't lost).
- The consumed-gates set (so already-fired OR/AND gates don't fire again).
- The state payload as JSON (Pydantic ``model_dump_json`` for BaseModel
  states; ``json.dumps(dataclasses.asdict(...))`` for dataclass states).
- A stable ``flow_id`` for correlation.
- Any steps paused by a ``requires_approval`` gate
  (:attr:`deferred_steps`) so resume can replay them under one or
  more :class:`FlowApprovalDecision` instances.

Persistence is **explicit** — there is no ``@persist`` decorator. The
developer calls :meth:`FlowCheckpoint.capture` (or constructs the
dataclass manually), serializes via :meth:`to_json`, stores wherever
they choose, and resumes via
:meth:`Runner.arun_flow_from_checkpoint`. This explicit-only model
forbids CrewAI's hidden auto-write behavior.

Captures between-step boundaries: completed steps, pending steps,
gate arrivals, consumed gates, and any HITL-deferred steps. The
``Flow`` class definition (decorators, state type) is NOT
serialised — the resuming side must reconstruct the same ``Flow``
class so the registry / transition table compile identically. Same
contract as :class:`TaskPipelineState`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from troopai.adk.flows.deferred import FlowApprovalDecision, FlowDeferralKind, FlowDeferredStep
from troopai.adk.flows.triggers import FlowTriggerEvent, FlowTriggerKind


@dataclass(frozen=True, kw_only=True)
class FlowCheckpoint:
    """Serializable mid-flow checkpoint.

    Captures everything the :class:`FlowExecutor` needs to resume the
    run from where it left off, with the exception of the state object
    itself (the developer rehydrates that explicitly before calling
    :meth:`Runner.arun_flow_from_checkpoint`).

    Attributes:
        flow_id: Stable identifier of the Flow instance — set by
            :attr:`Flow.flow_id` and preserved across the round-trip
            so consumers correlate the resume with the originating run.
        completed_steps: Tuple of step method names that finished before
            the checkpoint, in completion order. The resuming executor
            seeds these into its own ``completed_steps`` so any AND
            gates / direct listeners that depend on them are
            re-resolved correctly.
        pending_steps: Tuple of step method names that were pending at
            checkpoint time. The resuming executor seeds these into
            its ``pending`` deque to fire next.
        and_gate_arrivals: ``{gate_id: tuple-of-arrived-trigger-names}``
            mapping for partial AND-gate state. A gate that's half-met
            preserves its arrival set so the resume doesn't reset
            progress.
        consumed_gates: Tuple of ``gate_id`` strings for OR/AND gates
            that have already fired. The resuming executor seeds these
            into its ``consumed_gates`` so already-fired gates don't
            fire again.
        state_data: JSON payload of the state instance. For
            ``BaseModel`` states: produced via
            ``model_dump_json(mode="json")``. For ``@dataclass`` states:
            produced via ``json.dumps(dataclasses.asdict(state))``.
            The resuming side rehydrates the state instance manually
            (e.g., ``MyState.model_validate_json(checkpoint.state_data)``)
            and passes it as ``initial_state=`` when reconstructing the
            Flow.
        state_type_name: Fully-qualified class name of the state type
            (informational; rehydration requires the developer to
            supply the actual class — the framework never does runtime
            reflection on a string class name).
        deferred_steps: Tuple of :class:`FlowDeferredStep` instances
            captured because their ``requires_approval`` gate fired.
            Resume supplies one :class:`FlowApprovalDecision` per
            pending step. Empty tuple ⇒ no HITL pause.
        decisions: ``step_name → FlowApprovalDecision`` mapping
            recorded via :meth:`approve` / :meth:`reject`. Mutable
            on a frozen dataclass — both methods mutate the inner
            dict the way :meth:`RunState.approve` / :meth:`RunState.reject`
            mutate the tool-level approval maps on ``RunState``.
    """

    flow_id: str
    """Stable identifier of the Flow run."""

    completed_steps: tuple[str, ...]
    """Step method names that finished before the checkpoint, in order."""

    pending_steps: tuple[str, ...]
    """Step method names pending at checkpoint time."""

    and_gate_arrivals: dict[str, tuple[str, ...]]
    """``gate_id`` → arrived trigger names for partial AND-gate state."""

    consumed_gates: tuple[str, ...]
    """``gate_id`` for OR/AND gates that have already fired."""

    state_data: str
    """JSON-encoded state payload."""

    state_type_name: str = ""
    """Fully-qualified class name of the state type (informational)."""

    deferred_steps: tuple[FlowDeferredStep, ...] = ()
    """Steps paused by a ``requires_approval`` gate. Empty ⇒ no HITL pause."""

    pending_step_triggers: dict[str, tuple[FlowTriggerEvent, ...]] = field(default_factory=dict)
    """Trigger events for non-deferred pending steps at checkpoint time.

    Maps ``step_name → tuple[FlowTriggerEvent, ...]`` for every pending step
    that is NOT in :attr:`deferred_steps`.  On resume,
    :meth:`Runner._seed_executor_from_checkpoint` restores these into
    ``executor.pending_triggers`` so that :meth:`FlowExecutor._build_step_context`
    sees the same ``ctx.triggers`` tuple as the cold-start invocation would.

    Without this field, trigger events for non-deferred pending steps are lost
    on checkpoint round-trip, causing gate callables that branch on
    ``ctx.triggers`` to behave differently on resume than on cold start.
    """

    decisions: dict[str, FlowApprovalDecision] = field(default_factory=dict)
    """``step_name → FlowApprovalDecision`` recorded via :meth:`approve` / :meth:`reject`.

    Mirrors :attr:`RunState.deferred_tool_requests` / the
    ``approved_tools`` + ``rejected_tools`` channels on RunState —
    decisions baked into the resume payload itself, no separate
    approvals kwarg on :meth:`Runner.arun_flow_from_checkpoint`.

    The dict is mutable on a frozen dataclass: :meth:`approve` /
    :meth:`reject` mutate the inner mapping the way
    :meth:`RunState.approve` / :meth:`RunState.reject` mutate
    ``approved_tools`` / ``rejected_tools`` on the RunState instance.
    """

    @property
    def requires_action(self) -> bool:
        """``True`` when at least one deferred step has no recorded decision yet.

        Mirrors :attr:`RunResult.requires_action` at the checkpoint
        layer. Equivalent to
        ``any(d.step_name not in self.decisions for d in self.deferred_steps)``.
        """
        return any(d.step_name not in self.decisions for d in self.deferred_steps)

    def approve(
        self,
        deferred_step: FlowDeferredStep | str,
        *,
        approver_id: str | None = None,
        approver_role: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Record an approval decision for ``deferred_step``.

        Same shape as :meth:`RunState.approve` for tool HITL:
        mutates the checkpoint in place (the ``decisions`` mapping)
        rather than returning a new instance. The recorded
        :class:`FlowApprovalDecision` carries audit metadata
        (``approver_id``, ``approver_role``, ``reason``) but no
        routed message — approvals never need a routed explanation.

        Args:
            deferred_step: The :class:`FlowDeferredStep` to approve,
                or its ``step_name`` string. MUST name one of the
                entries in :attr:`deferred_steps`; otherwise
                ``ValueError``.
            approver_id: Optional opaque identifier (audit only).
            approver_role: Optional role / group (audit only).
            reason: Optional free-form rationale (audit only).

        Raises:
            ValueError: When ``deferred_step`` names no entry in
                :attr:`deferred_steps`.
        """
        step_name = self._require_known_step(deferred_step, method="approve")
        self.decisions[step_name] = FlowApprovalDecision(
            step_name=step_name,
            approved=True,
            approver_id=approver_id,
            approver_role=approver_role,
            reason=reason,
        )

    def reject(
        self,
        deferred_step: FlowDeferredStep | str,
        message: str | None = None,
        *,
        approver_id: str | None = None,
        approver_role: str | None = None,
        reason: str | None = None,
    ) -> None:
        """Record a rejection decision for ``deferred_step``.

        Same shape as :meth:`RunState.reject` for tool HITL.
        ``message`` is the *routed* explanation — surfaces through
        :class:`FlowStepRejectedEvent.message` and
        :class:`FlowStepRejected.decision_message` on the resume path.
        ``approver_id`` / ``approver_role`` / ``reason`` are
        audit-only and never routed.

        Args:
            deferred_step: The :class:`FlowDeferredStep` to reject,
                or its ``step_name`` string. MUST name one of the
                entries in :attr:`deferred_steps`; otherwise
                ``ValueError``.
            message: Optional routed explanation; surfaces through
                events / error handlers when the rejected step
                resumes.
            approver_id: Optional opaque identifier (audit only).
            approver_role: Optional role / group (audit only).
            reason: Optional free-form rationale (audit only).

        Raises:
            ValueError: When ``deferred_step`` names no entry in
                :attr:`deferred_steps`.
        """
        step_name = self._require_known_step(deferred_step, method="reject")
        self.decisions[step_name] = FlowApprovalDecision(
            step_name=step_name,
            approved=False,
            message=message,
            approver_id=approver_id,
            approver_role=approver_role,
            reason=reason,
        )

    def _require_known_step(self, deferred_step: FlowDeferredStep | str, *, method: str) -> str:
        """Resolve ``deferred_step`` to a known step name; raise when unknown or mistyped.

        Args:
            deferred_step: A :class:`FlowDeferredStep` or a bare
                ``step_name`` string.
            method: Calling method name (``"approve"`` / ``"reject"``),
                interpolated into the error message.

        Returns:
            The resolved step name.

        Raises:
            TypeError: When ``deferred_step`` is neither a string nor a
                :class:`FlowDeferredStep` — a bare ``AttributeError``
                from the ``.step_name`` access would read as an internal
                bug rather than a caller mistake.
            ValueError: When the resolved name is not in
                :attr:`deferred_steps`.
        """
        if isinstance(deferred_step, str):
            step_name = deferred_step
        elif isinstance(deferred_step, FlowDeferredStep):
            step_name = deferred_step.step_name
        else:
            raise TypeError(
                f"FlowCheckpoint.{method}: deferred_step must be a step-name string or a "
                f"FlowDeferredStep, got {type(deferred_step).__name__} ({deferred_step!r}).",
            )
        known = {d.step_name for d in self.deferred_steps}
        if step_name not in known:
            raise ValueError(
                f"FlowCheckpoint.{method}: step {step_name!r} is not in deferred_steps "
                f"{sorted(known)!r}. Pass one of the pending step names or its FlowDeferredStep.",
            )
        return step_name

    def to_json(self) -> str:
        """Serialize the checkpoint to a JSON string.

        The output is portable across processes. Load via
        :meth:`from_json`.

        Returns:
            A JSON string encoding every field. The ``state_data``
            field is already a JSON string and is nested as a string
            value (not parsed and re-stringified).
        """
        payload = {
            "flow_id": self.flow_id,
            "completed_steps": list(self.completed_steps),
            "pending_steps": list(self.pending_steps),
            "and_gate_arrivals": {k: list(v) for k, v in self.and_gate_arrivals.items()},
            "consumed_gates": list(self.consumed_gates),
            "state_data": self.state_data,
            "state_type_name": self.state_type_name,
            "deferred_steps": [_deferred_to_dict(d) for d in self.deferred_steps],
            "pending_step_triggers": {
                step: [{"name": t.name, "source_step": t.source_step, "kind": t.kind} for t in triggers]
                for step, triggers in self.pending_step_triggers.items()
            },
            "decisions": {k: _decision_to_dict(v) for k, v in self.decisions.items()},
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, payload: str) -> FlowCheckpoint:
        """Rehydrate a :class:`FlowCheckpoint` from :meth:`to_json` output.

        Required-field presence is enforced — a truncated or
        corrupted payload missing any of ``flow_id``,
        ``completed_steps``, ``pending_steps``, ``and_gate_arrivals``,
        ``consumed_gates``, or ``state_data`` raises
        :class:`ValueError`. Optional fields (``state_type_name``,
        ``deferred_steps``) default when absent.

        Args:
            payload: JSON string produced by a previous call to
                :meth:`to_json`.

        Returns:
            A reconstructed :class:`FlowCheckpoint` instance.

        Raises:
            ValueError: When the payload is missing a required field.
        """
        data = json.loads(payload)
        required = (
            "flow_id",
            "completed_steps",
            "pending_steps",
            "and_gate_arrivals",
            "consumed_gates",
            "state_data",
        )
        missing = [key for key in required if key not in data]
        if len(missing) > 0:
            raise ValueError(
                f"FlowCheckpoint payload missing required field(s): {missing!r}",
            )
        return cls(
            flow_id=data["flow_id"],
            completed_steps=tuple(data["completed_steps"]),
            pending_steps=tuple(data["pending_steps"]),
            and_gate_arrivals={k: tuple(v) for k, v in data["and_gate_arrivals"].items()},
            consumed_gates=tuple(data["consumed_gates"]),
            state_data=data["state_data"],
            state_type_name=data.get("state_type_name", ""),
            deferred_steps=tuple(_deferred_from_dict(d) for d in data.get("deferred_steps", [])),
            pending_step_triggers={
                step: tuple(_trigger_from_dict(t) for t in triggers)
                for step, triggers in data.get("pending_step_triggers", {}).items()
            },
            decisions={k: _decision_from_dict(v) for k, v in data.get("decisions", {}).items()},
        )

    @classmethod
    def capture(
        cls,
        *,
        flow_id: str,
        completed_steps: tuple[str, ...],
        pending_steps: tuple[str, ...],
        and_gate_arrivals: dict[str, set[str]],
        consumed_gates: set[str],
        state_data: str,
        state_type_name: str = "",
        deferred_steps: tuple[FlowDeferredStep, ...] = (),
        pending_step_triggers: dict[str, list[FlowTriggerEvent]] | None = None,
        decisions: dict[str, FlowApprovalDecision] | None = None,
    ) -> FlowCheckpoint:
        """Build a checkpoint from live executor state.

        Helper for the developer-driven persistence pattern: the
        executor exposes its internal state via this constructor so
        callers don't have to reach into private fields. Converts the
        executor's ``set`` data structures into the deterministic
        ``tuple`` forms used by the JSON format (sorted for stability).

        Args:
            flow_id: Stable identifier of the Flow run.
            completed_steps: Step names that finished, in order.
            pending_steps: Step names pending at this point.
            and_gate_arrivals: Executor's live ``{gate_id: set[name]}``.
            consumed_gates: Executor's live ``set[gate_id]``.
            state_data: JSON-encoded state payload.
            state_type_name: Fully-qualified state class name; defaults
                to empty (informational only).
            deferred_steps: Steps paused by a ``requires_approval``
                gate at this checkpoint.
            pending_step_triggers: ``{step_name: [FlowTriggerEvent, ...]}``
                provenance for non-deferred pending steps; ``None``
                stores an empty mapping (see
                :attr:`FlowCheckpoint.pending_step_triggers`).
            decisions: Pre-recorded ``{step_name: FlowApprovalDecision}``
                mapping to seed :attr:`FlowCheckpoint.decisions` with;
                ``None`` starts empty.

        Returns:
            A new frozen :class:`FlowCheckpoint`.
        """
        return cls(
            flow_id=flow_id,
            completed_steps=completed_steps,
            pending_steps=pending_steps,
            and_gate_arrivals={k: tuple(sorted(v)) for k, v in and_gate_arrivals.items()},
            consumed_gates=tuple(sorted(consumed_gates)),
            state_data=state_data,
            state_type_name=state_type_name,
            deferred_steps=deferred_steps,
            pending_step_triggers={step: tuple(triggers) for step, triggers in pending_step_triggers.items()}
            if pending_step_triggers is not None
            else {},
            decisions=dict(decisions) if decisions is not None else {},
        )


def _deferred_to_dict(deferred: FlowDeferredStep) -> dict[str, Any]:
    """Serialise one :class:`FlowDeferredStep` to a dict for JSON.

    ``triggers`` is encoded as a list of dicts (one per
    :class:`FlowTriggerEvent`). Timestamps are ISO 8601. The
    agent-bridge fields (``defer_key``, ``agent_run_state``) round-trip
    as raw strings.
    """
    policy = deferred.policy
    return {
        "step_name": deferred.step_name,
        "kind": deferred.kind,
        "triggers": [{"name": t.name, "source_step": t.source_step, "kind": t.kind} for t in deferred.triggers],
        "request_time": deferred.request_time.isoformat(),
        "deadline": deferred.deadline.isoformat() if deferred.deadline is not None else None,
        "policy": (
            {
                "quorum": policy.quorum,
                "allowed_roles": list(policy.allowed_roles),
                "deadline_seconds": policy.deadline_seconds,
                "auto_reject_on_expiry": policy.auto_reject_on_expiry,
                "metadata": dict(policy.metadata),
            }
            if policy is not None
            else None
        ),
        "metadata": dict(deferred.metadata),
        "defer_key": deferred.defer_key,
        "agent_run_state": deferred.agent_run_state,
    }


def _deferred_from_dict(data: dict[str, Any]) -> FlowDeferredStep:
    """Rehydrate a :class:`FlowDeferredStep` from its dict form.

    Raises :class:`ValueError` (not bare :class:`KeyError`) when a
    required field is missing — corrupted payload produces a
    flow-specific actionable message.
    """
    from troopai.adk.flows.approval_policy import FlowApprovalPolicy

    _require_field(data, "step_name", context="FlowDeferredStep")
    raw_time = data.get("request_time")
    parsed_time = datetime.fromisoformat(raw_time) if isinstance(raw_time, str) else datetime.now(UTC)
    raw_deadline = data.get("deadline")
    parsed_deadline = datetime.fromisoformat(raw_deadline) if isinstance(raw_deadline, str) else None
    triggers = tuple(_trigger_from_dict(t) for t in data.get("triggers", []))
    policy_data = data.get("policy")
    policy = (
        FlowApprovalPolicy(
            quorum=policy_data.get("quorum", 1),
            allowed_roles=tuple(policy_data.get("allowed_roles", [])),
            deadline_seconds=policy_data.get("deadline_seconds"),
            auto_reject_on_expiry=policy_data.get("auto_reject_on_expiry", True),
            metadata=dict(policy_data.get("metadata", {})),
        )
        if isinstance(policy_data, dict)
        else None
    )
    return FlowDeferredStep(
        step_name=data["step_name"],
        kind=_validate_deferral_kind(data.get("kind", "approval")),
        triggers=triggers,
        request_time=parsed_time,
        deadline=parsed_deadline,
        policy=policy,
        metadata=dict(data.get("metadata", {})),
        defer_key=data.get("defer_key"),
        agent_run_state=data.get("agent_run_state"),
    )


def _trigger_from_dict(data: dict[str, Any]) -> FlowTriggerEvent:
    """Rehydrate one :class:`FlowTriggerEvent` with loud missing-field validation."""
    _require_field(data, "name", context="FlowTriggerEvent")
    _require_field(data, "source_step", context="FlowTriggerEvent")
    _require_field(data, "kind", context="FlowTriggerEvent")
    return FlowTriggerEvent(
        name=data["name"],
        source_step=data["source_step"],
        kind=_validate_trigger_kind(data["kind"]),
    )


def _decision_to_dict(decision: FlowApprovalDecision) -> dict[str, Any]:
    """Serialise one :class:`FlowApprovalDecision` to a dict for JSON.

    Includes ``expired`` so the typed :attr:`FlowApprovalDecision.status`
    discriminator round-trips faithfully.
    """
    return {
        "step_name": decision.step_name,
        "approved": decision.approved,
        "message": decision.message,
        "approver_id": decision.approver_id,
        "approver_role": decision.approver_role,
        "reason": decision.reason,
        "decision_time": decision.decision_time.isoformat(),
        "expired": decision.expired,
        "metadata": dict(decision.metadata),
    }


def _decision_from_dict(data: dict[str, Any]) -> FlowApprovalDecision:
    """Rehydrate a :class:`FlowApprovalDecision` from its dict form.

    Raises :class:`ValueError` on missing required fields so a
    corrupted payload surfaces a flow-specific actionable message
    rather than a bare :class:`KeyError`.
    """
    _require_field(data, "step_name", context="FlowApprovalDecision")
    _require_field(data, "approved", context="FlowApprovalDecision")
    raw_time = data.get("decision_time")
    parsed_time = datetime.fromisoformat(raw_time) if isinstance(raw_time, str) else datetime.now(UTC)
    return FlowApprovalDecision(
        step_name=data["step_name"],
        approved=bool(data["approved"]),
        message=data.get("message"),
        approver_id=data.get("approver_id"),
        approver_role=data.get("approver_role"),
        reason=data.get("reason"),
        decision_time=parsed_time,
        expired=bool(data.get("expired", False)),
        metadata=dict(data.get("metadata", {})),
    )


def _require_field(data: dict[str, Any], field_name: str, *, context: str) -> None:
    """Raise :class:`ValueError` when ``field_name`` is missing from ``data``.

    Provides callers with a flow-specific actionable message instead
    of a bare :class:`KeyError` on the offending subscript.
    """
    if field_name not in data:
        raise ValueError(
            f"{context}: missing required field {field_name!r} in payload (keys present: {sorted(data.keys())!r}).",
        )


def _validate_deferral_kind(value: Any) -> FlowDeferralKind:
    """Narrow a deserialised string to the :data:`FlowDeferralKind` literal.

    Raises :class:`ValueError` on an unknown discriminator so a
    corrupted payload doesn't silently produce an invalid kind.
    """
    if value not in {"approval", "external_execution"}:
        raise ValueError(
            f"FlowDeferredStep.kind must be one of {{approval, external_execution}}, got {value!r}",
        )
    # The membership check above narrows to FlowDeferralKind at runtime;
    # pyright can't see the runtime equivalence between the set literal
    # and the Literal alias.
    return value  # type: ignore[no-any-return, return-value]


def _validate_trigger_kind(value: Any) -> FlowTriggerKind:
    """Narrow a deserialised string to the :data:`FlowTriggerKind` literal.

    Raises :class:`ValueError` on an unknown discriminator so a
    corrupted payload doesn't silently produce an invalid event.
    """
    if value not in {"step_completion", "route_label", "start"}:
        raise ValueError(f"FlowTriggerEvent.kind must be one of {{step_completion, route_label, start}}, got {value!r}")
    # The membership check above narrows to FlowTriggerKind at runtime;
    # pyright can't see the equivalence between the set literal and the
    # Literal alias.
    return value  # type: ignore[no-any-return, return-value]
