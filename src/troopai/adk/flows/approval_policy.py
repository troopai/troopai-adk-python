"""FlowApprovalPolicy — declarative approval rules for HITL steps.

Supplements the simple ``bool | Callable[[ctx], bool]`` form of
:attr:`FlowStep.__flow_requires_approval__` with a typed primitive
that expresses common policy shapes without bespoke callable code:

- Quorum-of-N approvers.
- Role-restricted approver pools.
- SLA deadlines with auto-reject on expiry.

Mirrors the typed-primitive philosophy already established by
:class:`FlowTriggerEvent`, :class:`FlowApprovalDecision`,
:class:`HandoffInputFilter`: prefer dataclasses over weak primitive
types, so callers can attach attributes (audit, telemetry, retry
budgets) to the policy itself rather than to ad-hoc context dicts.

The executor's gate-evaluation path treats the policy as an
*informational* primitive: deferrals halt with ``status="deferred"``
and the developer enforces quorum / role / deadline checks when
constructing the resume decision. The executor does not evaluate those
fields automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, kw_only=True)
class FlowApprovalPolicy:
    """Declarative policy attached to a deferred step's approval requirement.

    Attached to :attr:`FlowDeferredStep.policy` when the executor
    defers a step; surfaced through :class:`FlowCheckpoint` so the
    out-of-band approval driver (UI, Slack bot, approval service) can
    enforce quorum / role / deadline semantics before recording a
    decision.

    Attributes:
        quorum: Number of distinct approver decisions required before
            the step may proceed. ``1`` (default) is the standard
            single-approver case. Higher values declare quorum
            requirements that the approval driver MUST honour when
            constructing the resume :class:`FlowApprovalDecision`
            sequence. Cost-conservative default (no quorum overhead).
        allowed_roles: Tuple of role / group identifiers permitted to
            decide. Empty tuple (default) means *anyone* may decide —
            no role restriction. Non-empty values name the
            :attr:`FlowApprovalDecision.approver_role` strings that
            count as valid; decisions from outside the pool MUST be
            rejected by the driver.
        deadline_seconds: Optional SLA ceiling, in seconds, measured
            from :attr:`FlowDeferredStep.request_time`. ``None``
            (default) means no deadline. When set, the driver SHOULD
            auto-reject the step once
            ``now - request_time > deadline_seconds`` (assuming
            :attr:`auto_reject_on_expiry`).
        auto_reject_on_expiry: When ``deadline_seconds`` is set and
            elapses, controls whether the driver auto-issues a
            :class:`FlowApprovalDecision` with ``approved=False`` and
            ``message="deadline_expired"``. ``True`` (default) is the
            safe choice — failing closed on expired SLAs prevents a
            quiescent flow from hanging indefinitely. ``False`` keeps
            the deferral pending until a human resolves it.
        metadata: Open-ended developer payload — never read by the
            framework. Useful for tagging the policy with tenant IDs,
            ticket links, escalation hints, etc.
    """

    quorum: int = 1
    """Number of approver decisions required (≥1)."""

    allowed_roles: tuple[str, ...] = ()
    """Approver roles permitted to decide; empty ⇒ anyone."""

    deadline_seconds: float | None = None
    """SLA ceiling in seconds from request_time; ``None`` ⇒ no deadline."""

    auto_reject_on_expiry: bool = True
    """When deadline elapses: ``True`` auto-rejects; ``False`` keeps pending."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Open-ended audit / telemetry payload — parity with :attr:`FlowDeferredStep.metadata`."""

    def __post_init__(self) -> None:
        """Validate field invariants at construction time.

        Raises:
            ValueError: When ``quorum < 1`` or ``deadline_seconds`` is
                non-positive — explicit boundary checks at construction
                so silent misconfigurations fail loudly rather than
                propagating an unenforceable policy through the
                checkpoint round-trip.
        """
        if self.quorum < 1:
            raise ValueError(f"FlowApprovalPolicy.quorum must be >= 1, got {self.quorum}.")
        if self.deadline_seconds is not None and self.deadline_seconds <= 0:
            raise ValueError(
                f"FlowApprovalPolicy.deadline_seconds must be > 0 when set, got {self.deadline_seconds}.",
            )
