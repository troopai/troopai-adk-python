"""FlowTriggerEvent — typed payload for a single trigger arrival.

A trigger is whatever caused a downstream step to be scheduled — an
upstream step's completion, a router's returned route label, or the
unconditional ``@flow_start`` fire. The framework surfaces triggers
as :class:`FlowTriggerEvent` instances so consumers can distinguish
``"this fired because step X completed"`` from ``"this fired because
router Y returned label X"`` without re-walking the registry.

Used by:

- :attr:`FlowStepContext.triggers` — visible to step-level gate
  callables (``requires_approval``, ``enabled``) so the gate can
  branch on trigger provenance.
- :attr:`FlowDeferredStep.triggers` — preserved in the checkpoint
  so resume audit can reconstruct why a deferred step would have run.
- The streaming event surface (``FlowStepDeferredEvent`` etc.) —
  consumers of the event stream see the same typed payload, not a
  bare string.

The frozen-dataclass shape lets future fields (e.g. arrival
timestamp, producing-agent identity once Flow + Agent compose more
tightly) land without breaking existing introspection code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

FLOW_ERROR_TRIGGER: Final = "__error__"
"""Route literal fired when a step fails under ``error_policy="route_to_error_handler"``.

Declare an error-handler listener with ``@flow_listen(FLOW_ERROR_TRIGGER)``;
when a step body raises and the :class:`~troopai.adk.flows.config.FlowConfig`
uses ``error_policy="route_to_error_handler"``, the executor fires every
listener registered on this literal with the failed step recorded as the
trigger's ``source_step``. With no such listener, the policy falls back to
``"halt"`` semantics.
"""

type FlowTriggerKind = Literal["step_completion", "route_label", "start"]
"""Discriminator for how a trigger was produced.

- ``"step_completion"`` — an upstream step finished and the
  downstream listener fires on its name.
- ``"route_label"`` — a ``@flow_router`` returned a non-empty string
  and the downstream listener fires on that label. The producing
  router's method name is in :attr:`FlowTriggerEvent.source_step`.
- ``"start"`` — used internally for ``@flow_start`` steps; surfaced
  here for completeness but ``@flow_start`` typically has an empty
  :attr:`FlowStepContext.triggers` tuple rather than a synthetic
  start event. Reserved for future trigger surfaces.
"""


@dataclass(frozen=True, kw_only=True)
class FlowTriggerEvent:
    """One trigger arrival with full provenance.

    Frozen so consumers can keep references across awaits without
    worrying about mutation. The combination of :attr:`name`,
    :attr:`source_step`, and :attr:`kind` is enough to identify any
    trigger uniquely — gates and audit consumers can rely on
    equality / hashing for membership checks.

    Attributes:
        name: Canonical trigger name. For ``kind == "step_completion"``
            this is the upstream step's method name; for
            ``kind == "route_label"`` this is the literal string the
            router returned (the label downstream
            ``@flow_listen("label")`` matches on).
        source_step: Method name of the step that produced this
            trigger. For ``"step_completion"`` this equals
            :attr:`name`. For ``"route_label"`` this is the router's
            method name (the label lives in :attr:`name`).
        kind: How the trigger was produced — see
            :data:`FlowTriggerKind`.
    """

    name: str
    """Canonical trigger name (step name or route label)."""

    source_step: str
    """Producing step's method name (router for route labels)."""

    kind: FlowTriggerKind
    """How the trigger was produced."""
