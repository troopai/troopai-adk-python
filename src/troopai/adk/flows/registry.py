"""Registry — frozen snapshot of ``@flow_start`` / ``@flow_listen`` / ``@flow_router``
declarations on a :class:`Flow` subclass, plus the precomputed dispatch
table consumed by the :class:`FlowExecutor`.

Two layers:

1. :class:`FlowStepRegistry` — built ONCE by
   :class:`troopai.adk.flows.flow.FlowMeta` at class-creation time, from
   walking ``cls.__dict__``. Captures the raw declarations: which methods
   are starts, listeners, routers, plus their declared trigger specs.
2. :class:`FlowTransitionTable` — built per ``Runner.arun_flow`` call by
   :func:`build_transition_table` from the registry. Denormalizes into
   the dispatch indexes the executor uses at runtime.

The split is deliberate: the registry is class-level immutable data, while
the transition table is a derived, per-run artifact that can be re-built
cheaply (O(n) over registered steps). Keeping them separate lets the same
``Flow`` class be executed with different :class:`FlowConfig` values
(e.g., different ``max_listeners_per_step``) without re-walking the class
namespace.

No hidden behavior:

- :class:`FlowStepRegistry` validates at construction time and raises
  :class:`FlowDefinitionError` for any structural problem — never
  defaults silently or "fixes" the developer's wiring.
- :func:`build_transition_table` is a pure function — no globals, no
  cache, no side effects.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from troopai.adk.flows.combinators import And, Or
from troopai.adk.flows.exceptions import FlowDefinitionError

TriggerSpec = str | Or | And
"""A normalized trigger spec stored on a decorated method.

After decoration, callable method-references have been resolved to their
``__name__`` so only three forms reach the registry:

- ``str`` — a step name or a route label.
- :class:`Or` / :class:`And` — gate combinators.
"""


@dataclass(frozen=True)
class FlowStepRegistry:
    """Frozen registry of every Flow step declared on a class.

    Built by :class:`troopai.adk.flows.flow.FlowMeta` walking ``cls.__dict__``
    once at class-creation time. The metaclass collects function attributes
    set by the decorators in :mod:`troopai.adk.flows.decorators`, validates
    coherence, and stamps the resulting registry on the class as
    ``cls.__flow_registry__``.

    The registry is conceptually immutable. The ``listeners`` and
    ``routers`` dicts are populated at construction; never mutate them
    after the registry is built (the dataclass freeze prevents reassignment
    of the field, but not internal dict mutation — convention enforces this).

    Attributes:
        starts: Frozen set of method names decorated with ``@flow_start``.
        listeners: Mapping from listener method names to their declared
            trigger specs (one entry per declared trigger — currently
            always a one-element tuple since ``@flow_listen`` accepts a single
            trigger).
        routers: Mapping from router method names to their declared
            trigger specs (same shape as ``listeners``).

    Raises:
        FlowDefinitionError: When no ``@flow_start`` is declared, or when a
            method name appears in more than one role.
    """

    starts: frozenset[str]
    """Method names decorated with ``@flow_start``."""

    listeners: dict[str, tuple[TriggerSpec, ...]]
    """Listener method name → tuple of normalized trigger specs."""

    routers: dict[str, tuple[TriggerSpec, ...]]
    """Router method name → tuple of normalized trigger specs."""

    def __post_init__(self) -> None:
        """Validate :class:`FlowStepRegistry` structural integrity.

        Raises:
            FlowDefinitionError: When ``starts`` is empty, or when a
                method name appears in more than one of
                ``starts`` / ``listeners`` / ``routers``.
        """
        if len(self.starts) == 0:
            raise FlowDefinitionError(
                "Flow class must declare at least one @flow_start method.",
            )
        all_names = self.starts | set(self.listeners.keys()) | set(self.routers.keys())
        expected_total = len(self.starts) + len(self.listeners) + len(self.routers)
        if len(all_names) != expected_total:
            raise FlowDefinitionError(
                "Flow step method names must be unique across "
                "@flow_start, @flow_listen, @flow_router — a method may have at most one role.",
            )

    def step_names(self) -> frozenset[str]:
        """Return the set of all step names declared on the Flow class.

        Returns:
            A frozen set containing every name in ``starts``,
            ``listeners``, and ``routers``. Useful for the executor and
            for introspection tools.
        """
        return self.starts | frozenset(self.listeners.keys()) | frozenset(self.routers.keys())


@dataclass(frozen=True)
class GateSpec:
    """Spec for an AND or OR gate in the transition table.

    A gate is uniquely identified by the listener it gates plus the set
    of triggers. The gate id is the canonical ``"{listener}:{sorted_triggers}"``
    string used by the executor as a dict key for arrival tracking and
    consumption tracking.

    Attributes:
        listener_name: The method that fires when the gate is satisfied.
        triggers: Frozen set of trigger names the gate watches. AND gates
            require every trigger; OR gates require the first trigger.
        gate_id: Canonical id — set by :func:`_make_gate_id` to ensure
            it matches the index used in the table's ``_for`` indexes.
    """

    listener_name: str
    """The listener method this gate gates."""

    triggers: frozenset[str]
    """Trigger names the gate watches."""

    gate_id: str
    """Canonical gate id used by the executor for arrival/consumption tracking."""


def _make_gate_id(listener_name: str, kind: str, triggers: tuple[str, ...]) -> str:
    """Build the canonical gate id used as a dict key in the transition table.

    The id encodes the listener, the gate kind, and the sorted trigger
    names so the same gate definition always produces the same id. Used
    internally; not part of the public API.

    Args:
        listener_name: Method name of the listener the gate gates.
        kind: ``"and"`` or ``"or"``.
        triggers: Tuple of trigger names. Internally sorted for stability.

    Returns:
        A canonical string of the form
        ``"{listener_name}:{kind}:{sorted-trigger-csv}"``.
    """
    return f"{listener_name}:{kind}:{','.join(sorted(triggers))}"


@dataclass(frozen=True)
class FlowTransitionTable:
    """Precomputed dispatch table for :class:`FlowExecutor`.

    Built per-run from :class:`FlowStepRegistry` by
    :func:`build_transition_table`. The executor consults this table at
    every step completion / route resolution to decide which methods to
    fire next.

    The table is intentionally simple: dispatch is keyed by trigger name
    (either a step name OR a route label — the executor doesn't
    distinguish). When step ``X`` completes, the executor fires every
    listener and router in ``direct_listeners[X]`` / ``routers_for[X]``,
    plus updates any AND/OR gates that include ``X`` in their triggers.

    Attributes:
        starts: Tuple of method names to fire when the flow begins.
        direct_listeners: ``trigger_name → tuple[listener_name, ...]`` —
            listeners that fire on direct arrival of that trigger.
        routers_for: ``trigger_name → tuple[router_name, ...]`` — routers
            that fire on arrival of that trigger.
        and_gates: ``gate_id → GateSpec`` — every AND gate declared on
            any listener.
        and_gates_for: ``trigger_name → tuple[gate_id, ...]`` — index of
            which AND gates a given trigger participates in.
        or_gates: ``gate_id → GateSpec`` — every OR gate declared on any
            listener.
        or_gates_for: ``trigger_name → tuple[gate_id, ...]`` — index of
            which OR gates a given trigger participates in.
    """

    starts: tuple[str, ...]
    """Methods to fire when the flow begins."""

    direct_listeners: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """trigger_name → listeners firing on direct arrival."""

    routers_for: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """trigger_name → routers firing on direct arrival."""

    and_gates: dict[str, GateSpec] = field(default_factory=dict)
    """gate_id → AND GateSpec."""

    and_gates_for: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """trigger_name → AND gate_ids the trigger participates in."""

    or_gates: dict[str, GateSpec] = field(default_factory=dict)
    """gate_id → OR GateSpec."""

    or_gates_for: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """trigger_name → OR gate_ids the trigger participates in."""


def build_transition_table(
    registry: FlowStepRegistry,
    *,
    max_listeners_per_step: int = 20,
) -> FlowTransitionTable:
    """Compile :class:`FlowStepRegistry` into a runtime dispatch table.

    Pure function: same input → same output, no side effects. Enforces
    the ``max_listeners_per_step`` fan-out cap structurally — exceeding
    it raises :class:`FlowDefinitionError` here, NOT silently at runtime.
    The cap exists to prevent accidental explosions in mis-wired flows;
    developers who want wider fan-out raise it explicitly via
    :class:`FlowConfig`.

    Args:
        registry: The frozen registry built by ``FlowMeta``.
        max_listeners_per_step: Maximum number of listener / router /
            gate methods that may fire on a single trigger arrival.
            Cost-conservative default of 20. Raise via
            :class:`FlowConfig.max_listeners_per_step`.

    Returns:
        A frozen :class:`FlowTransitionTable` keyed by trigger name.

    Raises:
        FlowDefinitionError: When the fan-out for any trigger exceeds
            ``max_listeners_per_step``.
    """
    if max_listeners_per_step < 1:
        raise FlowDefinitionError(
            f"max_listeners_per_step must be >= 1; got {max_listeners_per_step}.",
        )

    direct: dict[str, list[str]] = defaultdict(list)
    routers_for: dict[str, list[str]] = defaultdict(list)
    and_gates: dict[str, GateSpec] = {}
    and_gates_for: dict[str, list[str]] = defaultdict(list)
    or_gates: dict[str, GateSpec] = {}
    or_gates_for: dict[str, list[str]] = defaultdict(list)

    for listener_name, triggers in registry.listeners.items():
        for trigger in triggers:
            if isinstance(trigger, str):
                direct[trigger].append(listener_name)
            elif isinstance(trigger, And):
                gate_id = _make_gate_id(listener_name, "and", trigger.triggers)
                and_gates[gate_id] = GateSpec(
                    listener_name=listener_name,
                    triggers=frozenset(trigger.triggers),
                    gate_id=gate_id,
                )
                for trigger_name in trigger.triggers:
                    and_gates_for[trigger_name].append(gate_id)
            else:
                gate_id = _make_gate_id(listener_name, "or", trigger.triggers)
                or_gates[gate_id] = GateSpec(
                    listener_name=listener_name,
                    triggers=frozenset(trigger.triggers),
                    gate_id=gate_id,
                )
                for trigger_name in trigger.triggers:
                    or_gates_for[trigger_name].append(gate_id)

    for router_name, triggers in registry.routers.items():
        for trigger in triggers:
            if isinstance(trigger, str):
                routers_for[trigger].append(router_name)
            else:
                # Routers gated by AND/OR are not supported: routers
                # produce a route label that downstream @flow_listen methods
                # consume, so a gate-gated router would need additional
                # semantics for gate-state-on-router-return.
                raise FlowDefinitionError(
                    f"@flow_router methods may only have string triggers; "
                    f"got combinator on router {router_name!r}. "
                    f"Use @flow_listen with a combinator and produce the route from a separate step.",
                )

    _enforce_fanout_cap(direct, routers_for, and_gates_for, or_gates_for, max_listeners_per_step)

    return FlowTransitionTable(
        starts=tuple(sorted(registry.starts)),
        direct_listeners={k: tuple(v) for k, v in direct.items()},
        routers_for={k: tuple(v) for k, v in routers_for.items()},
        and_gates=and_gates,
        and_gates_for={k: tuple(v) for k, v in and_gates_for.items()},
        or_gates=or_gates,
        or_gates_for={k: tuple(v) for k, v in or_gates_for.items()},
    )


def _enforce_fanout_cap(
    direct: dict[str, list[str]],
    routers_for: dict[str, list[str]],
    and_gates_for: dict[str, list[str]],
    or_gates_for: dict[str, list[str]],
    cap: int,
) -> None:
    """Raise :class:`FlowDefinitionError` if any trigger exceeds the fan-out cap.

    The cap counts the total number of distinct downstream methods that
    fire on a single trigger arrival: direct listeners + routers + gates
    that include the trigger. The check is structural — runs once at
    table-build time, not per invocation.

    Args:
        direct: trigger → direct listener names.
        routers_for: trigger → router names.
        and_gates_for: trigger → AND gate_ids that include the trigger.
        or_gates_for: trigger → OR gate_ids that include the trigger.
        cap: Maximum permitted total fan-out per trigger.

    Raises:
        FlowDefinitionError: When any trigger's combined fan-out exceeds
            ``cap``.
    """
    all_triggers = set(direct.keys()) | set(routers_for.keys()) | set(and_gates_for.keys()) | set(or_gates_for.keys())
    for trigger in all_triggers:
        total = (
            len(direct.get(trigger, []))
            + len(routers_for.get(trigger, []))
            + len(and_gates_for.get(trigger, []))
            + len(or_gates_for.get(trigger, []))
        )
        if total > cap:
            raise FlowDefinitionError(
                f"Trigger {trigger!r} would fan out to {total} downstream methods "
                f"(direct + routers + gates), exceeding max_listeners_per_step={cap}. "
                f"Raise FlowConfig.max_listeners_per_step to allow this fan-out.",
            )
