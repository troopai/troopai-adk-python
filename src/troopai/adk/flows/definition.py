"""Public immutable snapshot of a Flow's wiring topology.

:class:`FlowDefinition` captures the complete structural declaration of a
:class:`~troopai.adk.flows.flow.Flow` subclass in pure-data form: step names,
roles, trigger specs, gate specs, and router targets — with no callable
references and no back-reference to the originating ``Flow`` class or
instance. As a result it is:

- **Picklable** — every field is a primitive, tuple, frozenset, or a
  frozen dataclass composed of those. The mapping fields are wrapped in
  ``types.MappingProxyType`` which is also picklable.
- **Read-only** — the dataclass is frozen and the mapping fields are
  ``MappingProxyType`` views that reject in-place mutation.
- **Portable** — can be serialised, sent over a queue, or stored without
  importing any Flow-specific callable.

:func:`build_flow_definition` is a pure function: same
:class:`~troopai.adk.flows.registry.FlowStepRegistry` always produces the
same :class:`FlowDefinition`. :meth:`Flow.get_definition` is the ergonomic
entry point; the underlying function is public so tooling can call it
directly with a registry it already holds.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import override

from troopai.adk.flows.combinators import And
from troopai.adk.flows.exceptions import FlowDefinitionError
from troopai.adk.flows.registry import FlowStepRegistry, TriggerSpec, _make_gate_id


@dataclass(frozen=True)
class StepInfo:
    """Pure-data snapshot of a single decorated step.

    Carries the name, role discriminator, and raw trigger specs
    originally declared on the step. No callable reference; all
    trigger specs have already been normalised to name-strings,
    :class:`~troopai.adk.flows.combinators.Or`, or
    :class:`~troopai.adk.flows.combinators.And` by the time they reach
    the registry.

    Attributes:
        name: The step method name as a string.
        role: Decorator role — one of ``"start"``, ``"listen"``,
            ``"router"``.
        triggers: Tuple of normalised trigger specs declared on this
            step. Empty tuple for ``@flow_start`` steps (they have no
            upstream triggers). One entry per declared trigger for
            ``@flow_listen`` and ``@flow_router`` steps.
        description: Optional human-readable blurb attached via the
            ``description=`` decorator kwarg. ``None`` when the
            developer did not supply one.
    """

    name: str
    """Method name of the step."""

    role: str
    """Decorator role: ``"start"``, ``"listen"``, or ``"router"``."""

    triggers: tuple[TriggerSpec, ...]
    """Normalised trigger specs (names, Or, And). Empty for start steps."""

    description: str | None
    """Optional decorator description; ``None`` when not set."""


@dataclass(frozen=True)
class GateInfo:
    """Pure-data snapshot of an AND or OR gate.

    Mirrors :class:`~troopai.adk.flows.registry.GateSpec` but lives in the
    definition layer rather than the transition-table layer. The gate kind
    is encoded as a ``"and"`` / ``"or"`` string rather than a separate
    class so the type remains a plain frozen dataclass with no variant
    hierarchy.

    Attributes:
        gate_id: Canonical id of the form
            ``"{listener}:{kind}:{sorted-trigger-csv}"``.
        listener_name: The step that fires when the gate is satisfied.
        kind: Gate kind — ``"and"`` (all triggers required) or
            ``"or"`` (first trigger fires).
        triggers: Frozen set of trigger names the gate watches.
    """

    gate_id: str
    """Canonical gate id."""

    listener_name: str
    """Method that fires when the gate is satisfied."""

    kind: str
    """``"and"`` or ``"or"``."""

    triggers: frozenset[str]
    """Trigger names the gate watches."""


@dataclass(frozen=True)
class FlowDefinition:
    """Immutable pure-data snapshot of a Flow's wiring topology.

    Produced once per Flow class by :func:`build_flow_definition` /
    :meth:`~troopai.adk.flows.flow.Flow.get_definition`. Captures every
    structural decision made by the decorators — which steps exist,
    their roles, their triggers, the gate topology — with no callable
    references and no reference to the originating Flow class.

    Every field is either a primitive, a tuple, a frozenset, a
    :class:`~types.MappingProxyType` read-only view, or a frozen
    dataclass composed of those. A :class:`FlowDefinition` is picklable
    because :meth:`__reduce__` serialises the mapping fields as plain
    ``dict`` objects, which are re-wrapped into ``MappingProxyType`` on
    reconstruction. It can be used for:

    - Static validation before any run.
    - Visualisation without constructing or running a Flow.
    - Offline tooling, serialisation, and schema generation.

    Attributes:
        steps: Tuple of :class:`StepInfo` objects, one per decorated
            step, sorted by name for deterministic ordering.
        starts: Frozen set of step names decorated with ``@flow_start``.
        roles: Read-only mapping from step name to its role string
            (``"start"`` / ``"listen"`` / ``"router"``).
        direct_triggers: Read-only mapping from listener/router name to
            the tuple of plain string trigger names declared on it.
            Does not include gate triggers.
        gates: Tuple of :class:`GateInfo` objects covering both AND
            and OR gates, sorted by ``gate_id`` for determinism.
        router_triggers: Read-only mapping from router name to the
            tuple of plain string trigger names that activate it.
    """

    steps: tuple[StepInfo, ...]
    """All decorated steps, sorted by name."""

    starts: frozenset[str]
    """Names of ``@flow_start`` steps."""

    roles: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    """step_name → role string (read-only view)."""

    direct_triggers: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: MappingProxyType({}))
    """listener/router name → plain-string trigger names (read-only, no gate triggers)."""

    gates: tuple[GateInfo, ...] = field(default_factory=tuple)
    """AND and OR gates, sorted by gate_id."""

    router_triggers: Mapping[str, tuple[str, ...]] = field(default_factory=lambda: MappingProxyType({}))
    """router name → plain-string trigger names that activate it (read-only)."""

    def __post_init__(self) -> None:
        """Wrap any plain dict fields in :class:`MappingProxyType` for mutation resistance.

        The dataclass constructor accepts both plain ``dict`` and
        ``MappingProxyType`` values (so :func:`build_flow_definition` can
        pass dicts and ``__reduce__`` can reconstruct via the same path).
        This hook ensures the fields are always ``MappingProxyType`` on
        the live object, regardless of how the instance was constructed.
        ``object.__setattr__`` is required because the dataclass is
        ``frozen=True``.
        """
        if not isinstance(self.roles, MappingProxyType):
            object.__setattr__(self, "roles", MappingProxyType(dict(self.roles)))
        if not isinstance(self.direct_triggers, MappingProxyType):
            object.__setattr__(self, "direct_triggers", MappingProxyType(dict(self.direct_triggers)))
        if not isinstance(self.router_triggers, MappingProxyType):
            object.__setattr__(self, "router_triggers", MappingProxyType(dict(self.router_triggers)))

    @override
    def __reduce__(self) -> tuple[type, tuple[object, ...]]:
        """Return a picklable reconstruction tuple.

        :class:`~types.MappingProxyType` is not picklable directly.
        This method serialises the mapping fields as plain ``dict``
        objects; the constructor calls :meth:`__post_init__` which
        re-wraps them on reconstruction.

        Returns:
            A ``(cls, args)`` pair that ``pickle.loads`` calls as
            ``cls(*args)`` to reconstruct the instance.
        """
        return (
            self.__class__,
            (
                self.steps,
                self.starts,
                dict(self.roles),
                dict(self.direct_triggers),
                self.gates,
                dict(self.router_triggers),
            ),
        )

    def step_names(self) -> frozenset[str]:
        """Return a frozen set of all step names in this definition.

        Returns:
            Frozen set of every step method name, regardless of role.
        """
        return frozenset(s.name for s in self.steps)

    def steps_by_role(self, role: str) -> tuple[StepInfo, ...]:
        """Return the subset of steps matching ``role``.

        Args:
            role: One of ``"start"``, ``"listen"``, ``"router"``.

        Returns:
            Tuple of :class:`StepInfo` objects whose ``role`` matches,
            in the same sorted order as :attr:`steps`.
        """
        return tuple(s for s in self.steps if s.role == role)


def build_flow_definition(
    registry: FlowStepRegistry,
    *,
    descriptions: dict[str, str | None] | None = None,
) -> FlowDefinition:
    """Compile a :class:`FlowStepRegistry` into a :class:`FlowDefinition`.

    Pure function: same ``registry`` and ``descriptions`` always produce
    the same output. No side effects, no globals.

    Args:
        registry: The frozen step registry produced by
            :class:`~troopai.adk.flows.flow.FlowMeta` at class creation.
        descriptions: Optional mapping from step name to its
            ``description=`` kwarg value. When ``None``, all step
            descriptions are recorded as ``None``. Pass the result of
            :func:`~troopai.adk.flows.flow.collect_step_descriptions`
            (used by :meth:`~troopai.adk.flows.flow.Flow.get_definition`)
            to propagate decorator-level descriptions.

    Returns:
        A frozen :class:`FlowDefinition` reflecting the full wiring
        topology encoded in ``registry``.

    Raises:
        FlowDefinitionError: When a ``@flow_router`` step declares a
            combinator gate trigger — the same rejection
            :func:`~troopai.adk.flows.registry.build_transition_table`
            applies, so a definition never describes an unrunnable flow.
    """
    descs: dict[str, str | None] = descriptions if descriptions is not None else {}

    steps: list[StepInfo] = []
    roles: dict[str, str] = {}
    direct_triggers: dict[str, tuple[str, ...]] = {}
    router_triggers: dict[str, tuple[str, ...]] = {}
    gates: list[GateInfo] = []

    for name in sorted(registry.starts):
        steps.append(_build_start_step(name, descs))
        roles[name] = "start"

    for name in sorted(registry.listeners.keys()):
        step, plain, step_gates = _build_listener_step(name, registry.listeners[name], descs)
        steps.append(step)
        roles[name] = "listen"
        if plain:
            direct_triggers[name] = tuple(plain)
        gates.extend(step_gates)

    for name in sorted(registry.routers.keys()):
        step, plain_router = _build_router_step(name, registry.routers[name], descs)
        steps.append(step)
        roles[name] = "router"
        if plain_router:
            router_triggers[name] = tuple(plain_router)

    return FlowDefinition(
        steps=tuple(steps),
        starts=registry.starts,
        roles=MappingProxyType(roles),
        direct_triggers=MappingProxyType(direct_triggers),
        gates=tuple(sorted(gates, key=lambda g: g.gate_id)),
        router_triggers=MappingProxyType(router_triggers),
    )


def _build_start_step(name: str, descs: dict[str, str | None]) -> StepInfo:
    """Build a :class:`StepInfo` for a ``@flow_start`` step.

    Args:
        name: The step method name.
        descs: Description lookup from the class namespace.

    Returns:
        A :class:`StepInfo` with role ``"start"`` and empty triggers.
    """
    return StepInfo(name=name, role="start", triggers=(), description=descs.get(name))


def _build_listener_step(
    name: str,
    trigger_specs: tuple[TriggerSpec, ...],
    descs: dict[str, str | None],
) -> tuple[StepInfo, list[str], list[GateInfo]]:
    """Build a :class:`StepInfo` and extracted gate info for a ``@flow_listen`` step.

    Args:
        name: The step method name.
        trigger_specs: Normalised trigger specs from the registry.
        descs: Description lookup from the class namespace.

    Returns:
        Triple of the :class:`StepInfo`, a list of plain-string trigger
        names (no gate triggers), and a list of :class:`GateInfo` entries
        for any AND/OR triggers in the spec.
    """
    plain: list[str] = []
    gates: list[GateInfo] = []
    for spec in trigger_specs:
        if isinstance(spec, str):
            plain.append(spec)
        else:
            kind = "and" if isinstance(spec, And) else "or"
            gates.append(
                GateInfo(
                    gate_id=_make_gate_id(name, kind, spec.triggers),
                    listener_name=name,
                    kind=kind,
                    triggers=frozenset(spec.triggers),
                )
            )
    step = StepInfo(name=name, role="listen", triggers=trigger_specs, description=descs.get(name))
    return step, plain, gates


def _build_router_step(
    name: str,
    trigger_specs: tuple[TriggerSpec, ...],
    descs: dict[str, str | None],
) -> tuple[StepInfo, list[str]]:
    """Build a :class:`StepInfo` and plain trigger names for a ``@flow_router`` step.

    Args:
        name: The step method name.
        trigger_specs: Normalised trigger specs from the registry.
        descs: Description lookup from the class namespace.

    Returns:
        Pair of the :class:`StepInfo` and a list of plain-string trigger
        names activating this router.

    Raises:
        FlowDefinitionError: When any trigger spec is a combinator gate —
            same rejection as :func:`build_transition_table`, because a
            gate-gated router could never execute and a silently-built
            definition would describe an unrunnable flow.
    """
    for spec in trigger_specs:
        if not isinstance(spec, str):
            raise FlowDefinitionError(
                f"@flow_router methods may only have string triggers; "
                f"got combinator on router {name!r}. "
                f"Use @flow_listen with a combinator and produce the route from a separate step.",
            )
    plain: list[str] = [spec for spec in trigger_specs if isinstance(spec, str)]
    step = StepInfo(name=name, role="router", triggers=trigger_specs, description=descs.get(name))
    return step, plain
