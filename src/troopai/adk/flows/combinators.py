"""Combinators — :class:`Or` / :class:`And` gates built via operator overloads.

Combinators are constructed by the ``|`` and ``&`` operators on
:class:`troopai.adk.flows.flow_wrappers.FlowStep` instances (the wrapped
form of decorated methods)::

    @flow_listen(method_a | method_b)             # Or fires once on first arrival
    @flow_listen(method_a | method_b | method_c)  # Flattened: Or(("a","b","c"))
    @flow_listen(method_a & method_b)             # And fires once after BOTH arrive
    @flow_listen(method_a & method_b & method_c)  # Flattened: And(("a","b","c"))

OR semantics: the gate fires exactly ONCE per flow run, on the first
trigger arrival. Subsequent arrivals of other triggers in the same run
do NOT re-fire the listener — the gate is consumed.

AND semantics: the gate fires exactly ONCE per flow run, after the LAST
required trigger has arrived. Arrivals are tracked by name; the
listener fires on the arrival that completes the required set.

Mixed-type chains (``Or & x`` or ``And | x``) raise :class:`TypeError`
to keep operator-chain semantics unambiguous. For mixed shapes, build
the gate directly with the dataclass constructor::

    @flow_listen(Or(triggers=("a", "b")))   # explicit construction
    @flow_listen(And(triggers=("a", "b", "c")))

This codebase deliberately rejects function-style ``or_()`` / ``and_()``
constructors that CrewAI provides — operators alone are the fluent API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Or:
    """OR-gate combinator for :func:`flow_listen`.

    Fires the gated listener ONCE on the first trigger arrival in the
    flow run. The :class:`troopai.adk.flows.executor.FlowExecutor`
    enforces single-fire via its ``consumed_gates`` set (shared by OR
    and AND gates).

    Built fluently via ``method_a | method_b`` (operator on
    :class:`FlowStep`); flattens left-associatively when chained. May
    also be constructed directly with ``Or(triggers=(...))`` when the
    triggers are pure strings (e.g., route labels).

    Attributes:
        triggers: Tuple of step-name strings, length >= 2. Duplicates
            are rejected. Single-trigger ``Or`` is rejected (use
            ``@flow_listen(step)`` directly).
    """

    triggers: tuple[str, ...]
    """Trigger names monitored by this gate."""

    def __post_init__(self) -> None:
        """Validate the trigger tuple.

        Raises:
            ValueError: When fewer than 2 triggers or when duplicates exist.
        """
        if len(self.triggers) < 2:
            raise ValueError(
                f"Or requires at least 2 triggers; got {len(self.triggers)}. "
                f"For a single trigger, use @flow_listen(step) directly.",
            )
        if len(set(self.triggers)) != len(self.triggers):
            raise ValueError(
                f"Or triggers must be unique; got duplicates in {self.triggers!r}",
            )

    def __or__(self, other: Any) -> Or:
        """Flatten ``self | other`` into a wider :class:`Or` gate.

        Args:
            other: Another :class:`Or` (merged), a :class:`FlowStep`, a
                step-name string, or any callable with ``__name__``.

        Returns:
            A new :class:`Or` covering the union of triggers.

        Raises:
            TypeError: When ``other`` is an :class:`And` — mixing types
                in operator chains is ambiguous.
        """
        if isinstance(other, And):
            raise TypeError(
                "Cannot combine Or with And via operators — "
                "mixed-type chains are ambiguous. Construct the desired gate directly.",
            )
        from troopai.adk.flows.flow_wrappers import name_of

        if isinstance(other, Or):
            new = tuple(t for t in other.triggers if t not in self.triggers)
            return Or(triggers=(*self.triggers, *new))
        new_name = name_of(other)
        if new_name in self.triggers:
            return self
        return Or(triggers=(*self.triggers, new_name))

    def __and__(self, other: Any) -> Any:
        """Reject AND-combination with this Or via operators.

        Raises:
            TypeError: Always. Mixed-type operator chains are forbidden.
        """
        del other
        raise TypeError(
            "Cannot combine Or with another trigger via & — "
            "mixed-type chains are ambiguous. Construct And(...) directly for explicit AND shapes.",
        )


@dataclass(frozen=True)
class And:
    """AND-gate combinator for :func:`flow_listen`.

    Fires the gated listener ONCE after every trigger has arrived in the
    flow run. The :class:`troopai.adk.flows.executor.FlowExecutor` tracks
    arrivals by name and fires on the arrival that completes the
    required set; the gate is consumed after firing.

    Built fluently via ``method_a & method_b`` (operator on
    :class:`FlowStep`); flattens left-associatively when chained. May
    also be constructed directly with ``And(triggers=(...))``.

    Attributes:
        triggers: Tuple of step-name strings, length >= 2. Duplicates
            are rejected. Single-trigger ``And`` is rejected (use
            ``@flow_listen(step)`` directly).
    """

    triggers: tuple[str, ...]
    """Trigger names monitored by this gate."""

    def __post_init__(self) -> None:
        """Validate the trigger tuple.

        Raises:
            ValueError: When fewer than 2 triggers or when duplicates exist.
        """
        if len(self.triggers) < 2:
            raise ValueError(
                f"And requires at least 2 triggers; got {len(self.triggers)}. "
                f"For a single trigger, use @flow_listen(step) directly.",
            )
        if len(set(self.triggers)) != len(self.triggers):
            raise ValueError(
                f"And triggers must be unique; got duplicates in {self.triggers!r}",
            )

    def __and__(self, other: Any) -> And:
        """Flatten ``self & other`` into a wider :class:`And` gate.

        Args:
            other: Another :class:`And` (merged), a :class:`FlowStep`, a
                step-name string, or any callable with ``__name__``.

        Returns:
            A new :class:`And` covering the union of triggers.

        Raises:
            TypeError: When ``other`` is an :class:`Or` — mixing types
                in operator chains is ambiguous.
        """
        if isinstance(other, Or):
            raise TypeError(
                "Cannot combine And with Or via operators — "
                "mixed-type chains are ambiguous. Construct the desired gate directly.",
            )
        from troopai.adk.flows.flow_wrappers import name_of

        if isinstance(other, And):
            new = tuple(t for t in other.triggers if t not in self.triggers)
            return And(triggers=(*self.triggers, *new))
        new_name = name_of(other)
        if new_name in self.triggers:
            return self
        return And(triggers=(*self.triggers, new_name))

    def __or__(self, other: Any) -> Any:
        """Reject OR-combination with this And via operators.

        Raises:
            TypeError: Always. Mixed-type operator chains are forbidden.
        """
        raise TypeError(
            "Cannot combine And with another trigger via | — "
            "mixed-type chains are ambiguous. Construct Or(...) directly for explicit OR shapes.",
        )
