"""Unit tests for :class:`Or` / :class:`And` combinators.

Covers construction validation, the ``__or__`` / ``__and__`` operator
overloads, and the mixed-type rejection contract.
"""

from __future__ import annotations

import pytest

from troopai.adk.flows.combinators import And, Or


class TestOrConstruction:
    def test_two_distinct_triggers_ok(self) -> None:
        or_gate = Or(triggers=("a", "b"))
        assert or_gate.triggers == ("a", "b")

    def test_three_distinct_triggers_ok(self) -> None:
        or_gate = Or(triggers=("a", "b", "c"))
        assert or_gate.triggers == ("a", "b", "c")

    def test_single_trigger_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 2 triggers"):
            Or(triggers=("only_one",))

    def test_empty_triggers_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 2 triggers"):
            Or(triggers=())

    def test_duplicates_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            Or(triggers=("a", "a"))


class TestAndConstruction:
    def test_two_distinct_triggers_ok(self) -> None:
        and_gate = And(triggers=("a", "b"))
        assert and_gate.triggers == ("a", "b")

    def test_single_trigger_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 2 triggers"):
            And(triggers=("only_one",))

    def test_duplicates_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            And(triggers=("x", "x"))


class TestOrOperators:
    def test_or_or_string_flattens(self) -> None:
        gate = Or(triggers=("a", "b")) | "c"
        assert gate.triggers == ("a", "b", "c")

    def test_or_or_or_flattens(self) -> None:
        gate = Or(triggers=("a", "b")) | Or(triggers=("c", "d"))
        assert gate.triggers == ("a", "b", "c", "d")

    def test_or_or_existing_trigger_idempotent(self) -> None:
        gate = Or(triggers=("a", "b")) | "a"
        assert gate.triggers == ("a", "b")

    def test_or_and_raises(self) -> None:
        with pytest.raises(TypeError, match="mixed-type"):
            _ = Or(triggers=("a", "b")) & "c"

    def test_or_or_and_raises(self) -> None:
        with pytest.raises(TypeError, match="mixed-type"):
            _ = Or(triggers=("a", "b")) | And(triggers=("c", "d"))


class TestAndOperators:
    def test_and_and_string_flattens(self) -> None:
        gate = And(triggers=("a", "b")) & "c"
        assert gate.triggers == ("a", "b", "c")

    def test_and_and_and_flattens(self) -> None:
        gate = And(triggers=("a", "b")) & And(triggers=("c", "d"))
        assert gate.triggers == ("a", "b", "c", "d")

    def test_and_and_existing_trigger_idempotent(self) -> None:
        gate = And(triggers=("a", "b")) & "a"
        assert gate.triggers == ("a", "b")

    def test_and_or_raises(self) -> None:
        with pytest.raises(TypeError, match="mixed-type"):
            _ = And(triggers=("a", "b")) | "c"

    def test_and_and_or_raises(self) -> None:
        with pytest.raises(TypeError, match="mixed-type"):
            _ = And(triggers=("a", "b")) & Or(triggers=("c", "d"))
