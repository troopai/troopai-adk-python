"""Tests for ``JoinBarrier`` — AND/OR semantics, cycle-reset, and peek.

No I/O or LLM calls — the barrier is a pure in-memory data structure.
"""

from __future__ import annotations

from troopai.adk.graphs.join import JoinBarrier, JoinSemantics
from troopai.adk.orchestration.executable import NodeResult
from troopai.adk.types.tokens.llm_usage import LLMUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(text: str) -> NodeResult:
    return NodeResult(output=text, usage=LLMUsage(), final_text=text)


# ---------------------------------------------------------------------------
# AND semantics
# ---------------------------------------------------------------------------


class TestAndSemantics:
    def _barrier(self) -> JoinBarrier:
        return JoinBarrier(
            target="target",
            expected=frozenset({"a", "b"}),
            semantics=JoinSemantics.AND,
        )

    def test_not_ready_after_single_arrival(self) -> None:
        b = self._barrier()
        b.record("a", _result("from-a"))
        assert b.is_ready() is False

    def test_ready_after_both_arrivals(self) -> None:
        b = self._barrier()
        b.record("a", _result("from-a"))
        b.record("b", _result("from-b"))
        assert b.is_ready() is True

    def test_consume_returns_ordered_results_and_sources(self) -> None:
        b = self._barrier()
        b.record("b", _result("from-b"))
        b.record("a", _result("from-a"))
        results, sources = b.consume()
        # Sorted by source id: "a" < "b"
        assert sources == ["a", "b"]
        assert results[0].final_text == "from-a"
        assert results[1].final_text == "from-b"

    def test_consume_resets_arrivals(self) -> None:
        b = self._barrier()
        b.record("a", _result("x"))
        b.record("b", _result("y"))
        b.consume()
        assert b.is_ready() is False
        assert len(b.arrivals) == 0

    def test_second_cycle_after_reset(self) -> None:
        """After consume() the barrier must accept a fresh round."""
        b = self._barrier()
        b.record("a", _result("r1"))
        b.record("b", _result("r2"))
        b.consume()
        # Second round
        b.record("a", _result("r3"))
        assert b.is_ready() is False
        b.record("b", _result("r4"))
        assert b.is_ready() is True
        results, sources = b.consume()
        assert sources == ["a", "b"]
        assert results[0].final_text == "r3"
        assert results[1].final_text == "r4"

    def test_unexpected_source_ignored(self) -> None:
        """Arrivals from sources not in expected are silently dropped."""
        b = self._barrier()
        b.record("unknown", _result("x"))
        assert len(b.arrivals) == 0
        assert b.is_ready() is False


# ---------------------------------------------------------------------------
# OR semantics
# ---------------------------------------------------------------------------


class TestOrSemantics:
    def _barrier(self) -> JoinBarrier:
        return JoinBarrier(
            target="target",
            expected=frozenset({"a", "b", "c"}),
            semantics=JoinSemantics.OR,
        )

    def test_ready_after_single_arrival(self) -> None:
        b = self._barrier()
        b.record("a", _result("from-a"))
        assert b.is_ready() is True

    def test_ready_after_multiple_arrivals(self) -> None:
        b = self._barrier()
        b.record("a", _result("x"))
        b.record("b", _result("y"))
        assert b.is_ready() is True

    def test_not_ready_with_empty_arrivals(self) -> None:
        b = self._barrier()
        assert b.is_ready() is False


# ---------------------------------------------------------------------------
# record_skip — conditional-edge integration
# ---------------------------------------------------------------------------


class TestRecordSkip:
    """AND-join must not deadlock when an upstream conditional edge skips.

    Before the fix: a graph with two fan-ins where one edge has
    ``when=lambda r: False`` would leave the AND-barrier waiting
    forever for the skipped source. The loop would exit with
    ``NO_READY_NODES`` on a graph the developer considers correct.
    """

    def _barrier(self, semantics: JoinSemantics = JoinSemantics.AND) -> JoinBarrier:
        return JoinBarrier(
            target="target",
            expected=frozenset({"a", "b"}),
            semantics=semantics,
        )

    def test_and_ready_when_one_arrives_one_skips(self) -> None:
        b = self._barrier()
        b.record("a", _result("from-a"))
        b.record_skip("b")
        assert b.is_ready() is True

    def test_and_not_ready_when_all_skip(self) -> None:
        """All skipped = no real input = should not fire."""
        b = self._barrier()
        b.record_skip("a")
        b.record_skip("b")
        assert b.is_ready() is False

    def test_and_not_ready_with_partial_skip_no_arrival(self) -> None:
        b = self._barrier()
        b.record_skip("a")
        assert b.is_ready() is False

    def test_record_after_skip_clears_skip(self) -> None:
        """A later real arrival supersedes an earlier skip for the same source."""
        b = self._barrier()
        b.record_skip("a")
        b.record("a", _result("from-a"))
        assert "a" not in b.skipped
        assert "a" in b.arrivals

    def test_skip_after_arrival_is_noop(self) -> None:
        """Once a source has a real result, skip must not overwrite it."""
        b = self._barrier()
        b.record("a", _result("from-a"))
        b.record_skip("a")
        assert "a" in b.arrivals
        assert "a" not in b.skipped

    def test_skip_from_unexpected_source_ignored(self) -> None:
        b = self._barrier()
        b.record_skip("not-expected")
        assert len(b.skipped) == 0

    def test_consume_resets_skipped(self) -> None:
        b = self._barrier()
        b.record("a", _result("x"))
        b.record_skip("b")
        b.consume()
        assert len(b.skipped) == 0

    def test_or_semantics_unaffected_by_skip(self) -> None:
        """OR-join fires on any arrival; skips are recorded but not required."""
        b = self._barrier(JoinSemantics.OR)
        b.record_skip("a")
        assert b.is_ready() is False
        b.record("b", _result("y"))
        assert b.is_ready() is True


# ---------------------------------------------------------------------------
# record_fail followed by record_skip — transient predicate error in a cycle
# ---------------------------------------------------------------------------


class TestFailThenSkip:
    """A transient predicate error must not permanently lock an AND-join.

    Cyclic scenario: in superstep N the conditional edge A -> T raises,
    so ``record_fail("A")`` runs and the barrier stays un-ready (failed
    is non-empty), so the target never fires and ``consume()`` never
    clears the sets. In a later superstep the cycle re-fires A and the
    predicate cleanly returns ``False``, so ``record_skip("A")`` runs.
    A clean skip must clear the stale fail mark — otherwise ``is_ready``
    short-circuits on the non-empty ``failed`` set forever.
    """

    def _barrier(self, semantics: JoinSemantics = JoinSemantics.AND) -> JoinBarrier:
        return JoinBarrier(
            target="target",
            expected=frozenset({"a", "b"}),
            semantics=semantics,
        )

    def test_skip_clears_prior_fail(self) -> None:
        b = self._barrier()
        b.record_fail("a")
        assert "a" in b.failed
        b.record_skip("a")
        assert "a" not in b.failed
        assert "a" in b.skipped

    def test_and_ready_after_fail_then_skip_with_real_arrival(self) -> None:
        """fail("a") -> skip("a"), then a real arrival on "b" must fire."""
        b = self._barrier()
        b.record_fail("a")
        assert b.is_ready() is False
        b.record_skip("a")
        b.record("b", _result("from-b"))
        assert b.is_ready() is True
        results, sources = b.consume()
        assert sources == ["b"]
        assert results[0].final_text == "from-b"
