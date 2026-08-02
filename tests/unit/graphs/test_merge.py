"""Tests for the five built-in merge strategies.

Merge strategies are pure functions; no I/O, no LLM calls.  We construct
``NodeResult`` fixtures directly and verify the string output each strategy
produces.
"""

from __future__ import annotations

from troopai.adk.graphs.merge import DEFAULT_MERGE, Merge
from troopai.adk.orchestration.executable import NodeResult
from troopai.adk.types.tokens.llm_usage import LLMUsage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _result(text: str) -> NodeResult:
    """Build a minimal ``NodeResult`` with only ``final_text`` set."""
    return NodeResult(
        output=text,
        usage=LLMUsage(),
        final_text=text,
    )


_R1 = _result("alpha text")
_R2 = _result("beta text")

_SOURCES = ["src-a", "src-b"]


# ---------------------------------------------------------------------------
# last_wins
# ---------------------------------------------------------------------------


class TestLastWins:
    def test_returns_last_result_text(self) -> None:
        out = Merge.last_wins([_R1, _R2], _SOURCES)
        assert out == "beta text"

    def test_single_item_returns_it(self) -> None:
        out = Merge.last_wins([_R1], ["src-a"])
        assert out == "alpha text"

    def test_empty_returns_empty_string(self) -> None:
        out = Merge.last_wins([], [])
        assert out == ""

    def test_sources_arg_ignored(self) -> None:
        """last_wins does not use sources — verify no KeyError."""
        out = Merge.last_wins([_R1, _R2], [])
        assert out == "beta text"


# ---------------------------------------------------------------------------
# first_wins
# ---------------------------------------------------------------------------


class TestFirstWins:
    def test_returns_first_result_text(self) -> None:
        out = Merge.first_wins([_R1, _R2], _SOURCES)
        assert out == "alpha text"

    def test_single_item_returns_it(self) -> None:
        out = Merge.first_wins([_R2], ["src-b"])
        assert out == "beta text"

    def test_empty_returns_empty_string(self) -> None:
        out = Merge.first_wins([], [])
        assert out == ""


# ---------------------------------------------------------------------------
# concat_text
# ---------------------------------------------------------------------------


class TestConcatText:
    def test_joins_with_labels(self) -> None:
        out = Merge.concat_text([_R1, _R2], _SOURCES)
        assert isinstance(out, str)
        assert "[src-a]" in out
        assert "alpha text" in out
        assert "[src-b]" in out
        assert "beta text" in out

    def test_empty_text_results_omitted(self) -> None:
        empty_result = NodeResult(output="", final_text="", usage=LLMUsage())
        out = Merge.concat_text([empty_result, _R2], ["empty-src", "src-b"])
        assert isinstance(out, str)
        assert "[empty-src]" not in out
        assert "[src-b]" in out
        assert "beta text" in out

    def test_single_item(self) -> None:
        out = Merge.concat_text([_R1], ["src-a"])
        assert isinstance(out, str)
        assert out == "[src-a]\nalpha text"

    def test_empty_list(self) -> None:
        out = Merge.concat_text([], [])
        assert out == ""


# ---------------------------------------------------------------------------
# extend_items
# ---------------------------------------------------------------------------


class TestExtendItems:
    def test_returns_list(self) -> None:
        out = Merge.extend_items([_R1, _R2], _SOURCES)
        # Callables return empty new_items → flattened list is empty or Layer 1 params
        assert isinstance(out, list)

    def test_no_items_yields_empty_list(self) -> None:
        """Callable-backed results have empty new_items — extend flattens to []."""
        out = Merge.extend_items([_R1, _R2], _SOURCES)
        assert len(out) == 0

    def test_empty_results_yields_empty_list(self) -> None:
        out = Merge.extend_items([], [])
        assert out == []


# ---------------------------------------------------------------------------
# custom
# ---------------------------------------------------------------------------


class TestCustomMerge:
    def test_custom_wraps_and_returns_fn(self) -> None:
        def my_merge(results: list, sources: list) -> str:
            del results, sources
            return "custom-output"

        wrapped = Merge.custom(my_merge)
        out = wrapped([_R1, _R2], _SOURCES)
        assert out == "custom-output"

    def test_custom_lambda(self) -> None:
        fn = Merge.custom(lambda results, _sources: f"count={len(results)}")
        out = fn([_R1, _R2], _SOURCES)
        assert out == "count=2"


# ---------------------------------------------------------------------------
# DEFAULT_MERGE
# ---------------------------------------------------------------------------


class TestDefaultMerge:
    def test_default_merge_is_concat_text(self) -> None:
        """DEFAULT_MERGE MUST be Merge.concat_text per the module docstring."""
        assert DEFAULT_MERGE is Merge.concat_text
