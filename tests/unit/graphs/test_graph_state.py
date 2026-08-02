"""Tests for ``GraphState`` — record, serialisation roundtrip, and schema.

Uses a real minimal ``Graph`` so ``from_dict`` / ``from_json`` can validate
node ids against the roster.
"""

from __future__ import annotations

import json

import pytest

from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.state import GraphState
from troopai.adk.orchestration.executable import NodeResult
from troopai.adk.types.tokens.llm_usage import LLMUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop() -> str:
    return "noop"


def _make_graph() -> Graph:
    return Graph.new("state-test").node("a", _noop).node("b", _noop).edge("a", "b").entry("a").terminal("b").compile()


def _result(text: str, input_tokens: int = 10, output_tokens: int = 5) -> NodeResult:
    usage = LLMUsage(
        requests=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    return NodeResult(output=text, usage=usage, final_text=text)


# ---------------------------------------------------------------------------
# record()
# ---------------------------------------------------------------------------


class TestGraphStateRecord:
    def test_record_updates_node_results(self) -> None:
        g = _make_graph()
        state = GraphState(graph=g)
        r = _result("hello")
        state.record("a", r)
        assert "a" in state.node_results
        assert state.node_results["a"] is r

    def test_record_accumulates_cumulative_usage(self) -> None:
        g = _make_graph()
        state = GraphState(graph=g)
        state.record("a", _result("x", input_tokens=10, output_tokens=5))
        state.record("b", _result("y", input_tokens=20, output_tokens=8))
        assert state.cumulative_usage.input_tokens == 30
        assert state.cumulative_usage.output_tokens == 13
        assert state.cumulative_usage.total_tokens == 43

    def test_record_sets_per_node_usage(self) -> None:
        g = _make_graph()
        state = GraphState(graph=g)
        state.record("a", _result("x", input_tokens=7, output_tokens=3))
        assert "a" in state.per_node_usage
        assert state.per_node_usage["a"].input_tokens == 7

    def test_record_accumulates_per_node_on_re_fire(self) -> None:
        """Second record for the same node adds to per_node_usage."""
        g = _make_graph()
        state = GraphState(graph=g)
        state.record("a", _result("x", input_tokens=5, output_tokens=2))
        state.record("a", _result("y", input_tokens=3, output_tokens=1))
        assert state.per_node_usage["a"].input_tokens == 8

    def test_record_extends_all_items(self) -> None:
        """new_items from NodeResult are appended to all_items."""
        g = _make_graph()
        state = GraphState(graph=g)
        # Callable results have empty new_items — just check it doesn't crash
        state.record("a", _result("hello"))
        assert isinstance(state.all_items, list)

    def test_record_sets_produced_at_to_current_superstep(self) -> None:
        g = _make_graph()
        state = GraphState(graph=g)
        state.superstep = 3
        state.record("a", _result("hello"))
        assert state.produced_at == {"a": 3}

    def test_record_overwrites_produced_at_on_refire(self) -> None:
        g = _make_graph()
        state = GraphState(graph=g)
        state.superstep = 1
        state.record("a", _result("first"))
        state.superstep = 4
        state.record("a", _result("second"))
        assert state.produced_at["a"] == 4


# ---------------------------------------------------------------------------
# mark_version_consumed
# ---------------------------------------------------------------------------


class TestMarkVersionConsumed:
    def test_marks_version_seen(self) -> None:
        g = _make_graph()
        state = GraphState(graph=g, superstep=2)
        state.mark_version_consumed("b", "a")
        assert state.versions_seen["b"]["a"] == 2


# ---------------------------------------------------------------------------
# to_dict / from_dict roundtrip
# ---------------------------------------------------------------------------


class TestToDictFromDict:
    def _state_with_records(self, g: Graph) -> GraphState:
        state = GraphState(graph=g, thread_id="thr-001", superstep=3)
        state.record("a", _result("from-a", input_tokens=10, output_tokens=4))
        state.record("b", _result("from-b", input_tokens=6, output_tokens=2))
        state.mark_version_consumed("b", "a")
        state.terminal_outputs["b"] = "from-b"
        state.final_output = "from-b"
        state.status = "completed"
        return state

    def test_roundtrip_superstep(self) -> None:
        g = _make_graph()
        state = self._state_with_records(g)
        restored = GraphState.from_dict(state.to_dict(), g)
        assert restored.superstep == state.superstep

    def test_roundtrip_thread_id(self) -> None:
        g = _make_graph()
        state = self._state_with_records(g)
        restored = GraphState.from_dict(state.to_dict(), g)
        assert restored.thread_id == "thr-001"

    def test_roundtrip_node_results(self) -> None:
        g = _make_graph()
        state = self._state_with_records(g)
        restored = GraphState.from_dict(state.to_dict(), g)
        assert "a" in restored.node_results
        assert "b" in restored.node_results
        assert restored.node_results["a"].final_text == "from-a"
        assert restored.node_results["b"].final_text == "from-b"

    def test_roundtrip_versions_seen(self) -> None:
        g = _make_graph()
        state = self._state_with_records(g)
        restored = GraphState.from_dict(state.to_dict(), g)
        assert restored.versions_seen.get("b", {}).get("a") == 3

    def test_roundtrip_terminal_outputs(self) -> None:
        g = _make_graph()
        state = self._state_with_records(g)
        restored = GraphState.from_dict(state.to_dict(), g)
        assert restored.terminal_outputs.get("b") == "from-b"

    def test_roundtrip_cumulative_usage(self) -> None:
        g = _make_graph()
        state = self._state_with_records(g)
        restored = GraphState.from_dict(state.to_dict(), g)
        assert restored.cumulative_usage.input_tokens == 16
        assert restored.cumulative_usage.output_tokens == 6

    def test_roundtrip_status(self) -> None:
        g = _make_graph()
        state = self._state_with_records(g)
        restored = GraphState.from_dict(state.to_dict(), g)
        assert restored.status == "completed"


# ---------------------------------------------------------------------------
# to_json / from_json roundtrip
# ---------------------------------------------------------------------------


class TestToJsonFromJson:
    def test_roundtrip_basic(self) -> None:
        g = _make_graph()
        state = GraphState(graph=g, thread_id="json-thr", superstep=1)
        state.record("a", _result("hello"))
        raw = state.to_json()
        restored = GraphState.from_json(raw, g)
        assert restored.superstep == 1
        assert restored.thread_id == "json-thr"

    def test_to_json_is_bare_dict_no_version_key(self) -> None:
        g = _make_graph()
        state = GraphState(graph=g)
        payload = json.loads(state.to_json())
        assert "_schema_version" not in payload
        assert "data" not in payload  # no envelope wrapper
        assert payload == state.to_dict()

    def test_from_json_rejects_invalid_json(self) -> None:
        g = _make_graph()
        with pytest.raises(json.JSONDecodeError):
            GraphState.from_json("not json {[}", g)


# ---------------------------------------------------------------------------
# produced_at serialisation
# ---------------------------------------------------------------------------


class TestProducedAtSerialisation:
    def test_produced_at_round_trips(self) -> None:
        g = _make_graph()
        state = GraphState(graph=g)
        state.superstep = 2
        state.record("a", _result("x"))
        restored = GraphState.from_dict(state.to_dict(), g)
        assert restored.produced_at == {"a": 2}

    def test_from_dict_rejects_unknown_produced_at_id(self) -> None:
        g = _make_graph()
        with pytest.raises(ValueError, match="produced_at has unknown node id"):
            GraphState.from_dict({"produced_at": {"ghost": 1}}, g)

    def test_from_dict_tolerates_missing_produced_at(self) -> None:
        g = _make_graph()
        restored = GraphState.from_dict({"superstep": 5}, g)
        assert restored.produced_at == {}


# ---------------------------------------------------------------------------
# nested_graph_snapshots (PA4 — depth-N nested-graph defer/resume)
# ---------------------------------------------------------------------------


class TestNestedGraphSnapshots:
    """Round-trip and back-compat for the new nested_graph_snapshots slot."""

    def test_from_dict_tolerates_missing_nested_graph_snapshots(self) -> None:
        """Legacy checkpoints (no nested_graph_snapshots key) load with an empty dict."""
        g = _make_graph()
        restored = GraphState.from_dict({"superstep": 0}, g)
        assert restored.nested_graph_snapshots == {}

    def test_nested_graph_snapshots_round_trips_via_to_dict(self) -> None:
        """A populated nested_graph_snapshots round-trips through to_dict/from_dict."""
        inner = Graph.new("inner-rt").node("x", _noop).entry("x").terminal("x").compile()
        outer = Graph.new("outer-rt").node("g", inner).entry("g").terminal("g").compile()
        outer_state = GraphState(graph=outer)
        # Park an inner GraphState at the outer node "g".
        inner_state = GraphState(graph=inner)
        inner_state.superstep = 3
        outer_state.nested_graph_snapshots["g"] = inner_state

        restored = GraphState.from_dict(outer_state.to_dict(), outer)
        assert "g" in restored.nested_graph_snapshots
        assert restored.nested_graph_snapshots["g"].superstep == 3

    def test_from_dict_rejects_nested_graph_snapshot_on_non_graph_node(self) -> None:
        """Cross-reference check: nested_graph_snapshots key must point to a Graph-backed node."""
        # `_make_graph()` has nodes a + b, both callables — neither is a Graph.
        g = _make_graph()
        bad_payload = {
            "superstep": 0,
            "nested_graph_snapshots": {"a": {"superstep": 1}},  # "a" is a plain callable
        }
        with pytest.raises(ValueError, match="not a Graph"):
            GraphState.from_dict(bad_payload, g)
