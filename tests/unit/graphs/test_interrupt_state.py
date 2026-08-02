"""GraphState + interrupt/resume types — serialization + status allowlist."""

from __future__ import annotations

import json

import pytest

from troopai.adk.exceptions import TroopAIError
from troopai.adk.graphs import GraphResume, Interrupt, InterruptException
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.result import GraphRunStatus
from troopai.adk.graphs.state import GraphState


def _trivial_graph() -> Graph:
    return Graph.new("t1").node("a", lambda: "a-done").entry("a").terminal("a").compile()


def test_interrupt_exception_inherits_troopai_error() -> None:
    exc = InterruptException(Interrupt(node_id="a", question="approve?"))
    assert isinstance(exc, TroopAIError)
    assert exc.interrupt.node_id == "a"
    assert exc.interrupt.question == "approve?"


def test_graph_resume_frozen_dataclass_defaults() -> None:
    r = GraphResume()
    assert r.replies == {} and r.rejected == {}
    r2 = GraphResume(replies={"a": "yes"}, rejected={"b": "denied"})
    assert r2.replies == {"a": "yes"} and r2.rejected == {"b": "denied"}
    with pytest.raises((AttributeError, Exception)):
        r2.replies = {}  # type: ignore[misc]  # frozen


def test_graph_run_status_interrupted_value() -> None:
    assert GraphRunStatus.INTERRUPTED.value == "interrupted"


def test_graph_state_pending_interrupts_round_trip() -> None:
    g = _trivial_graph()
    s = GraphState(graph=g)
    s.pending_interrupts["a"] = Interrupt(
        node_id="a",
        question="approve?",
        kind="tool_approval",
        metadata={"tool": "x"},
    )
    s.status = "interrupted"
    d = s.to_dict()
    # no version key — same invariant as agent RunState
    assert "version" not in d and "schema_version" not in d
    restored = GraphState.from_dict(d, g)
    assert restored.status == "interrupted"
    assert restored.pending_interrupts == s.pending_interrupts
    # JSON round-trip (to_json/from_json must work identically)
    txt = s.to_json()
    assert isinstance(txt, str) and "version" not in json.loads(txt)
    restored2 = GraphState.from_json(txt, g)
    assert restored2.pending_interrupts == s.pending_interrupts


def test_graph_state_tolerant_loader_unknown_keys_ignored_and_missing_pending_defaults_empty() -> None:
    g = _trivial_graph()
    s = GraphState(graph=g)
    d = s.to_dict()
    d["future_field_we_dont_know"] = "ignored"  # tolerant
    d.pop("pending_interrupts", None)  # older payload without the field
    restored = GraphState.from_dict(d, g)
    assert restored.pending_interrupts == {}


def test_graph_run_result_carries_interrupts_field() -> None:
    from troopai.adk.graphs.result import GraphRunResult, GraphRunResultStreaming

    r = GraphRunResult(
        final_output=None,
        status=GraphRunStatus.COMPLETED,
        user_prompt="x",
    )
    assert r.interrupts == ()
    rs = GraphRunResultStreaming()
    assert rs.interrupts == ()
