"""Unit tests for ``_seed_barriers_from_checkpoint`` — barrier
reconstruction on resume from (produced_at, versions_seen)."""

from __future__ import annotations

from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.state import GraphState
from troopai.adk.orchestration.executable import NodeResult
from troopai.adk.run.graph_loop import (
    _build_join_barriers,
    _seed_barriers_from_checkpoint,
)
from troopai.adk.types.tokens.llm_usage import LLMUsage


def _noop() -> str:
    return "noop"


def _result(text: str) -> NodeResult:
    return NodeResult(output=text, usage=LLMUsage(), final_text=text)


def _linear_graph() -> Graph:
    return (
        Graph.new("seed-linear")
        .node("a", _noop)
        .node("b", _noop)
        .node("c", _noop)
        .edge("a", "b")
        .edge("b", "c")
        .entry("a")
        .terminal("c")
        .compile()
    )


async def test_seed_delivers_unconsumed_upstream() -> None:
    g = _linear_graph()
    state: GraphState = GraphState(graph=g)
    state.superstep = 1
    state.record("a", _result("a-out"))
    barriers = _build_join_barriers(g)
    await _seed_barriers_from_checkpoint(graph=g, state=state, barriers=barriers)
    assert barriers["b"].is_ready() is True
    assert barriers["c"].is_ready() is False


async def test_seed_skips_already_consumed_upstream() -> None:
    g = _linear_graph()
    state: GraphState = GraphState(graph=g)
    state.superstep = 1
    state.record("a", _result("a-out"))
    state.versions_seen = {"b": {"a": 1}}
    barriers = _build_join_barriers(g)
    await _seed_barriers_from_checkpoint(graph=g, state=state, barriers=barriers)
    assert barriers["b"].is_ready() is False


async def test_seed_redelivers_when_upstream_newer() -> None:
    g = _linear_graph()
    state: GraphState = GraphState(graph=g)
    state.superstep = 4
    state.record("a", _result("a-out-second"))
    state.versions_seen = {"b": {"a": 2}}
    barriers = _build_join_barriers(g)
    await _seed_barriers_from_checkpoint(graph=g, state=state, barriers=barriers)
    assert barriers["b"].is_ready() is True


async def test_seed_skips_unfired_upstream() -> None:
    g = _linear_graph()
    state: GraphState = GraphState(graph=g)
    barriers = _build_join_barriers(g)
    await _seed_barriers_from_checkpoint(graph=g, state=state, barriers=barriers)
    assert barriers["b"].is_ready() is False


async def test_seed_predicate_false_records_skip() -> None:
    g = (
        Graph.new("seed-pred")
        .node("a", _noop)
        .node("b", _noop)
        .edge("a", "b", when=lambda r: False)
        .entry("a")
        .terminal("b")
        .compile()
    )
    state: GraphState = GraphState(graph=g)
    state.superstep = 1
    state.record("a", _result("a-out"))
    barriers = _build_join_barriers(g)
    await _seed_barriers_from_checkpoint(graph=g, state=state, barriers=barriers)
    assert barriers["b"].is_ready() is False
