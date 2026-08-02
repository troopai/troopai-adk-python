"""Tests for ``InMemoryCheckpointer`` — hook-provider contract and CRUD.

Covers:
- save + load roundtrip.
- load returns None for unknown thread_id.
- load raises ValueError when graph_id mismatches.
- register() wires into a HookRegistry so on_node_end triggers save.
"""

from __future__ import annotations

import pytest

from troopai.adk.graphs.checkpointer import GraphCheckpoint
from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.hooks import HookRegistry
from troopai.adk.graphs.state import GraphState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop() -> str:
    return "noop"


def _make_graph(graph_id: str = "cp-test-graph") -> Graph:
    return Graph.new(graph_id).node("a", _noop).entry("a").terminal("a").compile()


def _make_checkpoint(
    thread_id: str = "thr-1",
    graph_id: str = "cp-test-graph",
    superstep: int = 1,
) -> GraphCheckpoint:
    state_dict: dict = {
        "thread_id": thread_id,
        "superstep": superstep,
        "node_results": {},
        "versions_seen": {},
        "all_items": [],
        "cumulative_usage": {"requests": 0, "total_tokens": 0, "input_tokens": 0, "output_tokens": 0},
        "per_node_usage": {},
        "terminal_outputs": {},
        "final_output": None,
        "status": "running",
        "error": None,
    }
    return GraphCheckpoint(
        thread_id=thread_id,
        graph_id=graph_id,
        state=state_dict,
        superstep=superstep,
    )


# ---------------------------------------------------------------------------
# Save + Load roundtrip
# ---------------------------------------------------------------------------


class TestSaveAndLoad:
    @pytest.mark.asyncio
    async def test_save_and_load_returns_state(self) -> None:
        cp = InMemoryCheckpointer()
        g = _make_graph()
        ckpt = _make_checkpoint()
        await cp.save(ckpt)
        restored = await cp.load("thr-1", g)
        assert restored is not None
        assert restored.superstep == 1

    @pytest.mark.asyncio
    async def test_load_unknown_thread_id_returns_none(self) -> None:
        cp = InMemoryCheckpointer()
        g = _make_graph()
        result = await cp.load("does-not-exist", g)
        assert result is None

    @pytest.mark.asyncio
    async def test_load_graph_id_mismatch_raises(self) -> None:
        cp = InMemoryCheckpointer()
        # Save checkpoint for "cp-test-graph"
        ckpt = _make_checkpoint(graph_id="cp-test-graph")
        await cp.save(ckpt)
        # Load with a different graph
        other_graph = _make_graph("other-graph-id")
        with pytest.raises(ValueError, match="graph_id"):
            await cp.load("thr-1", other_graph)

    @pytest.mark.asyncio
    async def test_later_save_overwrites_earlier(self) -> None:
        cp = InMemoryCheckpointer()
        g = _make_graph()
        await cp.save(_make_checkpoint(superstep=1))
        await cp.save(_make_checkpoint(superstep=5))
        restored = await cp.load("thr-1", g)
        assert restored is not None
        assert restored.superstep == 5


# ---------------------------------------------------------------------------
# list_checkpoints + delete
# ---------------------------------------------------------------------------


class TestListAndDelete:
    @pytest.mark.asyncio
    async def test_list_checkpoints_returns_thread_ids(self) -> None:
        cp = InMemoryCheckpointer()
        await cp.save(_make_checkpoint(thread_id="t1"))
        await cp.save(_make_checkpoint(thread_id="t2"))
        threads = await cp.list_checkpoints()
        assert "t1" in threads
        assert "t2" in threads

    @pytest.mark.asyncio
    async def test_list_checkpoints_empty(self) -> None:
        cp = InMemoryCheckpointer()
        threads = await cp.list_checkpoints()
        assert threads == []

    @pytest.mark.asyncio
    async def test_delete_removes_checkpoint(self) -> None:
        cp = InMemoryCheckpointer()
        g = _make_graph()
        await cp.save(_make_checkpoint())
        await cp.delete("thr-1")
        assert await cp.load("thr-1", g) is None

    @pytest.mark.asyncio
    async def test_delete_missing_is_noop(self) -> None:
        cp = InMemoryCheckpointer()
        # Must not raise
        await cp.delete("does-not-exist")


# ---------------------------------------------------------------------------
# Hook-provider contract
# ---------------------------------------------------------------------------


class TestHookProviderContract:
    def test_register_adds_to_registry(self) -> None:
        """register() must attach a GraphHooks impl so the registry can fan-out."""
        cp = InMemoryCheckpointer()
        registry = HookRegistry()
        cp.register(registry)
        # The registry should now have at least one hook
        assert len(registry._hooks) >= 1

    @pytest.mark.asyncio
    async def test_on_node_end_triggers_save(self) -> None:
        """Firing on_node_end via the registry must call cp.save when thread_id
        is set on the state."""
        from troopai.adk.orchestration.executable import NodeResult

        cp = InMemoryCheckpointer()
        registry = HookRegistry()
        cp.register(registry)

        g = _make_graph()
        state = GraphState(graph=g, thread_id="hook-thr", superstep=1)

        # Fire the hook — the checkpointer should persist state
        await registry.on_node_end(
            context=None,  # type: ignore[arg-type]
            state=state,
            node_id="a",
            result=NodeResult(output="x"),
        )

        restored = await cp.load("hook-thr", g)
        assert restored is not None

    @pytest.mark.asyncio
    async def test_on_node_end_no_save_when_no_thread_id(self) -> None:
        """When state.thread_id is None the checkpointer must skip saving."""
        from troopai.adk.orchestration.executable import NodeResult

        cp = InMemoryCheckpointer()
        registry = HookRegistry()
        cp.register(registry)

        g = _make_graph()
        state = GraphState(graph=g, thread_id=None, superstep=1)

        await registry.on_node_end(
            context=None,  # type: ignore[arg-type]
            state=state,
            node_id="a",
            result=NodeResult(output="x"),
        )

        threads = await cp.list_checkpoints()
        assert len(threads) == 0
