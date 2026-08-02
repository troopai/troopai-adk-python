"""Unit tests for tool_search.ToolSearchState reveal/reset semantics.

Focus: reveals performed inside an ``asyncio.gather``-spawned task must
survive back to the parent context. ``gather`` copies the context for
each spawned task, so a reveal that rebinds a ContextVar inside the task
is lost to the parent — which silently defeats ``tool_search`` whenever
the LLM emits it in a parallel tool batch. The state must instead mutate
a set the parent already bound (via ``reset()``) so in-place changes are
visible after the batch completes, while preserving per-run isolation.
"""

from __future__ import annotations

import asyncio
import json
from contextvars import copy_context
from typing import Any

from troopai.adk.tools import build_tool_search
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.tools.tool_search import ToolSearchState

MINIMAL_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def _make_tool(name: str, **overrides: Any) -> FunctionTool:
    defaults: dict[str, Any] = {
        "name": name,
        "schema": MINIMAL_SCHEMA,
        "description": f"description for {name}",
    }
    defaults.update(overrides)
    return FunctionTool(**defaults)


# ── Reveals survive a parallel (gather-spawned) tool batch ───────────


class TestRevealAcrossParallelBatch:
    """``reveal()`` inside a gather task must reach the parent context.

    Mirrors the parallel-tool-execution path: the Runner resets the
    revealed set in the run context, then ``execute_tool_calls`` spawns
    each tool via ``asyncio.gather`` (one Task per call, each with a
    copied context). When the gather-spawned ``tool_search`` reveals a
    tool, the parent context must observe it once the batch completes.
    """

    async def test_direct_reveal_in_gather_task_visible_to_parent(self) -> None:
        rare = _make_tool("rare_parallel", defer_loading=True)
        search = build_tool_search([rare])
        state = search.get_search_state()
        assert state is not None

        # Runner.arun() resets in the run context before launching the
        # parallel tool batch.
        state.reset()

        async def reveal_in_task(name: str) -> None:
            state.reveal(name)

        # asyncio.gather wraps each coroutine in its own Task, copying the
        # current context. With a frozenset rebind, the reveal would be
        # lost to the parent; with in-place mutation of the bound set it
        # survives.
        await asyncio.gather(reveal_in_task("rare_parallel"))

        assert "rare_parallel" in state.revealed, (
            "reveal() performed inside a gather-spawned task was lost to "
            "the parent context — tool_search is defeated in a parallel batch"
        )

    async def test_multiple_reveals_in_gather_batch_visible_to_parent(self) -> None:
        a = _make_tool("rare_a", defer_loading=True)
        b = _make_tool("rare_b", defer_loading=True)
        search = build_tool_search([a, b])
        state = search.get_search_state()
        assert state is not None

        state.reset()

        async def reveal_in_task(name: str) -> None:
            state.reveal(name)

        await asyncio.gather(
            reveal_in_task("rare_a"),
            reveal_in_task("rare_b"),
        )

        assert state.revealed == frozenset({"rare_a", "rare_b"}), (
            "reveals from sibling gather tasks did not all reach the parent"
        )

    async def test_on_invoke_in_gather_task_visible_to_parent(self) -> None:
        """End-to-end: the search tool's on_invoke, run inside a gather
        task, reveals the matched tool to the parent context."""
        rare = _make_tool("weather_lookup", defer_loading=True)
        search = build_tool_search([rare])
        state = search.get_search_state()
        assert state is not None

        state.reset()

        raw_args = json.dumps({"query": "weather", "top_k": 5})

        async def call_search() -> str:
            return await search.on_invoke(None, raw_args)  # type: ignore[arg-type]

        results = await asyncio.gather(call_search())
        decoded = json.loads(results[0])
        assert any(entry["name"] == "weather_lookup" for entry in decoded)

        assert "weather_lookup" in state.revealed, (
            "tool_search.on_invoke run in a parallel batch failed to reveal the matched tool to the parent context"
        )


# ── Per-run isolation is preserved by the in-place-mutation design ───


class TestPerRunIsolationPreserved:
    """The mutable-set design must not weaken cross-run isolation."""

    def test_snapshot_before_reveal_stays_empty(self) -> None:
        """A context snapshot taken before any reveal must not observe a
        later reveal in the originating context."""
        rare = _make_tool("rare_iso", defer_loading=True)
        search = build_tool_search([rare])
        state = search.get_search_state()
        assert state is not None

        # Snapshot the context before any reveal — simulates a concurrent
        # run scheduled before this run revealed anything.
        snapshot = copy_context()

        state.reveal("rare_iso")
        assert "rare_iso" in state.revealed

        seen: list[frozenset[str]] = []
        snapshot.run(lambda: seen.append(state.revealed))
        assert "rare_iso" not in seen[0], (
            "a reveal leaked into a context snapshot taken before it — per-run isolation is broken"
        )

    async def test_concurrent_runs_do_not_see_each_other(self) -> None:
        """Two concurrent runs (each its own task) that reset-then-reveal
        must each see only their own reveal."""
        rare = _make_tool("rare_conc", defer_loading=True)
        search = build_tool_search([rare])
        state = search.get_search_state()
        assert state is not None

        async def run_once(name: str) -> frozenset[str]:
            state.reset()
            await asyncio.sleep(0)
            state.reveal(name)
            await asyncio.sleep(0)
            return state.revealed

        results = await asyncio.gather(run_once("tool_x"), run_once("tool_y"))
        assert results[0] == frozenset({"tool_x"})
        assert results[1] == frozenset({"tool_y"})

    def test_revealed_returns_immutable_snapshot(self) -> None:
        """The read API returns a frozenset; mutating the result must not
        affect the backing set."""
        rare = _make_tool("rare_frozen", defer_loading=True)
        search = build_tool_search([rare])
        state = search.get_search_state()
        assert state is not None

        state.reset()
        state.reveal("rare_frozen")
        snapshot = state.revealed
        assert isinstance(snapshot, frozenset)
        # frozenset has no mutators; confirm a fresh read is independent.
        state.reveal("another")
        assert "another" not in snapshot
        assert "another" in state.revealed

    def test_reset_clears_then_allows_new_reveal(self) -> None:
        state = ToolSearchState(deferred={})
        state.reveal("first")
        assert "first" in state.revealed
        state.reset()
        assert state.revealed == frozenset()
        state.reveal("second")
        assert state.revealed == frozenset({"second"})

    def test_reveal_before_reset_is_isolated_to_context(self) -> None:
        """A reveal reached before any reset (lazy bind) is still
        context-local — a pre-bind snapshot must not see it."""
        state = ToolSearchState(deferred={})
        snapshot = copy_context()
        state.reveal("lazy")
        assert "lazy" in state.revealed
        seen: list[frozenset[str]] = []
        snapshot.run(lambda: seen.append(state.revealed))
        assert seen[0] == frozenset()
