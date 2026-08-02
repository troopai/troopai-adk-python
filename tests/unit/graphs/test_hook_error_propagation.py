"""Unit tests for ``propagate_errors`` on :class:`GraphHooks`.

Covers:
- A hook with ``propagate_errors=True`` causes the registry fan-out to
  re-raise, so the error reaches the caller.
- A hook with ``propagate_errors=False`` (the default) has its error
  logged and swallowed — the registry call completes normally.
- :class:`CheckpointerHooks` declares ``propagate_errors = True`` so
  a failed checkpointer save propagates to the caller.
- End-to-end path: a fake checkpointer whose ``save`` raises is
  registered on a :class:`HookRegistry`; firing ``on_node_end``
  propagates the error.
"""

from __future__ import annotations

from typing import Any, override

import pytest

from troopai.adk.graphs.checkpointers.hooks import CheckpointerHooks
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.hooks import GraphHooks, HookRegistry
from troopai.adk.graphs.state import GraphState
from troopai.adk.orchestration.executable import NodeResult
from troopai.adk.run.context import RunContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _noop() -> str:
    return "noop"


def _make_graph() -> Graph:
    return Graph.new("prop-test-graph").node("a", _noop).entry("a").terminal("a").compile()


def _make_ctx() -> RunContext[None]:
    return RunContext(context=None)  # type: ignore[arg-type]  # test scaffolding: hooks ignore the context payload


class _CriticalHook(GraphHooks[Any]):
    """Hook with ``propagate_errors=True`` that always raises."""

    propagate_errors = True

    @override
    async def on_node_end(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        node_id: str,
        result: NodeResult,
    ) -> None:
        del context, state, node_id, result
        raise RuntimeError("boom")


class _ObserverHook(GraphHooks[Any]):
    """Hook with ``propagate_errors=False`` (default) that always raises."""

    @override
    async def on_node_end(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        node_id: str,
        result: NodeResult,
    ) -> None:
        del context, state, node_id, result
        raise RuntimeError("observer-boom")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_critical_hook_propagates_error() -> None:
    """A hook with ``propagate_errors=True`` causes the fan-out to re-raise."""
    registry = HookRegistry()
    registry.add(_CriticalHook())

    g = _make_graph()
    state = GraphState(graph=g, thread_id="t1", superstep=1)

    with pytest.raises(RuntimeError, match="boom"):
        await registry.on_node_end(
            context=_make_ctx(),
            state=state,
            node_id="a",
            result=NodeResult(output="x"),
        )


async def test_observer_hook_error_is_swallowed() -> None:
    """A hook with ``propagate_errors=False`` has its error swallowed."""
    registry = HookRegistry()
    registry.add(_ObserverHook())

    g = _make_graph()
    state = GraphState(graph=g, thread_id="t2", superstep=1)

    # Must NOT raise — the observer error is logged and discarded.
    await registry.on_node_end(
        context=_make_ctx(),
        state=state,
        node_id="a",
        result=NodeResult(output="x"),
    )


def test_checkpointer_hooks_is_critical() -> None:
    """``CheckpointerHooks`` must have ``propagate_errors = True``."""
    assert CheckpointerHooks.propagate_errors is True


async def test_fake_checkpointer_save_error_propagates_via_registry() -> None:
    """End-to-end: a fake checkpointer whose ``save`` raises is registered
    on a HookRegistry; firing on_node_end propagates the error.
    """
    from troopai.adk.graphs.checkpointer import Checkpointer, GraphCheckpoint

    class _FailingCheckpointer:
        async def save(self, checkpoint: GraphCheckpoint) -> None:
            del checkpoint
            raise OSError("disk full")

        async def load(self, thread_id: str, graph: Graph) -> GraphCheckpoint | None:
            del thread_id, graph
            return None

        async def list_checkpoints(self) -> list[str]:
            return []

        async def delete(self, thread_id: str) -> None:
            del thread_id

        def register(self, registry: HookRegistry) -> None:
            registry.add(CheckpointerHooks(self))  # type: ignore[arg-type]  # fake checkpointer structurally satisfies the Checkpointer Protocol

    assert isinstance(_FailingCheckpointer(), Checkpointer)

    cp = _FailingCheckpointer()
    registry = HookRegistry()
    cp.register(registry)

    g = _make_graph()
    state = GraphState(graph=g, thread_id="fail-thr", superstep=1)

    with pytest.raises(OSError, match="disk full"):
        await registry.on_node_end(
            context=_make_ctx(),
            state=state,
            node_id="a",
            result=NodeResult(output="x"),
        )


async def test_observer_before_critical_does_not_block_propagation() -> None:
    """Observer hook fires first; its error is swallowed.  The subsequent
    critical hook still propagates its error.
    """
    registry = HookRegistry()
    registry.add(_ObserverHook())
    registry.add(_CriticalHook())

    g = _make_graph()
    state = GraphState(graph=g, thread_id="t3", superstep=1)

    with pytest.raises(RuntimeError, match="boom"):
        await registry.on_node_end(
            context=_make_ctx(),
            state=state,
            node_id="a",
            result=NodeResult(output="x"),
        )
