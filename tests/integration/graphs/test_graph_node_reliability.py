"""Integration tests for per-node timeout and retry enforcement in ``run_graph_loop``.

All nodes are plain Python callables — no LLM calls, no API keys required.
Tests verify that ``GraphConfig.per_node_timeout`` and
``GraphConfig.default_retry`` are enforced by the graph loop.
"""

from __future__ import annotations

import asyncio
import dataclasses

from troopai.adk.graphs.config import GraphConfig, NodeRetryPolicy
from troopai.adk.graphs.graph import Graph
from troopai.adk.run.config import DEFAULT_RUN_CONFIG
from troopai.adk.run.context import RunContext
from troopai.adk.run.graph_loop import run_graph_loop


async def test_node_timeout_fails_run() -> None:
    """A node that exceeds its per-attempt timeout must fail the run with GraphNodeTimeoutError."""

    async def slow() -> str:
        await asyncio.sleep(1.0)
        return "x"

    graph = (
        Graph.new("timeout-e2e")
        .node("a", slow)
        .node("b", lambda: "b-done")
        .edge("a", "b")
        .entry("a")
        .terminal("b")
        .with_config(GraphConfig(per_node_timeout=0.05))
        .compile()
    )
    ctx: RunContext = RunContext(context=None)
    res = await run_graph_loop(
        graph=graph,
        user_prompt="go",
        context=ctx,
        config=DEFAULT_RUN_CONFIG,
    )
    assert res.status.value == "failed"
    assert "GraphNodeTimeoutError" in (res.error or "")


async def test_retry_to_success_completes() -> None:
    """A node that fails once then succeeds is retried and the run completes."""
    call_count = 0

    def flaky() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("transient")
        return "done"

    graph = (
        Graph.new("retry-e2e")
        .node("a", flaky)
        .entry("a")
        .terminal("a")
        .with_config(GraphConfig(default_retry=NodeRetryPolicy(max_attempts=3, initial_backoff=0.001)))
        .compile()
    )
    ctx: RunContext = RunContext(context=None)
    res = await run_graph_loop(
        graph=graph,
        user_prompt="go",
        context=ctx,
        config=DEFAULT_RUN_CONFIG,
    )
    assert res.status.value == "completed"
    assert call_count == 2


async def test_no_policy_parity_propagates_original() -> None:
    """A node that raises with no retry/timeout policy re-raises the original exception.

    With neither knob opted into, the run fails with the original exception
    (RuntimeError), not a wrapper type like NodeRetriesExhaustedError.
    """

    def boom() -> str:
        raise RuntimeError("plain-failure")

    graph = Graph.new("parity-e2e").node("a", boom).entry("a").terminal("a").compile()
    ctx: RunContext = RunContext(context=None)
    res = await run_graph_loop(
        graph=graph,
        user_prompt="go",
        context=ctx,
        config=DEFAULT_RUN_CONFIG,
    )
    assert res.status.value == "failed"
    assert "RuntimeError" in (res.error or "")
    assert "plain-failure" in (res.error or "")
    assert "NodeRetriesExhaustedError" not in (res.error or "")


async def test_per_node_retry_override_beats_graph_default() -> None:
    """A per-node retry override allows more attempts than the graph default.

    The graph default allows only 1 attempt (``max_attempts=1``), which
    would fail a node that succeeds only on its 3rd call.  The node carries
    a per-node override (``max_attempts=4``), which is sufficient — the node
    retries twice and the graph completes.
    """
    call_count = 0

    def flaky_3rd() -> str:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("transient")
        return "ok"

    # Graph default: 1 attempt only — would fail without a per-node override.
    g = (
        Graph.new("per-node-retry-override")
        .node("a", flaky_3rd)
        .node("b", lambda: "b-done")
        .edge("a", "b")
        .entry("a")
        .terminal("b")
        .with_config(GraphConfig(default_retry=NodeRetryPolicy(max_attempts=1)))
        .compile()
    )

    # Attach a per-node retry override to node "a" via dataclasses.replace.
    # Graph is a frozen dataclass; Graph.nodes is a tuple[GraphNode, ...].
    node_a = g.get_node("a")
    node_a_override = dataclasses.replace(
        node_a,
        retry=NodeRetryPolicy(max_attempts=4, initial_backoff=0.001),
    )
    g = dataclasses.replace(
        g,
        nodes=tuple(node_a_override if n.id == "a" else n for n in g.nodes),
    )

    ctx: RunContext = RunContext(context=None)
    res = await run_graph_loop(
        graph=g,
        user_prompt="go",
        context=ctx,
        config=DEFAULT_RUN_CONFIG,
    )
    assert res.status.value == "completed"
    # Node "a" must have been called exactly 3 times:
    # graph default of 1 would have stopped at attempt 1; the per-node
    # override of 4 allowed the 2 retries that reached the 3rd call.
    assert call_count == 3


async def test_fail_fast_false_preserves_co_terminal_on_timeout() -> None:
    """Under ``fail_fast=False`` a timed-out co-terminal does not abort the run.

    Graph topology: ``entry -> slow`` and ``entry -> fast``; ``fast`` is the
    terminal.  ``slow`` exceeds ``per_node_timeout`` and is recorded as
    errored, but because ``fail_fast=False`` the loop does not cancel
    co-terminals.  ``fast`` completes; reaching the terminal triggers
    ``COMPLETED`` exit with ``final_output == "fast-done"``.
    """

    async def slow() -> str:
        await asyncio.sleep(1.0)
        return "late"

    # Both ``slow`` and ``fast`` are terminals so both have a path to a
    # terminal (required by the builder).  ``slow`` times out and is
    # recorded as errored — it never populates ``terminal_outputs``.
    # ``fast`` completes and is the only entry in ``terminal_outputs``.
    # With two declared terminals the loop returns ``final_output`` as a
    # dict keyed by whichever terminals fired; ``slow`` errored so the
    # dict contains only ``{"fast": "fast-done"}``.
    graph = (
        Graph.new("fail-fast-false-timeout")
        .node("entry", lambda: "go")
        .node("slow", slow)
        .node("fast", lambda: "fast-done")
        .edge("entry", "slow")
        .edge("entry", "fast")
        .entry("entry")
        .terminal("slow")
        .terminal("fast")
        .with_config(GraphConfig(per_node_timeout=0.05, fail_fast=False))
        .compile()
    )
    ctx: RunContext = RunContext(context=None)
    res = await run_graph_loop(
        graph=graph,
        user_prompt="go",
        context=ctx,
        config=DEFAULT_RUN_CONFIG,
    )
    # ``fast`` reached its terminal — the run completes despite ``slow`` timing out.
    assert res.status.value == "completed"
    # Only the fast terminal fired; slow errored and never wrote to terminal_outputs.
    assert res.final_output == {"fast": "fast-done"}
    # Explicit: slow timed out — it never produced a terminal output.
    assert res.state is not None
    assert "slow" not in res.state.terminal_outputs
