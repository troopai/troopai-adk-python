"""End-to-end graph run integration tests.

All nodes are plain Python callables — no LLM calls, no mocking required.
The tests exercise the BSP superstep driver (``run_graph_loop``) through
the ``Runner.arun_graph`` / ``Runner.configure().graph(...)`` entry points.

Covered scenarios:
- Linear chain (a → b → c)
- Parallel fan-out + AND-join with ``Merge.concat_text``
- Conditional routing (router → left | right)
- Fail-fast: a node raises, status is FAILED
- NO_READY_NODES: conditional graph where all predicates return False
- Checkpointer end-to-end (state persisted after graph_end)
- Profile runner path: ``Runner.configure(context=None).graph(g).arun("x")``
- Determinism: fan-out merged result is byte-identical across 5 runs
"""

from __future__ import annotations

import pytest

from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.merge import Merge
from troopai.adk.graphs.result import GraphRunStatus
from troopai.adk.run.runner import Runner

# ---------------------------------------------------------------------------
# Callable node helpers (all async, single str arg)
# ---------------------------------------------------------------------------


async def _prefix_a(text: str) -> str:
    return f"A:{text}"


async def _prefix_b(text: str) -> str:
    return f"B:{text}"


async def _prefix_c(text: str) -> str:
    return f"C:{text}"


async def _upper(text: str) -> str:
    return text.upper()


async def _lower(text: str) -> str:
    return text.lower()


async def _echo(text: str) -> str:
    return text


# ---------------------------------------------------------------------------
# Linear chain
# ---------------------------------------------------------------------------


class TestLinearChain:
    @pytest.mark.asyncio
    async def test_linear_three_nodes(self) -> None:
        g = (
            Graph.new("linear")
            .node("a", _prefix_a)
            .node("b", _prefix_b)
            .node("c", _prefix_c)
            .pipe("a", "b", "c")
            .entry("a")
            .terminal("c")
            .compile()
        )
        result = await Runner.arun_graph(g, "x")
        assert result.status == GraphRunStatus.COMPLETED
        # The output is a string produced by the terminal node "c".
        # Because concat_text is the default merge, each downstream node
        # sees its upstream's labeled text.  Node "a" sees "x" → produces
        # "A:x".  Node "b" sees "[a]\nA:x" → produces "B:[a]\nA:x".
        # Node "c" sees "[b]\nB:[a]\nA:x" → produces "C:[b]\nB:[a]\nA:x".
        final = result.final_output
        assert isinstance(final, str)
        assert "A:x" in final
        assert "B:" in final
        assert "C:" in final

    @pytest.mark.asyncio
    async def test_linear_total_supersteps(self) -> None:
        g = (
            Graph.new("linear-ss")
            .node("a", _prefix_a)
            .node("b", _prefix_b)
            .pipe("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        result = await Runner.arun_graph(g, "hi")
        # 2 nodes = 2 supersteps (each node fires in its own superstep)
        assert result.total_supersteps == 2

    @pytest.mark.asyncio
    async def test_linear_per_node_results(self) -> None:
        g = (
            Graph.new("linear-nodes")
            .node("a", _echo)
            .node("b", _upper)
            .pipe("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        result = await Runner.arun_graph(g, "hello")
        assert "a" in result.node_results
        assert "b" in result.node_results
        assert result.node_results["a"].final_text == "hello"
        # Node "b" uppercases whatever it receives. Because concat_text
        # is the default merge, "b" receives "[a]\nhello" so the output
        # is the upper-cased version of that label+text.
        b_text = result.node_results["b"].final_text
        assert b_text is not None
        assert "HELLO" in b_text


# ---------------------------------------------------------------------------
# Parallel fan-out + AND-join
# ---------------------------------------------------------------------------


class TestParallelFanOut:
    @pytest.mark.asyncio
    async def test_fanout_and_join_concat_text(self) -> None:
        """a → b, a → c; b,c → d (AND-join, Merge.concat_text)."""

        async def emit_b(text: str) -> str:
            return f"from-b:{text}"

        async def emit_c(text: str) -> str:
            return f"from-c:{text}"

        async def merge_node(text: str) -> str:
            return text  # just echo the merged text

        g = (
            Graph.new("fanout")
            .node("a", _echo)
            .node("b", emit_b)
            .node("c", emit_c)
            .node("d", merge_node, merge=Merge.concat_text)
            .edge("a", "b")
            .edge("a", "c")
            .edge("b", "d")
            .edge("c", "d")
            .entry("a")
            .terminal("d")
            .compile()
        )
        result = await Runner.arun_graph(g, "x")
        assert result.status == GraphRunStatus.COMPLETED
        final = result.final_output
        assert isinstance(final, str)
        # Both upstreams should appear in the merged text
        assert "from-b" in final
        assert "from-c" in final

    @pytest.mark.asyncio
    async def test_fanout_and_join_both_upstreams_fired(self) -> None:
        """Both b and c must have node_results after an AND-join graph."""
        g = (
            Graph.new("fanout-both")
            .node("a", _echo)
            .node("b", _upper)
            .node("c", _lower)
            .node("d", _echo, merge=Merge.concat_text)
            .edge("a", "b")
            .edge("a", "c")
            .edge("b", "d")
            .edge("c", "d")
            .entry("a")
            .terminal("d")
            .compile()
        )
        result = await Runner.arun_graph(g, "Test")
        assert "b" in result.node_results
        assert "c" in result.node_results


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------


class TestConditionalRouting:
    @pytest.mark.asyncio
    async def test_left_branch_fires(self) -> None:
        async def router(_: str) -> str:
            return "L"

        async def left(_: str) -> str:
            return "left-branch"

        async def right(_: str) -> str:
            return "right-branch"

        g = (
            Graph.new("cond-L")
            .node("router", router)
            .node("left", left)
            .node("right", right)
            .edge("router", "left", when=lambda r: r.output == "L")
            .edge("router", "right", when=lambda r: r.output == "R")
            .entry("router")
            .terminal("left", "right")
            .compile()
        )
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.COMPLETED
        # Multiple terminals → final_output is a dict {terminal_id: output}
        assert isinstance(result.final_output, dict)
        assert result.final_output.get("left") is not None
        # right branch must NOT have fired
        assert "right" not in result.node_results
        assert result.final_output.get("right") is None

    @pytest.mark.asyncio
    async def test_right_branch_fires(self) -> None:
        async def router(_: str) -> str:
            return "R"

        async def left(_: str) -> str:
            return "left-branch"

        async def right(_: str) -> str:
            return "right-branch"

        g = (
            Graph.new("cond-R")
            .node("router", router)
            .node("left", left)
            .node("right", right)
            .edge("router", "left", when=lambda r: r.output == "L")
            .edge("router", "right", when=lambda r: r.output == "R")
            .entry("router")
            .terminal("left", "right")
            .compile()
        )
        result = await Runner.arun_graph(g, "go")
        assert result.status == GraphRunStatus.COMPLETED
        # Multiple terminals → final_output is a dict {terminal_id: output}
        assert isinstance(result.final_output, dict)
        assert result.final_output.get("right") is not None
        assert "left" not in result.node_results
        assert result.final_output.get("left") is None

    @pytest.mark.asyncio
    async def test_only_selected_branch_in_per_node_usage(self) -> None:
        """per_node_usage must only have entries for nodes that actually fired."""

        async def router(_: str) -> str:
            return "L"

        async def left(_: str) -> str:
            return "left"

        async def right(_: str) -> str:
            return "right"

        g = (
            Graph.new("cond-usage")
            .node("router", router)
            .node("left", left)
            .node("right", right)
            .edge("router", "left", when=lambda r: r.output == "L")
            .edge("router", "right", when=lambda r: r.output == "R")
            .entry("router")
            .terminal("left", "right")
            .compile()
        )
        result = await Runner.arun_graph(g, "go")
        # "right" must not be present in per_node_usage (never fired)
        assert "right" not in result.per_node_usage


# ---------------------------------------------------------------------------
# Fail-fast
# ---------------------------------------------------------------------------


class TestFailFast:
    @pytest.mark.asyncio
    async def test_node_raises_status_failed(self) -> None:
        async def boom(_: str) -> str:
            raise RuntimeError("boom")

        g = Graph.new("fail").node("a", boom).entry("a").terminal("a").compile()
        result = await Runner.arun_graph(g, "x")
        assert result.status == GraphRunStatus.FAILED

    @pytest.mark.asyncio
    async def test_node_raises_error_message_contains_message(self) -> None:
        async def crasher(_: str) -> str:
            raise RuntimeError("boom")

        g = Graph.new("fail-msg").node("crasher", crasher).entry("crasher").terminal("crasher").compile()
        result = await Runner.arun_graph(g, "x")
        assert result.error is not None
        assert "boom" in result.error


# ---------------------------------------------------------------------------
# NO_READY_NODES exit
# ---------------------------------------------------------------------------


class TestNoReadyNodes:
    @pytest.mark.asyncio
    async def test_all_predicates_false_exits_no_ready_nodes(self) -> None:
        """When all conditional edges return False, exit NO_READY_NODES."""

        async def router(_: str) -> str:
            return "neither"

        async def left(_: str) -> str:
            return "left"

        async def right(_: str) -> str:
            return "right"

        g = (
            Graph.new("no-ready")
            .node("router", router)
            .node("left", left)
            .node("right", right)
            .edge("router", "left", when=lambda r: r.output == "L")
            .edge("router", "right", when=lambda r: r.output == "R")
            .entry("router")
            .terminal("left", "right")
            .compile()
        )
        result = await Runner.arun_graph(g, "x")
        assert result.status == GraphRunStatus.NO_READY_NODES


# ---------------------------------------------------------------------------
# Checkpointer end-to-end
# ---------------------------------------------------------------------------


class TestCheckpointerEndToEnd:
    @pytest.mark.asyncio
    async def test_checkpoint_saved_after_run(self) -> None:
        cp = InMemoryCheckpointer()
        g = Graph.new("cp-e2e").node("a", _echo).node("b", _upper).pipe("a", "b").entry("a").terminal("b").compile()
        await Runner.arun_graph(
            g,
            "hello",
            hooks=[cp],
            thread_id="t1",
        )
        restored = await cp.load("t1", g)
        assert restored is not None
        # The graph ran 2 supersteps
        assert restored.superstep >= 1

    @pytest.mark.asyncio
    async def test_checkpoint_state_has_correct_superstep(self) -> None:
        cp = InMemoryCheckpointer()
        g = Graph.new("cp-ss").node("a", _echo).entry("a").terminal("a").compile()
        await Runner.arun_graph(g, "x", hooks=[cp], thread_id="t2")
        restored = await cp.load("t2", g)
        assert restored is not None
        assert restored.superstep == 1  # Single-node graph runs in 1 superstep


# ---------------------------------------------------------------------------
# Builder path
# ---------------------------------------------------------------------------


class TestGraphRunnerPath:
    @pytest.mark.asyncio
    async def test_graph_runner_arun_equivalent_to_arun_graph(self) -> None:
        g = (
            Graph.new("builder-path")
            .node("a", _echo)
            .node("b", _upper)
            .pipe("a", "b")
            .entry("a")
            .terminal("b")
            .compile()
        )
        direct = await Runner.arun_graph(g, "hello")
        profile_result = await Runner.configure(context=None).graph(g).arun("hello")
        assert direct.final_output == profile_result.final_output
        assert direct.status == profile_result.status

    @pytest.mark.asyncio
    async def test_graph_runner_chaining_methods(self) -> None:
        g = Graph.new("profile-chain").node("n", _echo).entry("n").terminal("n").compile()
        result = await Runner.configure(context=None).graph(g).arun("world")
        assert result.status == GraphRunStatus.COMPLETED
        assert result.final_output == "world"


# ---------------------------------------------------------------------------
# Determinism of sorted apply
# ---------------------------------------------------------------------------


class TestDeterministicMerge:
    @pytest.mark.asyncio
    async def test_fanout_result_byte_identical_across_runs(self) -> None:
        """Merge result must be identical on every run (node-id-sorted apply)."""

        async def branch_x(_: str) -> str:
            return "X-branch"

        async def branch_y(_: str) -> str:
            return "Y-branch"

        async def joiner(text: str) -> str:
            return text

        g = (
            Graph.new("det")
            .node("entry", _echo)
            .node("bx", branch_x)
            .node("by", branch_y)
            .node("join", joiner, merge=Merge.concat_text)
            .edge("entry", "bx")
            .edge("entry", "by")
            .edge("bx", "join")
            .edge("by", "join")
            .entry("entry")
            .terminal("join")
            .compile()
        )

        results = []
        for _ in range(5):
            r = await Runner.arun_graph(g, "input")
            results.append(r.final_output)

        # All 5 runs must produce the same output
        assert all(o == results[0] for o in results)
        # Both branches must appear
        assert isinstance(results[0], str)
        assert "X-branch" in results[0]
        assert "Y-branch" in results[0]


# ---------------------------------------------------------------------------
# Single-node graph (entry == terminal)
# ---------------------------------------------------------------------------


class TestSingleNode:
    @pytest.mark.asyncio
    async def test_single_node_graph_runs(self) -> None:
        g = Graph.new("solo").node("only", _upper).entry("only").terminal("only").compile()
        result = await Runner.arun_graph(g, "hello")
        assert result.status == GraphRunStatus.COMPLETED
        assert result.final_output == "HELLO"
        assert result.total_supersteps == 1


# ---------------------------------------------------------------------------
# Conditional AND-join: one upstream fires, one is skipped
# ---------------------------------------------------------------------------


class TestConditionalAndJoin:
    """Regression test for the conditional-edge AND-join deadlock.

    Before the fix, an AND-join whose incoming edge set included a
    conditional edge with ``when=False`` would wait forever and exit
    the loop with ``NO_READY_NODES``. The fix records a ``record_skip``
    on the target barrier so AND-ready becomes
    ``(arrivals ∪ skipped) >= expected and len(arrivals) > 0``.
    """

    @pytest.mark.asyncio
    async def test_and_join_fires_when_one_parent_skipped(self) -> None:
        async def always(_: str) -> str:
            return "ALWAYS"

        async def gated(_: str) -> str:
            return "GATED"

        async def merger(text: str) -> str:
            return f"merged:{text}"

        g = (
            Graph.new("cond-and-join")
            .node("entry", _echo)
            .node("always", always)
            .node("gated", gated)
            .node("merge", merger, merge=Merge.last_wins)
            .edge("entry", "always")
            .edge("entry", "gated")
            .edge("always", "merge")
            .edge("gated", "merge", when=lambda _: False)
            .entry("entry")
            .terminal("merge")
            .compile()
        )

        result = await Runner.arun_graph(g, "hi")

        assert result.status == GraphRunStatus.COMPLETED
        assert isinstance(result.final_output, str)
        assert result.final_output.startswith("merged:")
        assert "ALWAYS" in result.final_output
        assert "GATED" not in result.final_output
