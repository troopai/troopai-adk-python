"""Unit tests for FunctionTool.defer_loading + build_tool_search()."""

from __future__ import annotations

import json
from contextvars import copy_context
from typing import Any
from unittest.mock import MagicMock

import pytest

from troopai.adk.run.llm_calls import build_tools
from troopai.adk.tools import build_tool_search, function_tool
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.tools.tool_search import reset_revealed_sets

MINIMAL_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def _make_tool(name: str, **overrides: Any) -> FunctionTool:
    defaults: dict[str, Any] = {
        "name": name,
        "schema": MINIMAL_SCHEMA,
        "description": f"description for {name}",
    }
    defaults.update(overrides)
    return FunctionTool(**defaults)


def _make_agent(tools: list[Any]) -> Any:
    a = MagicMock()
    a.tools = tools
    a.handoffs = None
    a.skills = []
    a.subagents = []
    return a


# ── FunctionTool.defer_loading field ────────────────────────────────


class TestDeferLoadingField:
    def test_default_false(self) -> None:
        assert _make_tool("t").defer_loading is False

    def test_explicit_true(self) -> None:
        assert _make_tool("t", defer_loading=True).defer_loading is True

    def test_decorator_passthrough(self) -> None:
        @function_tool(name="rare", defer_loading=True)
        def rare(x: str) -> str:
            return x

        assert rare.defer_loading is True


# ── build_tool_search ────────────────────────────────────────────────


class TestBuildToolSearch:
    def test_returns_function_tool(self) -> None:
        tool = build_tool_search([])
        assert isinstance(tool, FunctionTool)
        assert tool.name == "tool_search"

    def test_custom_name_and_description(self) -> None:
        tool = build_tool_search([], name="discover", description="custom")
        assert tool.name == "discover"
        assert tool.description == "custom"

    def test_search_state_attached(self) -> None:
        rare = _make_tool("rare", defer_loading=True)
        tool = build_tool_search([rare])
        state = tool.get_search_state()
        assert state is not None
        assert "rare" in state.deferred
        assert state.revealed == set()


# ── on_invoke matching ───────────────────────────────────────────────


class TestSearchInvocation:
    @pytest.mark.asyncio
    async def test_substring_match_in_name(self) -> None:
        rare = _make_tool("payment_processor", defer_loading=True)
        other = _make_tool("send_email", defer_loading=True)
        search = build_tool_search([rare, other])
        assert search.on_invoke is not None

        result = await search.on_invoke(MagicMock(), '{"query": "payment"}')
        names = [m["name"] for m in json.loads(result)]
        assert "payment_processor" in names
        assert "send_email" not in names

    @pytest.mark.asyncio
    async def test_match_in_description(self) -> None:
        rare = _make_tool(
            "alpha",
            description="Charges credit cards via the payment gateway.",
            defer_loading=True,
        )
        search = build_tool_search([rare])
        assert search.on_invoke is not None

        result = await search.on_invoke(MagicMock(), '{"query": "credit card"}')
        names = [m["name"] for m in json.loads(result)]
        assert names == ["alpha"]

    @pytest.mark.asyncio
    async def test_invocation_populates_revealed(self) -> None:
        rare = _make_tool("rare_one", defer_loading=True)
        search = build_tool_search([rare])
        assert search.on_invoke is not None
        assert search.get_search_state().revealed == set()

        await search.on_invoke(MagicMock(), '{"query": "rare"}')
        assert search.get_search_state().revealed == {"rare_one"}

    @pytest.mark.asyncio
    async def test_top_k_caps_results(self) -> None:
        tools = [_make_tool(f"tool_{i}", defer_loading=True) for i in range(10)]
        search = build_tool_search(tools)
        assert search.on_invoke is not None

        result = await search.on_invoke(MagicMock(), '{"query": "tool", "top_k": 3}')
        assert len(json.loads(result)) == 3

    @pytest.mark.asyncio
    async def test_no_matches_returns_empty(self) -> None:
        rare = _make_tool("alpha", defer_loading=True)
        search = build_tool_search([rare])
        assert search.on_invoke is not None

        result = await search.on_invoke(MagicMock(), '{"query": "totally_unrelated"}')
        assert json.loads(result) == []

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self) -> None:
        # An empty/whitespace query must not enumerate the catalogue.
        rare1 = _make_tool("rare1", defer_loading=True)
        rare2 = _make_tool("rare2", defer_loading=True)
        search = build_tool_search([rare1, rare2])
        assert search.on_invoke is not None

        for query in ("", "   ", "\n\t"):
            result = await search.on_invoke(MagicMock(), json.dumps({"query": query}))
            assert json.loads(result) == [], f"empty query {query!r} leaked catalogue"
        # Nothing was revealed by the empty query path
        assert search.get_search_state().revealed == set()

    @pytest.mark.asyncio
    async def test_top_k_clamped_high(self) -> None:
        # Even if the LLM bypasses the schema and submits top_k=99999,
        # the clamp keeps results bounded at 50.
        tools = [_make_tool(f"tool_{i}", defer_loading=True) for i in range(60)]
        search = build_tool_search(tools)
        assert search.on_invoke is not None

        result = await search.on_invoke(MagicMock(), '{"query": "tool", "top_k": 99999}')
        assert len(json.loads(result)) == 50

    @pytest.mark.asyncio
    async def test_top_k_clamped_low(self) -> None:
        # top_k=0 or negative is clamped to 1.
        rare = _make_tool("only_one", defer_loading=True)
        search = build_tool_search([rare])
        assert search.on_invoke is not None

        result = await search.on_invoke(MagicMock(), '{"query": "only", "top_k": -5}')
        assert len(json.loads(result)) == 1

    @pytest.mark.asyncio
    async def test_top_k_non_numeric_handled(self) -> None:
        search = build_tool_search([])
        assert search.on_invoke is not None
        result = await search.on_invoke(MagicMock(), '{"query": "x", "top_k": "many"}')
        assert "Invalid 'top_k'" in result

    @pytest.mark.asyncio
    async def test_invalid_json(self) -> None:
        search = build_tool_search([])
        assert search.on_invoke is not None
        result = await search.on_invoke(MagicMock(), "{not json")
        assert "Invalid JSON" in result


# ── build_tools() integration ────────────────────────────────────────


class TestBuildToolsFiltersDeferred:
    @pytest.mark.asyncio
    async def test_unrevealed_deferred_tool_filtered(self) -> None:
        regular = _make_tool("regular", on_invoke=MagicMock())
        rare = _make_tool("rare", defer_loading=True, on_invoke=MagicMock())
        search = build_tool_search([rare])
        agent = _make_agent([regular, rare, search])

        result = await build_tools(agent)
        assert result is not None
        names = [t.name for t in result if isinstance(t, FunctionTool)]
        # "regular" is included; "rare" hidden; "tool_search" included
        assert "regular" in names
        assert "rare" not in names
        assert "tool_search" in names

    @pytest.mark.asyncio
    async def test_revealed_tool_appears(self) -> None:
        regular = _make_tool("regular", on_invoke=MagicMock())
        rare = _make_tool("rare", defer_loading=True, on_invoke=MagicMock())
        search = build_tool_search([rare])

        # Drive the reveal through on_invoke (the actual code path)
        # rather than mutating state.revealed directly.
        assert search.on_invoke is not None
        await search.on_invoke(MagicMock(), '{"query": "rare"}')

        agent = _make_agent([regular, rare, search])
        result = await build_tools(agent)
        assert result is not None
        names = [t.name for t in result if isinstance(t, FunctionTool)]
        assert "rare" in names

    @pytest.mark.asyncio
    async def test_non_deferred_unaffected(self) -> None:
        # No search tool at all; non-deferred tools go through normally.
        regular_a = _make_tool("a", on_invoke=MagicMock())
        regular_b = _make_tool("b", on_invoke=MagicMock())
        agent = _make_agent([regular_a, regular_b])

        result = await build_tools(agent)
        assert result is not None
        names = [t.name for t in result if isinstance(t, FunctionTool)]
        assert names == ["a", "b"]

    @pytest.mark.asyncio
    async def test_multiple_search_tools_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        # Two search tools — first one wins, but the framework warns.
        rare = _make_tool("rare", defer_loading=True, on_invoke=MagicMock())
        search_a = build_tool_search([rare])
        search_b = build_tool_search([rare])
        agent = _make_agent([rare, search_a, search_b])

        import logging

        with caplog.at_level(logging.WARNING, logger="troopai.adk.tools.tool_search"):
            await build_tools(agent)
        assert any("tool_search instances" in r.message for r in caplog.records), (
            "expected a warning when multiple search tools are present"
        )


# ── find_revealed_deferred_tools helper ────────────────────────────


class TestFindRevealedHelper:
    def test_no_search_tool_returns_empty(self) -> None:
        from troopai.adk.tools.tool_search import find_revealed_deferred_tools

        regular = _make_tool("a")
        assert find_revealed_deferred_tools([regular]) == set()

    def test_returns_search_tools_revealed_set(self) -> None:
        from troopai.adk.tools.tool_search import find_revealed_deferred_tools

        rare = _make_tool("rare", defer_loading=True)
        search = build_tool_search([rare])
        search.get_search_state().reveal("rare")
        result = find_revealed_deferred_tools([rare, search])
        assert result == {"rare"}


# ── Executor execution gate ─────────────────────────────────────────


class TestExecutorGate:
    """Visibility filtering alone is not a security control; the
    executor must refuse to invoke a deferred tool whose name has not
    been revealed, even if the LLM emits a function-call to it."""

    @pytest.mark.asyncio
    async def test_unrevealed_deferred_tool_refused(self) -> None:
        from troopai.adk.hooks.hooks import RunHooks
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG
        from troopai.adk.run.context import RunContext
        from troopai.adk.run.tools_executor import execute_tool_calls
        from troopai.adk.types.responses.llm_response import (
            LLMResponseFunctionToolCall,
        )

        invocations: list[int] = []

        async def _handler(_ctx: Any, _raw: str) -> str:
            invocations.append(1)
            return "ran!"

        rare = _make_tool("rare_api", defer_loading=True, on_invoke=_handler)
        search = build_tool_search([rare])
        agent = _make_agent([rare, search])
        agent.tool_use_behavior = "run_llm_again"
        agent.llm = None
        agent.hooks = None

        # Simulate a prompt-injected LLM emitting a call to the
        # deferred tool WITHOUT having searched for it first.
        tc = LLMResponseFunctionToolCall(call_id="c1", name="rare_api", arguments="{}")
        results, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc],
            ctx_wrapper=RunContext(context=None),
            hooks=RunHooks(),
            config=DEFAULT_RUN_CONFIG,
            model="gpt-4o-mini",
        )
        # The handler must NOT have run.
        assert invocations == []
        # The LLM gets a "tool not found"-shaped response.
        assert "rare_api" in str(results[0].output)

    @pytest.mark.asyncio
    async def test_revealed_deferred_tool_executes(self) -> None:
        from troopai.adk.hooks.hooks import RunHooks
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG
        from troopai.adk.run.context import RunContext
        from troopai.adk.run.tools_executor import execute_tool_calls
        from troopai.adk.types.responses.llm_response import (
            LLMResponseFunctionToolCall,
        )

        invocations: list[int] = []

        async def _handler(_ctx: Any, _raw: str) -> str:
            invocations.append(1)
            return "ran!"

        rare = _make_tool("rare_api", defer_loading=True, on_invoke=_handler)
        search = build_tool_search([rare])
        # Drive reveal through the actual on_invoke path.
        assert search.on_invoke is not None
        await search.on_invoke(MagicMock(), '{"query": "rare"}')

        agent = _make_agent([rare, search])
        agent.tool_use_behavior = "run_llm_again"
        agent.llm = None
        agent.hooks = None

        tc = LLMResponseFunctionToolCall(call_id="c1", name="rare_api", arguments="{}")
        results, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc],
            ctx_wrapper=RunContext(context=None),
            hooks=RunHooks(),
            config=DEFAULT_RUN_CONFIG,
            model="gpt-4o-mini",
        )
        assert invocations == [1]
        assert "ran!" in str(results[0].output)


# ── Per-run isolation (ToolSearchState.revealed is per-context) ───────


class TestRevealedPerRunIsolation:
    """Verify that the revealed set is scoped per execution context.

    ``ToolSearchState.revealed`` uses a ``ContextVar`` so each
    ``Runner.arun()`` call (which runs in a fresh context copy) starts
    with an empty revealed set regardless of what prior runs revealed.
    """

    @pytest.mark.asyncio
    async def test_second_invocation_starts_empty(self) -> None:
        """A fresh context sees an empty revealed set even after a prior reveal.

        ``asyncio.create_task`` copies the context at scheduling time.
        If run #2 is scheduled BEFORE run #1 has revealed any tools, the
        copy has the default empty frozenset.  Because the
        ``ContextVar.set()`` call in ``reveal()`` only rebinds in the
        calling context, run #1's reveals are invisible to run #2.
        """
        rare = _make_tool("rare_one", defer_loading=True)
        search = build_tool_search([rare])
        assert search.on_invoke is not None
        state = search.get_search_state()
        assert state is not None

        # Snapshot the context BEFORE any reveal (simulates the context
        # copy that asyncio makes at task-creation time for run #2, while
        # run #1 has not yet revealed anything).
        ctx_for_run2 = copy_context()

        # Now reveal in the CURRENT context (simulates run #1 revealing).
        await search.on_invoke(MagicMock(), '{"query": "rare"}')
        assert "rare_one" in state.revealed

        # Run #2's context must still be empty.
        revealed_in_run2: list[frozenset[str]] = []
        ctx_for_run2.run(lambda: revealed_in_run2.append(state.revealed))

        assert "rare_one" not in revealed_in_run2[0], (
            "revealed set from run #1 was visible in run #2's context — "
            "ToolSearchState.revealed is not properly per-run isolated"
        )

    def test_fresh_context_has_empty_revealed(self) -> None:
        """A ``copy_context()`` snapshot taken before any reveal is empty.

        ``copy_context()`` is what ``asyncio.create_task`` calls
        internally — each task starts from the current context state at
        scheduling time, not the state after later mutations. This test
        verifies that a context snapshot taken before any reveal (which
        is what happens at the start of a new Runner.arun() call)
        correctly sees an empty revealed set.
        """
        rare = _make_tool("rare_tool", defer_loading=True)
        search = build_tool_search([rare])
        state = search.get_search_state()
        assert state is not None

        # Snapshot the context BEFORE any reveal (simulates the context
        # copy that asyncio makes at task-creation time for run #2,
        # while run #1 is still in progress).
        ctx_pre_reveal = copy_context()

        # Reveal a tool in the current context (simulates run #1 revealing).
        state.reveal("rare_tool")
        assert "rare_tool" in state.revealed

        # The pre-reveal snapshot must NOT see the reveal.
        revealed_in_snapshot: list[frozenset[str]] = []
        ctx_pre_reveal.run(lambda: revealed_in_snapshot.append(state.revealed))

        assert "rare_tool" not in revealed_in_snapshot[0], (
            "revealed set from run #1 was visible in a context snapshot "
            "taken before the reveal — per-run isolation is broken"
        )


# ── Sequential-await isolation (reset_revealed_sets) ─────────────────


class TestSequentialAwaitIsolation:
    """Verify that sequential ``await Runner.arun()`` calls (same coroutine,
    same asyncio context) each start with an empty revealed set.

    The ``ContextVar`` approach alone does NOT cover this case: sequential
    awaits share the same context, so ``ContextVar.set()`` from run #1 is
    still visible at the start of run #2.  The ``Runner`` must call
    ``reset_revealed_sets(agent.tools)`` at the start of every run to
    clear the state.
    """

    @pytest.mark.asyncio
    async def test_reset_clears_revealed_in_current_context(self) -> None:
        """``reset_revealed_sets`` must empty the revealed set in the
        current context, undoing any reveals from a prior run.

        This is the core fix: without calling ``reset_revealed_sets``,
        a second sequential ``await Runner.arun()`` call would see all
        tools that were revealed in the first call, bypassing the
        capability-gating intent of ``defer_loading``.
        """
        rare = _make_tool("rare_seq", defer_loading=True)
        search = build_tool_search([rare])
        assert search.on_invoke is not None
        state = search.get_search_state()
        assert state is not None

        # Simulate run #1: reveal a tool (as the LLM would via on_invoke).
        await search.on_invoke(MagicMock(), '{"query": "rare"}')
        assert "rare_seq" in state.revealed, "run #1 should have revealed rare_seq"

        # Simulate Runner.arun() start of run #2: reset revealed sets.
        reset_revealed_sets([rare, search])

        # After reset, the revealed set must be empty in the current context.
        assert state.revealed == frozenset(), (
            "revealed set was not cleared by reset_revealed_sets — "
            "sequential await Runner.arun() would carry over run #1 reveals"
        )

    @pytest.mark.asyncio
    async def test_reset_does_not_affect_other_search_tools(self) -> None:
        """Resetting one agent's tools must not affect another agent's search state."""
        rare_a = _make_tool("rare_a", defer_loading=True)
        search_a = build_tool_search([rare_a])
        assert search_a.on_invoke is not None

        rare_b = _make_tool("rare_b", defer_loading=True)
        search_b = build_tool_search([rare_b])
        assert search_b.on_invoke is not None

        # Reveal in both search tools.
        await search_a.on_invoke(MagicMock(), '{"query": "rare"}')
        await search_b.on_invoke(MagicMock(), '{"query": "rare"}')

        state_a = search_a.get_search_state()
        state_b = search_b.get_search_state()
        assert state_a is not None and "rare_a" in state_a.revealed
        assert state_b is not None and "rare_b" in state_b.revealed

        # Reset only agent A's tools.
        reset_revealed_sets([rare_a, search_a])

        # Only state_a should be cleared; state_b untouched.
        assert state_a.revealed == frozenset(), "state_a should have been reset"
        assert "rare_b" in state_b.revealed, "state_b should not have been reset"

    @pytest.mark.asyncio
    async def test_reset_allows_new_reveal_in_same_context(self) -> None:
        """After ``reset_revealed_sets``, reveals in run #2 work normally.

        The reset must not damage the ContextVar — new reveals after
        the reset are visible and the execution gate permits them.
        """
        rare = _make_tool("rare_reset", defer_loading=True)
        search = build_tool_search([rare])
        assert search.on_invoke is not None
        state = search.get_search_state()
        assert state is not None

        # run #1: reveal, then reset (simulating Runner.arun() start of run #2).
        await search.on_invoke(MagicMock(), '{"query": "rare"}')
        reset_revealed_sets([rare, search])

        # run #2: reveal again in the now-clean context.
        await search.on_invoke(MagicMock(), '{"query": "rare"}')
        assert "rare_reset" in state.revealed, (
            "reveal after reset should work — the ContextVar must still be functional"
        )

    def test_state_reset_method_directly(self) -> None:
        """``ToolSearchState.reset()`` clears the revealed set in the
        current context without affecting the ContextVar itself.
        """
        rare = _make_tool("direct_rare", defer_loading=True)
        search = build_tool_search([rare])
        state = search.get_search_state()
        assert state is not None

        state.reveal("direct_rare")
        assert "direct_rare" in state.revealed

        state.reset()
        assert state.revealed == frozenset(), "ToolSearchState.reset() must clear the revealed set"

    def test_reset_no_op_when_no_search_tool(self) -> None:
        """``reset_revealed_sets`` is a no-op when no search tool is present."""
        regular = _make_tool("regular_only")
        # Must not raise.
        reset_revealed_sets([regular])
