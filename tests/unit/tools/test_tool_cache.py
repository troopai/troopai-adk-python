"""Tests for FunctionTool caching."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from troopai.adk.agents import Agent
from troopai.adk.run.context import RunContext
from troopai.adk.run.tools_executor import execute_tool_calls
from troopai.adk.tools import Tool
from troopai.adk.tools.function_tool import FunctionTool, ToolCachePolicy
from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

# ── Helpers ──────────────────────────────────────────────────────────

_call_count = 0


def _make_agent(tools: Sequence[Tool]) -> Agent[Any]:
    return Agent(
        name="test_agent",
        system_prompt="test",
        tools=list(tools),
    )


def _make_tool_call(call_id: str, name: str, args: str = "{}") -> LLMResponseFunctionToolCall:
    return LLMResponseFunctionToolCall(call_id=call_id, name=name, arguments=args)


def _make_ctx(context: Any = None) -> RunContext[Any]:
    return RunContext(context=context)


def _make_hooks():
    from troopai.adk.hooks.hooks import RunHooks

    return RunHooks()


def _make_config():
    from troopai.adk.run.config import DEFAULT_RUN_CONFIG

    return DEFAULT_RUN_CONFIG


async def _counting_handler(_ctx, _raw_args):
    global _call_count
    _call_count += 1
    return f"result_{_call_count}"


async def _artifact_handler(_ctx, _raw_args):
    """Returns (content, artifact) tuple for content_and_artifact format."""
    global _call_count
    _call_count += 1
    return (f"summary_{_call_count}", {"full_data": [1, 2, 3], "call": _call_count})


# ── Tests ────────────────────────────────────────────────────────────


class TestToolCache:
    def setup_method(self):
        global _call_count
        _call_count = 0

    def test_cache_defaults_false(self) -> None:
        tool = FunctionTool(
            name="t",
            description="d",
            schema={"type": "object", "properties": {}},
        )
        assert tool.cache is False

    def test_cache_attr_exists(self) -> None:
        tool = FunctionTool(
            name="t",
            description="d",
            schema={"type": "object", "properties": {}},
            cache=True,
        )
        assert tool.cache is True
        assert tool.get_cached("any") is None  # empty cache

    @pytest.mark.asyncio
    async def test_cache_hit_skips_handler(self) -> None:
        tool = FunctionTool(
            name="cached_tool",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_counting_handler,
            cache=True,
        )
        agent = _make_agent([tool])
        tc = _make_tool_call("c1", "cached_tool", '{"x": 1}')
        ctx = _make_ctx()

        # First call — handler executes
        results1, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc],
            ctx_wrapper=ctx,
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        assert _call_count == 1

        # Second call in the same run — handler NOT called
        tc2 = _make_tool_call("c2", "cached_tool", '{"x": 1}')
        results2, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc2],
            ctx_wrapper=ctx,
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        assert _call_count == 1  # Still 1 — cached
        assert results2[0].output == results1[0].output

    @pytest.mark.asyncio
    async def test_different_args_not_cached(self) -> None:
        tool = FunctionTool(
            name="cached_tool",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_counting_handler,
            cache=True,
        )
        agent = _make_agent([tool])

        tc1 = _make_tool_call("c1", "cached_tool", '{"x": 1}')
        tc2 = _make_tool_call("c2", "cached_tool", '{"x": 2}')

        await execute_tool_calls(
            agent=agent,
            tool_calls=[tc1],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        assert _call_count == 1

        await execute_tool_calls(
            agent=agent,
            tool_calls=[tc2],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        assert _call_count == 2  # Different args, new call

    @pytest.mark.asyncio
    async def test_cache_true_is_run_scoped_by_default(self) -> None:
        tool = FunctionTool(
            name="cached_tool",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_counting_handler,
            cache=True,
        )
        agent = _make_agent([tool])

        await execute_tool_calls(
            agent=agent,
            tool_calls=[_make_tool_call("c1", "cached_tool", '{"x": 1}')],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        await execute_tool_calls(
            agent=agent,
            tool_calls=[_make_tool_call("c2", "cached_tool", '{"x": 1}')],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )

        assert _call_count == 2

    @pytest.mark.asyncio
    async def test_process_cache_policy_reuses_across_runs_with_canonical_json(self) -> None:
        tool = FunctionTool(
            name="cached_tool",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_counting_handler,
            cache=ToolCachePolicy(scope="process"),
        )
        agent = _make_agent([tool])

        results1, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[_make_tool_call("c1", "cached_tool", '{"x": 1, "y": 2}')],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        results2, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[_make_tool_call("c2", "cached_tool", '{"y":2,"x":1}')],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )

        assert _call_count == 1
        assert results2[0].output == results1[0].output

    @pytest.mark.asyncio
    async def test_process_cache_policy_evicts_least_recently_used_entry(self) -> None:
        tool = FunctionTool(
            name="cached_tool",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_counting_handler,
            cache=ToolCachePolicy(scope="process", max_entries=1),
        )
        agent = _make_agent([tool])

        await execute_tool_calls(
            agent=agent,
            tool_calls=[_make_tool_call("c1", "cached_tool", '{"x": 1}')],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        await execute_tool_calls(
            agent=agent,
            tool_calls=[_make_tool_call("c2", "cached_tool", '{"x": 2}')],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        await execute_tool_calls(
            agent=agent,
            tool_calls=[_make_tool_call("c3", "cached_tool", '{"x": 1}')],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )

        assert _call_count == 3

    @pytest.mark.asyncio
    async def test_process_cache_key_builder_can_isolate_tenants(self) -> None:
        def tenant_key(ctx, raw_args: str) -> str:
            return f"{ctx.context['tenant']}:{raw_args}"

        tool = FunctionTool(
            name="cached_tool",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_counting_handler,
            cache=ToolCachePolicy(scope="process", key_builder=tenant_key),
        )
        agent = _make_agent([tool])

        ctx_a1 = _make_ctx({"tenant": "a"})
        ctx_b = _make_ctx({"tenant": "b"})
        ctx_a2 = _make_ctx({"tenant": "a"})

        await execute_tool_calls(
            agent=agent,
            tool_calls=[_make_tool_call("c1", "cached_tool", '{"x": 1}')],
            ctx_wrapper=ctx_a1,
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        await execute_tool_calls(
            agent=agent,
            tool_calls=[_make_tool_call("c2", "cached_tool", '{"x": 1}')],
            ctx_wrapper=ctx_b,
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        await execute_tool_calls(
            agent=agent,
            tool_calls=[_make_tool_call("c3", "cached_tool", '{"x": 1}')],
            ctx_wrapper=ctx_a2,
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )

        assert _call_count == 2

    @pytest.mark.asyncio
    async def test_no_cache_when_disabled(self) -> None:
        tool = FunctionTool(
            name="uncached",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_counting_handler,
            cache=False,
        )
        agent = _make_agent([tool])
        tc = _make_tool_call("c1", "uncached", "{}")

        await execute_tool_calls(
            agent=agent,
            tool_calls=[tc],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        tc2 = _make_tool_call("c2", "uncached", "{}")
        await execute_tool_calls(
            agent=agent,
            tool_calls=[tc2],
            ctx_wrapper=_make_ctx(),
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        assert _call_count == 2  # No caching

    @pytest.mark.asyncio
    async def test_artifact_preserved_on_cache_hit(self) -> None:
        """content_and_artifact response format preserves artifact through cache (Bug 5)."""
        tool = FunctionTool(
            name="artifact_tool",
            description="d",
            schema={"type": "object", "properties": {}},
            on_invoke=_artifact_handler,
            cache=True,
            response_format="content_and_artifact",
        )
        agent = _make_agent([tool])
        ctx = _make_ctx()

        # First call — stores (result, artifact) in cache
        tc1 = _make_tool_call("c1", "artifact_tool", '{"q": "test"}')
        results1, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc1],
            ctx_wrapper=ctx,
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        assert _call_count == 1
        assert results1[0].artifact is not None
        assert isinstance(results1[0].artifact, dict)
        assert results1[0].artifact["full_data"] == [1, 2, 3]

        # Second call — cache hit should preserve artifact
        tc2 = _make_tool_call("c2", "artifact_tool", '{"q": "test"}')
        results2, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[tc2],
            ctx_wrapper=ctx,
            hooks=_make_hooks(),
            config=_make_config(),
            model="gpt-4o-mini",
        )
        assert _call_count == 1  # Still cached
        assert results2[0].output == results1[0].output
        assert results2[0].artifact is not None
        assert results2[0].artifact == results1[0].artifact
