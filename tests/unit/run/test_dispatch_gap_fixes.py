"""Regression tests for tool/LLM dispatch gaps in the run executor.

Each test class pins one dispatch bug so the fix cannot silently regress:

1. Skill-provided ``ExecutableBuiltinTool`` must be dispatchable — the
   builtin fallback searched only ``agent.tools`` and never ``agent.skills``,
   so a call to a skill's builtin resolved to "tool not found" even though
   ``build_tools()`` had offered it to the model.
2. Non-str tool results must still obey ``max_result_tokens`` — the cost cap
   was applied only to str results; non-str values were stringified and
   returned uncapped (both the main path and the HITL resume path).
3. HITL resumption must route through the shared wrapped path so it honors
   agent-global tool middleware, the per-tool timeout, ``content_and_artifact``
   unwrapping, and cooperative ``ToolRetry`` — calling ``on_invoke`` directly
   skipped all four.
4. The middleware malformed-JSON fallback must drain streaming iterators
   instead of leaking an undrained async generator.
5. The middleware terminal must serialise empty args to ``"{}"`` (valid JSON),
   never ``""``.
6. The skill-tools branch of ``build_tools`` must apply the same
   ``defer_loading`` visibility filter as the agent-tools branch.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from troopai.adk.agents.middleware import Middleware
from troopai.adk.exceptions import ToolRetry
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.config import RunConfig
from troopai.adk.run.context import RunContext
from troopai.adk.run.llm_calls import build_tools
from troopai.adk.run.tools_executor import (
    execute_approved_tool,
    execute_tool_calls,
    maybe_wrap_with_agent_middleware,
)
from troopai.adk.skills.skill import Skill
from troopai.adk.tools import build_tool_search, function_tool
from troopai.adk.tools.builtin.builtin_tool import ExecutableBuiltinTool
from troopai.adk.tools.deferred_tool import DeferredToolCall
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall
from troopai.adk.types.tools.tool_stream_event import ToolStreamEvent

MINIMAL_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


# ── Shared helpers ───────────────────────────────────────────────────


def _make_agent(
    tools: list[Any] | None = None,
    skills: list[Any] | None = None,
    middleware: Middleware | None = None,
) -> Any:
    return SimpleNamespace(
        name="test_agent",
        tools=tools if tools is not None else [],
        skills=skills if skills is not None else [],
        tool_use_behavior="run_llm_again",
        handoffs=None,
        llm=None,
        llm_config=None,
        output_schema=None,
        hooks=None,
        middleware=middleware if middleware is not None else Middleware(),
    )


def _make_ctx() -> Any:
    return RunContext(context=None)


def _make_tool_call(call_id: str, name: str, args: str = "{}") -> LLMResponseFunctionToolCall:
    return LLMResponseFunctionToolCall(call_id=call_id, name=name, arguments=args)


class _PassthroughToolMiddleware:
    """Minimal plumbing middleware that just forwards to the next link."""

    async def __call__(self, ctx: Any, tool: Any, args: dict[str, Any], next: Any) -> Any:
        return await next(ctx, tool, args)


class _RecordingToolMiddleware:
    """Records the name of each tool it wraps, then forwards."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def __call__(self, ctx: Any, tool: Any, args: dict[str, Any], next: Any) -> Any:
        self.seen.append(tool.name)
        return await next(ctx, tool, args)


# ── Finding 1: skill ExecutableBuiltinTool dispatch ──────────────────


class TestSkillBuiltinDispatch:
    async def test_skill_executable_builtin_is_dispatchable(self) -> None:
        """A call to a skill-provided ExecutableBuiltinTool executes it."""

        async def _builtin_invoke(_ctx: Any, _raw: str) -> str:
            return "builtin-ran"

        builtin = ExecutableBuiltinTool(
            name="mem_store",
            description="store a memory",
            schema=MINIMAL_SCHEMA,
            on_invoke=_builtin_invoke,
        )
        # ``Tool`` enumerates concrete ExecutableBuiltinTool subclasses; a bare
        # instance is a valid runtime stand-in (dispatch uses an isinstance
        # check), so type the skill tool list as ``list[Any]``.
        skill_tools: list[Any] = [builtin]
        skill = Skill(name="memkit", description="memory skill", tools=skill_tools)
        agent = _make_agent(tools=[], skills=[skill])
        config = RunConfig(fail_on_tool_error=False)

        results, deferred = await execute_tool_calls(
            agent=agent,
            tool_calls=[_make_tool_call("c1", "mem_store")],
            ctx_wrapper=_make_ctx(),
            hooks=RunHooks(),
            config=config,
            model="gpt-4o-mini",
        )

        assert deferred is None
        assert len(results) == 1
        # Pre-fix: the builtin fallback only scanned agent.tools, so this
        # resolved to a "tool not found" message instead of executing.
        assert results[0].output == "builtin-ran"


# ── Finding 2: non-str results obey max_result_tokens ────────────────


class TestNonStrResultCap:
    async def test_main_path_caps_non_str_result(self) -> None:
        """A large non-str result is truncated to max_result_tokens."""

        async def _big_handler(_ctx: Any, _raw: str) -> Any:
            return list(range(5000))  # non-str, far above the token cap

        tool = FunctionTool(
            name="big",
            description="returns a big list",
            schema=MINIMAL_SCHEMA,
            on_invoke=_big_handler,
            max_result_tokens=10,
        )
        agent = _make_agent(tools=[tool])
        config = RunConfig(fail_on_tool_error=False)

        results, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[_make_tool_call("c1", "big")],
            ctx_wrapper=_make_ctx(),
            hooks=RunHooks(),
            config=config,
            model="gpt-4o-mini",
        )

        # Pre-fix: non-str results were only ``str()``-ed, bypassing the cap.
        assert "[Result truncated" in results[0].output

    async def test_hitl_path_caps_non_str_result(self) -> None:
        """The HITL resume path caps a non-str result the same way."""

        async def _big_handler(_ctx: Any, _raw: str) -> Any:
            return list(range(5000))

        tool = FunctionTool(
            name="big",
            description="returns a big list",
            schema=MINIMAL_SCHEMA,
            on_invoke=_big_handler,
            max_result_tokens=10,
        )
        agent = _make_agent(tools=[tool])
        config = RunConfig(fail_on_tool_error=False)
        approved = DeferredToolCall(
            tool_call_id="c1",
            tool_name="big",
            tool_arguments={},
            raw_arguments="{}",
        )

        result, success = await execute_approved_tool(agent, approved, _make_ctx(), RunHooks(), config, None)

        assert success is True
        assert "[Result truncated" in result


# ── Finding 3: HITL resume routes through the shared wrapped path ─────


class TestApprovedToolWrappedPath:
    async def test_tool_retry_surfaces_hint_not_error(self) -> None:
        """ToolRetry on resumption returns the hint verbatim, not an error."""

        async def _retry_handler(_ctx: Any, _raw: str) -> str:
            raise ToolRetry("please call with a valid id")

        tool = FunctionTool(
            name="rt",
            description="retries",
            schema=MINIMAL_SCHEMA,
            on_invoke=_retry_handler,
        )
        agent = _make_agent(tools=[tool])
        config = RunConfig(fail_on_tool_error=False)
        approved = DeferredToolCall(tool_call_id="c1", tool_name="rt", tool_arguments={}, raw_arguments="{}")

        result, success = await execute_approved_tool(agent, approved, _make_ctx(), RunHooks(), config, None)

        assert success is True
        # Pre-fix: on_invoke was called directly; the re-raised ToolRetry was
        # caught by ``except Exception`` and turned into a tool-error message.
        assert result == "please call with a valid id"

    async def test_agent_global_middleware_runs_on_resume(self) -> None:
        """Agent-global tool middleware wraps the resumed tool call."""

        async def _ok_handler(_ctx: Any, _raw: str) -> str:
            return "ran"

        tool = FunctionTool(
            name="mw",
            description="ok",
            schema=MINIMAL_SCHEMA,
            on_invoke=_ok_handler,
        )
        recorder = _RecordingToolMiddleware()
        agent = _make_agent(tools=[tool], middleware=Middleware(tools=[recorder]))
        config = RunConfig(fail_on_tool_error=False)
        approved = DeferredToolCall(tool_call_id="c1", tool_name="mw", tool_arguments={}, raw_arguments="{}")

        result, success = await execute_approved_tool(agent, approved, _make_ctx(), RunHooks(), config, None)

        assert success is True
        assert result == "ran"
        # Pre-fix: on_invoke was called directly, so the middleware never ran.
        assert recorder.seen == ["mw"]

    async def test_timeout_enforced_on_resume(self) -> None:
        """A per-tool timeout fires on resumption instead of running unbounded."""

        async def _slow_handler(_ctx: Any, _raw: str) -> str:
            await asyncio.sleep(0.2)
            return "too late"

        tool = FunctionTool(
            name="slow",
            description="slow",
            schema=MINIMAL_SCHEMA,
            on_invoke=_slow_handler,
            timeout=0.01,
        )
        agent = _make_agent(tools=[tool])
        config = RunConfig(fail_on_tool_error=False)
        approved = DeferredToolCall(tool_call_id="c1", tool_name="slow", tool_arguments={}, raw_arguments="{}")

        result, success = await execute_approved_tool(agent, approved, _make_ctx(), RunHooks(), config, None)

        assert success is True
        # Pre-fix: no timeout was applied, so the handler ran to completion.
        assert result != "too late"
        assert "timed out" in result

    async def test_content_and_artifact_extracts_content_on_resume(self) -> None:
        """content_and_artifact tuple keeps the content, not the tuple repr."""

        async def _ca_handler(_ctx: Any, _raw: str) -> Any:
            return ("summary text", {"rows": [1, 2, 3]})

        tool = FunctionTool(
            name="ca",
            description="content+artifact",
            schema=MINIMAL_SCHEMA,
            on_invoke=_ca_handler,
            response_format="content_and_artifact",
        )
        agent = _make_agent(tools=[tool])
        config = RunConfig(fail_on_tool_error=False)
        approved = DeferredToolCall(tool_call_id="c1", tool_name="ca", tool_arguments={}, raw_arguments="{}")

        result, success = await execute_approved_tool(agent, approved, _make_ctx(), RunHooks(), config, None)

        assert success is True
        # Pre-fix: the whole (content, artifact) tuple was stringified.
        assert result == "summary text"


# ── Finding 4: middleware malformed-JSON fallback drains iterators ────


class TestMiddlewareMalformedJsonDrain:
    async def test_streaming_iterator_drained_on_malformed_json(self) -> None:
        """Malformed JSON in the middleware path still drains a streaming tool."""

        async def _streaming_handler(_ctx: Any, _raw: str) -> Any:
            async def _gen() -> Any:
                yield ToolStreamEvent(type="done", response="drained-final")

            return _gen()

        tool = SimpleNamespace(
            on_invoke=_streaming_handler,
            name="streamer",
            streaming=True,
            response_format="text",
        )
        wrapped = maybe_wrap_with_agent_middleware(tool, [_PassthroughToolMiddleware()])
        assert wrapped is not None

        ctx = SimpleNamespace(tool_call_id="c1")
        result = await wrapped(ctx, "{not valid json")

        # Pre-fix: the malformed-JSON branch returned the undrained async
        # generator, which the executor would stringify to a generator repr.
        assert not hasattr(result, "__aiter__")
        assert result == "drained-final"


# ── Finding 5: middleware terminal serialises empty args to "{}" ──────


class TestMiddlewareEmptyArgs:
    async def test_empty_args_serialised_as_object(self) -> None:
        """The middleware terminal passes "{}" to on_invoke for empty args."""
        seen: dict[str, str] = {}

        async def _record_handler(_ctx: Any, raw_args: str) -> str:
            seen["raw"] = raw_args
            return "ok"

        tool = FunctionTool(
            name="rec",
            description="records raw args",
            schema=MINIMAL_SCHEMA,
            on_invoke=_record_handler,
        )
        wrapped = maybe_wrap_with_agent_middleware(tool, [_PassthroughToolMiddleware()])
        assert wrapped is not None

        ctx = SimpleNamespace(tool_call_id="c1")
        await wrapped(ctx, "{}")

        # Pre-fix: empty args serialised to "" (invalid JSON) instead of "{}".
        assert seen["raw"] == "{}"


# ── Finding 6: skill defer_loading visibility filter ─────────────────


def _make_build_tools_agent(tools: list[Any], skills: list[Any]) -> Any:
    return SimpleNamespace(
        name="test_agent",
        tools=tools,
        skills=skills,
        llm=None,
        llm_config=None,
        handoffs=None,
        output_schema=None,
        system_prompt="test",
    )


class TestSkillDeferLoadingFilter:
    async def test_unrevealed_skill_deferred_tool_hidden(self) -> None:
        """An unrevealed skill defer_loading tool is not offered to the LLM."""

        @function_tool(name="deferred_skill_tool", defer_loading=True)
        def deferred_skill_tool(q: str) -> str:
            return "r"

        skill = Skill(name="s", description="d", tools=[deferred_skill_tool])
        agent = _make_build_tools_agent(tools=[], skills=[skill])

        result = await build_tools(agent)

        # Pre-fix: the skill branch skipped the defer_loading filter, so the
        # tool was always advertised.
        offered = [t for t in (result if result is not None else []) if isinstance(t, FunctionTool)]
        assert all(t.name != "deferred_skill_tool" for t in offered)

    async def test_revealed_skill_deferred_tool_appears(self) -> None:
        """Once revealed, the skill defer_loading tool is offered again."""

        @function_tool(name="deferred_skill_tool", defer_loading=True)
        def deferred_skill_tool(q: str) -> str:
            return "r"

        search = build_tool_search([])
        state = search.get_search_state()
        assert state is not None
        state.reveal("deferred_skill_tool")

        skill = Skill(name="s", description="d", tools=[deferred_skill_tool])
        agent = _make_build_tools_agent(tools=[search], skills=[skill])

        result = await build_tools(agent)

        assert result is not None
        offered = [t for t in result if isinstance(t, FunctionTool)]
        assert any(t.name == "deferred_skill_tool" for t in offered)
