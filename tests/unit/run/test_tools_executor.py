"""Regression tests for ``run/tools_executor.py``.

Covers two confirmed defects:

1. Agent-global tool middleware corrupting ``content_and_artifact``
   results — the middleware terminal/unwrap pair must preserve the
   artifact and surface the content string, matching the toolset-scoped
   middleware path. Without the fix the artifact is dropped and the LLM
   receives a stringified Python tuple.
2. Malformed LLM tool-call arguments crashing the run — the executor's
   pre-parse of ``tool_call.arguments`` must degrade to an empty dict
   instead of letting ``json.JSONDecodeError`` abort the whole run, on
   both the non-streaming and streaming paths.

End-to-end through ``execute_tool_calls`` /
``execute_tool_calls_streamed`` so the real wrapping + parse paths run.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from troopai.adk.agents.middleware import Middleware
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.config import RunConfig
from troopai.adk.run.context import RunContext
from troopai.adk.run.stream import RunResultStreaming
from troopai.adk.run.tools_executor import execute_tool_calls, execute_tool_calls_streamed
from troopai.adk.tools.function_tool import FunctionTool, function_tool
from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

# ── Helpers ──────────────────────────────────────────────────────────


def _make_agent(tools: list[FunctionTool], middleware: Middleware | None = None) -> Any:
    return SimpleNamespace(
        name="test_agent",
        tools=tools,
        skills=None,
        tool_use_behavior="run_llm_again",
        handoffs=None,
        llm=None,
        hooks=None,
        middleware=middleware if middleware is not None else Middleware(),
    )


def _make_ctx() -> RunContext[Any]:
    return RunContext(context=None)


def _make_hooks() -> RunHooks[Any]:
    return RunHooks()


def _make_streaming_result() -> RunResultStreaming:
    return RunResultStreaming(
        current_agent=SimpleNamespace(name="test_agent", handoffs=[]),  # type: ignore[arg-type]
        max_turns=3,
    )


_SENTINEL_ARTIFACT = {"rows": [1, 2, 3]}


async def _content_and_artifact_handler(_ctx: Any, _raw_args: str) -> tuple[str, Any]:
    """A ``content_and_artifact`` tool: LLM gets the summary, the app
    gets the structured artifact."""
    return ("summary for the llm", _SENTINEL_ARTIFACT)


def _make_artifact_tool() -> FunctionTool:
    return FunctionTool(
        name="rag",
        description="Returns a summary plus an application-side artifact.",
        schema={"type": "object", "properties": {}},
        on_invoke=_content_and_artifact_handler,
        response_format="content_and_artifact",
    )


class _PassthroughLoggingMiddleware:
    """Plumbing-only middleware that forwards the call unchanged.

    Stands in for ``ToolLoggingMiddleware`` — its presence is what
    triggers the agent-global wrapping path in the executor.
    """

    async def __call__(self, ctx: Any, tool: Any, args: dict[str, Any], next: Any) -> Any:
        return await next(ctx, tool, args)


# ── Finding 1: content_and_artifact × agent-global tool middleware ────


async def test_artifact_preserved_with_agent_middleware() -> None:
    """A ``content_and_artifact`` tool wrapped by agent-global tool
    middleware must surface the content string as ``output`` and keep
    the artifact on ``FunctionToolCallResult.artifact`` — not stringify
    the whole tuple and drop the artifact."""
    agent = _make_agent(
        [_make_artifact_tool()],
        middleware=Middleware(tools=[_PassthroughLoggingMiddleware()]),
    )
    call = LLMResponseFunctionToolCall(call_id="c1", name="rag", arguments="{}")

    results, _ = await execute_tool_calls(agent, [call], _make_ctx(), _make_hooks(), RunConfig())

    assert len(results) == 1
    result = results[0]
    # Output is the content string, NOT "('summary for the llm', {...})".
    assert result.output == "summary for the llm"
    assert "(" not in result.output  # no Python-tuple repr leaked to the LLM
    # The artifact survived the middleware chain.
    assert result.artifact == _SENTINEL_ARTIFACT


async def test_artifact_preserved_without_middleware_control() -> None:
    """Control: the same tool with NO agent-global middleware already
    works — confirms the middleware path was the source of corruption,
    not the tool itself."""
    agent = _make_agent([_make_artifact_tool()])
    call = LLMResponseFunctionToolCall(call_id="c1", name="rag", arguments="{}")

    results, _ = await execute_tool_calls(agent, [call], _make_ctx(), _make_hooks(), RunConfig())

    assert len(results) == 1
    assert results[0].output == "summary for the llm"
    assert results[0].artifact == _SENTINEL_ARTIFACT


# ── Finding 2: malformed JSON tool-call arguments ────────────────────
#
# Uses a ``@function_tool``-decorated tool — the production tool shape.
# Its ``on_invoke`` wrapper degrades malformed JSON to a recoverable
# error string. The defect being regression-tested is the executor's
# OWN pre-parse of ``tool_call.arguments`` (before the tool is ever
# invoked): pre-fix it raised ``JSONDecodeError`` and aborted the whole
# run before any of the tool's graceful handling could run.


@function_tool
def echo(text: str = "empty") -> str:
    """Echo the text argument.

    Args:
        text: The text to echo back.
    """
    return text


def _make_echo_tool() -> FunctionTool:
    assert isinstance(echo, FunctionTool)
    return echo


async def test_malformed_json_args_do_not_crash_run() -> None:
    """Malformed JSON in tool-call arguments must not raise
    ``JSONDecodeError`` out of the executor — the run continues with a
    recoverable result the LLM can retry."""
    agent = _make_agent([_make_echo_tool()])
    # Truncated / invalid JSON the way a weak or interrupted LLM emits.
    call = LLMResponseFunctionToolCall(call_id="c1", name="echo", arguments='{"text": "hi')

    results, deferred = await execute_tool_calls(agent, [call], _make_ctx(), _make_hooks(), RunConfig())

    assert deferred is None
    assert len(results) == 1
    # Run continued: a string result was produced rather than the run
    # aborting on the executor's pre-parse.
    assert isinstance(results[0].output, str)


async def test_malformed_json_args_degrade_to_empty_for_handler() -> None:
    """Fully invalid args must not crash the executor pre-parse either —
    the call still produces a result (no uncaught raise)."""
    agent = _make_agent([_make_echo_tool()])
    call = LLMResponseFunctionToolCall(call_id="c1", name="echo", arguments="{not json at all}")

    results, _ = await execute_tool_calls(agent, [call], _make_ctx(), _make_hooks(), RunConfig())

    assert len(results) == 1
    assert isinstance(results[0].output, str)


async def test_malformed_json_args_do_not_crash_streamed() -> None:
    """The streaming path pre-parses arguments for the TOOL_CALLED event;
    malformed JSON there must also degrade instead of crashing."""
    agent = _make_agent([_make_echo_tool()])
    call = LLMResponseFunctionToolCall(call_id="c1", name="echo", arguments='{"text": ')
    stream_result = _make_streaming_result()

    results, deferred = await execute_tool_calls_streamed(
        agent=agent,
        tool_calls=[call],
        ctx_wrapper=_make_ctx(),
        hooks=_make_hooks(),
        config=RunConfig(),
        result=stream_result,
    )

    assert deferred is None
    assert len(results) == 1
    assert isinstance(results[0].output, str)


async def test_valid_json_args_unaffected() -> None:
    """Sanity: well-formed JSON still parses and reaches the handler."""
    agent = _make_agent([_make_echo_tool()])
    call = LLMResponseFunctionToolCall(call_id="c1", name="echo", arguments='{"text": "hello"}')

    results, _ = await execute_tool_calls(agent, [call], _make_ctx(), _make_hooks(), RunConfig())

    assert len(results) == 1
    assert results[0].output == "hello"
