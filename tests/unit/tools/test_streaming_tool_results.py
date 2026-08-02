"""Unit tests for streaming tool results (#6).

Covers:

- ``ToolStreamEvent`` shape and discriminator literals.
- ``FunctionTool.__post_init__`` rejections of the four incoherent
  combinations (``cache``, ``cache_function``,
  ``response_format='content_and_artifact'``, ``return_direct``).
- ``drain_streaming_tool_value`` accumulates the ``"done"``
  payload and forwards non-terminal events to the active
  ``ToolStreamSink``.
- Pass-through behaviour for non-async-iterator values (so layered
  terminals can call the helper unconditionally).
- Warning emission when a streaming tool runs under the
  non-streaming path (no sink set).
- Middleware-preservation invariant: a ``ToolMiddleware`` registered
  on the agent observes the final accumulated value, not individual
  chunks.
- Full executor exercise via ``execute_tool_calls_streamed`` showing
  the consumer receives ``TOOL_CALLED`` → ``TOOL_PARTIAL_OUTPUT*`` →
  ``TOOL_OUTPUT`` while only one ``TOOL_OUTPUT`` carries the final
  accumulated value.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest

from troopai.adk.agents.middleware import Middleware
from troopai.adk.run.context import RunContext
from troopai.adk.run.stream import (
    CancelMode,
    RunItemStreamEvent,
    RunItemType,
    RunResultStreaming,
)
from troopai.adk.run.tools_executor import (
    _TOOL_STREAM_SINK,
    drain_streaming_tool_value,
    execute_tool_calls,
    execute_tool_calls_streamed,
    maybe_wrap_with_agent_middleware,
)
from troopai.adk.tools import (
    FunctionTool,
    ToolMiddlewareNext,
    function_tool,
)
from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall
from troopai.adk.types.tools import ToolStreamEvent

# ── ToolStreamEvent shape ─────────────────────────────────────────────


def test_tool_stream_event_part_delta_minimal() -> None:
    event = ToolStreamEvent(type="part_delta", delta="chunk")
    assert event.type == "part_delta"
    assert event.delta == "chunk"
    assert event.response is None


def test_tool_stream_event_done_carries_response() -> None:
    event = ToolStreamEvent(type="done", response="final value")
    assert event.type == "done"
    assert event.response == "final value"
    assert event.delta is None


def test_tool_stream_event_part_index_for_ordering() -> None:
    event = ToolStreamEvent(type="part_start", index=2)
    assert event.index == 2


# ── __post_init__ rejection of incoherent combinations ──────────────


@pytest.fixture
def streaming_user_function() -> Any:
    async def fn() -> AsyncIterator[ToolStreamEvent]:
        yield ToolStreamEvent(type="done", response="ok")

    return fn


def test_streaming_rejects_cache_true(streaming_user_function: Any) -> None:
    with pytest.raises(ValueError, match="streaming=True is incoherent with cache=True"):
        function_tool(name="t", description="d", streaming=True, cache=True)(streaming_user_function)


def test_streaming_rejects_cache_function(streaming_user_function: Any) -> None:
    def always_cache(args: str, result: str) -> bool:
        del args, result
        return True

    with pytest.raises(ValueError, match="streaming=True is incoherent with cache_function"):
        function_tool(name="t", description="d", streaming=True, cache_function=always_cache)(streaming_user_function)


def test_streaming_rejects_content_and_artifact(streaming_user_function: Any) -> None:
    with pytest.raises(ValueError, match="response_format='content_and_artifact'"):
        function_tool(
            name="t",
            description="d",
            streaming=True,
            response_format="content_and_artifact",
        )(streaming_user_function)


def test_streaming_rejects_return_direct(streaming_user_function: Any) -> None:
    with pytest.raises(ValueError, match="return_direct=True"):
        function_tool(name="t", description="d", streaming=True, return_direct=True)(streaming_user_function)


# ── drain_streaming_tool_value semantics ───────────────────────────


@pytest.mark.asyncio
async def test_drain_pass_through_for_non_iterator() -> None:
    # Strings, FunctionToolCallResult, and arbitrary scalars must
    # pass through unchanged so layered terminals can call the helper
    # unconditionally without coordinating which one is innermost.
    assert await drain_streaming_tool_value("scalar", "tool") == "scalar"
    assert await drain_streaming_tool_value(42, "tool") == 42


class _RecordingSink:
    """Captures events for assertion. Mirrors ``RunResultStreaming.put_event``."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def put_event(self, event: Any) -> None:
        self.events.append(event)


@pytest.mark.asyncio
async def test_drain_forwards_partial_events_and_accumulates_done() -> None:
    sink = _RecordingSink()
    token = _TOOL_STREAM_SINK.set(sink)
    try:

        async def gen() -> AsyncIterator[ToolStreamEvent]:
            yield ToolStreamEvent(type="part_start", index=0)
            yield ToolStreamEvent(type="part_delta", delta="A")
            yield ToolStreamEvent(type="part_delta", delta="B")
            yield ToolStreamEvent(type="done", response="A B C")

        final = await drain_streaming_tool_value(gen(), "tool_x")
    finally:
        _TOOL_STREAM_SINK.reset(token)

    assert final == "A B C"
    # Three non-done events forwarded; the done event is absorbed
    # into the final value rather than re-emitted as partial output.
    assert len(sink.events) == 3
    for event in sink.events:
        assert isinstance(event, RunItemStreamEvent)
        assert event.name == RunItemType.TOOL_PARTIAL_OUTPUT


@pytest.mark.asyncio
async def test_drain_warns_when_no_sink(caplog: pytest.LogCaptureFixture) -> None:
    async def gen() -> AsyncIterator[ToolStreamEvent]:
        yield ToolStreamEvent(type="part_delta", delta="x")
        yield ToolStreamEvent(type="done", response="final")

    with caplog.at_level(logging.WARNING, logger="troopai.adk.run.tools_executor"):
        final = await drain_streaming_tool_value(gen(), "tool_y")

    assert final == "final"
    assert any("Streaming tool 'tool_y'" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_drain_acloses_generator_on_sink_exception() -> None:
    """Sink raise mid-drain MUST close the producer's finally blocks.

    Without explicit ``aclose()``, the user's async generator stays
    abandoned until GC — leaking sockets / file handles / DB cursors
    held open in ``finally`` blocks for streaming tools.
    """
    cleanup_ran: list[bool] = []

    async def gen() -> AsyncIterator[ToolStreamEvent]:
        try:
            yield ToolStreamEvent(type="part_delta", delta="x")
            yield ToolStreamEvent(type="part_delta", delta="y")  # never reached
        finally:
            cleanup_ran.append(True)

    class _RaisingSink:
        async def put_event(self, event: Any) -> None:
            del event
            raise RuntimeError("sink saturated")

    token = _TOOL_STREAM_SINK.set(_RaisingSink())
    try:
        with pytest.raises(RuntimeError, match="sink saturated"):
            await drain_streaming_tool_value(gen(), "tool_acleanup")
    finally:
        _TOOL_STREAM_SINK.reset(token)

    assert cleanup_ran == [True], "producer's finally must run on sink exception"


# ── Middleware preservation: drain in terminal ──────────────────────


class _FinalResultRecorder:
    """Middleware that captures the result it receives via ``next``.

    The drain-in-terminal invariant says middleware sees the
    accumulated ``"done"`` payload, NOT individual chunks. This
    middleware records what it actually saw.
    """

    def __init__(self) -> None:
        self.observed_outputs: list[Any] = []

    async def __call__(
        self,
        ctx: ToolContext,
        tool: FunctionTool,
        args: dict[str, Any],
        next: ToolMiddlewareNext,
    ) -> Any:
        result = await next(ctx, tool, args)
        self.observed_outputs.append(result.output)
        return result


def _make_streaming_tool() -> FunctionTool:
    @function_tool(name="streamy", description="emits chunks then done", streaming=True)
    async def _streamy() -> AsyncIterator[ToolStreamEvent]:
        yield ToolStreamEvent(type="part_delta", delta="alpha ")
        yield ToolStreamEvent(type="part_delta", delta="beta")
        yield ToolStreamEvent(type="done", response="alpha beta — final")

    return _streamy


@pytest.mark.asyncio
async def test_middleware_observes_final_accumulated_value_not_chunks() -> None:
    tool = _make_streaming_tool()
    recorder = _FinalResultRecorder()

    invoke = maybe_wrap_with_agent_middleware(tool, [recorder])
    assert invoke is not None

    # No sink set: chunks discarded, drain still produces final value.
    output = await invoke(_make_tool_ctx(tool, "{}"), "{}")

    # Middleware saw exactly one result containing the final value.
    assert recorder.observed_outputs == ["alpha beta — final"]
    # Outer caller also gets the final value (executor strips back to .output).
    assert output == "alpha beta — final"


# ── Full executor exercise (streaming path) ─────────────────────────


def _make_tool_ctx(tool: FunctionTool, raw: str) -> ToolContext:
    return ToolContext(
        tool_name=tool.name,
        tool_call_id="call_1",
        tool_arguments={},
        raw_arguments=raw,
    )


def _make_executor_fixtures(
    tools: list[FunctionTool],
) -> tuple[Any, RunContext[Any], Any, Any, RunResultStreaming, list[LLMResponseFunctionToolCall]]:
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.run.config import DEFAULT_RUN_CONFIG

    agent = SimpleNamespace(
        name="streaming-test-agent",
        tools=tools,
        tool_use_behavior="run_llm_again",
        handoffs=None,
        llm=None,
        hooks=None,
        middleware=Middleware(),
    )
    ctx = RunContext(context=None)
    hooks = RunHooks()
    config = DEFAULT_RUN_CONFIG
    streaming_result = RunResultStreaming(current_agent=agent)  # type: ignore[arg-type]
    tool_call = LLMResponseFunctionToolCall(call_id="call_1", name=tools[0].name, arguments="{}")
    return agent, ctx, hooks, config, streaming_result, [tool_call]


@pytest.mark.asyncio
async def test_streaming_path_emits_partial_output_and_one_final() -> None:
    tool = _make_streaming_tool()
    agent, ctx, hooks, config, streaming_result, tool_calls = _make_executor_fixtures([tool])

    results, deferred = await execute_tool_calls_streamed(
        agent=agent,
        tool_calls=tool_calls,
        ctx_wrapper=ctx,
        hooks=hooks,
        config=config,
        result=streaming_result,
    )

    assert deferred is None
    assert len(results) == 1
    assert results[0].output == "alpha beta — final"

    # Drain the streaming queue (tool finished, but stream_events was
    # never started). complete() was not called — pull events with
    # get_nowait until empty.
    consumed: list[Any] = []
    while not streaming_result._event_queue.empty():
        consumed.append(streaming_result._event_queue.get_nowait())

    names = [event.name for event in consumed if isinstance(event, RunItemStreamEvent)]
    # One TOOL_CALLED, two TOOL_PARTIAL_OUTPUT (from the two
    # part_delta events; the done event is absorbed into the final
    # value), one TOOL_OUTPUT.
    assert names.count(RunItemType.TOOL_CALLED) == 1
    assert names.count(RunItemType.TOOL_PARTIAL_OUTPUT) == 2
    assert names.count(RunItemType.TOOL_OUTPUT) == 1

    output_events = [e for e in consumed if isinstance(e, RunItemStreamEvent) and e.name == RunItemType.TOOL_OUTPUT]
    assert output_events[0].item["output"] == "alpha beta — final"


@pytest.mark.asyncio
async def test_non_streaming_path_drains_silently_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    tool = _make_streaming_tool()
    agent, ctx, hooks, config, _streaming_result, tool_calls = _make_executor_fixtures([tool])

    # Verify the stream-sink contextvar is unset at the call site
    # (the non-streaming executor MUST NOT set it). The drain helper
    # then logs a warning per call.
    assert _TOOL_STREAM_SINK.get() is None

    with caplog.at_level(logging.WARNING, logger="troopai.adk.run.tools_executor"):
        results, deferred = await execute_tool_calls(
            agent=agent,
            tool_calls=tool_calls,
            ctx_wrapper=ctx,
            hooks=hooks,
            config=config,
        )

    assert deferred is None
    assert len(results) == 1
    assert results[0].output == "alpha beta — final"
    assert any("Streaming tool 'streamy'" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_cancelled_streaming_tool_call_does_not_emit_after_cancel() -> None:
    tool = _make_streaming_tool()
    agent, ctx, hooks, config, streaming_result, tool_calls = _make_executor_fixtures([tool])
    # Append a second tool call so we can confirm the cancel-mode
    # gate prevents launching the next tool. Both calls reference the
    # same tool name.
    tool_calls.append(LLMResponseFunctionToolCall(call_id="call_2", name=tool.name, arguments="{}"))

    streaming_result._cancel_mode = CancelMode.IMMEDIATE

    results, deferred = await execute_tool_calls_streamed(
        agent=agent,
        tool_calls=tool_calls,
        ctx_wrapper=ctx,
        hooks=hooks,
        config=config,
        result=streaming_result,
    )

    # Immediate cancel observed at the top of the loop; no tool calls run.
    assert results == []
    assert deferred is None
