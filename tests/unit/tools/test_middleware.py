"""Unit tests for the tool middleware module.

Covers:

- Chain composition order (outer-to-inner, then unwind)
- Args mutation propagating to inner middleware and the tool
- Result mutation propagating back to outer middleware
- ``ToolMiddlewareTermination`` short-circuiting the chain
- Exception propagation
- ``ToolLoggingMiddleware`` behaviour
- ``ToolMetricsMiddleware`` behaviour with a recorded sink
- ``WrapperToolset`` toolset-scoped middleware composing inside
  the agent-global chain at execution time
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.agents.middleware import Middleware
from troopai.adk.run.tools_executor import maybe_wrap_with_agent_middleware
from troopai.adk.tools import (
    FunctionTool,
    FunctionToolset,
    ToolLoggingMiddleware,
    ToolMetricsMiddleware,
    ToolMetricsRecorder,
    ToolMiddleware,
    ToolMiddlewareTermination,
    WrapperToolset,
    function_tool,
    wrap_tool_with_middleware,
)
from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.types.output.function_tool_call_result import FunctionToolCallResult


@function_tool(name="hello", description="greet")
def hello(name: str) -> str:
    return f"Hello, {name}!"


def _ctx(name: str = "World") -> ToolContext:
    import json

    return ToolContext(
        tool_name="hello",
        tool_call_id="call_1",
        tool_arguments={"name": name},
        raw_arguments=json.dumps({"name": name}),
    )


class _OrderRecorder:
    """Helper middleware that records its own pre/post tags."""

    def __init__(self, label: str, order: list[str]) -> None:
        self.label = label
        self.order = order

    async def __call__(
        self,
        ctx: ToolContext,
        tool: FunctionTool,
        args: dict[str, Any],
        next: Any,
    ) -> FunctionToolCallResult:
        self.order.append(f"+{self.label}")
        result = await next(ctx, tool, args)
        self.order.append(f"-{self.label}")
        return result


class TestChainComposition:
    async def test_outer_to_inner_then_unwind(self) -> None:
        order: list[str] = []
        agent = Agent(
            name="X",
            system_prompt="t",
            tools=[hello],
            middleware=Middleware(
                tools=[
                    _OrderRecorder("A", order),
                    _OrderRecorder("B", order),
                    _OrderRecorder("C", order),
                ],
            ),
        )
        wrapped = maybe_wrap_with_agent_middleware(hello, agent.middleware.tools)
        await wrapped(_ctx(), '{"name": "World"}')
        assert order == ["+A", "+B", "+C", "-C", "-B", "-A"]

    async def test_empty_middleware_returns_original(self) -> None:
        wrapped = maybe_wrap_with_agent_middleware(hello, [])
        # Empty middleware should hand back the exact tool.on_invoke
        # callable — zero overhead.
        assert wrapped is hello.on_invoke

    async def test_none_on_invoke_returns_none(self) -> None:
        bare_tool = FunctionTool(name="bare", schema={"type": "object", "properties": {}})
        wrapped = maybe_wrap_with_agent_middleware(bare_tool, [_OrderRecorder("A", [])])
        assert wrapped is None


class TestArgsMutation:
    async def test_middleware_can_mutate_args_before_next(self) -> None:
        class Upper:
            async def __call__(self, ctx, tool, args, next):  # type: ignore[no-untyped-def]
                if "name" in args:
                    args["name"] = args["name"].upper()
                return await next(ctx, tool, args)

        agent = Agent(name="X", system_prompt="t", tools=[hello], middleware=Middleware(tools=[Upper()]))
        wrapped = maybe_wrap_with_agent_middleware(hello, agent.middleware.tools)
        result = await wrapped(_ctx("world"), '{"name": "world"}')
        assert result == "Hello, WORLD!"


class TestResultMutation:
    async def test_middleware_can_transform_result(self) -> None:
        class Wrap:
            async def __call__(self, ctx, tool, args, next):  # type: ignore[no-untyped-def]
                result = await next(ctx, tool, args)
                return result.model_copy(update={"output": f"[wrapped] {result.output}"})

        agent = Agent(name="X", system_prompt="t", tools=[hello], middleware=Middleware(tools=[Wrap()]))
        wrapped = maybe_wrap_with_agent_middleware(hello, agent.middleware.tools)
        result = await wrapped(_ctx(), '{"name": "World"}')
        assert result == "[wrapped] Hello, World!"


class TestShortCircuit:
    async def test_middleware_skipping_next_returns_directly(self) -> None:
        class Cache:
            async def __call__(self, ctx, tool, args, next):  # type: ignore[no-untyped-def]
                return FunctionToolCallResult(
                    type="function_call_output",
                    call_id=ctx.tool_call_id or "",
                    output="[cached]",
                )

        agent = Agent(name="X", system_prompt="t", tools=[hello], middleware=Middleware(tools=[Cache()]))
        wrapped = maybe_wrap_with_agent_middleware(hello, agent.middleware.tools)
        result = await wrapped(_ctx(), '{"name": "World"}')
        assert result == "[cached]"

    async def test_middleware_termination_short_circuits(self) -> None:
        class CircuitBreaker:
            async def __call__(self, ctx, tool, args, next):  # type: ignore[no-untyped-def]
                raise ToolMiddlewareTermination(
                    FunctionToolCallResult(
                        type="function_call_output",
                        call_id=ctx.tool_call_id or "",
                        output="[breaker tripped]",
                    )
                )

        agent = Agent(
            name="X",
            system_prompt="t",
            tools=[hello],
            middleware=Middleware(tools=[CircuitBreaker()]),
        )
        wrapped = maybe_wrap_with_agent_middleware(hello, agent.middleware.tools)
        result = await wrapped(_ctx(), '{"name": "World"}')
        assert result == "[breaker tripped]"


class TestExceptionPropagation:
    async def test_inner_exception_propagates_to_outer(self) -> None:
        seen: list[str] = []

        class Outer:
            async def __call__(self, ctx, tool, args, next):  # type: ignore[no-untyped-def]
                try:
                    return await next(ctx, tool, args)
                except RuntimeError as e:
                    seen.append(str(e))
                    raise

        class Inner:
            async def __call__(self, ctx, tool, args, next):  # type: ignore[no-untyped-def]
                raise RuntimeError("inner boom")

        agent = Agent(
            name="X",
            system_prompt="t",
            tools=[hello],
            middleware=Middleware(tools=[Outer(), Inner()]),
        )
        wrapped = maybe_wrap_with_agent_middleware(hello, agent.middleware.tools)
        with pytest.raises(RuntimeError, match="inner boom"):
            await wrapped(_ctx(), '{"name": "World"}')
        assert seen == ["inner boom"]


class TestToolLoggingMiddleware:
    async def test_logs_start_and_end(self, caplog: pytest.LogCaptureFixture) -> None:
        agent = Agent(
            name="X",
            system_prompt="t",
            tools=[hello],
            middleware=Middleware(tools=[ToolLoggingMiddleware(level=logging.INFO)]),
        )
        wrapped = maybe_wrap_with_agent_middleware(hello, agent.middleware.tools)
        with caplog.at_level("INFO", logger="troopai.adk.tools.tool_middleware"):
            await wrapped(_ctx(), '{"name": "World"}')
        text = caplog.text
        assert "tool 'hello' starting" in text
        assert "tool 'hello' completed" in text

    async def test_log_args_optional(self, caplog: pytest.LogCaptureFixture) -> None:
        agent = Agent(
            name="X",
            system_prompt="t",
            tools=[hello],
            middleware=Middleware(tools=[ToolLoggingMiddleware(log_args=True, log_result=True)]),
        )
        wrapped = maybe_wrap_with_agent_middleware(hello, agent.middleware.tools)
        with caplog.at_level("INFO", logger="troopai.adk.tools.tool_middleware"):
            await wrapped(_ctx(), '{"name": "World"}')
        text = caplog.text
        assert "World" in text  # args logged
        assert "Hello, World!" in text  # result logged


class _RecordingRecorder:
    """In-memory ToolMetricsRecorder for tests."""

    def __init__(self) -> None:
        self.durations: list[tuple[str, float]] = []
        self.outcomes: list[tuple[str, bool]] = []

    def record_duration(self, tool_name: str, duration_seconds: float) -> None:
        self.durations.append((tool_name, duration_seconds))

    def record_outcome(self, tool_name: str, *, success: bool) -> None:
        self.outcomes.append((tool_name, success))


class TestToolMetricsMiddleware:
    async def test_recorder_protocol_acceptance(self) -> None:
        recorder = _RecordingRecorder()
        assert isinstance(recorder, ToolMetricsRecorder)

    async def test_records_duration_and_success(self) -> None:
        recorder = _RecordingRecorder()
        agent = Agent(
            name="X",
            system_prompt="t",
            tools=[hello],
            middleware=Middleware(tools=[ToolMetricsMiddleware(recorder=recorder)]),
        )
        wrapped = maybe_wrap_with_agent_middleware(hello, agent.middleware.tools)
        await wrapped(_ctx(), '{"name": "World"}')
        assert len(recorder.durations) == 1
        assert recorder.durations[0][0] == "hello"
        assert recorder.outcomes == [("hello", True)]

    async def test_records_duration_and_failure_on_exception(self) -> None:
        @function_tool(name="boom", description="fails")
        def boom() -> str:
            raise RuntimeError("nope")

        recorder = _RecordingRecorder()
        agent = Agent(
            name="X",
            system_prompt="t",
            tools=[boom],
            middleware=Middleware(tools=[ToolMetricsMiddleware(recorder=recorder)]),
        )
        wrapped = maybe_wrap_with_agent_middleware(boom, agent.middleware.tools)
        # With the default failure handler, a tool-body exception now
        # propagates (rather than being swallowed into a string) so the
        # executor can apply its retry-budget / fail_on_tool_error policy.
        # The metrics middleware observes the real failure in its ``except``
        # arm: it records the duration and a failed outcome, then re-raises.
        with pytest.raises(RuntimeError, match="nope"):
            await wrapped(
                ToolContext(tool_name="boom", tool_call_id="c", tool_arguments={}, raw_arguments=""),
                "",
            )
        assert len(recorder.durations) == 1
        assert recorder.outcomes == [("boom", False)]


class TestToolsetScopedMiddleware:
    """``WrapperToolset`` applies middleware to each materialised tool."""

    async def test_wrapper_toolset_applies_middleware_at_materialisation(self) -> None:
        order: list[str] = []
        ts = WrapperToolset(
            wrapped=FunctionToolset(tools=[hello]),
            middleware=[_OrderRecorder("toolset", order)],
        )
        out = await ts.get_tools()
        # The materialised tool's on_invoke goes through the toolset
        # middleware. Calling it directly drives the chain.
        invoke = out["hello"].on_invoke
        assert invoke is not None
        await invoke(_ctx(), '{"name": "World"}')
        assert order == ["+toolset", "-toolset"]

    async def test_wrap_tool_with_middleware_helper(self) -> None:
        order: list[str] = []
        wrapped = wrap_tool_with_middleware(hello, [_OrderRecorder("X", order)])
        assert wrapped.on_invoke is not None
        await wrapped.on_invoke(_ctx(), '{"name": "World"}')
        assert order == ["+X", "-X"]

    async def test_empty_middleware_returns_same_tool(self) -> None:
        assert wrap_tool_with_middleware(hello, []) is hello


class TestProtocolConformance:
    def test_tool_logging_middleware_is_tool_middleware(self) -> None:
        # Runtime-checkable Protocol — instances of the shipped
        # standard middleware must satisfy it.
        assert isinstance(ToolLoggingMiddleware(), ToolMiddleware)

    def test_tool_metrics_middleware_is_tool_middleware(self) -> None:
        assert isinstance(ToolMetricsMiddleware(recorder=_RecordingRecorder()), ToolMiddleware)


class TestContentAndArtifactThroughMiddleware:
    """Finding 6: artifact must survive the middleware chain."""

    async def test_artifact_preserved_after_middleware(self) -> None:
        """wrapped_on_invoke must return (output, artifact) when artifact is not None."""
        artifact_payload = [{"doc_id": "d1", "score": 0.99}]

        @function_tool(name="rag", response_format="content_and_artifact")
        def rag(query: str) -> tuple[str, Any]:
            return "Found 1 result", artifact_payload

        wrapped = wrap_tool_with_middleware(rag, [ToolLoggingMiddleware()])
        assert wrapped.on_invoke is not None
        result = await wrapped.on_invoke(_ctx("test_query"), '{"query":"test"}')
        # Result should be the tuple (output, artifact) so executor can
        # rebuild FunctionToolCallResult with both fields
        assert isinstance(result, tuple)
        assert len(result) == 2
        content, art = result
        assert content == "Found 1 result"
        assert art is artifact_payload

    async def test_text_tool_returns_string_not_tuple(self) -> None:
        """Normal text tools must still return a plain string, not a tuple."""

        @function_tool(name="greet")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        wrapped = wrap_tool_with_middleware(greet, [ToolLoggingMiddleware()])
        assert wrapped.on_invoke is not None
        result = await wrapped.on_invoke(_ctx("World"), '{"name":"World"}')
        assert isinstance(result, str)
        assert result == "Hello, World!"
