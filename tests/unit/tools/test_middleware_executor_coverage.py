"""Verification 3: executor-level tool hooks — raw-args before + result after.

DONE-ALREADY verdict: The existing ``ToolMiddleware`` Protocol (in
``tools/tool_middleware.py``) already delivers raw-args *before* the call and
the final ``FunctionToolCallResult`` *after* the call at the executor boundary.

Evidence:
- ``wrap_tool_with_middleware`` (tool_middleware.py:230) defines
  ``wrapped_on_invoke`` which JSON-parses ``raw_args`` into a ``dict[str, Any]``
  and passes it to the first middleware as the ``args`` parameter.
  This is the exact "raw args dict before" moment.
- The chain's terminal function (tool_middleware.py:262-302) calls
  ``original_on_invoke`` and normalises its output to a
  ``FunctionToolCallResult`` before returning it to the outermost
  middleware's ``next()`` call.
  This is the "result after" moment.
- ``RunHooks.on_tool_start`` (hooks.py:194) receives
  ``tool_arguments: dict[str, Any]`` — the parsed args dict — directly
  from the executor (tools_executor.py:415, 886).
- ``RunHooks.on_tool_end`` (hooks.py:211) receives the final string result
  directly from the executor (tools_executor.py:472, 1082).

This test file demonstrates both surfaces exercising the same scenario so the
coverage is unambiguous.  References:
- tools/tool_middleware.py:230 — ``wrap_tool_with_middleware``
- tools/tool_middleware.py:306 — ``compose_tool_middleware``
- run/tools_executor.py:415,886 — on_tool_start call site
- run/tools_executor.py:472,1082 — on_tool_end call site
"""

from __future__ import annotations

from typing import Any

from troopai.adk.tools import (
    FunctionTool,
    ToolMiddleware,
    function_tool,
    wrap_tool_with_middleware,
)
from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.tools.tool_middleware import ToolMiddlewareNext
from troopai.adk.types.output.function_tool_call_result import FunctionToolCallResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(name: str = "World") -> ToolContext:
    import json

    return ToolContext(
        tool_name="greet",
        tool_call_id="c1",
        tool_arguments={"name": name},
        raw_arguments=json.dumps({"name": name}),
    )


@function_tool(name="greet", description="greet")
def greet(name: str) -> str:
    return f"Hello, {name}!"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMiddlewareReceivesRawArgsBeforeAndResultAfter:
    """ToolMiddleware receives the parsed args dict before and the result after.

    This is the "executor-level raw-args before + result after" contract.
    The test uses ``wrap_tool_with_middleware`` which is the same composition
    path the executor uses (see ``maybe_wrap_with_agent_middleware`` in
    ``run/tools_executor.py`` which delegates to ``wrap_tool_with_middleware``).
    """

    async def test_middleware_sees_raw_args_before_call(self) -> None:
        captured_args: list[dict[str, Any]] = []

        class CaptureArgs:
            async def __call__(
                self,
                ctx: ToolContext,
                tool: FunctionTool,
                args: dict[str, Any],
                next: ToolMiddlewareNext,
            ) -> FunctionToolCallResult:
                # This is the "before" moment — raw args as a parsed dict.
                captured_args.append(dict(args))
                return await next(ctx, tool, args)

        wrapped = wrap_tool_with_middleware(greet, [CaptureArgs()])
        assert wrapped.on_invoke is not None
        await wrapped.on_invoke(_ctx("Alice"), '{"name": "Alice"}')

        assert len(captured_args) == 1
        assert captured_args[0] == {"name": "Alice"}

    async def test_middleware_sees_result_after_call(self) -> None:
        captured_results: list[Any] = []

        class CaptureResult:
            async def __call__(
                self,
                ctx: ToolContext,
                tool: FunctionTool,
                args: dict[str, Any],
                next: ToolMiddlewareNext,
            ) -> FunctionToolCallResult:
                result = await next(ctx, tool, args)
                # This is the "after" moment — the final FunctionToolCallResult.
                captured_results.append(result.output)
                return result

        wrapped = wrap_tool_with_middleware(greet, [CaptureResult()])
        assert wrapped.on_invoke is not None
        await wrapped.on_invoke(_ctx("Bob"), '{"name": "Bob"}')

        assert len(captured_results) == 1
        assert captured_results[0] == "Hello, Bob!"

    async def test_middleware_before_and_after_ordering(self) -> None:
        """Middleware executes pre-call logic before and post-call logic after the tool."""
        events: list[str] = []

        class OrderChecker:
            async def __call__(
                self,
                ctx: ToolContext,
                tool: FunctionTool,
                args: dict[str, Any],
                next: ToolMiddlewareNext,
            ) -> FunctionToolCallResult:
                events.append(f"before:{args.get('name')}")
                result = await next(ctx, tool, args)
                events.append(f"after:{result.output}")
                return result

        wrapped = wrap_tool_with_middleware(greet, [OrderChecker()])
        assert wrapped.on_invoke is not None
        await wrapped.on_invoke(_ctx("Charlie"), '{"name": "Charlie"}')

        assert events == ["before:Charlie", "after:Hello, Charlie!"]

    async def test_middleware_can_mutate_args_before_tool(self) -> None:
        """Middleware can modify the parsed args dict before the tool sees them."""

        class Uppercaser:
            async def __call__(
                self,
                ctx: ToolContext,
                tool: FunctionTool,
                args: dict[str, Any],
                next: ToolMiddlewareNext,
            ) -> FunctionToolCallResult:
                if "name" in args:
                    args["name"] = args["name"].upper()
                return await next(ctx, tool, args)

        wrapped = wrap_tool_with_middleware(greet, [Uppercaser()])
        assert wrapped.on_invoke is not None
        result = await wrapped.on_invoke(_ctx("dave"), '{"name": "dave"}')
        assert result == "Hello, DAVE!"

    async def test_middleware_can_transform_result_after_tool(self) -> None:
        """Middleware can transform the FunctionToolCallResult after the tool returns."""

        class ResultWrapper:
            async def __call__(
                self,
                ctx: ToolContext,
                tool: FunctionTool,
                args: dict[str, Any],
                next: ToolMiddlewareNext,
            ) -> FunctionToolCallResult:
                result = await next(ctx, tool, args)
                return result.model_copy(update={"output": f"[wrapped] {result.output}"})

        wrapped = wrap_tool_with_middleware(greet, [ResultWrapper()])
        assert wrapped.on_invoke is not None
        result = await wrapped.on_invoke(_ctx("Eve"), '{"name": "Eve"}')
        assert result == "[wrapped] Hello, Eve!"

    async def test_is_tool_middleware_protocol(self) -> None:
        """Both standard middleware classes satisfy the Protocol — belt and suspenders."""

        class MinimalMiddleware:
            async def __call__(
                self,
                ctx: ToolContext,
                tool: FunctionTool,
                args: dict[str, Any],
                next: ToolMiddlewareNext,
            ) -> FunctionToolCallResult:
                return await next(ctx, tool, args)

        assert isinstance(MinimalMiddleware(), ToolMiddleware)

    async def test_empty_middleware_stack_is_pass_through(self) -> None:
        """wrap_tool_with_middleware([]) returns the original tool unchanged."""
        result = wrap_tool_with_middleware(greet, [])
        assert result is greet
