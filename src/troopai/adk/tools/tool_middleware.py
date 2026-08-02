"""Tool middleware — composable interceptors around tool execution.

A `ToolMiddleware` wraps the call to `tool.on_invoke(...)` inside the
executor so cross-cutting concerns (logging, metrics, distributed
tracing, retries with backoff, agent-global arg injection) can run
without modifying every tool individually.

Middleware lives at two registration points:

1. **Agent-global** — `Agent(middleware=Middleware(tools=[Metrics(), Audit()]))`
   wraps every tool call across all toolsets and standalone tools.
   The `Middleware` config dataclass holds typed lists per layer;
   today only the `tools` slot is wired into the run loop.
2. **Toolset-scoped** — `WrapperToolset(wrapped=..., middleware=[Log()])`
   applies only to tools materialised from that wrapper.

Both surfaces use the same `ToolMiddleware` Protocol — no separate
"toolset middleware" concept. The executor composes both chains: the
agent-global middleware wraps the toolset-scoped middleware (which
already wraps the tool's own ``on_invoke``).

## Why a separate mechanism

Tool guardrails (`tool.guardrails.input`, `tool.guardrails.output`)
are **per-tool typed verdicts** with `.allow()` / `.reject_content()`
/ `.raise_exception()` semantics. They cannot share state between
pre and post (separate functions), cannot transform args going into
the call, cannot short-circuit inner middleware, and must be
re-declared on every tool to apply broadly. `RunHooks.on_tool_start`
/ `on_tool_end` are observation callbacks — they cannot modify args,
results, or control flow.

Middleware fills the gap: a single function that wraps the call
from before to after, owns shared state across pre and post (timer
start, span context), can mutate args going into the next link,
can transform the result coming back, and can short-circuit by
not calling `next()`. Microsoft Agent Framework's
`FunctionMiddleware` is the direct inspiration; Pydantic-AI's
`WrapperToolset.call_tool` override is an alternative model with
the same intent.

## Scope: function-level only

This module ships **function-level** middleware (Microsoft's
`FunctionMiddleware` equivalent) — wrapping a single tool call.
Agent-turn middleware (`AgentMiddleware`-equivalent, wrapping
`Runner.arun`) and LLM-call middleware (`ChatMiddleware`-equivalent,
wrapping `LLM.acomplete`) are separate future features that, if
shipped, would get their own typed contexts rather than being
retrofitted under a shared umbrella.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from troopai.adk.exceptions import TroopAIError
from troopai.adk.types.output.function_tool_call_result import FunctionToolCallResult

if TYPE_CHECKING:
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.tools.tool_context import ToolContext


logger = logging.getLogger(__name__)


ToolMiddlewareNext = Callable[
    ["ToolContext", "FunctionTool", dict[str, Any]],
    Awaitable["FunctionToolCallResult"],
]
"""The ``next`` callable passed to a middleware.

Invokes the rest of the chain (inner middleware, then the tool's own
``on_invoke``). A middleware that does not call ``next`` short-circuits
the chain — useful for caching layers that return a stored result
without invoking the underlying tool.
"""


@runtime_checkable
class ToolMiddleware(Protocol):
    """Plumbing-only interceptor around tool execution.

    Implementations wrap the call to ``tool.on_invoke``. Receive the
    tool context, the tool instance, the parsed args dict, and a
    ``next`` callable. Return the ``FunctionToolCallResult`` either
    by invoking ``next`` and possibly transforming the result, or by
    constructing one directly (short-circuit).

    Plumbing-only contract
    ----------------------
    Middleware MUST encode cross-cutting infrastructure:

      - Logging start/end of calls
      - Metrics emission (latency, outcome)
      - Distributed-tracing span open/close
      - Retry-with-backoff on transient failures
      - Cross-cutting argument injection (request_id, tenant_id,
        correlation IDs from ``RunContext``)
      - Caching (short-circuit on cache hit, populate on miss)

    Middleware MUST NOT encode policy or verdict. Anything that
    answers *"should this call proceed?"* or *"is this content
    acceptable?"* belongs in a typed verdict surface, not here:

      - PII / jailbreak / content filter -> ``ToolInputGuardrail`` or
        agent-level ``AgentInputGuardrail``, which return typed verdicts
        ``allow`` / ``reject_content`` / ``raise_exception``.
      - Approval gates -> ``FunctionTool.requires_approval`` (HITL
        deferral integrated with ``RunState`` resumption).
      - Schema validation -> ``FunctionTool.schema_enforcement``.
      - Rate limiting -> ``FunctionTool.rate_limit``.
      - Output remediation (re-prompt agent) ->
        ``AgentOutputGuardrail.remediation``.

    The decision rule: if the answer to *"should this call proceed?"*
    is involved, it is a guardrail. If only the surrounding plumbing
    (log lines, metric records, retry loop, span boundaries, request
    IDs, cache hits) changes, it is middleware.

    Registration
    ------------
    Implementations are registered via ``Agent.middleware.tools`` for
    agent-global wrapping or ``WrapperToolset.middleware`` for
    toolset-scoped wrapping. The agent-global chain composes outside
    the toolset-scoped chain, which composes outside the tool's own
    ``on_invoke``.

    Example: a logging middleware::

        class LogTimings:
            async def __call__(self, ctx, tool, args, next):
                start = time.monotonic()
                result = await next(ctx, tool, args)
                logger.info("%s took %.3fs", tool.name, time.monotonic() - start)
                return result
    """

    async def __call__(
        self,
        ctx: ToolContext,
        tool: FunctionTool,
        args: dict[str, Any],
        next: ToolMiddlewareNext,
    ) -> FunctionToolCallResult: ...


class ToolMiddlewareTermination(TroopAIError):
    """Short-circuit signal raised by middleware to stop the chain.

    Carries the result the executor should treat as the tool's
    output. Useful for circuit-breaker / cache-hit / replay middleware
    that wants to return a stored result without re-invoking the
    tool.

    ``compose_tool_middleware`` catches this exception **at the
    raising middleware's frame** and returns the carried result as a
    normal value through the rest of the chain. Outer middleware
    therefore sees the carried result as a regular ``await next(…)``
    return — post-call code (timer stop, span close, metrics
    ``record_outcome(success=True)``) runs cleanly. This is the
    typical request-pipeline shape; without it, outer
    metrics/tracing middleware would mis-classify a cache hit as a
    ``BaseException`` failure.
    """

    def __init__(self, result: FunctionToolCallResult) -> None:
        super().__init__("Middleware short-circuit")
        self.result = result


def compose_tool_middleware(
    middleware: list[ToolMiddleware],
    terminal: ToolMiddlewareNext,
) -> ToolMiddlewareNext:
    """Compose a middleware list around a terminal handler.

    The first middleware in the list runs first (outermost); the
    last middleware runs last (innermost, just before ``terminal``).
    This matches the intuition of a request pipeline: the entry of
    the request walks the chain top-to-bottom; the response walks
    back bottom-to-top.

    Empty list returns ``terminal`` unchanged.

    Args:
        middleware: Outer-to-inner ordered list of middleware.
        terminal: The function that runs after the entire chain
            has been traversed (typically the actual tool's
            ``on_invoke`` invocation).

    Returns:
        A ``next`` callable that, when invoked, runs the full chain
        followed by ``terminal``.
    """
    chain: ToolMiddlewareNext = terminal
    # Walk the list right-to-left so the first item ends up wrapping
    # all later items (outermost). Each iteration captures the current
    # ``chain`` in a closure as the next-link for ``mw``. The
    # try/except converts ``ToolMiddlewareTermination`` raised by
    # ``_mw`` (or any inner link) into a normal return value, so outer
    # middleware sees a clean ``await next(…)`` result and runs its
    # post-call code (timer stop, metrics ``success=True``) cleanly.
    for mw in reversed(middleware):
        next_link = chain

        async def wrapped(
            ctx: ToolContext,
            tool: FunctionTool,
            args: dict[str, Any],
            *,
            _mw: ToolMiddleware = mw,
            _next: ToolMiddlewareNext = next_link,
        ) -> FunctionToolCallResult:
            try:
                return await _mw(ctx, tool, args, _next)
            except ToolMiddlewareTermination as term:
                return term.result

        chain = wrapped
    return chain


def wrap_tool_with_middleware(
    tool: FunctionTool,
    middleware: list[ToolMiddleware],
) -> FunctionTool:
    """Return a clone of ``tool`` whose ``on_invoke`` runs through ``middleware``.

    Used by ``WrapperToolset`` to apply toolset-scoped middleware at
    materialisation time. The returned tool has the same name and
    surface API as the original; only the ``on_invoke`` callable is
    replaced with the wrapped version.

    Internal state (``_agent``, ``_cache``, ``_rate_state``,
    ``_search_state``) is preserved by reference.

    Args:
        tool: The tool to wrap.
        middleware: Outer-to-inner ordered list. Empty list is a
            pass-through and returns the original tool unchanged.

    Returns:
        A new ``FunctionTool`` with the wrapped ``on_invoke``.
    """
    if len(middleware) == 0:
        return tool

    original_on_invoke = tool.on_invoke
    if original_on_invoke is None:
        # No invoker means the tool is never executable; the chain has
        # nothing to wrap. Return as-is rather than build a chain that
        # would crash on call.
        return tool

    async def terminal(
        ctx: ToolContext,
        tool_inner: FunctionTool,
        args: dict[str, Any],
    ) -> FunctionToolCallResult:
        # ``tool_inner`` mirrors the protocol shape; the actual call
        # uses the closure-captured ``original_on_invoke``.
        del tool_inner
        # Re-serialise the (possibly mutated) args dict back to JSON
        # so ``original_on_invoke`` sees the JSON-string interface it
        # was authored for. Middleware that mutated ``args`` therefore
        # changes what the tool sees. An empty dict serialises to
        # ``"{}"`` (a valid no-arg object), never ``""`` — the latter
        # is not JSON and would diverge from the no-middleware path.
        raw = json.dumps(args)
        out = await original_on_invoke(ctx, raw)
        # Drain streaming-tool iterators so middleware sees the
        # final accumulated value, not chunks. Imported lazily to
        # avoid a tools_executor import cycle (tool_middleware is
        # imported by tools_executor).
        from troopai.adk.run.tools_executor import drain_streaming_tool_value

        out = await drain_streaming_tool_value(out, tool.name)
        if isinstance(out, FunctionToolCallResult):
            return out
        # Build a minimal result with the call_id taken from the
        # context; the executor overlays max_result_tokens trimming
        # and JSON minification on top of whatever we return.
        # Only ``content_and_artifact`` tools return a 2-tuple
        # (content_str, artifact); preserve the artifact in the Result
        # so downstream middleware and wrapped_on_invoke can forward it.
        # A text-format tool may legitimately return a 2-tuple value
        # (e.g. coordinates), so guard on the declared contract to match
        # the no-middleware executor path — otherwise the second element
        # is silently diverted into the artifact channel.
        if tool.response_format == "content_and_artifact" and isinstance(out, tuple) and len(out) == 2:
            content, artifact = out
            return FunctionToolCallResult(
                type="function_call_output",
                call_id=ctx.tool_call_id or "",
                output=content if isinstance(content, str) else str(content),
                artifact=artifact,
            )
        return FunctionToolCallResult(
            type="function_call_output",
            call_id=ctx.tool_call_id or "",
            output=out if isinstance(out, str) else str(out),
        )

    chain = compose_tool_middleware(middleware, terminal)

    async def wrapped_on_invoke(ctx: ToolContext, raw_args: str) -> Any:
        try:
            parsed: dict[str, Any] = json.loads(raw_args) if len(raw_args) > 0 else {}
        except json.JSONDecodeError:
            # Defer to the original tool, which will surface the JSON
            # error via the standard error path. A streaming tool's
            # ``on_invoke`` returns an async iterator even here, so drain
            # it to the final value — mirroring the terminal closure —
            # rather than handing an undrained iterator back to the
            # executor, which treats a middleware-wrapped tool as already
            # drained.
            from troopai.adk.run.tools_executor import drain_streaming_tool_value

            out = await original_on_invoke(ctx, raw_args)
            return await drain_streaming_tool_value(out, tool.name)
        # ``ToolMiddlewareTermination`` is caught and unwrapped inside
        # ``compose_tool_middleware`` so the chain returns a normal
        # ``FunctionToolCallResult``; outer code receives the cached/
        # synthetic result like any other return value.
        result = await chain(ctx, tool, parsed)
        # The chain Protocol guarantees we have a FunctionToolCallResult
        # here; the executor's normal post-processing wraps the raw
        # output value (minify, max_result_tokens, etc.).
        # For content_and_artifact tools the artifact must survive the
        # middleware chain: return (output, artifact) so the executor
        # rebuilds the FunctionToolCallResult with both fields intact.
        if result.artifact is not None:
            return (result.output, result.artifact)
        return result.output

    return tool.clone(on_invoke=wrapped_on_invoke)


# ---------------------------------------------------------------------------
# Standard middleware shipped out of the box
# ---------------------------------------------------------------------------


@dataclass
class ToolLoggingMiddleware:
    """Single-purpose baseline that logs the start and end of each call.

    Replaces the ad-hoc ``logger.info`` lines users would otherwise
    sprinkle through their tools. Uses the project's standard
    ``logger = logging.getLogger(__name__)`` pattern.

    Attributes:
        level: Log level for both start and end records. Defaults
            to ``logging.INFO``.
        log_args: When ``True``, the args dict is included in the
            start log record. Default ``False`` to avoid leaking
            potentially sensitive parameters.
        log_result: When ``True``, the result is included in the end
            log record. Default ``False`` for the same reason.
    """

    level: int = logging.INFO
    """Log level for both start and end records."""

    log_args: bool = False
    """Include the args dict in the start record."""

    log_result: bool = False
    """Include the result in the end record."""

    async def __call__(
        self,
        ctx: ToolContext,
        tool: FunctionTool,
        args: dict[str, Any],
        next: ToolMiddlewareNext,
    ) -> FunctionToolCallResult:
        if self.log_args:
            logger.log(self.level, "tool '%s' starting with args=%r", tool.name, args)
        else:
            logger.log(self.level, "tool '%s' starting", tool.name)
        result = await next(ctx, tool, args)
        if self.log_result:
            logger.log(self.level, "tool '%s' completed: %r", tool.name, result.output)
        else:
            logger.log(self.level, "tool '%s' completed", tool.name)
        return result


@runtime_checkable
class ToolMetricsRecorder(Protocol):
    """Protocol for ``ToolMetricsMiddleware``'s metrics sink.

    The framework does not bundle a metrics implementation —
    OpenTelemetry, StatsD, Prometheus, or a custom in-memory recorder
    can all conform to this two-method shape. Plug in whichever
    matches your observability stack.
    """

    def record_duration(self, tool_name: str, duration_seconds: float) -> None:
        """Record a tool invocation's wall-clock duration."""

    def record_outcome(self, tool_name: str, *, success: bool) -> None:
        """Record whether a tool invocation succeeded or raised."""


@dataclass
class ToolMetricsMiddleware:
    """Records per-tool latency and success/failure to a recorder.

    The recorder is a small Protocol with two methods —
    ``record_duration`` and ``record_outcome`` — so users can plug
    in StatsD, OpenTelemetry, Prometheus, or an in-memory test
    recorder without the framework taking a dependency on any
    specific metrics library.

    Attributes:
        recorder: The metrics sink. See :class:`ToolMetricsRecorder`.
    """

    recorder: ToolMetricsRecorder
    """The metrics sink that receives duration and outcome records."""

    async def __call__(
        self,
        ctx: ToolContext,
        tool: FunctionTool,
        args: dict[str, Any],
        next: ToolMiddlewareNext,
    ) -> FunctionToolCallResult:
        start = time.monotonic()
        try:
            result = await next(ctx, tool, args)
        except BaseException:
            self.recorder.record_duration(tool.name, time.monotonic() - start)
            self.recorder.record_outcome(tool.name, success=False)
            raise
        self.recorder.record_duration(tool.name, time.monotonic() - start)
        self.recorder.record_outcome(tool.name, success=True)
        return result
