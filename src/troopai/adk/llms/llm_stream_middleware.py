"""Streaming LLM middleware — interceptors around ``LLM.acomplete(stream=True)``.

Sibling Protocol of :class:`~troopai.adk.llms.llm_middleware.LLMMiddleware`
that wraps the streaming-side ``LLM.acomplete(stream=True)`` call.
The non-streaming Protocol returns ``LLMResponse``; this one returns
``AsyncIterator[LLMStreamEvent]``. Two non-overlapping return types
mean two Protocols rather than a polymorphic union — the type
checker never has to narrow at the call site, and existing
``LLMMiddleware`` implementations are untouched (zero breaking
change).

Registered via ``Agent.middleware.stream_llms``. The chain is
composed inside ``call_llm_streamed`` in ``troopai.adk.run.llm_calls``.

## Plumbing-only contract

Same forbidden-vs-allowed split as the non-streaming Protocol —
logging, metrics, tracing, retries, and replay caching are
allowed; verdict-shaped logic (PII, jailbreak, content filter,
budget enforcement) belongs in guardrails or
``RunConfig.usage_limits``. The plumbing-only contract is
codified in the project rules and applies unchanged across all
streaming paths.

## Iterator passthrough idiom

The terminal returns the iterator from ``llm.acomplete(stream=True)``.
A wrapping middleware that wants to inspect events re-yields them::

    class CountTokens:
        async def __call__(self, agent, messages, llm_config, ctx, next):
            inner = await next(messages, llm_config)

            async def relay():
                async for event in inner:
                    if event.type == "part_delta" and event.delta:
                        self._tokens += 1
                    yield event

            return relay()

A short-circuit middleware can return a pre-built iterator::

    class CachedReplay:
        async def __call__(self, agent, messages, llm_config, ctx, next):
            if cache_hit := self._cache.get(...):
                return self._replay(cache_hit)
            return await next(messages, llm_config)

## Pairing with the non-streaming sibling

Use :func:`make_logging_middlewares` to register one shared
implementation on both ``Agent.middleware.llms`` and
``Agent.middleware.stream_llms``. Two separate classes back the
shared helper because Protocol satisfaction depends on
``__call__``'s return-type identity — a single class cannot
structurally satisfy both (``LLMResponse`` vs.
``AsyncIterator[LLMStreamEvent]``).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from troopai.adk.exceptions import TroopAIError

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.llms.llm_config import LLMConfig
    from troopai.adk.llms.llm_middleware import LLMMiddleware, LLMMiddlewareNext
    from troopai.adk.run.context import RunContext
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.responses.llm_response import LLMResponse, LLMStreamEvent


logger = logging.getLogger(__name__)


LLMStreamMiddlewareNext = Callable[
    ["list[LLMInputContentItem]", "LLMConfig | None"],
    Awaitable["AsyncIterator[LLMStreamEvent]"],
]
"""The ``next`` callable passed to an ``LLMStreamMiddleware``.

Awaiting ``next`` returns the inner ``AsyncIterator[LLMStreamEvent]``.
The middleware then either re-yields events (after inspection /
metric-gathering) or returns the iterator unchanged. Not calling
``next`` short-circuits the chain — useful for cache replay.
"""


@runtime_checkable
class LLMStreamMiddleware(Protocol):
    """Plumbing-only interceptor around a single streaming LLM call.

    Implementations wrap one ``LLM.acomplete(..., stream=True)`` call
    and return an ``AsyncIterator[LLMStreamEvent]``. Receive the
    active agent, the message list, the resolved ``LLMConfig`` (or
    ``None``), the run context, and a ``next`` callable.

    See module docstring for the iterator passthrough idiom.
    """

    async def __call__(
        self,
        agent: Agent,
        messages: list[LLMInputContentItem],
        llm_config: LLMConfig | None,
        context: RunContext[Any],
        next: LLMStreamMiddlewareNext,
    ) -> AsyncIterator[LLMStreamEvent]: ...


class LLMStreamMiddlewareTermination(TroopAIError):
    """Short-circuit signal raised by streaming middleware to stop the chain.

    Carries the iterator the runner should consume in place of the
    rest of the chain. Useful for cached / deterministic-replay
    middleware that wants to surface a pre-built event stream
    without invoking the provider.

    Caught at the raising middleware's frame inside
    :func:`compose_llm_stream_middleware`; outer middleware sees the
    carried iterator as a regular ``await next(…)`` return.

    Synthetic-iterator audit note
    -----------------------------
    A pre-built iterator carrying ``LLMStreamEvent`` events with
    ``type="part_start"`` / ``type="part_delta"`` and an
    :class:`~troopai.adk.types.responses.llm_response.LLMResponseFunctionToolCall`
    in the ``part`` field — or a terminal ``"done"`` event whose
    ``response`` carries function-call parts — will cause the runner
    to dispatch those tool calls exactly as if the provider had
    emitted them. This is the design point that makes deterministic
    replay possible on the streaming path; it is also a real
    authority the middleware author is taking on. Treat synthetic-
    iterator middleware (cache, replay) as a security-relevant
    surface, mirroring the non-streaming sibling
    :class:`~troopai.adk.llms.llm_middleware.LLMMiddlewareTermination`.
    """

    def __init__(self, response: AsyncIterator[LLMStreamEvent]) -> None:
        super().__init__("LLMStreamMiddleware short-circuit")
        self.response = response


def compose_llm_stream_middleware(
    middleware: list[LLMStreamMiddleware],
    terminal: LLMStreamMiddlewareNext,
    *,
    agent: Agent,
    context: RunContext[Any],
) -> LLMStreamMiddlewareNext:
    """Compose a streaming middleware list around a terminal handler.

    Mirrors :func:`~troopai.adk.llms.llm_middleware.compose_llm_middleware`
    structurally; the only difference is the return type of every
    link is ``AsyncIterator[LLMStreamEvent]`` instead of
    ``LLMResponse``.

    Empty list returns ``terminal`` unchanged (zero-overhead path).

    Args:
        middleware: Outer-to-inner ordered list of streaming middleware.
        terminal: The function that runs after the entire chain has been
            traversed (the actual ``llm.acomplete(stream=True)`` call).
        agent: The active agent. Bound here so the returned ``next``
            callable matches the simple ``(messages, llm_config) ->
            iterator`` shape callers expect.
        context: The ``RunContext`` to thread through every middleware
            call.

    Returns:
        A ``next`` callable that, when awaited, runs the full chain
        followed by ``terminal`` and returns the resulting
        ``AsyncIterator[LLMStreamEvent]``.
    """
    if len(middleware) == 0:
        return terminal

    chain: LLMStreamMiddlewareNext = terminal
    for mw in reversed(middleware):
        next_link = chain

        async def wrapped(
            messages: list[LLMInputContentItem],
            llm_config: LLMConfig | None,
            *,
            _mw: LLMStreamMiddleware = mw,
            _next: LLMStreamMiddlewareNext = next_link,
            _agent: Agent = agent,
            _ctx: RunContext[Any] = context,
        ) -> AsyncIterator[LLMStreamEvent]:
            try:
                return await _mw(_agent, messages, llm_config, _ctx, _next)
            except LLMStreamMiddlewareTermination as term:
                return term.response

        chain = wrapped
    return chain


# ---------------------------------------------------------------------------
# Standard middleware shipped out of the box
# ---------------------------------------------------------------------------


@dataclass
class LLMStreamLoggingMiddleware:
    """Logs the start and end of each streaming LLM call.

    Records one INFO line at call start (before the iterator is
    consumed), then drains the inner iterator while re-yielding
    events to the outer consumer, then records one INFO line at
    call end carrying the part_delta count.

    Attributes:
        level: Log level for both records.
        log_delta_count: Whether to count ``"part_delta"`` events
            and include the count in the end record.
    """

    level: int = logging.INFO
    """Log level for both start and end records."""

    log_delta_count: bool = True
    """Whether to count ``part_delta`` events for the end record."""

    async def __call__(
        self,
        agent: Agent,
        messages: list[LLMInputContentItem],
        llm_config: LLMConfig | None,
        context: RunContext[Any],
        next: LLMStreamMiddlewareNext,
    ) -> AsyncIterator[LLMStreamEvent]:
        del context
        logger.log(
            self.level,
            "llm stream call starting (agent='%s', messages=%d)",
            agent.name,
            len(messages),
        )
        inner = await next(messages, llm_config)

        async def relay() -> AsyncIterator[LLMStreamEvent]:
            delta_count = 0
            try:
                async for event in inner:
                    if self.log_delta_count and event.type == "part_delta":
                        delta_count += 1
                    yield event
            except (GeneratorExit, asyncio.CancelledError):
                # Consumer closed/cancelled the stream — not a call failure, so
                # do not log a phantom "failed" record. Re-raise per protocol.
                raise
            except BaseException:
                if self.log_delta_count:
                    logger.log(
                        self.level,
                        "llm stream call failed (agent='%s', deltas_before_error=%d)",
                        agent.name,
                        delta_count,
                    )
                else:
                    logger.log(
                        self.level,
                        "llm stream call failed (agent='%s')",
                        agent.name,
                    )
                raise
            if self.log_delta_count:
                logger.log(
                    self.level,
                    "llm stream call completed (agent='%s', deltas=%d)",
                    agent.name,
                    delta_count,
                )
            else:
                logger.log(
                    self.level,
                    "llm stream call completed (agent='%s')",
                    agent.name,
                )

        return relay()


@runtime_checkable
class LLMStreamMetricsRecorder(Protocol):
    """Protocol for ``LLMStreamMetricsMiddleware``'s metrics sink.

    Two-method shape mirroring :class:`LLMMetricsRecorder` — pluggable
    against StatsD, OpenTelemetry, Prometheus, or an in-memory test
    recorder.
    """

    def record_duration(self, model: str, duration_seconds: float) -> None:
        """Record a single streaming call's wall-clock duration (start to drain end)."""

    def record_outcome(self, model: str, *, success: bool) -> None:
        """Record whether a single streaming call drained cleanly or raised."""


@dataclass
class LLMStreamMetricsMiddleware:
    """Records per-call wall-clock duration and drain success.

    Duration covers the entire stream (start of the call until the
    last event is yielded). Outcome is ``success=False`` if the
    iterator raises during drain — captured by wrapping each
    ``async for`` in a try/except.

    Attributes:
        recorder: The metrics sink. See :class:`LLMStreamMetricsRecorder`.
    """

    recorder: LLMStreamMetricsRecorder
    """The metrics sink that receives duration and outcome records."""

    async def __call__(
        self,
        agent: Agent,
        messages: list[LLMInputContentItem],
        llm_config: LLMConfig | None,
        context: RunContext[Any],
        next: LLMStreamMiddlewareNext,
    ) -> AsyncIterator[LLMStreamEvent]:
        del context
        failure_label = agent.llm if isinstance(agent.llm, str) else "unknown"
        start = time.monotonic()
        try:
            inner = await next(messages, llm_config)
        except BaseException:
            self.recorder.record_duration(failure_label, time.monotonic() - start)
            self.recorder.record_outcome(failure_label, success=False)
            raise

        recorder = self.recorder

        async def relay() -> AsyncIterator[LLMStreamEvent]:
            try:
                async for event in inner:
                    yield event
            except (GeneratorExit, asyncio.CancelledError):
                # Consumer closed or cancelled the stream — that is not a drain
                # failure, so record no outcome (recording success=False here
                # would report a phantom failure). Re-raise to honor the
                # generator/cancellation protocol.
                raise
            except BaseException:
                recorder.record_duration(failure_label, time.monotonic() - start)
                recorder.record_outcome(failure_label, success=False)
                raise
            recorder.record_duration(failure_label, time.monotonic() - start)
            recorder.record_outcome(failure_label, success=True)

        return relay()


# ---------------------------------------------------------------------------
# Cross-path factory: paired non-streaming + streaming middleware
# ---------------------------------------------------------------------------


def make_logging_middlewares(
    log: logging.Logger | None = None,
    *,
    level: int = logging.INFO,
) -> tuple[LLMMiddleware, LLMStreamMiddleware]:
    """Return paired ``(LLMMiddleware, LLMStreamMiddleware)`` loggers.

    Use this to register one logger across both LLM-call paths in
    one call. The returned objects are two separate classes (Protocol
    satisfaction depends on ``__call__``'s return-type identity, so a
    single class cannot structurally satisfy both Protocols), but
    they delegate to the same shared logger and prefix.

    Args:
        log: Logger to write to. Defaults to this module's logger.
        level: Log level for both records.

    Returns:
        ``(non_streaming_middleware, streaming_middleware)`` tuple.
        Register each in its respective slot::

            from troopai.adk.agents.middleware import Middleware
            from troopai.adk.llms.llm_stream_middleware import make_logging_middlewares

            ns, st = make_logging_middlewares()
            agent = Agent(
                name="...",
                middleware=Middleware(llms=[ns], stream_llms=[st]),
            )
    """
    from troopai.adk.llms.llm_middleware import LLMLoggingMiddleware

    # Type the locals against the Protocols so each branch's
    # concrete class is assignment-compatible without ``# type: ignore``.
    # Both branches return Protocol-conformant instances; the type
    # checker validates that nominally per branch.
    non_streaming: LLMMiddleware
    streaming: LLMStreamMiddleware
    if log is None:
        non_streaming = LLMLoggingMiddleware(level=level)
        streaming = LLMStreamLoggingMiddleware(level=level)
    else:
        # Caller supplied a non-default logger. Both shipped baselines
        # write via their own module's logger, so route through the
        # ``_LoggingViaTarget`` / ``_StreamingLoggingViaTarget`` helper
        # classes that take an explicit logger argument.
        non_streaming = _LoggingViaTarget(log, level)
        streaming = _StreamingLoggingViaTarget(log, level)
    return non_streaming, streaming


@dataclass
class _LoggingViaTarget:
    """Non-streaming logger middleware that writes to a caller-supplied logger.

    Internal helper for :func:`make_logging_middlewares` when a
    caller passes a non-default logger. Public API is the factory.
    """

    target_log: logging.Logger
    level: int = logging.INFO

    async def __call__(
        self,
        agent: Agent,
        messages: list[LLMInputContentItem],
        llm_config: LLMConfig | None,
        context: RunContext[Any],
        next: LLMMiddlewareNext,
    ) -> LLMResponse:
        del context
        self.target_log.log(self.level, "llm call starting (agent='%s', messages=%d)", agent.name, len(messages))
        response = await next(messages, llm_config)
        self.target_log.log(self.level, "llm call completed (agent='%s', model='%s')", agent.name, response.model)
        return response


@dataclass
class _StreamingLoggingViaTarget:
    """Streaming logger middleware that writes to a caller-supplied logger.

    Internal helper for :func:`make_logging_middlewares` when a
    caller passes a non-default logger. Public API is the factory.
    """

    target_log: logging.Logger
    level: int = logging.INFO

    async def __call__(
        self,
        agent: Agent,
        messages: list[LLMInputContentItem],
        llm_config: LLMConfig | None,
        context: RunContext[Any],
        next: LLMStreamMiddlewareNext,
    ) -> AsyncIterator[LLMStreamEvent]:
        del context
        target_log = self.target_log
        level = self.level
        target_log.log(level, "llm stream call starting (agent='%s', messages=%d)", agent.name, len(messages))
        inner = await next(messages, llm_config)

        async def relay() -> AsyncIterator[LLMStreamEvent]:
            count = 0
            try:
                async for event in inner:
                    if event.type == "part_delta":
                        count += 1
                    yield event
            except (GeneratorExit, asyncio.CancelledError):
                # Consumer closed/cancelled the stream — not a call failure, so
                # do not log a phantom "failed" record. Re-raise per protocol.
                raise
            except BaseException:
                target_log.log(
                    level,
                    "llm stream call failed (agent='%s', deltas_before_error=%d)",
                    agent.name,
                    count,
                )
                raise
            target_log.log(level, "llm stream call completed (agent='%s', deltas=%d)", agent.name, count)

        return relay()
