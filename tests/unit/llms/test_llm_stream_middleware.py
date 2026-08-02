"""Unit tests for the streaming LLM-middleware module.

Covers:

- ``compose_llm_stream_middleware`` chain order (outer-to-inner,
  then unwind on iterator drain).
- Empty-list zero-overhead identity.
- Iterator passthrough idiom: middleware re-yields events.
- ``LLMStreamMiddlewareTermination`` short-circuiting the chain.
- ``LLMStreamLoggingMiddleware`` log records via ``caplog``.
- ``LLMStreamMetricsMiddleware`` recorder calls.
- ``make_logging_middlewares`` factory: paired registration backed
  by independent classes (Protocol satisfaction is per-class).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, cast

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.llms.llm_middleware import LLMMiddleware
from troopai.adk.llms.llm_stream_middleware import (
    LLMStreamLoggingMiddleware,
    LLMStreamMetricsMiddleware,
    LLMStreamMiddleware,
    LLMStreamMiddlewareTermination,
    compose_llm_stream_middleware,
    make_logging_middlewares,
)
from troopai.adk.run.context import RunContext
from troopai.adk.types.input import LLMInputContentItem
from troopai.adk.types.responses.llm_response import LLMResponse, LLMStreamEvent


def _agent(name: str = "X", llm: str = "gpt-4o-mini") -> Agent:
    return Agent(name=name, system_prompt="t", llm=llm)


def _ctx() -> RunContext[Any]:
    return RunContext.make(None)


def _response() -> LLMResponse:
    return LLMResponse(response_id="r1", model="gpt-4o-mini")


def _make_event(typ: str, **kwargs: Any) -> LLMStreamEvent:
    # Helper for terser construction across tests.
    return LLMStreamEvent(type=typ, **kwargs)  # type: ignore[arg-type]


async def _scripted_terminal(
    events: list[LLMStreamEvent],
) -> AsyncIterator[LLMStreamEvent]:
    for event in events:
        yield event


# ── Chain composition ──────────────────────────────────────────────


class TestChainComposition:
    async def test_empty_list_returns_terminal_unchanged(self) -> None:
        agent = _agent()
        ctx = _ctx()

        async def terminal(
            messages: list[LLMInputContentItem],
            llm_config: LLMConfig | None,
        ) -> AsyncIterator[LLMStreamEvent]:
            return _scripted_terminal([_make_event("done", response=_response())])

        chain = compose_llm_stream_middleware([], terminal, agent=agent, context=ctx)
        # Identity (the same callable, no wrapping).
        assert chain is terminal

    async def test_outer_to_inner_then_unwind_on_drain(self) -> None:
        order: list[str] = []
        agent = _agent()
        ctx = _ctx()

        class Recorder:
            def __init__(self, label: str) -> None:
                self.label = label

            async def __call__(
                self,
                a: Agent,
                m: list[LLMInputContentItem],
                cfg: LLMConfig | None,
                c: RunContext[Any],
                next: Any,
            ) -> AsyncIterator[LLMStreamEvent]:
                order.append(f"+{self.label}")
                inner = await next(m, cfg)

                async def relay() -> AsyncIterator[LLMStreamEvent]:
                    async for event in inner:
                        yield event
                    order.append(f"-{self.label}")

                return relay()

        async def terminal(m: list[LLMInputContentItem], cfg: LLMConfig | None) -> AsyncIterator[LLMStreamEvent]:
            order.append("=terminal")
            return _scripted_terminal([_make_event("done", response=_response())])

        chain = compose_llm_stream_middleware(
            [Recorder("A"), Recorder("B"), Recorder("C")],
            terminal,
            agent=agent,
            context=ctx,
        )

        result = await chain([], None)
        # Drain to fire the "-X" tags.
        async for _ in result:
            pass

        # Outer A wraps B wraps C wraps terminal. Pre-yield order is
        # outer-to-inner; post-drain order is inner-to-outer.
        assert order == ["+A", "+B", "+C", "=terminal", "-C", "-B", "-A"]


class TestTermination:
    async def test_short_circuit_returns_carried_iterator(self) -> None:
        agent = _agent()
        ctx = _ctx()
        cached_event = _make_event("done", response=_response())

        async def cached_iter() -> AsyncIterator[LLMStreamEvent]:
            yield cached_event

        class Cache:
            async def __call__(
                self,
                a: Agent,
                m: list[LLMInputContentItem],
                cfg: LLMConfig | None,
                c: RunContext[Any],
                next: Any,
            ) -> AsyncIterator[LLMStreamEvent]:
                raise LLMStreamMiddlewareTermination(cached_iter())

        terminal_called = []

        async def terminal(m: list[LLMInputContentItem], cfg: LLMConfig | None) -> AsyncIterator[LLMStreamEvent]:
            terminal_called.append(True)
            return _scripted_terminal([])

        chain = compose_llm_stream_middleware([Cache()], terminal, agent=agent, context=ctx)
        result = await chain([], None)
        consumed = [event async for event in result]

        assert consumed == [cached_event]
        assert terminal_called == []  # short-circuit never reaches terminal


# ── LLMStreamLoggingMiddleware ─────────────────────────────────────


class TestLLMStreamLoggingMiddleware:
    async def test_start_and_end_records_with_delta_count(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        events = [
            _make_event("part_start", index=0),
            _make_event("part_delta", index=0, delta="A"),
            _make_event("part_delta", index=0, delta="B"),
            _make_event("done", response=_response()),
        ]

        async def terminal(m: list[LLMInputContentItem], cfg: LLMConfig | None) -> AsyncIterator[LLMStreamEvent]:
            return _scripted_terminal(events)

        mw = LLMStreamLoggingMiddleware()
        with caplog.at_level(logging.INFO, logger="troopai.adk.llms.llm_stream_middleware"):
            chain = compose_llm_stream_middleware([mw], terminal, agent=_agent(), context=_ctx())
            result = await chain([], None)
            consumed = [event async for event in result]

        assert len(consumed) == 4

        starting = [r for r in caplog.records if "stream call starting" in r.message]
        completed = [r for r in caplog.records if "stream call completed" in r.message]
        assert len(starting) == 1
        assert len(completed) == 1
        assert "deltas=2" in completed[0].message  # two part_delta events

    async def test_failed_record_emitted_when_iterator_raises(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Completion log must be emitted even when the stream raises mid-way.

        Regression: the completion log inside relay() was only reachable after
        the async for exhausted normally; a mid-stream exception bypassed it so
        operators saw start but never end.
        """

        async def bad_terminal(m: list[LLMInputContentItem], cfg: LLMConfig | None) -> AsyncIterator[LLMStreamEvent]:
            async def gen() -> AsyncIterator[LLMStreamEvent]:
                yield _make_event("part_delta", index=0, delta="x")
                raise RuntimeError("mid-stream boom")

            return gen()

        mw = LLMStreamLoggingMiddleware(log_delta_count=True)
        with caplog.at_level(logging.INFO, logger="troopai.adk.llms.llm_stream_middleware"):
            chain = compose_llm_stream_middleware([mw], bad_terminal, agent=_agent(), context=_ctx())
            result = await chain([], None)

            with pytest.raises(RuntimeError, match="mid-stream boom"):
                async for _ in result:
                    pass

        # Both a starting record AND a failed record must be present.
        starting = [r for r in caplog.records if "stream call starting" in r.message]
        failed = [r for r in caplog.records if "stream call failed" in r.message]
        assert len(starting) == 1, "expected a 'starting' log record"
        assert len(failed) == 1, f"expected a 'failed' log record, got records: {[r.message for r in caplog.records]}"


# ── LLMStreamMetricsMiddleware ─────────────────────────────────────


class _RecordingMetricsRecorder:
    def __init__(self) -> None:
        self.durations: list[tuple[str, float]] = []
        self.outcomes: list[tuple[str, bool]] = []

    def record_duration(self, model: str, duration_seconds: float) -> None:
        self.durations.append((model, duration_seconds))

    def record_outcome(self, model: str, *, success: bool) -> None:
        self.outcomes.append((model, success))


class TestLLMStreamMetricsMiddleware:
    async def test_records_success_after_clean_drain(self) -> None:
        recorder = _RecordingMetricsRecorder()
        events = [_make_event("done", response=_response())]

        async def terminal(m: list[LLMInputContentItem], cfg: LLMConfig | None) -> AsyncIterator[LLMStreamEvent]:
            return _scripted_terminal(events)

        mw = LLMStreamMetricsMiddleware(recorder=recorder)
        chain = compose_llm_stream_middleware([mw], terminal, agent=_agent(), context=_ctx())
        result = await chain([], None)
        async for _ in result:
            pass

        assert len(recorder.durations) == 1
        assert recorder.outcomes == [("gpt-4o-mini", True)]

    async def test_records_failure_when_iterator_raises(self) -> None:
        recorder = _RecordingMetricsRecorder()

        async def bad_terminal(m: list[LLMInputContentItem], cfg: LLMConfig | None) -> AsyncIterator[LLMStreamEvent]:
            async def gen() -> AsyncIterator[LLMStreamEvent]:
                yield _make_event("part_delta", delta="x")
                raise RuntimeError("provider blew up")

            return gen()

        mw = LLMStreamMetricsMiddleware(recorder=recorder)
        chain = compose_llm_stream_middleware([mw], bad_terminal, agent=_agent(), context=_ctx())
        result = await chain([], None)

        with pytest.raises(RuntimeError, match="provider blew up"):
            async for _ in result:
                pass

        assert len(recorder.durations) == 1
        assert recorder.outcomes == [("gpt-4o-mini", False)]

    async def test_cancelled_error_not_recorded_as_failure(self) -> None:
        """A CancelledError mid-drain is not a call failure — record nothing.

        Regression: the relay's ``except BaseException`` recorded success=False
        on cancellation, reporting a phantom failure in the metrics.
        """
        import asyncio

        recorder = _RecordingMetricsRecorder()

        async def cancelling_terminal(
            m: list[LLMInputContentItem], cfg: LLMConfig | None
        ) -> AsyncIterator[LLMStreamEvent]:
            async def gen() -> AsyncIterator[LLMStreamEvent]:
                yield _make_event("part_delta", index=0, delta="x")
                raise asyncio.CancelledError

            return gen()

        mw = LLMStreamMetricsMiddleware(recorder=recorder)
        chain = compose_llm_stream_middleware([mw], cancelling_terminal, agent=_agent(), context=_ctx())
        result = await chain([], None)

        with pytest.raises(asyncio.CancelledError):
            async for _ in result:
                pass

        # Neither a success nor a (phantom) failure outcome was recorded.
        assert recorder.outcomes == []

    async def test_early_close_not_recorded_as_failure(self) -> None:
        """Closing the stream early (GeneratorExit) must not record a failure."""
        recorder = _RecordingMetricsRecorder()

        async def terminal(m: list[LLMInputContentItem], cfg: LLMConfig | None) -> AsyncIterator[LLMStreamEvent]:
            async def gen() -> AsyncIterator[LLMStreamEvent]:
                yield _make_event("part_delta", index=0, delta="x")
                yield _make_event("part_delta", index=0, delta="y")
                yield _make_event("done", response=_response())

            return gen()

        mw = LLMStreamMetricsMiddleware(recorder=recorder)
        chain = compose_llm_stream_middleware([mw], terminal, agent=_agent(), context=_ctx())
        result = await chain([], None)

        stream = cast("AsyncGenerator[LLMStreamEvent, None]", result)
        await stream.__anext__()  # consume one event
        await stream.aclose()  # abandon the stream early → GeneratorExit in relay

        assert ("gpt-4o-mini", False) not in recorder.outcomes


# ── Factory: paired (LLMMiddleware, LLMStreamMiddleware) ───────────


class TestMakeLoggingMiddlewares:
    def test_returns_paired_protocol_satisfying_classes(self) -> None:
        ns, st = make_logging_middlewares()

        # The non-streaming half satisfies LLMMiddleware Protocol.
        assert isinstance(ns, LLMMiddleware)
        # The streaming half satisfies LLMStreamMiddleware Protocol.
        assert isinstance(st, LLMStreamMiddleware)
        # They are NOT the same class — Protocol satisfaction is
        # per-class, and the two return types
        # (LLMResponse vs AsyncIterator[LLMStreamEvent]) cannot
        # coexist on a single __call__ signature.
        assert type(ns) is not type(st)

    def test_factory_accepts_custom_logger(self) -> None:
        custom = logging.getLogger("custom-llm-logger")
        ns, st = make_logging_middlewares(custom, level=logging.DEBUG)

        # The factory returns wrapper instances that delegate to
        # the supplied logger; verify we still satisfy the Protocols.
        assert isinstance(ns, LLMMiddleware)
        assert isinstance(st, LLMStreamMiddleware)
