"""Unit tests for the LLM-middleware module.

Covers:

- ``compose_llm_middleware`` chain order (outer-to-inner, then unwind)
- Empty-list zero-overhead identity
- Messages visibility inside middleware
- Response transformation propagating back through outer middleware
- ``LLMMiddlewareTermination`` short-circuiting the chain
- Streaming-mode skip with debug log
- ``LLMLoggingMiddleware`` log records via ``caplog``
- ``LLMMetricsMiddleware`` recorder calls

The wiring inside ``call_llm`` (compose chain around ``llm.acomplete``)
is exercised end-to-end in
``tests/integration/test_all_middleware_layers.py``.
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.llms.llm_middleware import (
    LLMLoggingMiddleware,
    LLMMetricsMiddleware,
    LLMMetricsRecorder,
    LLMMiddleware,
    LLMMiddlewareTermination,
    compose_llm_middleware,
)
from troopai.adk.run.context import RunContext
from troopai.adk.run.llm_calls import call_llm_streamed
from troopai.adk.types.input import LLMInputContentItem
from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText
from troopai.adk.types.tokens import LLMUsage


def _agent(name: str = "X", llm: str = "gpt-4o-mini") -> Agent:
    return Agent(name=name, system_prompt="t", llm=llm)


def _ctx() -> RunContext[Any]:
    return RunContext.make(None)


def _response(text: str = "hi", model: str = "gpt-4o-mini") -> LLMResponse:
    return LLMResponse(
        response_id="r1",
        model=model,
        response=[LLMResponseText(type="text", text=text)],
        usage=LLMUsage(requests=1, input_tokens=1, output_tokens=1, total_tokens=2),
    )


class _OrderRecorder:
    def __init__(self, label: str, order: list[str]) -> None:
        self.label = label
        self.order = order

    async def __call__(
        self,
        agent: Agent,
        messages: list[LLMInputContentItem],
        llm_config: LLMConfig | None,
        context: RunContext[Any],
        next: Any,
    ) -> LLMResponse:
        self.order.append(f"+{self.label}")
        response = await next(messages, llm_config)
        self.order.append(f"-{self.label}")
        return response


class TestChainComposition:
    async def test_outer_to_inner_then_unwind(self) -> None:
        order: list[str] = []
        agent = _agent()
        ctx = _ctx()

        async def terminal(msgs: list[LLMInputContentItem], cfg: LLMConfig | None) -> LLMResponse:
            order.append("=terminal")
            return _response()

        chain = compose_llm_middleware(
            [
                _OrderRecorder("A", order),
                _OrderRecorder("B", order),
                _OrderRecorder("C", order),
            ],
            terminal,
            agent=agent,
            context=ctx,
        )
        await chain([], None)
        assert order == ["+A", "+B", "+C", "=terminal", "-C", "-B", "-A"]

    async def test_empty_list_returns_terminal_unchanged(self) -> None:
        agent = _agent()
        ctx = _ctx()

        async def terminal(msgs, cfg):  # type: ignore[no-untyped-def]
            return _response()

        chain = compose_llm_middleware([], terminal, agent=agent, context=ctx)
        assert chain is terminal


class TestMessagesVisibility:
    async def test_middleware_sees_messages(self) -> None:
        seen: list[int] = []

        class Inspect:
            async def __call__(self, agent, messages, llm_config, context, next):  # type: ignore[no-untyped-def]
                seen.append(len(messages))
                return await next(messages, llm_config)

        async def terminal(msgs, cfg):  # type: ignore[no-untyped-def]
            return _response()

        chain = compose_llm_middleware([Inspect()], terminal, agent=_agent(), context=_ctx())
        await chain([{"role": "user", "content": "a"}, {"role": "user", "content": "b"}], None)
        assert seen == [2]


class TestResponseTransformation:
    async def test_middleware_can_overwrite_response(self) -> None:
        class Overwrite:
            async def __call__(self, agent, messages, llm_config, context, next):  # type: ignore[no-untyped-def]
                response = await next(messages, llm_config)
                # Replace the model field in-place via the inner field
                # — LLMResponse is a non-frozen dataclass.
                response.model = "REPLACED"
                return response

        async def terminal(msgs, cfg):  # type: ignore[no-untyped-def]
            return _response(model="ORIGINAL")

        chain = compose_llm_middleware([Overwrite()], terminal, agent=_agent(), context=_ctx())
        out = await chain([], None)
        assert out.model == "REPLACED"


class TestTermination:
    async def test_termination_returns_through_outer_middleware(self) -> None:
        """``compose_llm_middleware`` catches the termination at the
        raising middleware's frame and converts it into a normal
        return. Outer middleware therefore sees the carried response
        as a regular ``await next(…)`` result and runs its post-call
        code (e.g. metrics ``record_outcome(success=True)``).
        """
        terminal_mock = AsyncMock()
        sentinel = _response(text="from-cache")
        order: list[str] = []

        class CacheHit:
            async def __call__(self, agent, messages, llm_config, context, next):  # type: ignore[no-untyped-def]
                raise LLMMiddlewareTermination(sentinel)

        chain = compose_llm_middleware(
            [_OrderRecorder("A", order), CacheHit()],
            terminal_mock,
            agent=_agent(),
            context=_ctx(),
        )
        response = await chain([], None)
        assert response is sentinel
        # Outer "+A" fired AND "-A" fired (chain unwound via normal
        # return). Terminal was NOT called.
        assert order == ["+A", "-A"]
        terminal_mock.assert_not_called()


class TestStreamingPathSkip:
    async def test_streaming_path_emits_debug_skip(
        self,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Configure an agent with non-empty llms middleware list.
        from troopai.adk.agents.middleware import Middleware

        called_count: list[int] = []

        class InvasiveMW:
            async def __call__(self, agent, messages, llm_config, context, next):  # type: ignore[no-untyped-def]
                called_count.append(1)
                return await next(messages, llm_config)

        agent = Agent(
            name="StreamingAgent",
            system_prompt="t",
            llm="gpt-4o-mini",
            middleware=Middleware(llms=[InvasiveMW()]),
        )

        # Stub out everything past the debug-skip point so we can
        # focus on the skip behaviour. Any further machinery is
        # exercised by the integration test.
        async def fake_acomplete(*args, **kwargs):  # type: ignore[no-untyped-def]
            # Coroutine function (no yield) — ``await llm.acomplete(...)``
            # executes this body. Raising before returning lets the
            # test assert on the skip-debug log without exercising the
            # downstream stream-consumption machinery.
            raise RuntimeError("stop test before stream consumption")

        fake_llm = MagicMock()
        fake_llm.acomplete = fake_acomplete
        monkeypatch.setattr(
            "troopai.adk.run.llm_calls.resolve_llm",
            lambda agent, config: fake_llm,
        )
        monkeypatch.setattr(
            "troopai.adk.run.llm_calls.build_tools",
            AsyncMock(return_value=None),
        )
        monkeypatch.setattr(
            "troopai.adk.run.llm_calls.resolve_output_schema",
            lambda agent: None,
        )
        monkeypatch.setattr(
            "troopai.adk.run.llm_calls.resolve_llm_config",
            lambda agent, tool_choice_override=None: None,
        )

        from troopai.adk.run.config import RunConfig
        from troopai.adk.run.stream import RunResultStreaming

        result = MagicMock(spec=RunResultStreaming)

        with (
            caplog.at_level(logging.WARNING, logger="troopai.adk.run.llm_calls"),
            pytest.raises(RuntimeError, match="stop test"),
        ):
            await call_llm_streamed(
                agent=agent,
                messages=[],
                config=RunConfig(),
                result=result,
            )

        # Middleware was NOT invoked despite being registered.
        assert called_count == []
        # Mismatch warning WAS emitted (non-streaming middleware
        # registered while making a streaming call → user notices
        # the path mismatch and registers via stream_llms instead).
        skip_records = [r for r in caplog.records if "Non-streaming LLMMiddleware does not apply" in r.message]
        assert len(skip_records) == 1


class TestLLMLoggingMiddleware:
    async def test_start_and_end_records_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        async def terminal(msgs, cfg):  # type: ignore[no-untyped-def]
            return _response()

        with caplog.at_level(logging.INFO, logger="troopai.adk.llms.llm_middleware"):
            chain = compose_llm_middleware([LLMLoggingMiddleware()], terminal, agent=_agent("L"), context=_ctx())
            await chain([], None)

        starting = [r for r in caplog.records if "llm call starting" in r.message]
        completed = [r for r in caplog.records if "llm call completed" in r.message]
        assert len(starting) == 1
        assert "L" in starting[0].message
        assert len(completed) == 1


class TestLLMMetricsMiddleware:
    async def test_recorder_called_on_success(self) -> None:
        durations: list[tuple[str, float]] = []
        outcomes: list[tuple[str, bool]] = []

        class Sink:
            def record_duration(self, model: str, duration_seconds: float) -> None:
                durations.append((model, duration_seconds))

            def record_outcome(self, model: str, *, success: bool) -> None:
                outcomes.append((model, success))

        sink = Sink()
        assert isinstance(sink, LLMMetricsRecorder)

        async def terminal(msgs, cfg):  # type: ignore[no-untyped-def]
            return _response(model="m1")

        chain = compose_llm_middleware(
            [LLMMetricsMiddleware(recorder=sink)],
            terminal,
            agent=_agent(),
            context=_ctx(),
        )
        await chain([], None)
        assert durations == [("m1", durations[0][1])]
        assert outcomes == [("m1", True)]

    async def test_recorder_called_on_failure(self) -> None:
        outcomes: list[tuple[str, bool]] = []

        class Sink:
            def record_duration(self, model: str, duration_seconds: float) -> None:
                pass

            def record_outcome(self, model: str, *, success: bool) -> None:
                outcomes.append((model, success))

        async def terminal(msgs, cfg):  # type: ignore[no-untyped-def]
            raise RuntimeError("provider down")

        chain = compose_llm_middleware(
            [LLMMetricsMiddleware(recorder=Sink())],
            terminal,
            agent=_agent("err-agent", llm="gpt-x"),
            context=_ctx(),
        )
        with pytest.raises(RuntimeError):
            await chain([], None)
        # On failure, the model label is the agent's llm string fallback
        # (we never got a response.model).
        assert outcomes == [("gpt-x", False)]


class TestProtocolStructuralConformance:
    def test_logging_middleware_satisfies_protocol(self) -> None:
        assert isinstance(LLMLoggingMiddleware(), LLMMiddleware)

    def test_metrics_middleware_satisfies_protocol(self) -> None:
        class _NoopSink:
            def record_duration(self, model: str, duration_seconds: float) -> None:
                pass

            def record_outcome(self, model: str, *, success: bool) -> None:
                pass

        assert isinstance(LLMMetricsMiddleware(recorder=_NoopSink()), LLMMiddleware)
