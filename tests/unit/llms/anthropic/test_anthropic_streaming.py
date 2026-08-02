"""Streaming-path regression tests for ``AnthropicLLM``.

Pins the bug fixes documented in the rewrite of
``anthropic_model.py::_stream``:

1. ``signature_delta`` no longer crashes with IndexError when it
   arrives before the matching thinking part has been appended.
2. ``RawMessageStartEvent.message.usage.input_tokens`` is preserved
   on the final ``LLMResponse.usage`` instead of always reporting 0.
3. ``RawMessageDeltaEvent.delta.stop_reason`` lands on
   ``LLMResponse.finish_reason``.
4. Block-index ordering is stable across interleaved deltas.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest
from anthropic.types import (
    Message,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStopEvent,
    SignatureDelta,
    StopReason,
    TextBlock,
    TextDelta,
    ThinkingBlock,
    ThinkingDelta,
    Usage,
)
from anthropic.types.raw_message_delta_event import Delta as RawMessageDeltaEventDelta

from troopai.adk.llms.anthropic import AnthropicLLM
from troopai.adk.types.responses.llm_response import (
    LLMResponseReasoning,
    LLMResponseText,
    LLMStreamEvent,
)


class _FakeAsyncStream:
    """Minimal AsyncStream stand-in that yields a fixed list of events.

    The model code's ``isinstance(stream_iter, AsyncStream)`` assertion
    is bypassed by patching ``_call_messages`` directly rather than
    going through the SDK boundary — that lets tests exercise the
    event-handling logic without instantiating the real
    :class:`anthropic.AsyncStream` (which has constructor requirements
    we don't want to mock).
    """

    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.closed = False

    def __aiter__(self) -> _FakeAsyncStream:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration as e:
            raise StopAsyncIteration from e

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def _patch_async_stream(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Replace ``anthropic_model.AsyncStream`` with the test stand-in.

    Used to bypass the model code's
    ``isinstance(stream_iter, AsyncStream)`` assertion at the SDK
    boundary without instantiating the real
    :class:`anthropic.AsyncStream`. ``monkeypatch`` restores the
    original symbol after the test, preventing cross-test bleed.
    """
    from troopai.adk.llms.anthropic import anthropic_model

    monkeypatch.setattr(anthropic_model, "AsyncStream", _FakeAsyncStream)
    yield


def _install_streaming_mock(llm: AnthropicLLM, events: list[Any]) -> AsyncMock:
    """Patch ``llm._call_messages`` to return a fake stream.

    Avoids the SDK's :class:`anthropic.AsyncStream` instance check by
    monkey-patching the boundary helper rather than the SDK itself.
    The fixture ``_patch_async_stream`` covers the matching
    isinstance assertion on the model module — every test that uses
    this helper MUST request the fixture.
    """
    fake_stream = _FakeAsyncStream(events)
    call_mock = AsyncMock(return_value=fake_stream)
    # ``method-assign``: replacing the bound method on a single test
    # instance is the standard pytest mocking idiom; the type checker
    # warns because ``_call_messages`` is declared on the class.
    llm._call_messages = call_mock  # type: ignore[method-assign]
    return call_mock


def _start_event(input_tokens: int = 12) -> RawMessageStartEvent:
    return RawMessageStartEvent(
        type="message_start",
        message=Message(
            id="msg_stream",
            type="message",
            role="assistant",
            model="claude-sonnet-4",
            content=[],
            stop_reason=None,
            stop_sequence=None,
            usage=Usage(input_tokens=input_tokens, output_tokens=0),
        ),
    )


def _delta_event(
    output_tokens: int,
    stop_reason: StopReason = "end_turn",
) -> RawMessageDeltaEvent:
    return RawMessageDeltaEvent(
        type="message_delta",
        delta=RawMessageDeltaEventDelta(stop_reason=stop_reason, stop_sequence=None),
        usage=MessageDeltaUsage(output_tokens=output_tokens),
    )


@pytest.mark.usefixtures("_patch_async_stream")
class TestSignatureDeltaBugFix:
    async def test_signature_delta_before_block_stop_does_not_crash(self) -> None:
        # Reproduce the order: thinking block start -> signature_delta
        # -> thinking_delta -> block_stop. The prior implementation
        # would IndexError on signature_delta.
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        events = [
            _start_event(input_tokens=20),
            RawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=ThinkingBlock(type="thinking", thinking="", signature=""),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=SignatureDelta(type="signature_delta", signature="sig-part-1"),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=ThinkingDelta(type="thinking_delta", thinking="reasoning..."),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=SignatureDelta(type="signature_delta", signature="-part-2"),
            ),
            RawContentBlockStopEvent(type="content_block_stop", index=0),
            _delta_event(output_tokens=8),
            RawMessageStopEvent(type="message_stop"),
        ]
        _install_streaming_mock(llm, events)

        stream = await llm.acomplete(messages="reason", stream=True)
        collected: list[Any] = []
        async for event in stream:
            collected.append(event)

        # Final 'done' event must carry an LLMResponse with one
        # reasoning part holding the merged signature.
        done = collected[-1]
        assert done.type == "done"
        assert done.response is not None
        reasoning_parts = [p for p in done.response.response if isinstance(p, LLMResponseReasoning)]
        assert len(reasoning_parts) == 1
        assert reasoning_parts[0].thinking == "reasoning..."
        assert reasoning_parts[0].signature == "sig-part-1-part-2"


@pytest.mark.usefixtures("_patch_async_stream")
class TestStreamingUsageInputTokens:
    async def test_input_tokens_captured_from_message_start(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        events = [
            _start_event(input_tokens=42),
            RawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=TextBlock(type="text", text="", citations=None),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=TextDelta(type="text_delta", text="hello"),
            ),
            RawContentBlockStopEvent(type="content_block_stop", index=0),
            _delta_event(output_tokens=7, stop_reason="end_turn"),
        ]
        _install_streaming_mock(llm, events)

        stream = await llm.acomplete(messages="hi", stream=True)
        done_event = None
        async for event in stream:
            if event.type == "done":
                done_event = event

        assert done_event is not None
        assert done_event.response is not None
        usage = done_event.response.usage
        assert usage is not None
        assert usage.input_tokens == 42  # bug fix: was 0 before
        assert usage.output_tokens == 7
        assert usage.total_tokens == 49


@pytest.mark.usefixtures("_patch_async_stream")
class TestStreamClosesUnderlyingStream:
    """The SDK stream is closed on both normal completion and early abandon."""

    @staticmethod
    def _text_events() -> list[Any]:
        return [
            _start_event(input_tokens=5),
            RawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=TextBlock(type="text", text="", citations=None),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=TextDelta(type="text_delta", text="hi"),
            ),
            RawContentBlockStopEvent(type="content_block_stop", index=0),
            _delta_event(output_tokens=3),
            RawMessageStopEvent(type="message_stop"),
        ]

    async def test_closed_after_full_consume(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        fake_stream = _FakeAsyncStream(self._text_events())
        llm._call_messages = AsyncMock(return_value=fake_stream)  # type: ignore[method-assign]

        stream = await llm.acomplete(messages="hi", stream=True)
        async for _event in stream:
            pass

        assert fake_stream.closed is True

    async def test_closed_on_early_abandon(self) -> None:
        # Exercise the ``_stream`` generator directly: aclosing it while
        # suspended mid-flight must run its ``finally`` and close the SDK
        # stream. (The public ``acomplete`` return wraps ``_stream`` in the
        # shared streaming-error contract, which does not itself forward
        # aclose to sub-iterators — that is a separate concern; here we pin
        # that ``_stream`` releases the stream when it is closed.)
        from troopai.adk.llms.llm_config import LLMConfig

        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        fake_stream = _FakeAsyncStream(self._text_events())
        llm._call_messages = AsyncMock(return_value=fake_stream)  # type: ignore[method-assign]

        # ``_stream`` is declared ``AsyncIterator`` but is an async
        # generator at runtime; narrow so ``aclose()`` is visible.
        gen = cast(
            "AsyncGenerator[LLMStreamEvent, None]",
            llm._stream(
                messages=[],
                max_tokens=100,
                system=None,
                tools=None,
                tool_choice=None,
                thinking=None,
                config=LLMConfig(),
                service_tier=None,
                effort=None,
                mid_system=False,
                extra_body={},
            ),
        )
        async for _event in gen:
            break  # abandon the stream mid-flight
        await gen.aclose()

        assert fake_stream.closed is True


@pytest.mark.usefixtures("_patch_async_stream")
class TestStreamingFinishReason:
    async def test_stop_reason_propagated_to_finish_reason(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        events = [
            _start_event(),
            RawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=TextBlock(type="text", text="", citations=None),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=TextDelta(type="text_delta", text="x"),
            ),
            RawContentBlockStopEvent(type="content_block_stop", index=0),
            _delta_event(output_tokens=3, stop_reason="max_tokens"),
        ]
        _install_streaming_mock(llm, events)

        stream = await llm.acomplete(messages="hi", stream=True)
        done_event = None
        async for event in stream:
            if event.type == "done":
                done_event = event

        assert done_event is not None
        assert done_event.response is not None
        assert done_event.response.finish_reason == "max_tokens"


@pytest.mark.usefixtures("_patch_async_stream")
class TestStreamingPartOrdering:
    async def test_text_delta_yields_part_delta_event(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        events = [
            _start_event(),
            RawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=TextBlock(type="text", text="", citations=None),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=TextDelta(type="text_delta", text="Hello"),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=TextDelta(type="text_delta", text=", world!"),
            ),
            RawContentBlockStopEvent(type="content_block_stop", index=0),
            _delta_event(output_tokens=3),
        ]
        _install_streaming_mock(llm, events)

        stream = await llm.acomplete(messages="hi", stream=True)
        deltas: list[str] = []
        done_event = None
        async for event in stream:
            if event.type == "part_delta" and event.delta is not None:
                deltas.append(event.delta)
            elif event.type == "done":
                done_event = event

        assert deltas == ["Hello", ", world!"]
        assert done_event is not None
        assert done_event.response is not None
        text_parts = [p for p in done_event.response.response if isinstance(p, LLMResponseText)]
        assert len(text_parts) == 1
        assert text_parts[0].text == "Hello, world!"


@pytest.mark.usefixtures("_patch_async_stream")
class TestAnthropicStreamingErrorContract:
    async def test_midstream_error_emits_done_error_then_reraises(self) -> None:
        """A mid-stream SDK failure yields a terminal done(finish_reason='error') then re-raises.

        Proves the cross-provider ``stream_with_error_contract`` wrapper is
        wired into ``AnthropicLLM.acomplete``'s streaming path.
        """

        class _RaisingStream(_FakeAsyncStream):
            def __init__(self) -> None:
                super().__init__([])

            async def __anext__(self) -> Any:
                raise RuntimeError("anthropic stream blew up")

        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        llm._call_messages = AsyncMock(return_value=_RaisingStream())  # type: ignore[method-assign]

        stream = await llm.acomplete(messages="hi", stream=True)
        collected: list[Any] = []
        with pytest.raises(RuntimeError, match="anthropic stream blew up"):
            async for event in stream:
                collected.append(event)

        done = [e for e in collected if e.type == "done"]
        assert len(done) == 1
        assert done[0].response is not None
        assert done[0].response.finish_reason == "error"
