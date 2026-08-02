"""Tests for StreamedAudioResult: ordering, lifecycle balance, errors."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import cast, override

from troopai.adk.voice.events import VoiceStreamEvent
from troopai.adk.voice.result import DEFAULT_AUDIO_QUEUE_SIZE, StreamedAudioResult
from troopai.adk.voice.tts import TTSModel, TTSModelSettings


class WordTTS(TTSModel):
    """Synthesizes a segment to its UTF-8 bytes in one chunk."""

    @property
    @override
    def model_name(self) -> str:
        return "word-tts"

    @override
    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        yield text.encode("utf-8")


class FloodTTS(TTSModel):
    """Emits far more audio chunks than the output queue can hold.

    Used to drive the synthesis task into a blocked ``put()`` on a full
    bounded output queue, so that abandoning the consumer exercises the
    cancellation / cleanup path.
    """

    @property
    @override
    def model_name(self) -> str:
        return "flood-tts"

    @override
    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        for _ in range(DEFAULT_AUDIO_QUEUE_SIZE * 4):
            yield b"x"


class BoomTTS(TTSModel):
    """Raises after the first (empty) chunk to exercise error handling."""

    @property
    @override
    def model_name(self) -> str:
        return "boom-tts"

    @override
    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        yield b""
        raise RuntimeError("synth boom")


def _normalize(event: VoiceStreamEvent) -> str:
    if event.type == "voice_stream_event_audio":
        return "audio"
    if event.type == "voice_stream_event_lifecycle":
        return event.event
    return "error"


async def _collect(result: StreamedAudioResult) -> list[VoiceStreamEvent]:
    return [event async for event in result.stream()]


async def test_lifecycle_brackets_audio_in_order():
    result = StreamedAudioResult(WordTTS(), TTSModelSettings())
    result.start()
    await result.start_turn()
    await result.add_text("This is a long enough sentence. ")
    await result.add_text("Second part is also long enough. ")
    await result.end_turn()
    await result.complete()

    events = await _collect(result)
    assert [_normalize(e) for e in events] == [
        "turn_started",
        "audio",
        "audio",
        "turn_ended",
        "session_ended",
    ]
    audio = b"".join(e.data for e in events if e.type == "voice_stream_event_audio")
    assert audio == b"This is a long enough sentence.Second part is also long enough."


async def test_empty_turn_still_emits_balanced_lifecycle():
    result = StreamedAudioResult(WordTTS(), TTSModelSettings())
    result.start()
    await result.start_turn()
    await result.end_turn()
    await result.complete()

    events = await _collect(result)
    assert [_normalize(e) for e in events] == ["turn_started", "turn_ended", "session_ended"]


async def test_short_turn_is_flushed_on_end():
    result = StreamedAudioResult(WordTTS(), TTSModelSettings())
    result.start()
    await result.start_turn()
    await result.add_text("Hi.")  # below the splitter minimum
    await result.end_turn()
    await result.complete()

    events = await _collect(result)
    audio = b"".join(e.data for e in events if e.type == "voice_stream_event_audio")
    assert audio == b"Hi."


async def test_synthesis_error_surfaces_as_error_event():
    result = StreamedAudioResult(BoomTTS(), TTSModelSettings())
    result.start()
    await result.start_turn()
    await result.add_text("Long enough sentence to synthesize now. ")
    await result.end_turn()
    await result.complete()

    events = await _collect(result)
    errors = [e for e in events if e.type == "voice_stream_event_error"]
    assert len(errors) == 1
    assert isinstance(errors[0].error, RuntimeError)


async def test_producer_error_surfaces_as_error_event():
    result = StreamedAudioResult(WordTTS(), TTSModelSettings())
    result.start()
    await result.add_error(ValueError("producer failed"))

    events = await _collect(result)
    assert len(events) == 1
    assert events[0].type == "voice_stream_event_error"
    assert isinstance(events[0].error, ValueError)


async def test_abandoned_stream_does_not_deadlock():
    """A consumer that reads one event then closes the stream must not hang.

    Regression for two interacting bugs: (1) the synthesis task, cancelled
    by ``_cleanup`` while blocked on a full output queue, re-blocked forever
    in its ``finally`` trying to enqueue the terminal sentinel; (2)
    ``_cleanup`` suppressed *every* ``CancelledError`` while awaiting that
    task — including the one ``wait_for`` raises on timeout — so the hang
    was silently absorbed after the full timeout rather than surfaced.

    ``wait_for`` bounds the hang, and the elapsed-time assertion fails a
    regression loudly (buggy cleanup runs to the ~3s timeout; the fix
    returns near-instantly).
    """
    result = StreamedAudioResult(FloodTTS(), TTSModelSettings())
    result.start()
    await result.start_turn()
    await result.add_text("A sentence long enough to synthesize now. ")
    await result.end_turn()
    await result.complete()

    stream = result.stream()
    first = await stream.__anext__()
    assert first is not None

    # Let synthesis flood the bounded output queue so the synth task is
    # blocked on a full-queue put() when _cleanup cancels it — that is the
    # precondition for the deadlock this test guards.
    for _ in range(1000):
        if result._event_queue.full():
            break
        await asyncio.sleep(0)
    assert result._event_queue.full()

    # Closing the generator runs its finally -> _cleanup, which cancels the
    # flooded synthesis task. Must return promptly, not block to the timeout.
    # stream() is an async generator at runtime (declared as the wider
    # AsyncIterator, which omits aclose()).
    gen = cast("AsyncGenerator[VoiceStreamEvent, None]", stream)
    started = time.monotonic()
    await asyncio.wait_for(gen.aclose(), timeout=3.0)
    elapsed = time.monotonic() - started
    assert elapsed < 1.0, f"stream cleanup took {elapsed:.2f}s — deadlock/foreign-cancel regression"
