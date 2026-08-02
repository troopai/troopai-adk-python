"""Streamed audio result — turns agent text into ordered speech events.

A :class:`StreamedAudioResult` is the object a caller iterates to hear an
agent speak. It sits between the producer (the pipeline, feeding agent
text turn by turn) and the consumer (the caller, playing audio):

1. The producer pushes text deltas via :meth:`add_text`, bracketed by
   :meth:`start_turn` / :meth:`end_turn`, and finally :meth:`complete`.
2. Text accumulates and is cut into synthesis-ready segments by the
   configured splitter; whole segments and turn/session markers flow
   through one internal queue.
3. A single background synthesis task drains that queue **in order**,
   calling the TTS model per segment and forwarding lifecycle markers,
   so audio never interleaves across segments.
4. The caller reads the resulting :class:`~troopai.adk.voice.events.VoiceStreamEvent`
   stream via :meth:`stream`.

The output queue is bounded, so a slow consumer back-pressures synthesis
rather than letting audio accumulate without limit. ``turn_started`` and
``turn_ended`` are emitted as a balanced pair even when a turn produces
no audio.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from troopai.adk.voice.events import (
    VoiceStreamEvent,
    VoiceStreamEventAudio,
    VoiceStreamEventError,
    VoiceStreamEventLifecycle,
)
from troopai.adk.voice.splitter import sentence_splitter

if TYPE_CHECKING:
    from troopai.adk.voice.splitter import TextSplitter
    from troopai.adk.voice.tts import TTSModel, TTSModelSettings

logger = logging.getLogger(__name__)

DEFAULT_AUDIO_QUEUE_SIZE = 64
"""Max buffered output audio events before synthesis back-pressures."""

DEFAULT_SEGMENT_QUEUE_SIZE = 256
"""Max buffered pending text segments before the producer back-pressures."""


@dataclass(frozen=True)
class _StreamDone:
    """Internal sentinel marking the end of an internal queue."""


_STREAM_DONE = _StreamDone()

_SegmentItem = str | VoiceStreamEventLifecycle | VoiceStreamEventError | _StreamDone
_OutputItem = VoiceStreamEvent | _StreamDone


class StreamedAudioResult:
    """Ordered speech-event stream produced from streamed agent text.

    Construct one per pipeline run, call :meth:`start` to launch
    synthesis, drive it with the producer methods, and let the caller
    consume :meth:`stream`.
    """

    def __init__(self, tts_model: TTSModel, tts_settings: TTSModelSettings) -> None:
        self._tts_model = tts_model
        self._tts_settings = tts_settings
        splitter = tts_settings.text_splitter
        self._splitter: TextSplitter = splitter if splitter is not None else sentence_splitter()
        self._buffer = ""
        self._segment_queue: asyncio.Queue[_SegmentItem] = asyncio.Queue(maxsize=DEFAULT_SEGMENT_QUEUE_SIZE)
        self._event_queue: asyncio.Queue[_OutputItem] = asyncio.Queue(maxsize=DEFAULT_AUDIO_QUEUE_SIZE)
        self._synth_task: asyncio.Task[None] | None = None
        self._producer_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle wiring (called by the pipeline)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the background synthesis task. Requires a running loop."""
        if self._synth_task is None:
            self._synth_task = asyncio.create_task(self._synthesis_loop())

    def set_producer_task(self, task: asyncio.Task[None]) -> None:
        """Register the producer task so it is cancelled on consumer exit."""
        self._producer_task = task

    # ------------------------------------------------------------------
    # Producer interface (called by the pipeline's producer coroutine)
    # ------------------------------------------------------------------

    async def start_turn(self) -> None:
        """Mark the beginning of a speaking turn."""
        await self._segment_queue.put(VoiceStreamEventLifecycle(event="turn_started"))

    async def add_text(self, text: str) -> None:
        """Buffer a text delta and release any complete segment for synthesis."""
        if len(text) == 0:
            return
        self._buffer += text
        ready, remainder = self._splitter(self._buffer)
        self._buffer = remainder
        if len(ready) > 0:
            await self._segment_queue.put(ready)

    async def end_turn(self) -> None:
        """Flush the turn's trailing text and mark the turn ended."""
        remaining = self._buffer.strip()
        self._buffer = ""
        if len(remaining) > 0:
            await self._segment_queue.put(remaining)
        await self._segment_queue.put(VoiceStreamEventLifecycle(event="turn_ended"))

    async def complete(self) -> None:
        """Mark the whole session ended; no further turns will be added."""
        await self._segment_queue.put(VoiceStreamEventLifecycle(event="session_ended"))
        await self._segment_queue.put(_STREAM_DONE)

    async def add_error(self, error: Exception) -> None:
        """Surface a producer-side error on the stream and end it."""
        await self._segment_queue.put(VoiceStreamEventError(error=error))
        await self._segment_queue.put(_STREAM_DONE)

    # ------------------------------------------------------------------
    # Consumer interface
    # ------------------------------------------------------------------

    async def stream(self) -> AsyncIterator[VoiceStreamEvent]:
        """Yield speech events until the session ends.

        Yields:
            Each :class:`~troopai.adk.voice.events.VoiceStreamEvent` in
            order: lifecycle markers, audio chunks, and a terminal error
            if one occurred.
        """
        try:
            while True:
                item = await self._event_queue.get()
                if isinstance(item, _StreamDone):
                    break
                yield item
        finally:
            await self._cleanup()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _synthesis_loop(self) -> None:
        """Drain pending segments in order, synthesizing audio per segment.

        Lifecycle and error markers pass through verbatim; text segments
        are synthesized to audio. A :class:`_StreamDone` ends the loop.
        This is the sole writer to the output queue, so audio never
        interleaves across segments.
        """
        try:
            while True:
                item = await self._segment_queue.get()
                if isinstance(item, _StreamDone):
                    break
                if isinstance(item, (VoiceStreamEventLifecycle, VoiceStreamEventError)):
                    await self._event_queue.put(item)
                    continue
                await self._synthesize_segment(item)
        except asyncio.CancelledError:
            # The consumer abandoned the stream, so it will never read a
            # terminal sentinel — and the bounded output queue may be full,
            # which would make a put() here block forever (deadlocking the
            # awaiting _cleanup). Propagate the cancellation WITHOUT
            # enqueuing _STREAM_DONE. The sentinel is only needed on the
            # paths where a consumer is still reading (below).
            raise
        except Exception as exc:
            logger.exception("Voice synthesis failed")
            await self._event_queue.put(VoiceStreamEventError(error=exc))
            await self._event_queue.put(_STREAM_DONE)
        else:
            await self._event_queue.put(_STREAM_DONE)

    async def _synthesize_segment(self, text: str) -> None:
        """Synthesize one text segment, forwarding each audio chunk."""
        async for chunk in self._tts_model.run(text, self._tts_settings):
            if len(chunk) > 0:
                await self._event_queue.put(VoiceStreamEventAudio(data=chunk))

    async def _cleanup(self) -> None:
        """Cancel the producer and synthesis tasks on consumer exit.

        Only the cancellation this method itself requests on each task is
        swallowed. If THIS coroutine is cancelled from the outside while
        awaiting a task, that foreign cancellation is re-raised rather than
        suppressed, so an external ``cancel()`` on the consumer is honored.
        """
        for task in (self._producer_task, self._synth_task):
            if task is None or task.done():
                continue
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                current = asyncio.current_task()
                if current is not None and current.cancelling() > 0:
                    # This coroutine was itself cancelled from the outside,
                    # not just `task` — propagate that cancellation.
                    raise
