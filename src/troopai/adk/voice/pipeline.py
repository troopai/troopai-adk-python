"""The voice pipeline — wires STT, an agent workflow, and TTS together.

:class:`VoicePipeline` is the single entry point: hand it audio, get back
a :class:`~troopai.adk.voice.result.StreamedAudioResult` to play.

It supports two input shapes:

- :class:`~troopai.adk.voice.audio.AudioInput` — one captured
  utterance, transcribed once, answered once, spoken once.
- :class:`~troopai.adk.voice.audio.StreamedAudioInput` — a live
  microphone stream, segmented into turns by a realtime transcription
  session, each turn answered and spoken in sequence.

The speech-to-text and text-to-speech models are **required and
explicit** — the pipeline never silently falls back to a default
provider. Production happens on a background task so ``run`` returns
immediately and the caller streams audio as it is produced.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from troopai.adk.tracing import custom_span
from troopai.adk.voice.audio import AudioInput, StreamedAudioInput
from troopai.adk.voice.pipeline_config import VoicePipelineConfig
from troopai.adk.voice.result import StreamedAudioResult

if TYPE_CHECKING:
    from troopai.adk.voice.stt import STTModel
    from troopai.adk.voice.tts import TTSModel
    from troopai.adk.voice.workflow import VoiceWorkflow

logger = logging.getLogger(__name__)


class VoicePipeline:
    """Speech-to-text → agent workflow → text-to-speech orchestration.

    Args:
        workflow: Produces the text to speak for each transcript.
        stt_model: Transcribes audio. Required — no default provider.
        tts_model: Synthesizes speech. Required — no default provider.
        config: Optional pipeline configuration; a default is used when
            omitted.
    """

    def __init__(
        self,
        *,
        workflow: VoiceWorkflow,
        stt_model: STTModel,
        tts_model: TTSModel,
        config: VoicePipelineConfig | None = None,
    ) -> None:
        self._workflow = workflow
        self._stt_model = stt_model
        self._tts_model = tts_model
        self._config = config if config is not None else VoicePipelineConfig()

    async def run(self, audio_input: AudioInput | StreamedAudioInput) -> StreamedAudioResult:
        """Start processing audio and return a streamable speech result.

        Args:
            audio_input: A buffered utterance (:class:`AudioInput`) or a
                live stream (:class:`StreamedAudioInput`).

        Returns:
            A :class:`StreamedAudioResult`; iterate its
            :meth:`~StreamedAudioResult.stream` to receive speech events.
        """
        result = StreamedAudioResult(self._tts_model, self._config.tts_settings)
        result.start()
        if isinstance(audio_input, AudioInput):
            producer = self._run_single_turn(audio_input, result)
        else:
            producer = self._run_multi_turn(audio_input, result)
        result.set_producer_task(asyncio.create_task(producer))
        return result

    async def _run_single_turn(self, audio_input: AudioInput, result: StreamedAudioResult) -> None:
        """Transcribe one utterance, answer it, and speak the answer."""
        try:
            transcription = await self._transcribe(audio_input)
            await self._speak_turn(transcription, result)
            await result.complete()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Buffered voice run failed")
            await result.add_error(exc)

    async def _run_multi_turn(self, audio_input: StreamedAudioInput, result: StreamedAudioResult) -> None:
        """Open a realtime session and answer each detected turn in order."""
        try:
            await self._greet(result)
            session = await self._stt_model.create_session(audio_input, self._config.stt_settings)
            try:
                async for transcription in session.transcribe_turns():
                    await self._speak_turn(transcription, result)
            finally:
                await session.close()
            await result.complete()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Streamed voice run failed")
            await result.add_error(exc)

    async def _speak_turn(self, transcription: str, result: StreamedAudioResult) -> None:
        """Run the workflow for one transcript, streaming text into audio."""
        await result.start_turn()
        try:
            async for text in self._workflow.run(transcription):
                await result.add_text(text)
        finally:
            # Always close the turn so ``turn_started`` / ``turn_ended`` stay
            # a balanced pair even when the workflow raises mid-stream.
            await result.end_turn()

    async def _greet(self, result: StreamedAudioResult) -> None:
        """Speak the workflow's opening greeting, if it yields one."""
        greeting = [text async for text in self._workflow.on_start()]
        if len(greeting) == 0:
            return
        await result.start_turn()
        for text in greeting:
            await result.add_text(text)
        await result.end_turn()

    async def _transcribe(self, audio_input: AudioInput) -> str:
        """Transcribe a buffered utterance under a tracing span."""
        with custom_span("voice.transcription", data={"workflow_name": self._config.workflow_name}) as span:
            transcription = await self._stt_model.transcribe(audio_input, self._config.stt_settings)
            if self._config.trace_include_sensitive_data:
                span.data.data["transcription"] = transcription
            return transcription
