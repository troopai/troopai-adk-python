"""Native OpenAI speech-to-text model.

Implements the framework-owned :class:`~troopai.adk.voice.stt.STTModel`
ABC against the ``openai`` SDK (buffered transcription) and the OpenAI
realtime transcription websocket (continuous, turn-by-turn). The
``openai`` SDK and the realtime websocket are confined to this provider
module; everything the pipeline sees is the provider-agnostic ABC.

The realtime path uses the ``websockets`` library directly — it needs
precise control over the transcription-intent handshake — and is gated
behind the ``voice`` extra. Audio is consumed as raw PCM bytes.

Refs:
    - Transcription API: https://platform.openai.com/docs/api-reference/audio/createTranscription
    - Realtime transcription: https://platform.openai.com/docs/guides/realtime-transcription
    - OpenAI SDK: https://github.com/openai/openai-python
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, override

from openai import omit

from troopai.adk.voice.audio import DEFAULT_SAMPLE_RATE
from troopai.adk.voice.exceptions import STTError, STTWebsocketError
from troopai.adk.voice.stt import StreamedTranscriptionSession, STTModel

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from troopai.adk.voice.audio import AudioInput, StreamedAudioInput
    from troopai.adk.voice.stt import STTModelSettings

logger = logging.getLogger(__name__)

DEFAULT_STT_MODEL = "gpt-4o-transcribe"
"""Default OpenAI transcription model."""

REALTIME_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
"""Realtime transcription websocket endpoint."""

DEFAULT_TURN_DETECTION_TYPE = "semantic_vad"
"""Server-side turn detection used when settings name none."""

EVENT_INACTIVITY_TIMEOUT = 5.0
"""Seconds to wait for a trailing transcript after the audio stream ends."""

_TRANSCRIPT_COMPLETED_EVENTS = frozenset(
    {
        "conversation.item.input_audio_transcription.completed",
        "input_audio_transcription_completed",
    }
)
"""Realtime event types that carry a finalized turn transcript."""


class OpenAISTTModel(STTModel):
    """OpenAI speech-to-text with buffered and realtime transcription.

    Args:
        model: OpenAI transcription model ID.
        api_key: API key. Falls back to ``OPENAI_API_KEY``.
        base_url: Optional custom base URL for the buffered HTTP path
            (the realtime websocket always targets the public endpoint).
        organization: Optional OpenAI organization ID.
        project: Optional OpenAI project ID.
        max_retries: SDK-level retries for transient errors (default 0).
    """

    def __init__(
        self,
        model: str = DEFAULT_STT_MODEL,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        max_retries: int = 0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._organization = organization
        self._project = project
        self._max_retries = max_retries
        self._client: AsyncOpenAI | None = None

    @property
    @override
    def model_name(self) -> str:
        return self._model

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-initialize and cache the OpenAI async client."""
        client = self._client
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as exc:
                raise ImportError(
                    "The 'openai' package is required for OpenAISTTModel. "
                    "Install it with: pip install 'troopai-adk-python[voice]'"
                ) from exc

            client = AsyncOpenAI(
                api_key=self._api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=self._base_url,
                organization=self._organization,
                project=self._project,
                max_retries=self._max_retries,
            )
            self._client = client
        return client

    @override
    async def transcribe(self, audio: AudioInput, settings: STTModelSettings) -> str:
        """Transcribe one buffered utterance via the HTTP API."""
        client = self._get_client()
        filename, wav_bytes, content_type = audio.to_upload()
        try:
            response = await client.audio.transcriptions.create(
                model=self._model,
                file=(filename, wav_bytes, content_type),
                prompt=settings.prompt if settings.prompt is not None else omit,
                language=settings.language if settings.language is not None else omit,
                temperature=settings.temperature if settings.temperature is not None else omit,
            )
        except Exception as exc:
            raise STTError(f"OpenAI transcription failed: {exc}") from exc
        return response.text

    @override
    async def create_session(
        self, audio: StreamedAudioInput, settings: STTModelSettings
    ) -> StreamedTranscriptionSession:
        """Open a realtime transcription session over a continuous stream."""
        api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
        if api_key is None or len(api_key) == 0:
            raise STTWebsocketError("An OpenAI API key is required for realtime transcription (set OPENAI_API_KEY).")
        return OpenAISTTSession(audio=audio, model=self._model, settings=settings, api_key=api_key)


class OpenAISTTSession(StreamedTranscriptionSession):
    """A realtime OpenAI transcription session over a websocket.

    Audio is base64-PCM streamed to the server, which segments speech by
    voice-activity detection and returns one transcript per turn. The
    session ends when the audio stream closes and no further transcript
    arrives within :data:`EVENT_INACTIVITY_TIMEOUT`.
    """

    def __init__(
        self,
        *,
        audio: StreamedAudioInput,
        model: str,
        settings: STTModelSettings,
        api_key: str,
    ) -> None:
        self._audio = audio
        self._model = model
        self._settings = settings
        self._api_key = api_key
        self._websocket: Any = None
        self._sender_task: asyncio.Task[None] | None = None
        self._input_done = asyncio.Event()
        self._closed = False

    @override
    async def transcribe_turns(self) -> AsyncIterator[str]:
        websocket = await self._connect()
        self._websocket = websocket

        from websockets.exceptions import ConnectionClosed

        self._sender_task = asyncio.create_task(self._send_audio(websocket))
        try:
            while True:
                try:
                    if self._input_done.is_set():
                        raw = await asyncio.wait_for(websocket.recv(), timeout=EVENT_INACTIVITY_TIMEOUT)
                    else:
                        raw = await websocket.recv()
                except TimeoutError:
                    break
                except ConnectionClosed:
                    break

                event = json.loads(raw)
                event_type = event.get("type", "")
                if event_type in _TRANSCRIPT_COMPLETED_EVENTS:
                    transcript = event.get("transcript", "")
                    if isinstance(transcript, str) and len(transcript) > 0:
                        yield transcript
                elif event_type == "error":
                    raise STTWebsocketError(f"Realtime transcription error: {event.get('error')}")
        finally:
            await self.close()

    @override
    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        sender = self._sender_task
        if sender is not None and not sender.done():
            sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sender
        websocket = self._websocket
        if websocket is not None:
            with contextlib.suppress(Exception):
                await websocket.close()

    async def _connect(self) -> Any:
        """Open the websocket and send the transcription-session config."""
        try:
            import websockets
        except ImportError as exc:
            raise STTWebsocketError(
                "Realtime speech-to-text requires 'websockets'. Install it with: pip install 'troopai-adk-python[voice]'"
            ) from exc

        headers = {"Authorization": f"Bearer {self._api_key}", "OpenAI-Beta": "realtime=v1"}
        try:
            websocket = await websockets.connect(REALTIME_URL, additional_headers=headers)
            await websocket.send(json.dumps(self._session_update_payload()))
        except Exception as exc:
            raise STTWebsocketError(f"Could not open realtime transcription session: {exc}") from exc
        return websocket

    async def _send_audio(self, websocket: Any) -> None:
        """Stream PCM chunks to the server until the audio input ends."""
        try:
            async for chunk in self._audio.iter_chunks():
                payload = json.dumps(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
                await websocket.send(payload)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Realtime audio sender stopped on error")
        finally:
            # Signal the reader to switch to its inactivity timeout so the
            # turn loop can drain any trailing transcript, then exit.
            self._input_done.set()

    def _session_update_payload(self) -> dict[str, Any]:
        """Build the ``session.update`` payload from the agnostic settings."""
        detection = self._settings.turn_detection
        if detection is not None:
            turn_detection: dict[str, Any] = {"type": detection.mode.value}
            if detection.threshold is not None:
                turn_detection["threshold"] = detection.threshold
            if detection.prefix_padding_ms is not None:
                turn_detection["prefix_padding_ms"] = detection.prefix_padding_ms
            if detection.silence_duration_ms is not None:
                turn_detection["silence_duration_ms"] = detection.silence_duration_ms
        else:
            turn_detection = {"type": DEFAULT_TURN_DETECTION_TYPE}

        transcription: dict[str, Any] = {"model": self._model}
        if self._settings.prompt is not None:
            transcription["prompt"] = self._settings.prompt
        if self._settings.language is not None:
            transcription["language"] = self._settings.language

        return {
            "type": "session.update",
            "session": {
                "type": "transcription",
                "audio": {
                    "input": {
                        "format": {"type": "audio/pcm", "rate": DEFAULT_SAMPLE_RATE},
                        "transcription": transcription,
                        "turn_detection": turn_detection,
                    }
                },
            },
        }
