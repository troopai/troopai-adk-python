"""Tests for the native OpenAI speech models (STT + TTS).

The ``openai`` client and the realtime websocket are mocked — no network
calls. Covers buffered transcription, streamed synthesis, the realtime
session protocol, and the API-key guard.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.llms.openai.openai_stt import OpenAISTTModel, OpenAISTTSession
from troopai.adk.llms.openai.openai_tts import OpenAITTSModel
from troopai.adk.voice.audio import AudioInput, StreamedAudioInput
from troopai.adk.voice.exceptions import STTWebsocketError
from troopai.adk.voice.stt import STTModelSettings, TurnDetection, TurnDetectionMode
from troopai.adk.voice.tts import TTSModelSettings


class _FakeStreamingResponse:
    """Mimics the OpenAI streaming-speech async context manager."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aenter__(self) -> _FakeStreamingResponse:
        return self

    async def __aexit__(self, *args: object) -> bool:
        return False

    async def iter_bytes(self, chunk_size: int) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _FakeWebSocket:
    """Mimics a ``websockets`` connection for the realtime session."""

    def __init__(self, incoming: list[str]) -> None:
        self._incoming = list(incoming)
        self.sent: list[str] = []
        self.closed = False

    async def send(self, data: str) -> None:
        self.sent.append(data)

    async def recv(self) -> str:
        import asyncio

        await asyncio.sleep(0)  # yield so the audio sender task can run
        if len(self._incoming) > 0:
            return self._incoming.pop(0)
        from websockets.exceptions import ConnectionClosed

        raise ConnectionClosed(None, None)

    async def close(self) -> None:
        self.closed = True


async def test_tts_streams_pcm_chunks_with_mapped_params():
    fake_client = MagicMock()
    create = MagicMock(return_value=_FakeStreamingResponse([b"aa", b"bb"]))
    fake_client.audio.speech.with_streaming_response.create = create

    with patch("openai.AsyncOpenAI", return_value=fake_client):
        model = OpenAITTSModel()
        settings = TTSModelSettings(voice="nova", instructions="speak calmly", speed=1.25)
        chunks = [chunk async for chunk in model.run("hello", settings)]

    assert chunks == [b"aa", b"bb"]
    kwargs = create.call_args.kwargs
    assert kwargs["response_format"] == "pcm"
    assert kwargs["voice"] == "nova"
    assert kwargs["input"] == "hello"
    assert kwargs["speed"] == 1.25
    assert kwargs["extra_body"]["instructions"] == "speak calmly"


async def test_tts_omits_instructions_when_unset():
    fake_client = MagicMock()
    create = MagicMock(return_value=_FakeStreamingResponse([b"x"]))
    fake_client.audio.speech.with_streaming_response.create = create

    with patch("openai.AsyncOpenAI", return_value=fake_client):
        model = OpenAITTSModel()
        _ = [chunk async for chunk in model.run("hi", TTSModelSettings())]

    assert create.call_args.kwargs["extra_body"] == {}


async def test_stt_transcribes_buffer_with_mapped_params():
    fake_client = MagicMock()
    fake_client.audio.transcriptions.create = AsyncMock(return_value=SimpleNamespace(text="recognized text"))

    with patch("openai.AsyncOpenAI", return_value=fake_client):
        model = OpenAISTTModel()
        text = await model.transcribe(AudioInput(data=b"\x00\x00"), STTModelSettings(language="en"))

    assert text == "recognized text"
    kwargs = fake_client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-transcribe"
    assert kwargs["language"] == "en"


async def test_create_session_requires_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = OpenAISTTModel(api_key=None)
    with pytest.raises(STTWebsocketError):
        await model.create_session(StreamedAudioInput(), STTModelSettings())


async def test_realtime_session_yields_transcripts(monkeypatch: pytest.MonkeyPatch):
    incoming = [
        json.dumps({"type": "transcription_session.created"}),
        json.dumps({"type": "conversation.item.input_audio_transcription.completed", "transcript": "first"}),
        json.dumps({"type": "conversation.item.input_audio_transcription.completed", "transcript": "second"}),
    ]
    fake_ws = _FakeWebSocket(incoming)

    async def fake_connect(url: str, **kwargs: Any) -> _FakeWebSocket:
        return fake_ws

    monkeypatch.setattr("websockets.connect", fake_connect)

    audio = StreamedAudioInput()
    await audio.add_audio(b"\x00\x00")
    await audio.add_audio(None)

    session = OpenAISTTSession(
        audio=audio,
        model="gpt-4o-transcribe",
        settings=STTModelSettings(turn_detection=TurnDetection(mode=TurnDetectionMode.SEMANTIC_VAD)),
        api_key="test-key",
    )
    transcripts = [turn async for turn in session.transcribe_turns()]

    assert transcripts == ["first", "second"]
    first_sent = json.loads(fake_ws.sent[0])
    assert first_sent["type"] == "session.update"
    assert first_sent["session"]["audio"]["input"]["turn_detection"]["type"] == "semantic_vad"
    assert any(json.loads(message).get("type") == "input_audio_buffer.append" for message in fake_ws.sent)
    assert fake_ws.closed is True
