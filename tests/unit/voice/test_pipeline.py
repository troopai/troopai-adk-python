"""Tests for VoicePipeline buffered + streamed orchestration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import override

from troopai.adk.voice.audio import AudioInput, StreamedAudioInput
from troopai.adk.voice.events import VoiceStreamEvent
from troopai.adk.voice.exceptions import STTError
from troopai.adk.voice.pipeline import VoicePipeline
from troopai.adk.voice.result import StreamedAudioResult
from troopai.adk.voice.stt import StreamedTranscriptionSession, STTModel, STTModelSettings
from troopai.adk.voice.tts import TTSModel, TTSModelSettings
from troopai.adk.voice.workflow import VoiceWorkflow


class FakeSession(StreamedTranscriptionSession):
    def __init__(self, turns: list[str]) -> None:
        self._turns = turns
        self.closed = False

    @override
    async def transcribe_turns(self) -> AsyncIterator[str]:
        for turn in self._turns:
            yield turn

    @override
    async def close(self) -> None:
        self.closed = True


class FakeSTT(STTModel):
    def __init__(self, *, transcript: str = "hello world", turns: list[str] | None = None) -> None:
        self._transcript = transcript
        self._session = FakeSession(turns if turns is not None else [])

    @property
    @override
    def model_name(self) -> str:
        return "fake-stt"

    @override
    async def transcribe(self, audio: AudioInput, settings: STTModelSettings) -> str:
        return self._transcript

    @override
    async def create_session(
        self, audio: StreamedAudioInput, settings: STTModelSettings
    ) -> StreamedTranscriptionSession:
        return self._session


class BoomSTT(STTModel):
    @property
    @override
    def model_name(self) -> str:
        return "boom-stt"

    @override
    async def transcribe(self, audio: AudioInput, settings: STTModelSettings) -> str:
        raise STTError("stt unavailable")

    @override
    async def create_session(
        self, audio: StreamedAudioInput, settings: STTModelSettings
    ) -> StreamedTranscriptionSession:
        raise STTError("stt unavailable")


class FakeTTS(TTSModel):
    @property
    @override
    def model_name(self) -> str:
        return "fake-tts"

    @override
    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        yield text.encode("utf-8")


class EchoWorkflow(VoiceWorkflow):
    @override
    async def run(self, transcription: str) -> AsyncIterator[str]:
        yield transcription + " "


class BoomWorkflow(VoiceWorkflow):
    """Yields one chunk, then raises mid-turn."""

    @override
    async def run(self, transcription: str) -> AsyncIterator[str]:
        del transcription
        yield "partial answer "
        raise RuntimeError("workflow boom")


def _normalize(event: VoiceStreamEvent) -> str:
    if event.type == "voice_stream_event_audio":
        return "audio"
    if event.type == "voice_stream_event_lifecycle":
        return event.event
    return "error"


async def _collect(result: StreamedAudioResult) -> list[VoiceStreamEvent]:
    return [event async for event in result.stream()]


async def test_buffered_pipeline_speaks_transcript():
    pipeline = VoicePipeline(
        workflow=EchoWorkflow(),
        stt_model=FakeSTT(transcript="hello world"),
        tts_model=FakeTTS(),
    )
    result = await pipeline.run(AudioInput(data=b"\x00\x00" * 50))
    events = await _collect(result)

    assert [_normalize(e) for e in events] == ["turn_started", "audio", "turn_ended", "session_ended"]
    audio = b"".join(e.data for e in events if e.type == "voice_stream_event_audio")
    assert audio == b"hello world"


async def test_streamed_pipeline_answers_each_turn():
    stt = FakeSTT(turns=["first turn here", "second turn here"])
    pipeline = VoicePipeline(workflow=EchoWorkflow(), stt_model=stt, tts_model=FakeTTS())
    result = await pipeline.run(StreamedAudioInput())
    events = await _collect(result)

    normalized = [_normalize(e) for e in events]
    assert normalized.count("turn_started") == 2
    assert normalized.count("turn_ended") == 2
    assert normalized[-1] == "session_ended"
    assert stt._session.closed is True


async def test_pipeline_surfaces_stt_error():
    pipeline = VoicePipeline(workflow=EchoWorkflow(), stt_model=BoomSTT(), tts_model=FakeTTS())
    result = await pipeline.run(AudioInput(data=b"\x00\x00"))
    events = await _collect(result)

    assert any(e.type == "voice_stream_event_error" for e in events)


async def test_workflow_error_mid_turn_still_closes_the_turn():
    """A workflow that raises mid-turn must still emit ``turn_ended`` so the
    ``turn_started`` / ``turn_ended`` pair stays balanced (plus a terminal
    error). Without the try/finally in ``_speak_turn`` the turn is left
    open."""
    pipeline = VoicePipeline(workflow=BoomWorkflow(), stt_model=FakeSTT(transcript="hi"), tts_model=FakeTTS())
    result = await pipeline.run(AudioInput(data=b"\x00\x00"))
    events = await _collect(result)

    normalized = [_normalize(e) for e in events]
    assert normalized.count("turn_started") == 1
    assert normalized.count("turn_started") == normalized.count("turn_ended")
    assert any(e.type == "voice_stream_event_error" for e in events)
