"""Tests for the voice audio containers and PCM helpers."""

from __future__ import annotations

import io
import wave

import pytest

from troopai.adk.voice.audio import AudioInput, StreamedAudioInput, pcm16_from_float32, pcm16_from_int16


def test_to_wav_bytes_roundtrips_pcm():
    pcm = b"\x01\x00\x02\x00\x03\x00"
    audio = AudioInput(data=pcm, sample_rate=24000)
    with wave.open(io.BytesIO(audio.to_wav_bytes()), "rb") as wav_file:
        assert wav_file.getframerate() == 24000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.readframes(wav_file.getnframes()) == pcm


def test_to_upload_shape():
    name, payload, content_type = AudioInput(data=b"\x00\x00").to_upload()
    assert name == "audio.wav"
    assert content_type == "audio/wav"
    assert isinstance(payload, bytes)


async def test_streamed_audio_input_iterates_until_sentinel():
    stream = StreamedAudioInput()
    await stream.add_audio(b"a")
    await stream.add_audio(b"b")
    await stream.add_audio(None)
    chunks = [chunk async for chunk in stream.iter_chunks()]
    assert chunks == [b"a", b"b"]


def test_pcm16_from_int16_roundtrips():
    np = pytest.importorskip("numpy")
    samples = np.array([0, 1, -1, 32767, -32768], dtype=np.int16)
    out = pcm16_from_int16(samples)
    assert np.frombuffer(out, dtype="<i2").tolist() == [0, 1, -1, 32767, -32768]


def test_pcm16_from_float32_clips_and_scales():
    np = pytest.importorskip("numpy")
    samples = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)
    out = np.frombuffer(pcm16_from_float32(samples), dtype="<i2")
    assert out[0] == 0
    assert out[1] == 32767
    assert out[2] == -32767
    assert out[3] == 32767  # 2.0 clipped to 1.0 before scaling
