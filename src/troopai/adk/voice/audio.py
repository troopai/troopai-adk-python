"""Audio containers for the voice pipeline.

Audio is represented as raw little-endian PCM ``bytes`` throughout the
public surface — no third-party array library is required to *use* the
voice pipeline. ``numpy`` is an optional convenience: :func:`pcm16_from_float32`
and :func:`pcm16_from_int16` convert sample arrays captured from a
microphone library into the PCM bytes the models consume, and raise a
guiding :class:`ImportError` when ``numpy`` is absent.

Two shapes exist:

- :class:`AudioInput` — a complete, already-captured utterance. Fed to
  :meth:`~troopai.adk.voice.stt.STTModel.transcribe` for a single
  buffered turn.
- :class:`StreamedAudioInput` — an append-as-you-go microphone stream.
  Fed to :meth:`~troopai.adk.voice.stt.STTModel.create_session` for
  continuous, multi-turn transcription. ``add_audio(None)`` signals the
  end of the stream.
"""

from __future__ import annotations

import io
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncio

DEFAULT_SAMPLE_RATE = 24000
"""Sample rate (Hz) the OpenAI speech models emit and consume."""

DEFAULT_CHANNELS = 1
"""Mono audio — one channel."""

DEFAULT_SAMPLE_WIDTH = 2
"""Bytes per sample — 16-bit signed PCM."""


@dataclass
class AudioInput:
    """A complete captured utterance as raw PCM bytes.

    Attributes:
        data: Little-endian PCM sample bytes (16-bit signed by default).
        sample_rate: Samples per second.
        channels: Number of interleaved channels (1 = mono).
        sample_width: Bytes per sample (2 = 16-bit).
    """

    data: bytes
    """Little-endian PCM sample bytes."""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    """Samples per second."""

    channels: int = DEFAULT_CHANNELS
    """Number of interleaved channels (1 = mono)."""

    sample_width: int = DEFAULT_SAMPLE_WIDTH
    """Bytes per sample (2 = 16-bit signed PCM)."""

    def to_wav_bytes(self) -> bytes:
        """Wrap the raw PCM data in a WAV container.

        Transcription HTTP endpoints expect a self-describing audio
        file rather than headerless PCM. This wraps the buffer in a
        minimal WAV container using the stdlib :mod:`wave` module — no
        third-party dependency.

        Returns:
            The PCM data wrapped as in-memory WAV bytes.
        """
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(self.data)
        return buffer.getvalue()

    def to_upload(self, filename: str = "audio.wav") -> tuple[str, bytes, str]:
        """Return a ``(filename, wav_bytes, content_type)`` upload tuple.

        Shaped for the ``file=`` parameter of an HTTP transcription
        client, which accepts a ``(name, bytes, mime)`` triple.

        Args:
            filename: Name reported to the upload endpoint.

        Returns:
            A tuple of ``(filename, wav_bytes, "audio/wav")``.
        """
        return (filename, self.to_wav_bytes(), "audio/wav")


class StreamedAudioInput:
    """An append-as-you-go microphone stream backed by an async queue.

    Producers call :meth:`add_audio` with PCM byte chunks as they are
    captured, then :meth:`add_audio` with ``None`` to close the stream.
    A realtime transcription session consumes the chunks via
    :meth:`iter_chunks`.
    """

    def __init__(self) -> None:
        # Imported lazily so the dataclass module above stays import-light;
        # the queue is only meaningful once an event loop is running.
        import asyncio

        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    @property
    def queue(self) -> asyncio.Queue[bytes | None]:
        """The underlying chunk queue (``None`` is the end-of-stream sentinel)."""
        return self._queue

    async def add_audio(self, audio: bytes | None) -> None:
        """Append a PCM chunk, or ``None`` to signal end-of-stream.

        Args:
            audio: A chunk of little-endian PCM bytes, or ``None`` to
                close the stream.
        """
        await self._queue.put(audio)

    async def iter_chunks(self) -> AsyncIterator[bytes]:
        """Yield PCM chunks until the end-of-stream sentinel arrives.

        Yields:
            Each appended PCM byte chunk in order. The iterator returns
            when ``add_audio(None)`` is observed.
        """
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            yield chunk


def pcm16_from_int16(samples: Any) -> bytes:
    """Convert a ``numpy`` int16 sample array to little-endian PCM bytes.

    Args:
        samples: A ``numpy.ndarray`` of ``int16`` samples (typed ``Any``
            so the public signature does not require the optional
            ``numpy`` dependency).

    Returns:
        Little-endian 16-bit PCM bytes.

    Raises:
        ImportError: When ``numpy`` is not installed.
    """
    np = _require_numpy()
    array = np.asarray(samples, dtype=np.int16)
    pcm: bytes = array.astype("<i2").tobytes()
    return pcm


def pcm16_from_float32(samples: Any) -> bytes:
    """Convert a ``numpy`` float32 sample array (``[-1.0, 1.0]``) to PCM bytes.

    Args:
        samples: A ``numpy.ndarray`` of ``float32`` samples in the
            range ``[-1.0, 1.0]`` (typed ``Any`` so the public signature
            does not require the optional ``numpy`` dependency).

    Returns:
        Little-endian 16-bit PCM bytes.

    Raises:
        ImportError: When ``numpy`` is not installed.
    """
    np = _require_numpy()
    array = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm: bytes = (array * 32767.0).astype("<i2").tobytes()
    return pcm


def _require_numpy() -> Any:
    """Import and return the ``numpy`` module, or raise a guiding install hint.

    Returns ``Any`` so callers can use the module's array API without the
    framework taking a hard typing dependency on the optional package.
    """
    try:
        import numpy
    except ImportError as exc:
        raise ImportError(
            "Converting sample arrays to PCM requires 'numpy'. Install the voice extra: "
            "pip install 'troopai-adk-python[voice]'. Raw PCM bytes need no extra."
        ) from exc
    return numpy
