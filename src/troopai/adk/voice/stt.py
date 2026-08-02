"""Speech-to-text abstractions — framework-owned, provider-agnostic.

:class:`STTModel` is the speech-to-text counterpart to the
:class:`~troopai.adk.llms.llm.LLM` ABC: the pipeline talks to it,
never to a provider SDK. Concrete implementations (e.g. the OpenAI
models under ``llms/openai/``) convert these provider-agnostic settings
to their wire format internally.

Two transcription paths exist:

- :meth:`STTModel.transcribe` — one buffered utterance to one transcript.
- :meth:`STTModel.create_session` — a long-lived realtime session that
  segments a continuous microphone stream into one transcript per turn.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.voice.audio import AudioInput, StreamedAudioInput


class TurnDetectionMode(StrEnum):
    """How a realtime session decides one speaking turn has ended.

    Provider-agnostic; each realtime implementation maps these to its
    own voice-activity-detection wire shape.
    """

    SERVER_VAD = "server_vad"
    """Energy-threshold voice-activity detection on silence boundaries."""

    SEMANTIC_VAD = "semantic_vad"
    """Model-based detection of when the speaker has finished a thought."""


@dataclass
class TurnDetection:
    """Voice-activity-detection settings for a realtime session.

    Attributes:
        mode: The detection strategy.
        threshold: Activation energy threshold in ``[0.0, 1.0]`` for
            ``SERVER_VAD``; ``None`` uses the provider default.
        prefix_padding_ms: Milliseconds of audio retained before detected
            speech; ``None`` uses the provider default.
        silence_duration_ms: Trailing silence that ends a turn; ``None``
            uses the provider default.
    """

    mode: TurnDetectionMode = TurnDetectionMode.SERVER_VAD
    """The detection strategy."""

    threshold: float | None = None
    """Activation energy threshold in ``[0.0, 1.0]`` (``SERVER_VAD``)."""

    prefix_padding_ms: int | None = None
    """Milliseconds of audio retained before detected speech."""

    silence_duration_ms: int | None = None
    """Trailing silence (ms) that ends a turn."""


@dataclass
class STTModelSettings:
    """Provider-agnostic knobs for a transcription request.

    Attributes:
        prompt: Optional text biasing the transcription toward expected
            vocabulary or spelling.
        language: Optional ISO-639-1 language code; ``None`` lets the
            model auto-detect.
        temperature: Optional decoding temperature.
        turn_detection: Realtime-session voice-activity detection; ignored
            by the buffered :meth:`STTModel.transcribe` path.
    """

    prompt: str | None = None
    """Optional text biasing transcription toward expected vocabulary."""

    language: str | None = None
    """Optional ISO-639-1 language code; ``None`` auto-detects."""

    temperature: float | None = None
    """Optional decoding temperature."""

    turn_detection: TurnDetection | None = None
    """Realtime-session voice-activity detection (buffered path ignores it)."""


class StreamedTranscriptionSession(ABC):
    """A live transcription session over a continuous audio stream.

    Produced by :meth:`STTModel.create_session`. Iterating
    :meth:`transcribe_turns` yields one transcript per detected speaking
    turn until the audio stream ends. :meth:`close` releases the
    underlying connection.
    """

    @abstractmethod
    def transcribe_turns(self) -> AsyncIterator[str]:
        """Yield one transcript per detected speaking turn.

        Yields:
            The transcript text for each completed turn, in order.
        """

    @abstractmethod
    async def close(self) -> None:
        """Close the session and release its underlying connection."""


class STTModel(ABC):
    """Abstract speech-to-text model.

    Implementations live in their provider's module (e.g.
    ``llms/openai/``) and convert :class:`STTModelSettings` to provider
    wire parameters internally.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The provider model identifier used for transcription."""

    @abstractmethod
    async def transcribe(self, audio: AudioInput, settings: STTModelSettings) -> str:
        """Transcribe one complete buffered utterance.

        Args:
            audio: The captured utterance.
            settings: Transcription knobs.

        Returns:
            The transcribed text.

        Raises:
            STTError: When transcription fails.
        """

    @abstractmethod
    async def create_session(
        self, audio: StreamedAudioInput, settings: STTModelSettings
    ) -> StreamedTranscriptionSession:
        """Open a realtime session over a continuous audio stream.

        Args:
            audio: The append-as-you-go microphone stream.
            settings: Transcription knobs, including
                :attr:`STTModelSettings.turn_detection`.

        Returns:
            A live :class:`StreamedTranscriptionSession`.

        Raises:
            STTWebsocketError: When the realtime session cannot be opened.
        """
