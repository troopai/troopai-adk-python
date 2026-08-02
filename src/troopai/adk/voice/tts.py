"""Text-to-speech abstractions — framework-owned, provider-agnostic.

:class:`TTSModel` is the text-to-speech counterpart to the
:class:`~troopai.adk.llms.llm.LLM` ABC. The pipeline streams agent
text into :meth:`TTSModel.run`, which yields synthesized PCM audio
chunks. Concrete implementations live in their provider's module (e.g.
``llms/openai/``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.voice.splitter import TextSplitter


@dataclass
class TTSModelSettings:
    """Provider-agnostic knobs for a synthesis request.

    Attributes:
        voice: Provider voice name; ``None`` lets the implementation pick
            its documented default voice.
        speed: Optional playback speed multiplier; ``None`` uses the
            provider default.
        instructions: Optional style/delivery instructions. ``None`` by
            default — the framework never injects a hidden instruction
            prompt the developer did not write.
        text_splitter: How streamed text is cut into synthesis segments;
            ``None`` uses the default sentence splitter.
    """

    voice: str | None = None
    """Provider voice name; ``None`` uses the implementation default."""

    speed: float | None = None
    """Optional playback speed multiplier."""

    instructions: str | None = None
    """Optional style/delivery instructions; ``None`` injects nothing."""

    text_splitter: TextSplitter | None = None
    """How streamed text is segmented; ``None`` uses the sentence splitter."""


class TTSModel(ABC):
    """Abstract text-to-speech model.

    Implementations live in their provider's module (e.g.
    ``llms/openai/``) and convert :class:`TTSModelSettings` to provider
    wire parameters internally.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """The provider model identifier used for synthesis."""

    @abstractmethod
    def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        """Synthesize one text segment, streaming PCM audio chunks.

        Args:
            text: The text to synthesize.
            settings: Synthesis knobs (voice, speed, instructions).

        Yields:
            Little-endian 16-bit PCM audio byte chunks.

        Raises:
            TTSError: When synthesis fails.
        """
