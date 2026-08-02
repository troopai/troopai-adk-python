"""Native OpenAI text-to-speech model.

Implements the framework-owned :class:`~troopai.adk.voice.tts.TTSModel`
ABC against ``openai.AsyncOpenAI().audio.speech`` — the ``openai`` SDK is
confined to this provider module. Audio streams back as raw 24 kHz PCM
bytes so it slots straight into the voice pipeline without a decode step.

Refs:
    - Speech API: https://platform.openai.com/docs/api-reference/audio/createSpeech
    - OpenAI SDK: https://github.com/openai/openai-python
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, override

from openai import omit

from troopai.adk.voice.exceptions import TTSError
from troopai.adk.voice.tts import TTSModel

if TYPE_CHECKING:
    from openai import AsyncOpenAI

    from troopai.adk.voice.tts import TTSModelSettings

logger = logging.getLogger(__name__)

DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
"""Default OpenAI synthesis model."""

DEFAULT_VOICE = "ash"
"""Default voice when settings do not name one (a synthesis request requires a voice)."""

TTS_CHUNK_SIZE = 1024
"""Bytes per streamed audio chunk."""


class OpenAITTSModel(TTSModel):
    """OpenAI text-to-speech, streaming raw PCM audio.

    Args:
        model: OpenAI TTS model ID.
        api_key: API key. Falls back to ``OPENAI_API_KEY``.
        base_url: Optional custom base URL (Azure OpenAI, proxy).
        organization: Optional OpenAI organization ID.
        project: Optional OpenAI project ID.
        max_retries: SDK-level retries for transient errors (default 0).
    """

    def __init__(
        self,
        model: str = DEFAULT_TTS_MODEL,
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
                    "The 'openai' package is required for OpenAITTSModel. "
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
    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        """Synthesize ``text`` and stream PCM audio chunks.

        ``instructions`` are forwarded only when the developer set them —
        no hidden style prompt is injected.
        """
        client = self._get_client()
        voice = settings.voice if settings.voice is not None else DEFAULT_VOICE
        extra_body: dict[str, Any] = {}
        if settings.instructions is not None:
            extra_body["instructions"] = settings.instructions

        try:
            async with client.audio.speech.with_streaming_response.create(
                model=self._model,
                voice=voice,
                input=text,
                response_format="pcm",
                speed=settings.speed if settings.speed is not None else omit,
                extra_body=extra_body,
            ) as response:
                async for chunk in response.iter_bytes(chunk_size=TTS_CHUNK_SIZE):
                    yield chunk
        except Exception as exc:
            raise TTSError(f"OpenAI text-to-speech failed: {exc}") from exc
