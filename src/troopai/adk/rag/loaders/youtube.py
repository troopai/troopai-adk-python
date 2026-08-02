"""YouTube loaders — index video transcripts.

:class:`YoutubeVideoLoader` loads a single video's transcript;
:class:`YoutubeChannelLoader` enumerates a channel's uploads (via pytube) and
loads each transcript, skipping videos with none. Transcripts come from
``youtube-transcript-api``; channel enumeration from ``pytube``. Install the
``rag-youtube`` extra. ``max_videos`` bounds channel ingestion.
"""

from __future__ import annotations

import asyncio
import logging
from typing import ClassVar, override
from urllib.parse import parse_qs, urlparse

from troopai.adk.exceptions.exceptions import DocumentLoadError
from troopai.adk.rag.document import LoadedDocument
from troopai.adk.rag.loaders.base import DocumentLoader

logger = logging.getLogger(__name__)


_PATH_ID_PREFIXES: tuple[str, ...] = ("/shorts/", "/embed/", "/live/")
"""``youtube.com`` path forms that carry the video id as the first segment."""


def _extract_video_id(source: str) -> str:
    """Extract the 11-character video id from a YouTube URL or raw id.

    Handles ``youtu.be`` short links, ``watch?v=`` query URLs, and the
    ``/shorts/``, ``/embed/``, and ``/live/`` path forms — every non-channel
    shape the loader registry routes here.
    """
    parsed = urlparse(source)
    if len(parsed.scheme) == 0:
        return source
    if "youtu.be" in parsed.netloc:
        candidate = parsed.path.lstrip("/")
        return candidate.split("/")[0] if len(candidate) > 0 else source
    values = parse_qs(parsed.query).get("v")
    if values is not None and len(values) > 0:
        return values[0]
    for prefix in _PATH_ID_PREFIXES:
        if parsed.path.startswith(prefix):
            candidate = parsed.path[len(prefix) :].split("/")[0]
            if len(candidate) > 0:
                return candidate
    raise DocumentLoadError(source, f"Could not extract a video id from: {source}")


def _fetch_transcript(video_id: str, source: str) -> str:
    """Fetch and join a video's transcript text (runs in a worker thread)."""
    from youtube_transcript_api import (  # pyright: ignore[reportMissingImports]
        YouTubeTranscriptApi,
        YouTubeTranscriptApiException,
    )

    try:
        fetched = YouTubeTranscriptApi().fetch(video_id)
    except YouTubeTranscriptApiException as exc:
        raise DocumentLoadError(source, f"No transcript for video {video_id}: {exc}") from exc
    return " ".join(snippet.text for snippet in fetched if len(snippet.text.strip()) > 0)


class YoutubeVideoLoader(DocumentLoader):
    """Loads a single YouTube video's transcript (via youtube-transcript-api)."""

    requires_packages: ClassVar[tuple[str, ...]] = ("youtube_transcript_api",)
    install_extra: ClassVar[str] = "rag-youtube"

    def __init__(self) -> None:
        """Verify youtube-transcript-api is importable.

        Raises:
            ImportError: If youtube-transcript-api is not installed.
        """
        self.ensure_dependencies()

    @override
    async def load(self, source: str) -> list[LoadedDocument]:
        """Load the transcript of video ``source`` as one document.

        Args:
            source: A YouTube watch URL, ``youtu.be`` URL, or raw video id.

        Returns:
            A single-element list, or an empty list if the transcript is empty.

        Raises:
            DocumentLoadError: If the id cannot be parsed or no transcript exists.
        """
        video_id = _extract_video_id(source)
        text = await asyncio.to_thread(_fetch_transcript, video_id, source)
        if len(text.strip()) == 0:
            logger.debug("YoutubeVideoLoader: %s transcript empty", video_id)
            return []
        return [LoadedDocument(content=text, source=source, metadata={"video_id": video_id})]


def _channel_video_urls(source: str, max_videos: int) -> list[str]:
    """Enumerate a channel's video URLs via pytube (runs in a worker thread)."""
    from pytube import Channel  # pyright: ignore[reportMissingImports]

    try:
        channel = Channel(source)
        return list(channel.video_urls[:max_videos])
    except Exception as exc:  # pytube raises a broad range of network/parse errors
        raise DocumentLoadError(source, f"Could not enumerate channel {source}: {exc}") from exc


class YoutubeChannelLoader(DocumentLoader):
    """Loads transcripts for a YouTube channel's uploads (pytube + transcript-api).

    Attributes:
        max_videos: Maximum number of videos enumerated per channel.
    """

    requires_packages: ClassVar[tuple[str, ...]] = ("youtube_transcript_api", "pytube")
    install_extra: ClassVar[str] = "rag-youtube"

    def __init__(self, *, max_videos: int = 20) -> None:
        """
        Args:
            max_videos: Maximum number of videos enumerated per channel.

        Raises:
            ImportError: If youtube-transcript-api / pytube are not installed.
            ValueError: If ``max_videos`` is not positive.
        """
        if max_videos <= 0:
            raise ValueError(f"YoutubeChannelLoader.max_videos must be > 0, got {max_videos}")
        self.max_videos = max_videos
        self.ensure_dependencies()

    @override
    async def load(self, source: str) -> list[LoadedDocument]:
        """Load transcripts for up to ``max_videos`` of channel ``source``.

        Args:
            source: A YouTube channel URL.

        Returns:
            One document per video that has a transcript (others skipped).

        Raises:
            DocumentLoadError: If the channel cannot be enumerated.
        """
        urls = await asyncio.to_thread(_channel_video_urls, source, self.max_videos)
        documents: list[LoadedDocument] = []
        for url in urls:
            video_id = _extract_video_id(url)
            try:
                text = await asyncio.to_thread(_fetch_transcript, video_id, url)
            except DocumentLoadError as exc:
                logger.warning("YoutubeChannelLoader: skipping %s: %s", url, exc)
                continue
            if len(text.strip()) > 0:
                documents.append(LoadedDocument(content=text, source=url, metadata={"video_id": video_id}))
        logger.debug("YoutubeChannelLoader: %s -> %d transcript(s)", source, len(documents))
        return documents
