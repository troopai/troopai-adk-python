"""Events streamed from a running voice pipeline.

:meth:`~troopai.adk.voice.result.StreamedAudioResult.stream` yields a
union of three event variants, discriminated by their ``type`` field:

- :class:`VoiceStreamEventAudio` — a chunk of synthesized PCM audio to
  play.
- :class:`VoiceStreamEventLifecycle` — a turn/session boundary marker,
  useful for driving UI state (e.g. a "speaking" indicator).
- :class:`VoiceStreamEventError` — an error raised mid-stream; the
  stream ends after it.

These are plain ``@dataclass`` discriminated-union members, mirroring the
runner's own boundary stream events
(:class:`~troopai.adk.run.stream.RawResponseStreamEvent` and friends)
rather than the validation-heavy response types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class VoiceStreamEventAudio:
    """A chunk of synthesized PCM audio ready for playback.

    Attributes:
        data: Little-endian 16-bit PCM audio bytes at the synthesizer's
            sample rate.
        type: Discriminator. Always ``"voice_stream_event_audio"``.
    """

    data: bytes
    """Little-endian 16-bit PCM audio bytes."""

    type: Literal["voice_stream_event_audio"] = "voice_stream_event_audio"


@dataclass
class VoiceStreamEventLifecycle:
    """A turn or session boundary marker.

    ``turn_started`` and ``turn_ended`` always arrive in balanced pairs —
    a turn that produces no audio (e.g. the agent answered with a silent
    tool call) still emits both, so consumers can rely on the pairing to
    drive UI state. ``session_ended`` is emitted once, last.

    Attributes:
        event: Which boundary this marks.
        type: Discriminator. Always ``"voice_stream_event_lifecycle"``.
    """

    event: Literal["turn_started", "turn_ended", "session_ended"]
    """Which boundary this marks."""

    type: Literal["voice_stream_event_lifecycle"] = "voice_stream_event_lifecycle"


@dataclass
class VoiceStreamEventError:
    """An error raised while the pipeline was producing audio.

    The stream ends after this event — no further audio or lifecycle
    events follow.

    Attributes:
        error: The exception that ended the stream.
        type: Discriminator. Always ``"voice_stream_event_error"``.
    """

    error: Exception
    """The exception that ended the stream."""

    type: Literal["voice_stream_event_error"] = "voice_stream_event_error"


VoiceStreamEvent = VoiceStreamEventAudio | VoiceStreamEventLifecycle | VoiceStreamEventError
"""Discriminated union of every event a voice pipeline streams."""
