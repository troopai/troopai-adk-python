"""Configuration for a :class:`~troopai.adk.voice.pipeline.VoicePipeline`.

Trace-capture of transcripts and audio defaults to **off**: a span never
records the user's words or voice unless the developer explicitly opts
in. This keeps the conservative default — the developer never has to opt
*out* of capturing sensitive data they did not choose to capture.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from troopai.adk.voice.stt import STTModelSettings
from troopai.adk.voice.tts import TTSModelSettings


@dataclass
class VoicePipelineConfig:
    """Tunables shared across a voice pipeline run.

    Attributes:
        stt_settings: Settings passed to every transcription call.
        tts_settings: Settings passed to every synthesis call.
        workflow_name: Span name identifying this pipeline in traces.
        trace_include_sensitive_data: When ``True``, attach transcripts
            to spans. Defaults to ``False`` (no transcript capture).
        trace_include_sensitive_audio_data: When ``True``, attach audio
            byte sizes/metadata to spans. Defaults to ``False``.
    """

    stt_settings: STTModelSettings = field(default_factory=STTModelSettings)
    """Settings passed to every transcription call."""

    tts_settings: TTSModelSettings = field(default_factory=TTSModelSettings)
    """Settings passed to every synthesis call."""

    workflow_name: str = "Voice Agent"
    """Span name identifying this pipeline in traces."""

    trace_include_sensitive_data: bool = False
    """When ``True``, attach transcripts to spans (default ``False``)."""

    trace_include_sensitive_audio_data: bool = False
    """When ``True``, attach audio metadata to spans (default ``False``)."""
