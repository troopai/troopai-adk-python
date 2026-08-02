"""Pydantic models + JSON schemas describing memory pipeline artifacts.

Layer-1 framework types: no provider SDK imports, no session
dependency, no LLM wiring. The the extraction step and the consolidation step modules read
and write these structures; the pipeline orchestrator
(``SandboxMemoryManager``) coordinates them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ROLLOUT_EXTRACTION_ARTIFACTS_JSON_SCHEMA",
    "ROLLOUT_EXTRACTION_ARTIFACTS_STRICT_FORMAT",
    "MemoryGenerationResult",
    "RolloutExtractionArtifacts",
    "RolloutTerminalMetadata",
    "TerminalState",
]


TerminalState = Literal[
    "completed",
    "interrupted",
    "cancelled",
    "failed",
    "max_turns_exceeded",
    "guardrail_tripped",
]
"""Coarse outcome classification for a single agent rollout."""


class RolloutTerminalMetadata(BaseModel):
    """How a rollout ended (success / failure / abort / etc.).

    Attributes:
        terminal_state: Outcome classification — drives the extraction step's
            triage section and the consolidation step's signal filtering.
        exception_type: Fully-qualified class name when the run
            ended with an exception, else ``None``.
        exception_message: One-line description of the exception, else ``None``.
        has_final_output: True iff the run produced a final output
            payload (even on failure, agents can yield partial output).
    """

    model_config = ConfigDict(frozen=True)

    terminal_state: TerminalState
    """Outcome classification."""

    exception_type: str | None = None
    """FQN of the terminating exception, if any."""

    exception_message: str | None = None
    """One-line description of the terminating exception."""

    has_final_output: bool = False
    """True iff the run produced a final output payload."""


class RolloutExtractionArtifacts(BaseModel):
    """Extraction-step output — what the extractor LLM emits per rollout.

    Attributes:
        rollout_slug: File-safe identifier (alphanumeric + ``-`` /
            ``_``, ≤80 chars). Used as the suffix of the
            rollout-summary filename.
        rollout_summary: Comprehensive task-grouped summary of the
            rollout — what the user asked, what the agent did, what
            went well, what failed.
        raw_memory: Distilled high-signal observations the next
            agent should know — preferences, reusable knowledge,
            failure modes. YAML frontmatter + bullet content.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rollout_slug: str = Field(..., min_length=1, max_length=80)
    """File-safe slug used as filename suffix."""

    rollout_summary: str
    """Comprehensive task-grouped rollout description."""

    raw_memory: str
    """Distilled high-signal observations for future runs."""


ROLLOUT_EXTRACTION_ARTIFACTS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "rollout_slug": {"type": "string", "minLength": 1, "maxLength": 80},
        "rollout_summary": {"type": "string"},
        "raw_memory": {"type": "string"},
    },
    "required": ["rollout_slug", "rollout_summary", "raw_memory"],
}
"""Strict JSON Schema for ``RolloutExtractionArtifacts`` — bind to providers that support `response_format=json_schema`."""


ROLLOUT_EXTRACTION_ARTIFACTS_STRICT_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "name": "sandbox_memory_rollout_extraction_artifacts",
    "description": "Sandbox memory rollout extraction artifacts.",
    "schema": ROLLOUT_EXTRACTION_ARTIFACTS_JSON_SCHEMA,
    "strict": True,
}
"""Provider-ready text-format block for ``RolloutExtractionArtifacts``."""


class MemoryGenerationResult(BaseModel):
    """Aggregate result of one ``SandboxMemoryManager.flush()`` pass.

    Attributes:
        rollouts_processed: Number of per-rollout extraction calls
            executed during this flush.
        consolidation_skipped: True iff the consolidation step was a
            no-op (no raw memories changed since the last flush).
        consolidated_at: ISO 8601 timestamp marking the end of the
            flush, or ``None`` when the consolidation step was skipped.
    """

    model_config = ConfigDict(frozen=True)

    rollouts_processed: int = 0
    """Number of per-rollout extraction calls executed during this flush."""

    consolidation_skipped: bool = False
    """True iff the consolidation step had no meaningful work to do."""

    consolidated_at: str | None = None
    """ISO 8601 timestamp marking flush completion (None if the consolidation step skipped)."""
