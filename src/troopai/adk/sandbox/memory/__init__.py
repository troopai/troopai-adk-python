"""Sandbox memory subsystem — durable cross-run agent memory.

A two-phase LLM-driven pipeline that converts agent rollouts
(per-run transcripts) into durable consolidated memories the next
run can read back.

Pipeline shape:

1. Each run appends a JSONL rollout segment to
   ``{sessions_dir}/{rollout_id}.jsonl``.
2. At session-stop, ``SandboxMemoryManager.flush()`` walks the
   accumulated rollout files:
   * extraction step: each rollout file is sent to the
     extraction model; the LLM emits a ``RolloutExtractionArtifacts``
     (rollout_slug + rollout_summary + raw_memory) which lands as
     ``raw_memories/{rollout_id}.md`` and
     ``rollout_summaries/{rollout_id}_{slug}.md``.
   * consolidation step: the top-N raw memories are merged
     into ``memories/raw_memories.md``, then sent to the consolidation
     model with the prior ``MEMORY.md`` + ``memory_summary.md``;
     the model rewrites both consolidation artifacts in place.

Cross-references:
* ``troopai.adk.sandbox.capabilities.memory.MemoryCapability`` — the
  capability surface that wires generation into the agent loop.
"""

from __future__ import annotations

from troopai.adk.sandbox.memory.consolidation import run_consolidation
from troopai.adk.sandbox.memory.interface import (
    ROLLOUT_EXTRACTION_ARTIFACTS_JSON_SCHEMA,
    MemoryGenerationResult,
    RolloutExtractionArtifacts,
    RolloutTerminalMetadata,
)
from troopai.adk.sandbox.memory.manager import SandboxMemoryManager
from troopai.adk.sandbox.memory.prompts import (
    render_memory_consolidation_prompt,
    render_memory_read_prompt,
    render_rollout_extraction_prompt,
    render_rollout_extraction_user_prompt,
)
from troopai.adk.sandbox.memory.rollout_extraction import run_rollout_extraction
from troopai.adk.sandbox.memory.rollouts import (
    build_rollout_payload,
    dump_rollout_json,
    terminal_metadata_for_exception,
    terminal_metadata_for_result,
)
from troopai.adk.sandbox.memory.storage import (
    ConsolidationInputSelection,
    ConsolidationSelectionItem,
    SandboxMemoryStorage,
)

__all__ = [
    "ROLLOUT_EXTRACTION_ARTIFACTS_JSON_SCHEMA",
    "ConsolidationInputSelection",
    "ConsolidationSelectionItem",
    "MemoryGenerationResult",
    "RolloutExtractionArtifacts",
    "RolloutTerminalMetadata",
    "SandboxMemoryManager",
    "SandboxMemoryStorage",
    "build_rollout_payload",
    "dump_rollout_json",
    "render_memory_consolidation_prompt",
    "render_memory_read_prompt",
    "render_rollout_extraction_prompt",
    "render_rollout_extraction_user_prompt",
    "run_consolidation",
    "run_rollout_extraction",
    "terminal_metadata_for_exception",
    "terminal_metadata_for_result",
]
