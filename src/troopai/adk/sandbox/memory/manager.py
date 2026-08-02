"""``SandboxMemoryManager`` — orchestrates the two-step pipeline.

Lifecycle:

1. Each run that should produce memory calls
   ``enqueue_rollout(rollout_id, payload_jsonl)`` once. The
   manager appends the JSONL line to the per-rollout file under
   ``{sessions_dir}/`` and records the rollout for extraction.
2. ``flush()`` runs the extraction step on every pending rollout,
   persists the artifacts, then runs the consolidation step once
   to rewrite the consolidated memory files.

The manager does NOT own provider wiring; ``RolloutExtractionLLMCaller``
and ``MemoryConsolidationRunner`` are injected so callers can plug in our
``Runner.arun`` flow, a stub, or a different framework entirely.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import ValidationError

from troopai.adk.sandbox.memory.consolidation import (
    MemoryConsolidationRunner,
    run_consolidation,
)
from troopai.adk.sandbox.memory.interface import (
    MemoryGenerationResult,
    RolloutTerminalMetadata,
)
from troopai.adk.sandbox.memory.rollout_extraction import (
    RolloutExtractionLLMCaller,
    run_rollout_extraction,
)
from troopai.adk.sandbox.memory.storage import SandboxMemoryStorage, make_safe_slug

if TYPE_CHECKING:
    from troopai.adk.sandbox.capabilities.memory import MemoryLayoutConfig
    from troopai.adk.sandbox.clients.session import BaseSandboxSession

logger = logging.getLogger(__name__)

__all__ = ["MAX_EXTRACTION_ATTEMPTS_PER_ROLLOUT", "SandboxMemoryManager"]

_ROLLOUT_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

MAX_EXTRACTION_ATTEMPTS_PER_ROLLOUT: int = 3
"""Hard cap on extraction retries per rollout — beyond this we quarantine."""


class SandboxMemoryManager:
    """Coordinates per-rollout extraction and the consolidation step.

    Construct ONE manager per ``BaseSandboxSession`` — the manager
    owns the lock that serializes ``flush()`` and the pending map
    that the extraction step consumes.
    """

    def __init__(
        self,
        *,
        session: BaseSandboxSession,
        layout: MemoryLayoutConfig,
        extraction_llm: RolloutExtractionLLMCaller,
        consolidation_runner: MemoryConsolidationRunner,
        max_raw_memories_for_consolidation: int = 256,
        extra_prompt: str | None = None,
    ) -> None:
        self._session = session
        self._storage = SandboxMemoryStorage(session=session, layout=layout)
        self._extraction_llm = extraction_llm
        self._consolidation_runner = consolidation_runner
        self._max_raw_memories_for_consolidation = max_raw_memories_for_consolidation
        self._extra_prompt = extra_prompt
        # Pending rollouts plus their optional seed metadata supplied at enqueue.
        self._pending_rollouts: dict[str, RolloutTerminalMetadata | None] = {}
        self._attempt_counts: dict[str, int] = {}
        self._flush_lock = asyncio.Lock()
        self._layout_lock = asyncio.Lock()
        self._layout_ready = False

    @property
    def storage(self) -> SandboxMemoryStorage:
        """Return the storage helper for ad-hoc workspace I/O."""
        return self._storage

    async def enqueue_rollout(
        self,
        *,
        rollout_id: str,
        payload_jsonl: str,
        terminal_metadata: RolloutTerminalMetadata | None = None,
    ) -> None:
        """Append a rollout JSONL segment + mark the rollout for extraction.

        Args:
            rollout_id: Stable identifier scoping the rollout.
            payload_jsonl: One JSONL line terminating in ``\\n``.
            terminal_metadata: Source-of-truth terminal classification —
                when supplied here the manager prefers it over the
                JSONL-derived value (covers crashed runs that never
                wrote a terminal block).
        """
        _validate_rollout_id(rollout_id)
        await self._ensure_layout()
        await self._storage.append_rollout_segment(rollout_id=rollout_id, payload_jsonl=payload_jsonl)
        # Last-write-wins for the same rollout_id keeps the most recent
        # caller-supplied metadata if any.
        existing = self._pending_rollouts.get(rollout_id)
        self._pending_rollouts[rollout_id] = terminal_metadata or existing

    async def flush(
        self,
        *,
        run_consolidation_pass: bool = True,
    ) -> MemoryGenerationResult:
        """Extract every pending rollout, then optionally consolidate.

        Args:
            run_consolidation_pass: When ``False``, only the extraction
                step runs — useful in tests or pipelines that want to
                gate consolidation behind a downstream signal.
        """
        async with self._flush_lock:
            await self._ensure_layout()

            extraction_count = await self._run_extraction_pass()
            if not run_consolidation_pass:
                return MemoryGenerationResult(
                    rollouts_processed=extraction_count,
                    consolidation_skipped=True,
                )

            selection = await self._storage.build_consolidation_selection(
                max_raw_memories_for_consolidation=self._max_raw_memories_for_consolidation,
            )
            if len(selection.selected) == 0:
                return MemoryGenerationResult(
                    rollouts_processed=extraction_count,
                    consolidation_skipped=True,
                )

            rebuilt = await self._storage.rebuild_raw_memories(selected_items=selection.selected)
            if not rebuilt:
                return MemoryGenerationResult(
                    rollouts_processed=extraction_count,
                    consolidation_skipped=True,
                )

            await run_consolidation(
                memory_root=str(self._storage.memories_dir),
                selection=selection,
                runner=self._consolidation_runner,
                extra_prompt=self._extra_prompt,
            )
            await self._storage.write_consolidation_selection(selected_items=selection.selected)
            return MemoryGenerationResult(
                rollouts_processed=extraction_count,
                consolidation_skipped=False,
                consolidated_at=datetime.now(tz=UTC).isoformat(),
            )

    async def _ensure_layout(self) -> None:
        # The lock serializes concurrent enqueues during cold-start; without
        # it two concurrent `enqueue_rollout` calls can both observe
        # _layout_ready=False, both run ``storage.ensure_layout()``, and
        # both flip the flag — benign on local backends but liable to
        # double-create + permission-race on hosted ones.
        async with self._layout_lock:
            if self._layout_ready:
                return
            await self._storage.ensure_layout()
            self._layout_ready = True

    async def _run_extraction_pass(self) -> int:
        if len(self._pending_rollouts) == 0:
            return 0
        # Snapshot + clear so a concurrent enqueue lands in a fresh queue.
        snapshot = dict(self._pending_rollouts)
        self._pending_rollouts.clear()
        pending_ids = sorted(snapshot.keys())
        processed = 0
        for rollout_id in pending_ids:
            seed_metadata = snapshot[rollout_id]
            try:
                if await self._process_rollout(rollout_id, seed_metadata=seed_metadata):
                    processed += 1
                # Successful pass clears the retry counter even on no-op
                # responses — the gating contract treats no-op as terminal.
                self._attempt_counts.pop(rollout_id, None)
            except asyncio.CancelledError:
                # Propagate shutdown signals — re-enqueue the current rollout
                # AND all not-yet-processed rollouts from the snapshot so that
                # every pending rollout survives the cancellation and the next
                # flush retries it. The snapshot was cleared at the top of this
                # method, so unprocessed entries would otherwise be permanently
                # orphaned.
                current_idx = pending_ids.index(rollout_id)
                for remaining_id in pending_ids[current_idx:]:
                    if remaining_id not in self._pending_rollouts:
                        self._pending_rollouts[remaining_id] = snapshot[remaining_id]
                raise
            except (ValueError, ValidationError, KeyError, TypeError):
                # Treat schema / programming-error failures as permanent
                # so a poisoned LLM response doesn't loop forever.
                logger.exception("extract[%s] failed permanently — quarantining", rollout_id)
                self._attempt_counts.pop(rollout_id, None)
            except Exception:
                attempts = self._attempt_counts.get(rollout_id, 0) + 1
                self._attempt_counts[rollout_id] = attempts
                if attempts >= MAX_EXTRACTION_ATTEMPTS_PER_ROLLOUT:
                    logger.exception(
                        "extract[%s] exhausted retries (%d) — quarantining",
                        rollout_id,
                        attempts,
                    )
                    self._attempt_counts.pop(rollout_id, None)
                else:
                    logger.exception(
                        "extract[%s] transient failure (attempt %d/%d) — will retry",
                        rollout_id,
                        attempts,
                        MAX_EXTRACTION_ATTEMPTS_PER_ROLLOUT,
                    )
                    self._pending_rollouts[rollout_id] = seed_metadata
        return processed

    async def _process_rollout(
        self,
        rollout_id: str,
        *,
        seed_metadata: RolloutTerminalMetadata | None,
    ) -> bool:
        rollout_path = self._storage.sessions_dir / f"{rollout_id}.jsonl"
        contents = await self._storage.read_text(rollout_path)
        # Prefer the caller-supplied metadata; fall back to JSONL-derived.
        terminal_metadata = seed_metadata or _terminal_metadata_from_jsonl(contents, rollout_id=rollout_id)
        artifacts = await run_rollout_extraction(
            rollout_id=rollout_id,
            rollout_contents=contents,
            terminal_metadata=terminal_metadata,
            llm=self._extraction_llm,
            extra_prompt=self._extra_prompt,
        )
        if artifacts is None:
            return False
        safe_slug = make_safe_slug(artifacts.rollout_slug)
        # Single source of truth for the on-disk filename — both the
        # storage write and the frontmatter reference must use the
        # SAME sanitized slug, otherwise consolidation would search for a
        # file that never existed.
        rollout_summary_filename = f"{rollout_id}_{safe_slug}.md"
        await self._storage.write_rollout_summary(
            rollout_id=rollout_id,
            slug=safe_slug,
            body=artifacts.rollout_summary,
        )
        raw_memory = _ensure_metadata(
            artifacts.raw_memory,
            rollout_id=rollout_id,
            rollout_path=str(rollout_path),
            rollout_summary_file=rollout_summary_filename,
            terminal_state=terminal_metadata.terminal_state,
        )
        await self._storage.write_raw_memory(rollout_id=rollout_id, body=raw_memory)
        return True


def _validate_rollout_id(rollout_id: str) -> None:
    if not _ROLLOUT_ID_PATTERN.match(rollout_id):
        raise ValueError(f"rollout_id must match [A-Za-z0-9._-]{{1,128}}, got {rollout_id!r}")


def _terminal_metadata_from_jsonl(contents: str, *, rollout_id: str) -> RolloutTerminalMetadata:
    """Walk the JSONL tail-first; pick the most recent valid ``terminal_metadata`` block.

    When no metadata can be recovered (empty JSONL, malformed lines,
    schema mismatch), return a SYNTHESIZED ``"failed"`` classification
    so downstream extraction sees an explicit failure signal instead
    of a silent ``"completed"`` lie that pollutes durable memory.
    """
    for raw_line in reversed(contents.splitlines()):
        line = raw_line.strip()
        if len(line) == 0:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        meta = obj.get("terminal_metadata")
        if isinstance(meta, dict):
            try:
                return RolloutTerminalMetadata.model_validate(meta)
            except ValidationError as exc:
                logger.warning(
                    "rollout[%s]: terminal_metadata failed validation: %s",
                    rollout_id,
                    exc,
                )
                continue
    logger.warning(
        "rollout[%s]: no terminal_metadata recovered from JSONL — synthesizing 'failed'",
        rollout_id,
    )
    return RolloutTerminalMetadata(
        terminal_state="failed",
        exception_message="memory: terminal_metadata not found in JSONL",
        has_final_output=False,
    )


_METADATA_PLACEHOLDER_VALUES = frozenset({"unknown", "null", "none"})

_FRAMEWORK_OWNED_KEYS = frozenset({"rollout_id", "rollout_path", "rollout_summary_file", "terminal_state"})
"""Keys whose value is framework ground-truth — the LLM never gets to override them.

The on-disk filename is derived from these (``raw_memories/{rollout_id}.md``,
``rollout_summaries/{rollout_id}_{slug}.md``). If an LLM-emitted frontmatter
value were kept, consolidation would resolve the path from the frontmatter and
search for a file that was never written, silently dropping the memory.
"""


def _ensure_metadata(
    raw_memory: str,
    *,
    rollout_id: str,
    rollout_path: str,
    rollout_summary_file: str,
    terminal_state: str,
) -> str:
    """Guarantee the raw_memory frontmatter has the fields the consolidation step needs.

    Framework-owned keys (``rollout_id``, ``rollout_path``,
    ``rollout_summary_file``, ``terminal_state``) are ALWAYS rewritten with the
    authoritative values — any LLM-emitted line for those keys is stripped so the
    frontmatter cannot diverge from the on-disk filename. ``updated_at`` is filled
    only when the LLM left it missing/blank/placeholder.
    """
    authoritative = {
        "rollout_id": rollout_id,
        "rollout_path": rollout_path,
        "rollout_summary_file": rollout_summary_file,
        "terminal_state": terminal_state,
    }
    # Drop any LLM-emitted lines for keys the framework owns outright.
    kept_lines = [
        line for line in raw_memory.splitlines() if not any(line.startswith(f"{key}:") for key in _FRAMEWORK_OWNED_KEYS)
    ]
    prefix_lines = [f"{key}: {value}" for key, value in authoritative.items()]
    if not _has_usable_updated_at(kept_lines):
        prefix_lines.append(f"updated_at: {datetime.now(tz=UTC).isoformat()}")
    body = "\n".join(kept_lines).lstrip("\n")
    return "\n".join([*prefix_lines, body])


def _has_usable_updated_at(lines: list[str]) -> bool:
    """True iff some line carries a non-blank, non-placeholder ``updated_at:`` value."""
    for line in lines:
        if line.startswith("updated_at:"):
            raw_val = line.split(":", 1)[1].strip()
            if len(raw_val) > 0 and raw_val.lower() not in _METADATA_PLACEHOLDER_VALUES:
                return True
    return False
