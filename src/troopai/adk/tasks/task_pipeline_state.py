"""TaskPipelineState — serializable resume state for TaskPipeline runs.

A :class:`TaskPipelineState` records the completed slots and the next
task index for a partially-executed :class:`TaskPipeline`. Persisting
the state between processes lets a long pipeline survive a crash or
a deliberate pause/resume across sessions.

Scope:

- The state captures **between-task boundaries only**. Mid-turn HITL
  resume — pausing inside an agent loop and resuming from a captured
  :class:`~troopai.adk.run.state.RunState` — is intentionally NOT
  supported. The resume entry point starts the next-indexed task
  from scratch.
- :attr:`TaskOutput.new_items` is not serialized. The audit trail
  for completed tasks lives on the
  :class:`~troopai.adk.session.session.Session` if one is attached;
  resume rebuilds it from there.
- :attr:`Task.skip_if` / :attr:`Task.metadata` / :attr:`Task.agent`
  are NOT serialized. Resume requires the developer to reconstruct
  the same :class:`TaskPipeline` definition on the resuming side
  (same Agent identities, same Task definitions) — analogous to
  :class:`~troopai.adk.run.state.RunState`'s "same Agent definition
  on both sides" contract.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, override

if TYPE_CHECKING:
    from troopai.adk.tasks.task_output import TaskOutput


@dataclass(frozen=True, kw_only=True)
class TaskPipelineState:
    """Serializable mid-pipeline checkpoint.

    Attributes:
        pipeline_id: Stable identifier for the pipeline run. The
            developer chooses the format (UUID, timestamp, business
            correlation id); the framework does not generate one.
            Carried through round-trips so the resuming side can
            correlate the resume with the originating run.
        slots: Per-task :class:`TaskOutput` slots in input order. Each
            slot covers a task that finished (either ran or skipped)
            BEFORE the checkpoint. Same indexing as
            :attr:`TaskPipeline.tasks`.
        resume_index: Index of the next task to execute on resume.
            ``len(slots) == resume_index`` is the invariant after a
            clean checkpoint at a task boundary. Resume from index 0
            re-runs the entire pipeline. For DAG-shaped pipelines
            (any task declaring :attr:`Task.depends_on`) the
            :attr:`completed_task_ids` set is authoritative on what
            to skip; ``resume_index`` is then advisory only.
        completed_task_ids: Stable IDs of every task that finished
            before the checkpoint (whether ran or skipped). Used by
            DAG-resume to identify which level-N tasks to re-run
            (those whose ID is NOT in this set). Empty tuple ⇒
            positional-index semantics apply.
        metadata: Open-ended developer tag dict. Surfaced verbatim on
            the resumed :class:`TaskPipelineResult`. Useful for
            tenant ids, request-correlation tags, etc.
    """

    pipeline_id: str
    """Stable identifier chosen by the developer."""

    slots: tuple[TaskOutput, ...]
    """Per-task TaskOutput slots in input order."""

    resume_index: int
    """Index of the next task to execute on resume."""

    completed_task_ids: tuple[str, ...] = ()
    """Task IDs that finished before the checkpoint (DAG resume only)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Open-ended developer metadata."""

    @override
    def __repr__(self) -> str:
        """One-line checkpoint summary: id, slots, resume point.

        ``completed`` counts :attr:`completed_task_ids` — the
        authoritative skip-set on DAG resume.
        """
        parts: list[str] = [
            f"pipeline_id={self.pipeline_id!r}",
            f"slots={len(self.slots)}",
            f"resume_index={self.resume_index}",
            f"completed={len(self.completed_task_ids)}",
        ]
        return f"TaskPipelineState({', '.join(parts)})"

    def __post_init__(self) -> None:
        """Validate :class:`TaskPipelineState` construction.

        Raises:
            ValueError: When :attr:`resume_index` is negative or
                inconsistent with :attr:`slots` length.
        """
        if self.resume_index < 0:
            raise ValueError(
                f"TaskPipelineState.resume_index must be non-negative, got {self.resume_index}",
            )
        if self.resume_index > len(self.slots):
            raise ValueError(
                f"TaskPipelineState.resume_index ({self.resume_index}) cannot exceed "
                f"len(slots) ({len(self.slots)}) — slots must cover every task before the resume point.",
            )

    def to_json(self) -> str:
        """Serialize to a JSON string.

        The output is portable across processes. Load via
        :meth:`from_json`.

        Returns:
            A compact JSON string representing this checkpoint. All
            values are JSON-native scalars, lists, or dicts.
        """
        payload = {
            "pipeline_id": self.pipeline_id,
            "resume_index": self.resume_index,
            "metadata": dict(self.metadata),
            "slots": [slot.to_dict() for slot in self.slots],
            "completed_task_ids": list(self.completed_task_ids),
        }
        return json.dumps(payload)

    @classmethod
    def from_json(cls, payload: str) -> TaskPipelineState:
        """Rehydrate from a :meth:`to_json` output.

        Required fields (``pipeline_id``, ``slots``, ``resume_index``)
        MUST be present in the payload; their absence raises
        :class:`ValueError` so a truncated or corrupted state cannot
        silently load as a partial / empty pipeline. ``metadata`` and
        ``completed_task_ids`` are optional with empty defaults.

        Args:
            payload: A JSON string previously produced by
                :meth:`to_json`.

        Returns:
            A :class:`TaskPipelineState` rehydrated from ``payload``.

        Raises:
            ValueError: When ``payload`` is not valid JSON or when any
                of the required fields (``pipeline_id``, ``slots``,
                ``resume_index``) is missing.
        """
        from troopai.adk.tasks.task_output import TaskOutput

        data = json.loads(payload)
        required = ("pipeline_id", "slots", "resume_index")
        missing = [key for key in required if key not in data]
        if len(missing) > 0:
            raise ValueError(
                f"TaskPipelineState payload missing required field(s): {missing!r}",
            )
        slots = tuple(TaskOutput.from_dict(s) for s in data["slots"])
        return cls(
            pipeline_id=data["pipeline_id"],
            slots=slots,
            resume_index=data["resume_index"],
            completed_task_ids=tuple(data.get("completed_task_ids", [])),
            metadata=dict(data.get("metadata", {})),
        )
