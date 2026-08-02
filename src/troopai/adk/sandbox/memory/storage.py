"""Read and write sandbox memory files via a ``BaseSandboxSession``.

The storage layer owns the on-workspace layout:

```
{memories_dir}/
    MEMORY.md
    memory_summary.md
    raw_memories.md                     # merged consolidation input
    raw_memories/{rollout_id}.md        # the extraction step outputs
    rollout_summaries/{rid}_{slug}.md   # the extraction step outputs
    skills/                             # optional the consolidation step outputs
    consolidation_selection.json            # state of the last consolidation
{sessions_dir}/
    {rollout_id}.jsonl                  # the extraction step inputs
```

All reads / writes go through ``BaseSandboxSession`` primitives so
the same code works against every backend (local, Docker, K8s,
hosted). Concurrent calls to ``ensure_layout`` are serialized by an
``asyncio.Lock`` so two enqueues can't race the first-time mkdir.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from troopai.adk.exceptions.exceptions import WorkspaceReadNotFoundError

if TYPE_CHECKING:
    from troopai.adk.sandbox.capabilities.memory import MemoryLayoutConfig
    from troopai.adk.sandbox.clients.session import BaseSandboxSession

logger = logging.getLogger(__name__)

__all__ = [
    "ConsolidationInputSelection",
    "ConsolidationSelectionItem",
    "SandboxMemoryStorage",
    "decode_payload",
    "make_safe_slug",
]


def decode_payload(payload: object) -> str:
    """Turn the result of ``stream.read()`` into a UTF-8 string.

    Raises ``TypeError`` for non-bytes/str payloads — the session
    contract says read() returns one of those; fall-through to
    ``str()`` would leak ``repr()`` debris into durable memory.
    """
    if isinstance(payload, str):
        return payload
    if isinstance(payload, bytes | bytearray):
        return bytes(payload).decode("utf-8", errors="replace")
    raise TypeError(
        f"BaseSandboxSession.read() returned unexpected type {type(payload).__name__} — "
        "expected str | bytes | bytearray"
    )


def make_safe_slug(slug: str) -> str:
    """Normalize a slug to file-safe ASCII; falls back to ``rollout``.

    Shared between the storage layer's filename construction and the
    extraction-result metadata so the on-disk file and the
    ``rollout_summary_file`` frontmatter cannot diverge.
    """
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in slug)[:80]
    if len(safe) == 0:
        return "rollout"
    return safe


@dataclass(frozen=True, kw_only=True)
class ConsolidationSelectionItem:
    """One rollout queued for the consolidation step."""

    rollout_id: str
    """Stable identifier for the rollout."""

    updated_at: str
    """ISO 8601 timestamp from the raw_memory's frontmatter."""

    rollout_path: str
    """Path of the raw rollout JSONL inside the workspace."""

    rollout_summary_file: str
    """Filename of the per-rollout summary under ``rollout_summaries/``."""

    terminal_state: str
    """Terminal state classification (``"completed"`` / ``"failed"`` / ...)."""

    def to_dict(self) -> dict[str, str]:
        return {
            "rollout_id": self.rollout_id,
            "updated_at": self.updated_at,
            "rollout_path": self.rollout_path,
            "rollout_summary_file": self.rollout_summary_file,
            "terminal_state": self.terminal_state,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ConsolidationSelectionItem | None:
        rollout_id = str(payload.get("rollout_id") or "").strip()
        rollout_summary_file = str(payload.get("rollout_summary_file") or "").strip()
        if len(rollout_id) == 0 or len(rollout_summary_file) == 0:
            return None
        return cls(
            rollout_id=rollout_id,
            updated_at=str(payload.get("updated_at") or "").strip(),
            rollout_path=str(payload.get("rollout_path") or "").strip(),
            rollout_summary_file=rollout_summary_file,
            terminal_state=str(payload.get("terminal_state") or "").strip(),
        )


@dataclass(frozen=True, kw_only=True)
class ConsolidationInputSelection:
    """What the consolidation step sees: which rollouts are queued + delta vs the prior run."""

    selected: list[ConsolidationSelectionItem]
    """Rollouts included in this consolidation pass."""

    retained_rollout_ids: set[str]
    """Rollout IDs that were ALSO in the prior selection."""

    removed: list[ConsolidationSelectionItem]
    """Rollouts present in the prior selection but not this one (evicted)."""


class SandboxMemoryStorage:
    """Read / write the memory layout via a ``BaseSandboxSession``."""

    def __init__(
        self,
        *,
        session: BaseSandboxSession,
        layout: MemoryLayoutConfig,
    ) -> None:
        self._session = session
        self._layout = layout
        self._layout_lock = asyncio.Lock()
        # One lock per rollout_id so the append read-modify-write is atomic
        # against concurrent appends for the SAME rollout (interleaving at the
        # read/write await points would otherwise clobber a segment). Distinct
        # rollouts stay unserialized.
        self._append_locks: dict[str, asyncio.Lock] = {}

    @property
    def sessions_dir(self) -> Path:
        """Workspace-relative directory for rollout JSONL segments."""
        return Path(self._layout.sessions_dir)

    @property
    def memories_dir(self) -> Path:
        """Workspace-relative directory for consolidated memory artifacts."""
        return Path(self._layout.memories_dir)

    @property
    def raw_memories_dir(self) -> Path:
        """Where the extraction step writes per-rollout raw memories."""
        return self.memories_dir / "raw_memories"

    @property
    def rollout_summaries_dir(self) -> Path:
        """Where the extraction step writes per-rollout summaries."""
        return self.memories_dir / "rollout_summaries"

    @property
    def skills_dir(self) -> Path:
        """Where the consolidation step may write optional reusable skill packets."""
        return self.memories_dir / "skills"

    @property
    def memory_md_path(self) -> Path:
        """Canonical durable memory file rewritten by the consolidation step."""
        return self.memories_dir / "MEMORY.md"

    @property
    def memory_summary_path(self) -> Path:
        """Short summary injected into the agent system prompt."""
        return self.memories_dir / "memory_summary.md"

    @property
    def raw_memories_md_path(self) -> Path:
        """Merged the current input — concatenation of selected raw memories."""
        return self.memories_dir / "raw_memories.md"

    @property
    def consolidation_selection_path(self) -> Path:
        """JSON record of which rollouts were consolidated last."""
        return self.memories_dir / "consolidation_selection.json"

    async def ensure_layout(self) -> None:
        """Materialize every required directory + stub the consolidation files.

        Holds its own lock (distinct from the manager-level one, which
        guards the manager's cold-start flag) so direct storage callers
        without a manager also get serialized layout creation.
        """
        async with self._layout_lock:
            await asyncio.gather(
                self._session.mkdir(self.sessions_dir, parents=True),
                self._session.mkdir(self.memories_dir, parents=True),
                self._session.mkdir(self.raw_memories_dir, parents=True),
                self._session.mkdir(self.rollout_summaries_dir, parents=True),
                self._session.mkdir(self.skills_dir, parents=True),
            )
            await self._ensure_text_file(self.memory_md_path)
            await self._ensure_text_file(self.memory_summary_path)

    async def _ensure_text_file(self, path: Path) -> None:
        try:
            handle = await self._session.read(path)
            handle.close()
            return
        except (FileNotFoundError, WorkspaceReadNotFoundError):
            await self._session.write(path, io.BytesIO(b""))

    async def read_text(self, path: Path) -> str:
        """Return ``path``'s contents as a UTF-8 string."""
        handle = await self._session.read(path)
        try:
            return decode_payload(handle.read())
        finally:
            handle.close()

    async def write_text(self, path: Path, text: str) -> None:
        """Overwrite ``path`` with ``text`` (UTF-8 encoded)."""
        await self._session.write(path, io.BytesIO(text.encode("utf-8")))

    async def append_rollout_segment(
        self,
        *,
        rollout_id: str,
        payload_jsonl: str,
    ) -> Path:
        """Append a JSONL segment to the per-rollout sessions file.

        The first call materializes the parent directory; subsequent
        calls read the existing contents, concatenate, and write back
        through the session's ``write`` primitive. A per-rollout lock
        serializes the read-modify-write so concurrent appends for the
        same rollout cannot clobber each other's segment.
        """
        if len(rollout_id.strip()) == 0:
            raise ValueError("append_rollout_segment: rollout_id must be non-empty")
        await self._session.mkdir(self.sessions_dir, parents=True)
        path = self.sessions_dir / f"{rollout_id}.jsonl"
        async with self._append_locks.setdefault(rollout_id, asyncio.Lock()):
            existing = ""
            try:
                existing = await self.read_text(path)
            except (FileNotFoundError, WorkspaceReadNotFoundError):
                existing = ""
            new_contents = existing + payload_jsonl
            await self.write_text(path, new_contents)
        return path

    async def write_raw_memory(
        self,
        *,
        rollout_id: str,
        body: str,
    ) -> Path:
        """Drop a per-rollout ``raw_memory`` body at ``raw_memories/{rollout_id}.md``."""
        path = self.raw_memories_dir / f"{rollout_id}.md"
        await self.write_text(path, body)
        return path

    async def write_rollout_summary(
        self,
        *,
        rollout_id: str,
        slug: str,
        body: str,
    ) -> Path:
        """Drop a per-rollout ``rollout_summary`` body at ``rollout_summaries/{id}_{slug}.md``.

        ``slug`` MUST already be sanitized via ``make_safe_slug`` —
        the caller owns the mapping so frontmatter references and
        the on-disk file name cannot diverge.
        """
        path = self.rollout_summaries_dir / f"{rollout_id}_{slug}.md"
        await self.write_text(path, body)
        return path

    async def build_consolidation_selection(
        self,
        *,
        max_raw_memories_for_consolidation: int,
    ) -> ConsolidationInputSelection:
        """Pick the top N raw memories and diff against the prior selection."""
        current_items = await self._list_current_selection_items()
        selected = current_items[:max_raw_memories_for_consolidation]
        prior_selected = await self.read_consolidation_selection()
        selected_rollout_ids = {item.rollout_id for item in selected}
        prior_rollout_ids = {item.rollout_id for item in prior_selected}
        return ConsolidationInputSelection(
            selected=selected,
            retained_rollout_ids=selected_rollout_ids & prior_rollout_ids,
            removed=[item for item in prior_selected if item.rollout_id not in selected_rollout_ids],
        )

    async def rebuild_raw_memories(
        self,
        *,
        selected_items: list[ConsolidationSelectionItem],
    ) -> bool:
        """Merge the selected raw memories into ``memories/raw_memories.md``.

        Returns ``True`` iff at least one selected item was readable.
        """
        chunks: list[str] = []
        for item in selected_items:
            raw_memory_path = self.raw_memories_dir / f"{item.rollout_id}.md"
            try:
                chunks.append((await self.read_text(raw_memory_path)).rstrip("\n"))
            except (FileNotFoundError, WorkspaceReadNotFoundError):
                continue
        if len(chunks) == 0:
            return False
        await self.write_text(self.raw_memories_md_path, "\n\n".join(chunks))
        return True

    async def read_consolidation_selection(self) -> list[ConsolidationSelectionItem]:
        """Load the prior consolidation-step selection state, or ``[]`` when absent / malformed.

        Malformed JSON / unexpected schema logs at WARNING and tries
        to move the corrupted file aside (``.corrupt.<timestamp>``)
        so we keep a forensic trail of state loss — silent return
        would otherwise destroy the eviction record on the next
        successful flush.
        """
        try:
            raw_payload = await self.read_text(self.consolidation_selection_path)
        except (FileNotFoundError, WorkspaceReadNotFoundError):
            return []
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError as exc:
            await self._quarantine_selection_file(
                reason=f"json decode failed: {exc}",
            )
            return []
        if not isinstance(payload, dict):
            await self._quarantine_selection_file(reason="payload is not an object")
            return []
        selected = payload.get("selected")
        if not isinstance(selected, list):
            await self._quarantine_selection_file(reason="'selected' is not a list")
            return []
        items: list[ConsolidationSelectionItem] = []
        for entry in selected:
            if not isinstance(entry, dict):
                continue
            item = ConsolidationSelectionItem.from_dict(entry)
            if item is not None:
                items.append(item)
        return items

    async def write_consolidation_selection(
        self,
        *,
        selected_items: list[ConsolidationSelectionItem],
    ) -> None:
        """Persist the current consolidation-step selection so the next flush can diff."""
        payload = {
            "updated_at": datetime.now(tz=UTC).isoformat(),
            "selected": [item.to_dict() for item in selected_items],
        }
        await self.write_text(
            self.consolidation_selection_path,
            json.dumps(payload, indent=2) + "\n",
        )

    async def _quarantine_selection_file(self, *, reason: str) -> None:
        """Move a corrupted ``consolidation_selection.json`` aside with a timestamp.

        Best-effort: a backend that can't perform the move logs and
        continues — the corrupted file then gets overwritten on the
        next successful flush, which is the least-bad outcome.
        """
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        quarantine = self.consolidation_selection_path.with_suffix(f".json.corrupt.{timestamp}")
        logger.warning(
            "consolidation_selection.json malformed (%s); quarantining to %s",
            reason,
            quarantine.as_posix(),
        )
        try:
            raw = await self.read_text(self.consolidation_selection_path)
            await self.write_text(quarantine, raw)
        except Exception as exc:
            logger.warning(
                "consolidation_selection.json quarantine failed: %s — corrupted file will be overwritten on next flush",
                exc,
            )

    async def _list_current_selection_items(self) -> list[ConsolidationSelectionItem]:
        try:
            entries = await self._session.ls(self.raw_memories_dir)
        except (FileNotFoundError, WorkspaceReadNotFoundError):
            return []
        except OSError as exc:
            logger.error(
                "consolidation: ls(raw_memories_dir) failed with OSError — treating as empty: %s",
                exc,
            )
            return []

        items: list[tuple[tuple[int, str], str, ConsolidationSelectionItem]] = []
        skipped: list[str] = []
        for entry in entries:
            if entry.is_directory:
                continue
            # ``FileEntry`` declares ``name: str`` via __slots__, so direct attribute
            # access is the typed-surface path; the prior defensive getattr() hid
            # the contract from the static checker.
            name = entry.name
            if not name.endswith(".md"):
                continue
            try:
                raw_memory = (await self.read_text(self.raw_memories_dir / name)).rstrip("\n")
            except (FileNotFoundError, WorkspaceReadNotFoundError):
                continue
            item = _extract_selection_item(raw_memory)
            if item is None:
                skipped.append(name)
                continue
            items.append((_updated_at_sort_key(raw_memory), item.rollout_id, item))
        if len(skipped) > 0:
            logger.warning(
                "consolidation: %d raw-memory file(s) missing required frontmatter and were skipped: %s",
                len(skipped),
                ", ".join(skipped),
            )
        items.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in items]


def _updated_at_sort_key(raw_memory: str) -> tuple[int, str]:
    for line in raw_memory.splitlines():
        if line.startswith("updated_at:"):
            _, value = line.split(":", maxsplit=1)
            updated_at = value.strip()
            if len(updated_at) == 0 or updated_at == "unknown":
                return (0, "")
            return (1, updated_at)
    return (0, "")


def _extract_selection_item(raw_memory: str) -> ConsolidationSelectionItem | None:
    rollout_id = _extract_metadata_value(raw_memory, "rollout_id")
    rollout_summary_file = _extract_metadata_value(raw_memory, "rollout_summary_file")
    if len(rollout_id) == 0 or len(rollout_summary_file) == 0:
        return None
    return ConsolidationSelectionItem(
        rollout_id=rollout_id,
        updated_at=_extract_metadata_value(raw_memory, "updated_at"),
        rollout_path=_extract_metadata_value(raw_memory, "rollout_path"),
        rollout_summary_file=rollout_summary_file,
        terminal_state=_extract_metadata_value(raw_memory, "terminal_state"),
    )


def _extract_metadata_value(raw_memory: str, key: str) -> str:
    prefix = f"{key}:"
    for line in raw_memory.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    return ""
