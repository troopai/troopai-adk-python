"""``MemoryCapability`` — workspace-persisted run memory.

Memory artifacts live in the sandbox workspace under
``{layout.memories_dir}/`` (default ``memories/``). The capability
reads ``memory_summary.md`` into the system prompt on each run and
appends a JSONL run-segment to ``{layout.sessions_dir}/{rollout-id}.jsonl``.

Memory generation runs at session-stop time in two passes — raw
capture then LLM-driven consolidation — both configured via
``MemoryGenerateConfig``. The helpers here ship the persistence +
read paths; the consolidation prompt is the caller's responsibility.
"""

from __future__ import annotations

import json
import uuid
from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal, override

from pydantic import BaseModel, ConfigDict, Field

from troopai.adk.sandbox.capabilities.base import SandboxCapability

if TYPE_CHECKING:
    from troopai.adk.types.sandbox.manifest import Manifest

__all__ = [
    "MemoryCapability",
    "MemoryGenerateConfig",
    "MemoryLayoutConfig",
    "MemoryReadConfig",
    "resolve_conversation_id",
]


_MEMORY_SUMMARY_MAX_CHARS = 60_000
"""Approx 15k tokens at ~4 chars/token — keeps the memory summary within a single-message budget."""


def _validate_relative_no_dotdot(name: str, label: str) -> None:
    if len(name) == 0:
        raise ValueError(f"{label} must be non-empty")
    if name.startswith("/") or name.startswith("\\"):
        raise ValueError(f"{label} must be workspace-relative, got {name!r}")
    if ".." in name or "\x00" in name:
        raise ValueError(f"{label} must not contain '..' or null bytes; got {name!r}")


class MemoryLayoutConfig(BaseModel):
    """Filesystem layout for memory artifacts inside the workspace.

    Attributes:
        memories_dir: Workspace-relative directory for memory files
            (``MEMORY.md``, ``memory_summary.md``, ``raw_memories.md``).
            Must be a non-empty path with no ``..`` traversal.
        sessions_dir: Workspace-relative directory for per-rollout JSONL
            session files. Same path constraints as ``memories_dir``.
    """

    model_config = ConfigDict(frozen=True)

    memories_dir: str = "memories"
    sessions_dir: str = "sessions"

    @override
    def model_post_init(self, context: Any) -> None:
        del context
        _validate_relative_no_dotdot(self.memories_dir, "MemoryLayoutConfig.memories_dir")
        _validate_relative_no_dotdot(self.sessions_dir, "MemoryLayoutConfig.sessions_dir")


class MemoryReadConfig(BaseModel):
    """Read-side memory behavior.

    Attributes:
        live_update: When True, the agent may update the memory files
            mid-run when it spots stale or incorrect entries.
    """

    model_config = ConfigDict(frozen=True)

    live_update: bool = True
    """When True, the agent may update memory mid-run if it spots stale entries."""


class MemoryGenerateConfig(BaseModel):
    """Generation-side memory behavior.

    Attributes:
        max_raw_memories_for_consolidation: Cap on the number of raw-memory
            entries forwarded to the consolidation step in one flush. Beyond
            the cap the oldest entries are trimmed.
        extra_prompt: Optional extra text appended to the consolidation
            prompt to steer what the LLM retains or discards.
    """

    model_config = ConfigDict(frozen=True)

    max_raw_memories_for_consolidation: int = 256
    extra_prompt: str | None = None


def resolve_conversation_id(
    *,
    conversation_id: str | None = None,
    sdk_session_id: str | None = None,
    run_group_id: str | None = None,
) -> str:
    """Return the conversation ID following the documented priority.

    Order: ``conversation_id`` > ``sdk_session_id`` > ``run_group_id``
    > a freshly-generated UUID4.
    """
    if conversation_id is not None and len(conversation_id) > 0:
        return conversation_id
    if sdk_session_id is not None and len(sdk_session_id) > 0:
        return sdk_session_id
    if run_group_id is not None and len(run_group_id) > 0:
        return run_group_id
    return f"troopai-mem-{uuid.uuid4().hex[:12]}"


class MemoryCapability(SandboxCapability):
    """Capability that wires workspace-persisted memory.

    Attributes:
        type: Discriminator. Always ``"memory"``.
        layout: Filesystem directory layout for memory artifacts inside
            the workspace.
        read: Read-side behavior (inject summary into prompt, live
            update). ``None`` disables prompt injection.
        generate: Generation-side behavior (flush + consolidation
            parameters). ``None`` disables memory generation.
    """

    type: Literal["memory"] = "memory"

    layout: MemoryLayoutConfig = Field(default_factory=MemoryLayoutConfig)
    read: MemoryReadConfig | None = Field(default_factory=MemoryReadConfig)
    generate: MemoryGenerateConfig | None = Field(default_factory=MemoryGenerateConfig)

    @override
    def required_capability_types(self) -> set[str]:
        if self.read is None:
            return set()
        if self.read.live_update:
            return {"shell", "filesystem"}
        return {"shell"}

    @override
    def process_manifest(self, manifest: Manifest) -> Manifest:
        from troopai.adk.types.sandbox.entries import Dir
        from troopai.adk.types.sandbox.manifest import Manifest as _Manifest

        for label, path in (
            ("memories_dir", self.layout.memories_dir),
            ("sessions_dir", self.layout.sessions_dir),
        ):
            if path in manifest.entries:
                raise ValueError(
                    f"MemoryCapability.layout.{label} {path!r} overlaps an existing manifest entry",
                )
        new_entries = dict(manifest.entries)
        new_entries[self.layout.memories_dir] = Dir(children={})
        new_entries[self.layout.sessions_dir] = Dir(children={})
        return _Manifest(
            root=manifest.root,
            entries=new_entries,
            environment=manifest.environment,
            users=manifest.users,
            groups=manifest.groups,
            extra_path_grants=manifest.extra_path_grants,
            remote_mount_command_allowlist=manifest.remote_mount_command_allowlist,
        )

    @override
    async def instructions(self, manifest: Manifest | None) -> str | None:
        _ = manifest
        if self.read is None or self.session is None:
            return None
        from troopai.adk.exceptions.exceptions import WorkspaceReadNotFoundError

        summary_path = f"{self.layout.memories_dir}/memory_summary.md"
        try:
            stream = await self.session.read(summary_path)
        except WorkspaceReadNotFoundError:
            return None
        try:
            payload = stream.read()
        finally:
            stream.close()
        text = payload.decode("utf-8", errors="replace").strip()
        if len(text) == 0:
            return None
        if len(text) > _MEMORY_SUMMARY_MAX_CHARS:
            text = (
                text[: _MEMORY_SUMMARY_MAX_CHARS // 2]
                + "\n[... truncated ...]\n"
                + text[-_MEMORY_SUMMARY_MAX_CHARS // 2 :]
            )
        live_hint = (
            " You may update `memories/MEMORY.md` when you discover that "
            "memory is stale or the user asks for an update."
            if self.read.live_update
            else ""
        )
        return f"Persistent memory from prior runs (treat as guidance, not ground truth):{live_hint}\n\n{text}"

    async def append_run_segment(
        self,
        *,
        rollout_id: str,
        segment: dict[str, Any],
    ) -> None:
        """Append ``segment`` as one JSON line to the rollout file."""
        if self.session is None:
            raise ValueError(
                "MemoryCapability.append_run_segment requires a bound session",
            )
        path = f"{self.layout.sessions_dir}/{rollout_id}.jsonl"
        from troopai.adk.exceptions.exceptions import WorkspaceReadNotFoundError

        existing = b""
        try:
            stream = await self.session.read(path)
        except WorkspaceReadNotFoundError:
            stream = None
        if stream is not None:
            try:
                existing = stream.read()
            finally:
                stream.close()
        line = (json.dumps(segment, ensure_ascii=False) + "\n").encode("utf-8")
        await self.session.write(path, BytesIO(existing + line))

    async def read_raw_memories(self) -> str:
        """Read the raw-memories accumulator file used to feed consolidation."""
        if self.session is None:
            raise ValueError("MemoryCapability.read_raw_memories requires a bound session")
        from troopai.adk.exceptions.exceptions import WorkspaceReadNotFoundError

        path = f"{self.layout.memories_dir}/raw_memories.md"
        try:
            stream = await self.session.read(path)
        except WorkspaceReadNotFoundError:
            return ""
        try:
            raw_bytes: bytes = stream.read()
            return raw_bytes.decode("utf-8", errors="replace")
        finally:
            stream.close()

    async def write_consolidated_memory(
        self,
        *,
        memory_md: str,
        memory_summary_md: str,
    ) -> None:
        """Write the consolidation-step output: MEMORY.md + memory_summary.md."""
        if self.session is None:
            raise ValueError(
                "MemoryCapability.write_consolidated_memory requires a bound session",
            )
        await self.session.write(
            f"{self.layout.memories_dir}/MEMORY.md",
            BytesIO(memory_md.encode("utf-8")),
        )
        await self.session.write(
            f"{self.layout.memories_dir}/memory_summary.md",
            BytesIO(memory_summary_md.encode("utf-8")),
        )
