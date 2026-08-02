"""Tests for MemoryCapability runtime (TM.1-TM.5)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.exceptions.exceptions import WorkspaceReadNotFoundError
from troopai.adk.sandbox.capabilities.memory import (
    MemoryCapability,
    MemoryLayoutConfig,
    MemoryReadConfig,
    resolve_conversation_id,
)
from troopai.adk.types.sandbox.entries import Dir
from troopai.adk.types.sandbox.manifest import Manifest


class TestMemoryLayoutConfigValidation:
    def test_default_construction(self) -> None:
        cfg = MemoryLayoutConfig()
        assert cfg.memories_dir == "memories"
        assert cfg.sessions_dir == "sessions"

    def test_absolute_memories_dir_rejected(self) -> None:
        with pytest.raises(ValueError, match="workspace-relative"):
            MemoryLayoutConfig(memories_dir="/tmp/memories")

    def test_dotdot_sessions_dir_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"\.\."):
            MemoryLayoutConfig(sessions_dir="../sessions")

    def test_empty_memories_dir_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            MemoryLayoutConfig(memories_dir="")


class TestResolveConversationId:
    def test_explicit_wins(self) -> None:
        assert (
            resolve_conversation_id(
                conversation_id="explicit",
                sdk_session_id="sess",
                run_group_id="group",
            )
            == "explicit"
        )

    def test_sdk_session_second(self) -> None:
        assert resolve_conversation_id(sdk_session_id="sess", run_group_id="group") == "sess"

    def test_group_third(self) -> None:
        assert resolve_conversation_id(run_group_id="group") == "group"

    def test_falls_back_to_generated(self) -> None:
        cid = resolve_conversation_id()
        assert cid.startswith("troopai-mem-")
        assert len(cid) > len("troopai-mem-")


class TestProcessManifest:
    def test_reserves_both_dirs(self) -> None:
        cap = MemoryCapability()
        result = cap.process_manifest(Manifest())
        assert "memories" in result.entries
        assert "sessions" in result.entries
        assert isinstance(result.entries["memories"], Dir)
        assert isinstance(result.entries["sessions"], Dir)

    def test_overlap_raises(self) -> None:
        cap = MemoryCapability()
        manifest = Manifest(entries={"memories": Dir(children={})})
        with pytest.raises(ValueError, match="overlaps"):
            cap.process_manifest(manifest)

    def test_custom_layout(self) -> None:
        cap = MemoryCapability(
            layout=MemoryLayoutConfig(
                memories_dir="custom_mem",
                sessions_dir="custom_sess",
            ),
        )
        result = cap.process_manifest(Manifest())
        assert "custom_mem" in result.entries
        assert "custom_sess" in result.entries


class TestInstructions:
    @pytest.mark.asyncio
    async def test_unbound_returns_none(self) -> None:
        cap = MemoryCapability()
        assert await cap.instructions(Manifest()) is None

    @pytest.mark.asyncio
    async def test_missing_summary_returns_none(self) -> None:
        cap = MemoryCapability()
        session = MagicMock()
        session.read = AsyncMock(
            side_effect=WorkspaceReadNotFoundError("memory_summary.md"),
        )
        cap.bind(session)
        assert await cap.instructions(Manifest()) is None

    @pytest.mark.asyncio
    async def test_existing_summary_returned(self) -> None:
        cap = MemoryCapability()
        session = MagicMock()
        stream = MagicMock()
        stream.read = MagicMock(return_value=b"prior run learned X")
        stream.close = MagicMock()
        session.read = AsyncMock(return_value=stream)
        cap.bind(session)
        result = await cap.instructions(Manifest())
        assert result is not None
        assert "prior run learned X" in result

    @pytest.mark.asyncio
    async def test_live_update_hint_present(self) -> None:
        cap = MemoryCapability(read=MemoryReadConfig(live_update=True))
        session = MagicMock()
        stream = MagicMock()
        stream.read = MagicMock(return_value=b"summary text")
        stream.close = MagicMock()
        session.read = AsyncMock(return_value=stream)
        cap.bind(session)
        result = await cap.instructions(Manifest())
        assert result is not None
        assert "MEMORY.md" in result

    @pytest.mark.asyncio
    async def test_live_update_hint_absent_when_disabled(self) -> None:
        cap = MemoryCapability(read=MemoryReadConfig(live_update=False))
        session = MagicMock()
        stream = MagicMock()
        stream.read = MagicMock(return_value=b"summary text")
        stream.close = MagicMock()
        session.read = AsyncMock(return_value=stream)
        cap.bind(session)
        result = await cap.instructions(Manifest())
        assert result is not None
        assert "MEMORY.md" not in result

    @pytest.mark.asyncio
    async def test_long_summary_truncated(self) -> None:
        cap = MemoryCapability()
        session = MagicMock()
        stream = MagicMock()
        stream.read = MagicMock(return_value=b"x" * 200_000)
        stream.close = MagicMock()
        session.read = AsyncMock(return_value=stream)
        cap.bind(session)
        result = await cap.instructions(Manifest())
        assert result is not None
        assert "truncated" in result


class TestAppendRunSegment:
    @pytest.mark.asyncio
    async def test_first_segment_creates_file(self) -> None:
        cap = MemoryCapability()
        session = MagicMock()
        session.read = AsyncMock(side_effect=WorkspaceReadNotFoundError("not yet"))
        session.write = AsyncMock()
        cap.bind(session)
        await cap.append_run_segment(
            rollout_id="run-1",
            segment={"role": "user", "content": "hi"},
        )
        session.write.assert_awaited_once()
        call_path = session.write.call_args.args[0]
        assert call_path == "sessions/run-1.jsonl"
        written_bytes = session.write.call_args.args[1].read()
        line = json.loads(written_bytes.decode("utf-8").rstrip("\n"))
        assert line == {"role": "user", "content": "hi"}

    @pytest.mark.asyncio
    async def test_subsequent_segment_appends(self) -> None:
        cap = MemoryCapability()
        session = MagicMock()
        existing = b'{"role": "user", "content": "first"}\n'
        stream = MagicMock()
        stream.read = MagicMock(return_value=existing)
        stream.close = MagicMock()
        session.read = AsyncMock(return_value=stream)
        session.write = AsyncMock()
        cap.bind(session)
        await cap.append_run_segment(
            rollout_id="run-1",
            segment={"role": "assistant", "content": "second"},
        )
        written_bytes = session.write.call_args.args[1].read()
        lines = written_bytes.decode("utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["content"] == "first"
        assert json.loads(lines[1])["content"] == "second"

    @pytest.mark.asyncio
    async def test_unbound_raises(self) -> None:
        cap = MemoryCapability()
        with pytest.raises(ValueError, match="bound session"):
            await cap.append_run_segment(rollout_id="x", segment={})


class TestConsolidatedMemoryIO:
    @pytest.mark.asyncio
    async def test_write_consolidated(self) -> None:
        cap = MemoryCapability()
        session = MagicMock()
        session.write = AsyncMock()
        cap.bind(session)
        await cap.write_consolidated_memory(
            memory_md="# MEMORY\nlessons",
            memory_summary_md="summary",
        )
        # Two writes — MEMORY.md and memory_summary.md.
        assert session.write.await_count == 2
        paths = [call.args[0] for call in session.write.call_args_list]
        assert "memories/MEMORY.md" in paths
        assert "memories/memory_summary.md" in paths

    @pytest.mark.asyncio
    async def test_read_raw_memories_missing_returns_empty(self) -> None:
        cap = MemoryCapability()
        session = MagicMock()
        session.read = AsyncMock(side_effect=WorkspaceReadNotFoundError("nope"))
        cap.bind(session)
        assert await cap.read_raw_memories() == ""

    @pytest.mark.asyncio
    async def test_read_raw_memories_present(self) -> None:
        cap = MemoryCapability()
        session = MagicMock()
        stream = MagicMock()
        stream.read = MagicMock(return_value=b"raw memory line\n")
        stream.close = MagicMock()
        session.read = AsyncMock(return_value=stream)
        cap.bind(session)
        assert (await cap.read_raw_memories()).strip() == "raw memory line"
