"""End-to-end tests for ``SandboxMemoryManager`` with a fake session + LLMs."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

import pytest

from troopai.adk.sandbox.capabilities.memory import MemoryLayoutConfig
from troopai.adk.sandbox.memory import SandboxMemoryManager
from troopai.adk.sandbox.memory.rollouts import build_rollout_payload, dump_rollout_json


class _FakeSession:
    """In-memory session backing the manager's I/O calls."""

    def __init__(self) -> None:
        self._files: dict[str, bytes] = {}
        self._dirs: set[str] = set()

    def normalize_path(self, path: Path | str, *, for_write: bool = False) -> Path:
        _ = for_write
        return Path(path)

    async def mkdir(self, path: Path | str, *, parents: bool = False, user: Any = None) -> None:
        _ = parents, user
        self._dirs.add(str(path))

    async def write(self, path: Path | str, data: Any, *, user: Any = None) -> None:
        _ = user
        payload = data.read() if hasattr(data, "read") else data
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self._files[str(path)] = payload

    async def read(self, path: Path | str, *, user: Any = None) -> io.BytesIO:
        _ = user
        key = str(path)
        if key not in self._files:
            raise FileNotFoundError(key)
        return io.BytesIO(self._files[key])

    async def ls(self, path: Path | str, *, user: Any = None) -> list[Any]:
        _ = user
        prefix = f"{path}/"

        class _Entry:
            def __init__(self, name: str) -> None:
                self.name = name
                self.is_directory = False
                self.size_bytes = -1

        names: list[Any] = []
        for key in self._files:
            if key.startswith(prefix):
                names.append(_Entry(key[len(prefix) :]))
        return names

    async def rm(self, path: Path | str, *, recursive: bool = False, user: Any = None) -> None:
        _ = recursive, user
        self._files.pop(str(path), None)


@pytest.fixture
def fake_session() -> _FakeSession:
    return _FakeSession()


@pytest.mark.asyncio
async def test_enqueue_appends_jsonl_segment(fake_session: _FakeSession) -> None:
    async def fake_extract(_s: str, _u: str) -> str:
        return json.dumps({"rollout_slug": "", "rollout_summary": "", "raw_memory": ""})

    async def fake_consolidate(_s: str, _u: str) -> None:
        return None

    manager = SandboxMemoryManager(
        session=fake_session,  # type: ignore[arg-type]
        layout=MemoryLayoutConfig(),
        extraction_llm=fake_extract,
        consolidation_runner=fake_consolidate,
    )
    payload = build_rollout_payload(
        rollout_id="rollout-1",
        inputs="hi",
        new_items=[],
        final_output="ok",
    )
    await manager.enqueue_rollout(
        rollout_id="rollout-1",
        payload_jsonl=dump_rollout_json(payload),
    )
    assert "sessions/rollout-1.jsonl" in fake_session._files
    contents = fake_session._files["sessions/rollout-1.jsonl"].decode("utf-8")
    assert "rollout-1" in contents


@pytest.mark.asyncio
async def test_flush_runs_extraction_and_records_results(fake_session: _FakeSession) -> None:
    async def fake_extract(_s: str, _u: str) -> str:
        return json.dumps(
            {
                "rollout_slug": "demo-task",
                "rollout_summary": "User asked X; agent did Y.",
                "raw_memory": "## Notes\n- thing",
            }
        )

    consolidate_calls: list[tuple[str, str]] = []

    async def fake_consolidate(system: str, user: str) -> None:
        consolidate_calls.append((system, user))

    manager = SandboxMemoryManager(
        session=fake_session,  # type: ignore[arg-type]
        layout=MemoryLayoutConfig(),
        extraction_llm=fake_extract,
        consolidation_runner=fake_consolidate,
    )
    payload = build_rollout_payload(
        rollout_id="rollout-1",
        inputs="hi",
        new_items=[],
        final_output="ok",
    )
    await manager.enqueue_rollout(
        rollout_id="rollout-1",
        payload_jsonl=dump_rollout_json(payload),
    )
    result = await manager.flush()
    assert result.rollouts_processed == 1
    assert result.consolidation_skipped is False
    assert result.consolidated_at is not None
    # the consolidation phase was invoked because raw_memories.md rebuild succeeded.
    assert len(consolidate_calls) == 1
    # The raw memory landed on the workspace with frontmatter metadata.
    raw_path = "memories/raw_memories/rollout-1.md"
    assert raw_path in fake_session._files
    raw = fake_session._files[raw_path].decode("utf-8")
    assert "rollout_id: rollout-1" in raw
    assert "rollout_path: sessions/rollout-1.jsonl" in raw


@pytest.mark.asyncio
async def test_flush_skips_consolidation_on_empty_selection(fake_session: _FakeSession) -> None:
    async def fake_extract(_s: str, _u: str) -> str:
        return json.dumps({"rollout_slug": "", "rollout_summary": "", "raw_memory": ""})

    consolidate_called = False

    async def fake_consolidate(_s: str, _u: str) -> None:
        nonlocal consolidate_called
        consolidate_called = True

    manager = SandboxMemoryManager(
        session=fake_session,  # type: ignore[arg-type]
        layout=MemoryLayoutConfig(),
        extraction_llm=fake_extract,
        consolidation_runner=fake_consolidate,
    )
    payload = build_rollout_payload(
        rollout_id="rollout-1",
        inputs="hi",
        new_items=[],
        final_output="ok",
    )
    await manager.enqueue_rollout(
        rollout_id="rollout-1",
        payload_jsonl=dump_rollout_json(payload),
    )
    result = await manager.flush()
    assert result.rollouts_processed == 0
    assert result.consolidation_skipped is True
    assert not consolidate_called


@pytest.mark.asyncio
async def test_flush_can_skip_consolidation_explicitly(fake_session: _FakeSession) -> None:
    async def fake_extract(_s: str, _u: str) -> str:
        return json.dumps(
            {
                "rollout_slug": "demo",
                "rollout_summary": "summary",
                "raw_memory": "## Notes",
            }
        )

    consolidate_called = False

    async def fake_consolidate(_s: str, _u: str) -> None:
        nonlocal consolidate_called
        consolidate_called = True

    manager = SandboxMemoryManager(
        session=fake_session,  # type: ignore[arg-type]
        layout=MemoryLayoutConfig(),
        extraction_llm=fake_extract,
        consolidation_runner=fake_consolidate,
    )
    payload = build_rollout_payload(
        rollout_id="rollout-1",
        inputs="hi",
        new_items=[],
        final_output="ok",
    )
    await manager.enqueue_rollout(
        rollout_id="rollout-1",
        payload_jsonl=dump_rollout_json(payload),
    )
    result = await manager.flush(run_consolidation_pass=False)
    assert result.consolidation_skipped is True
    assert not consolidate_called


class TestEnsureMetadata:
    """Unit tests for the module-private _ensure_metadata helper."""

    _KWARGS: dict[str, str] = {
        "rollout_id": "r1",
        "rollout_path": "sessions/r1.jsonl",
        "rollout_summary_file": "rollout_summaries/r1_summary.md",
        "terminal_state": "completed",
    }

    def _call(self, raw_memory: str) -> str:
        from troopai.adk.sandbox.memory.manager import _ensure_metadata

        return _ensure_metadata(raw_memory, **self._KWARGS)  # type: ignore[arg-type]

    def test_blank_updated_at_is_replaced(self) -> None:
        """Regression: 'updated_at: ' (blank) was treated as present and never refreshed."""
        raw = "rollout_id: r1\nupdated_at: \nrollout_path: sessions/r1.jsonl\nrollout_summary_file: rollout_summaries/r1_summary.md\nterminal_state: completed\n## Notes\n- thing\n"
        result = self._call(raw)
        # updated_at must now carry a real ISO timestamp (no longer blank).
        for line in result.splitlines():
            if line.startswith("updated_at:"):
                value = line.split(":", 1)[1].strip()
                assert len(value) > 0, "updated_at must not be blank after ensure_metadata"
                assert value != "unknown"
                return
        raise AssertionError("updated_at field not found in result")

    def test_unknown_updated_at_is_replaced(self) -> None:
        """Regression: 'updated_at: unknown' was treated as present and never refreshed."""
        raw = "rollout_id: r1\nupdated_at: unknown\nrollout_path: sessions/r1.jsonl\nrollout_summary_file: rollout_summaries/r1_summary.md\nterminal_state: completed\n## Notes\n"
        result = self._call(raw)
        for line in result.splitlines():
            if line.startswith("updated_at:"):
                value = line.split(":", 1)[1].strip()
                assert value not in ("unknown", "null", "none", "")
                return
        raise AssertionError("updated_at field not found in result")

    def test_valid_updated_at_is_preserved(self) -> None:
        """A real ISO timestamp must be left unchanged (no spurious double-inject)."""
        ts = "2024-01-15T10:30:00+00:00"
        raw = f"rollout_id: r1\nupdated_at: {ts}\nrollout_path: sessions/r1.jsonl\nrollout_summary_file: rollout_summaries/r1_summary.md\nterminal_state: completed\n"
        result = self._call(raw)
        updated_at_values = [
            line.split(":", 1)[1].strip() for line in result.splitlines() if line.startswith("updated_at:")
        ]
        assert updated_at_values == [ts], "valid updated_at must be preserved exactly once"

    @staticmethod
    def _value_for(result: str, key: str) -> str | None:
        prefix = f"{key}:"
        for line in result.splitlines():
            if line.startswith(prefix):
                return line.split(":", 1)[1].strip()
        return None

    @staticmethod
    def _count_lines_for(result: str, key: str) -> int:
        prefix = f"{key}:"
        return sum(1 for line in result.splitlines() if line.startswith(prefix))

    def test_llm_supplied_rollout_id_is_overwritten_by_framework_value(self) -> None:
        """Regression: an LLM-hallucinated rollout_id must NOT win over the real one.

        The on-disk file is written to ``raw_memories/{real_rollout_id}.md`` while
        consolidation resolves the path from the frontmatter rollout_id. If the
        LLM's id were kept, that path would never exist and the memory would be
        silently dropped from consolidation.
        """
        raw = (
            "rollout_id: hallucinated-id\n"
            "rollout_path: /tmp/made-up.jsonl\n"
            "rollout_summary_file: wrong_summary.md\n"
            "terminal_state: completed\n"
            "updated_at: 2024-01-15T10:30:00+00:00\n"
            "## Notes\n- thing\n"
        )
        result = self._call(raw)
        # Exactly one line per framework key, carrying the authoritative value.
        for key in ("rollout_id", "rollout_path", "rollout_summary_file"):
            assert self._count_lines_for(result, key) == 1, f"duplicate {key} frontmatter line"
        assert self._value_for(result, "rollout_id") == self._KWARGS["rollout_id"]
        assert self._value_for(result, "rollout_path") == self._KWARGS["rollout_path"]
        assert self._value_for(result, "rollout_summary_file") == self._KWARGS["rollout_summary_file"]
        # The hallucinated values must be entirely gone.
        assert "hallucinated-id" not in result
        assert "/tmp/made-up.jsonl" not in result
        assert "wrong_summary.md" not in result

    def test_llm_supplied_terminal_state_is_overwritten(self) -> None:
        """The framework's terminal_state classification overrides the LLM's guess."""
        raw = "rollout_id: r1\nterminal_state: failed\n## Notes\n- thing\n"
        result = self._call(raw)
        assert self._count_lines_for(result, "terminal_state") == 1
        assert self._value_for(result, "terminal_state") == self._KWARGS["terminal_state"]

    def test_body_content_survives_frontmatter_rewrite(self) -> None:
        """Stripping framework keys must not touch the markdown body."""
        raw = "rollout_id: bogus\n## Notes\n- keep this line\n- and this one\n"
        result = self._call(raw)
        assert "## Notes" in result
        assert "- keep this line" in result
        assert "- and this one" in result


@pytest.mark.asyncio
async def test_flush_keeps_memory_when_llm_hallucinates_rollout_id(fake_session: _FakeSession) -> None:
    """End-to-end: an LLM that invents a rollout_id must not get its memory dropped.

    The file is written under the real rollout_id; consolidation resolves the
    raw_memory path from the frontmatter rollout_id. With the framework value
    forced into the frontmatter, rebuild_raw_memories finds the file and the
    memory reaches consolidation instead of being silently skipped.
    """

    async def fake_extract(_s: str, _u: str) -> str:
        return json.dumps(
            {
                "rollout_slug": "demo-task",
                "rollout_summary": "User asked X; agent did Y.",
                # The LLM hallucinates a rollout_id that does NOT match the real one.
                "raw_memory": "rollout_id: not-the-real-id\nterminal_state: failed\n## Notes\n- item",
            }
        )

    consolidate_calls: list[tuple[str, str]] = []

    async def fake_consolidate(system: str, user: str) -> None:
        consolidate_calls.append((system, user))

    manager = SandboxMemoryManager(
        session=fake_session,  # type: ignore[arg-type]
        layout=MemoryLayoutConfig(),
        extraction_llm=fake_extract,
        consolidation_runner=fake_consolidate,
    )
    payload = build_rollout_payload(rollout_id="real-rollout", inputs="hi", new_items=[], final_output="ok")
    await manager.enqueue_rollout(rollout_id="real-rollout", payload_jsonl=dump_rollout_json(payload))

    result = await manager.flush()

    # The memory survived: consolidation ran and raw_memories.md was rebuilt with the body.
    assert result.consolidation_skipped is False
    assert len(consolidate_calls) == 1
    merged = fake_session._files["memories/raw_memories.md"].decode("utf-8")
    assert "- item" in merged
    # The frontmatter now carries the real id, matching the on-disk filename.
    assert "rollout_id: real-rollout" in merged
    assert "not-the-real-id" not in merged


def test_invalid_rollout_id_rejected() -> None:
    import asyncio as _aio

    async def runner() -> None:
        async def fake_extract(_s: str, _u: str) -> str:
            return "{}"

        async def fake_consolidate(_s: str, _u: str) -> None:
            return None

        manager = SandboxMemoryManager(
            session=_FakeSession(),  # type: ignore[arg-type]
            layout=MemoryLayoutConfig(),
            extraction_llm=fake_extract,
            consolidation_runner=fake_consolidate,
        )
        with pytest.raises(ValueError, match="rollout_id"):
            await manager.enqueue_rollout(rollout_id="bad rollout id!", payload_jsonl="x")

    _aio.run(runner())


class TestCancelledErrorPreservesAllPendingRollouts:
    """Regression tests for the CancelledError re-enqueue bug in _run_extraction_pass."""

    def _make_manager(
        self,
        fake_session: _FakeSession,
        extraction_side_effect: object,
    ) -> SandboxMemoryManager:
        async def fake_consolidate(_s: str, _u: str) -> None:
            return None

        return SandboxMemoryManager(
            session=fake_session,  # type: ignore[arg-type]
            layout=MemoryLayoutConfig(),
            extraction_llm=extraction_side_effect,  # type: ignore[arg-type]
            consolidation_runner=fake_consolidate,
        )

    async def test_cancelled_mid_batch_re_enqueues_all_remaining(self, fake_session: _FakeSession) -> None:
        """All rollouts after the cancelled one must be re-enqueued, not just the current."""
        call_count = 0

        async def fake_extract(system: str, user: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                # Cancel on the second rollout.
                raise asyncio.CancelledError()
            return json.dumps(
                {
                    "rollout_slug": "slug",
                    "rollout_summary": "summary",
                    "raw_memory": "## Notes\n- item",
                }
            )

        manager = self._make_manager(fake_session, fake_extract)
        # Enqueue 4 rollouts sorted as r1, r2, r3, r4.
        for i in range(1, 5):
            payload = build_rollout_payload(rollout_id=f"r{i}", inputs="q", new_items=[], final_output="a")
            await manager.enqueue_rollout(rollout_id=f"r{i}", payload_jsonl=dump_rollout_json(payload))

        with pytest.raises(asyncio.CancelledError):
            await manager.flush(run_consolidation_pass=False)

        # r1 was processed before the cancel; r2, r3, r4 must be re-enqueued.
        # (r2 is the one that raised, r3 and r4 were never visited.)
        pending = set(manager._pending_rollouts.keys())
        assert "r2" in pending, "cancelled rollout must be re-enqueued"
        assert "r3" in pending, "unprocessed rollout r3 must be re-enqueued"
        assert "r4" in pending, "unprocessed rollout r4 must be re-enqueued"

    async def test_cancelled_on_first_rollout_re_enqueues_all(self, fake_session: _FakeSession) -> None:
        """Cancellation on the very first rollout must preserve all of them."""

        async def always_cancel(system: str, user: str) -> str:
            raise asyncio.CancelledError()

        manager = self._make_manager(fake_session, always_cancel)
        for i in range(1, 4):
            payload = build_rollout_payload(rollout_id=f"r{i}", inputs="q", new_items=[], final_output="a")
            await manager.enqueue_rollout(rollout_id=f"r{i}", payload_jsonl=dump_rollout_json(payload))

        with pytest.raises(asyncio.CancelledError):
            await manager.flush(run_consolidation_pass=False)

        pending = set(manager._pending_rollouts.keys())
        assert pending == {"r1", "r2", "r3"}
