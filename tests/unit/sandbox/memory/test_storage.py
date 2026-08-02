"""Unit tests for ``SandboxMemoryStorage``.

Focus: the append path's atomicity under concurrent same-rollout writes.
A naive read-concat-write would interleave at the read/write await points
and clobber a JSONL segment; the per-rollout lock prevents that.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from typing import Any

from troopai.adk.sandbox.capabilities.memory import MemoryLayoutConfig
from troopai.adk.sandbox.memory.storage import SandboxMemoryStorage


class _YieldingSession:
    """In-memory session that yields control inside ``read``/``write``.

    The ``await asyncio.sleep(0)`` inside each primitive forces the event
    loop to hand control to any other ready coroutine — exactly the window
    an unlocked read-modify-write would interleave through.
    """

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
        await asyncio.sleep(0)
        self._files[str(path)] = payload

    async def read(self, path: Path | str, *, user: Any = None) -> io.BytesIO:
        _ = user
        key = str(path)
        await asyncio.sleep(0)
        if key not in self._files:
            raise FileNotFoundError(key)
        return io.BytesIO(self._files[key])


def _storage(session: _YieldingSession) -> SandboxMemoryStorage:
    return SandboxMemoryStorage(
        session=session,  # type: ignore[arg-type]
        layout=MemoryLayoutConfig(),
    )


async def test_concurrent_appends_same_rollout_keep_every_segment() -> None:
    """Two concurrent appends for one rollout_id must not lose a segment.

    Without the per-rollout lock the two coroutines both read the empty
    file, each concatenates its own line, and the second write overwrites
    the first — one line is permanently lost. The lock serializes the
    read-modify-write so both lines survive.
    """
    session = _YieldingSession()
    storage = _storage(session)

    await asyncio.gather(
        storage.append_rollout_segment(rollout_id="r1", payload_jsonl='{"seg": 1}\n'),
        storage.append_rollout_segment(rollout_id="r1", payload_jsonl='{"seg": 2}\n'),
    )

    contents = session._files["sessions/r1.jsonl"].decode("utf-8")
    lines = [line for line in contents.splitlines() if len(line) > 0]
    assert len(lines) == 2, f"expected both segments to survive, got: {contents!r}"
    assert '{"seg": 1}' in contents
    assert '{"seg": 2}' in contents


async def test_many_concurrent_appends_same_rollout_keep_every_segment() -> None:
    """Stress the same race with a larger fan-out for a tighter interleave."""
    session = _YieldingSession()
    storage = _storage(session)

    payloads = [f'{{"seg": {i}}}\n' for i in range(12)]
    await asyncio.gather(*(storage.append_rollout_segment(rollout_id="r1", payload_jsonl=p) for p in payloads))

    contents = session._files["sessions/r1.jsonl"].decode("utf-8")
    lines = [line for line in contents.splitlines() if len(line) > 0]
    assert len(lines) == len(payloads), f"lost a segment: {contents!r}"
    for i in range(12):
        assert f'{{"seg": {i}}}' in contents


async def test_distinct_rollouts_are_not_serialized_against_each_other() -> None:
    """Each rollout writes to its own file; distinct ids stay independent."""
    session = _YieldingSession()
    storage = _storage(session)

    await asyncio.gather(
        storage.append_rollout_segment(rollout_id="r1", payload_jsonl='{"a": 1}\n'),
        storage.append_rollout_segment(rollout_id="r2", payload_jsonl='{"b": 2}\n'),
    )

    assert session._files["sessions/r1.jsonl"].decode("utf-8") == '{"a": 1}\n'
    assert session._files["sessions/r2.jsonl"].decode("utf-8") == '{"b": 2}\n'


async def test_empty_rollout_id_rejected() -> None:
    """A blank rollout_id is a boundary error, not a silent no-op."""
    session = _YieldingSession()
    storage = _storage(session)

    try:
        await storage.append_rollout_segment(rollout_id="   ", payload_jsonl="x\n")
    except ValueError as exc:
        assert "rollout_id" in str(exc)
    else:
        raise AssertionError("expected ValueError for blank rollout_id")
