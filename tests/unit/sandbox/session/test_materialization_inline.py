"""Tests for materialization.inline — File + Dir materializers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from troopai.adk.sandbox.session.materialization.inline import (
    materialize_dir,
    materialize_file,
)
from troopai.adk.types.sandbox.entries import Dir, File


def _recording_session() -> Any:
    class _Rec:
        def __init__(self) -> None:
            self.writes: dict[str, bytes] = {}
            self.mkdirs: list[tuple[str, bool]] = []
            self.runs: list[tuple[str, ...]] = []

        async def write(self, path: object, data: Any, *, user: object = None) -> None:
            self.writes[str(path)] = data.read()

        async def mkdir(self, path: object, *, parents: bool = False, user: object = None) -> None:
            self.mkdirs.append((str(path), parents))

        async def run(
            self, *command: object, timeout: float | None = None, shell: bool = True, user: object = None
        ) -> Any:
            self.runs.append(tuple(str(c) for c in command))
            return SimpleNamespace(exit_code=0, stdout=b"", stderr=b"")

    return _Rec()


class TestMaterializeFile:
    async def test_writes_content_and_returns_record(self) -> None:
        session = _recording_session()
        result = await materialize_file(session, "pkg/mod.py", File(content=b"print(1)\n"))
        assert session.writes == {"pkg/mod.py": b"print(1)\n"}
        assert result.path == "pkg/mod.py"
        assert result.size_bytes == len(b"print(1)\n")
        assert result.is_directory is False
        # chmod applied after the write.
        assert session.runs == [("chmod", "644", "pkg/mod.py")]

    async def test_normalizes_key(self) -> None:
        session = _recording_session()
        result = await materialize_file(session, "a/./b.txt", File(content=b"x"))
        assert "a/b.txt" in session.writes
        assert result.path == "a/b.txt"


class TestMaterializeDir:
    async def test_mkdirs_parents_and_returns_directory_record(self) -> None:
        session = _recording_session()
        result = await materialize_dir(session, "out/logs", Dir())
        assert session.mkdirs == [("out/logs", True)]
        assert result.path == "out/logs"
        assert result.size_bytes == 0
        assert result.is_directory is True
        # Dir entry: 0o644 read bits → traverse-augmented to 0o755
        # (a non-traversable directory cannot hold its children).
        assert session.runs == [("chmod", "755", "out/logs")]
