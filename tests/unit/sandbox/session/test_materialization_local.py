"""Tests for materialization.local — LocalFile/LocalDir host read + defense.

Uses real ``tmp_path`` files, directories, and symlinks (hermetic)
so the filesystem-security boundary is exercised against actual
inodes, not mocks.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from troopai.adk.exceptions.exceptions import LocalArtifactError
from troopai.adk.sandbox.session.materialization.local import (
    materialize_local_dir,
    materialize_local_file,
    resolve_host_source,
)
from troopai.adk.types.sandbox.entries import LocalDir, LocalFile
from troopai.adk.types.sandbox.workspace_paths import SandboxPathGrant


def _recording_session() -> Any:
    class _Rec:
        def __init__(self) -> None:
            self.writes: dict[str, bytes] = {}
            self.mkdirs: list[str] = []
            self.runs: list[tuple[str, ...]] = []

        async def write(self, path: object, data: Any, *, user: object = None) -> None:
            self.writes[str(path)] = data.read()

        async def mkdir(self, path: object, *, parents: bool = False, user: object = None) -> None:
            self.mkdirs.append(str(path))

        async def run(
            self, *command: object, timeout: float | None = None, shell: bool = True, user: object = None
        ) -> Any:
            self.runs.append(tuple(str(c) for c in command))
            return SimpleNamespace(exit_code=0, stdout=b"", stderr=b"")

    return _Rec()


class TestResolveHostSource:
    def test_relative_src_under_base(self, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("x")
        out = resolve_host_source(Path("a.txt"), base_dir=tmp_path, grants=[])
        assert out == Path(os.path.abspath(tmp_path / "a.txt"))

    def test_absolute_src_under_base(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("x")
        assert resolve_host_source(f, base_dir=tmp_path, grants=[]) == Path(os.path.abspath(f))

    def test_outside_base_without_grant_rejected(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("x")
        with pytest.raises(LocalArtifactError, match="outside the host"):
            resolve_host_source(outside, base_dir=base, grants=[])

    def test_outside_base_with_grant_allowed(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        base.mkdir()
        granted = tmp_path / "granted"
        granted.mkdir()
        f = granted / "a.txt"
        f.write_text("x")
        out = resolve_host_source(f, base_dir=base, grants=[SandboxPathGrant(path=str(granted))])
        assert out == Path(os.path.abspath(f))

    def test_symlinked_parent_component_rejected(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        base.mkdir()
        real = tmp_path / "real"
        real.mkdir()
        (real / "secret.txt").write_text("s")
        os.symlink(real, base / "link")
        with pytest.raises(LocalArtifactError, match="symlink"):
            resolve_host_source(base / "link" / "secret.txt", base_dir=base, grants=[])

    def test_symlinked_component_inside_grant_rejected(self, tmp_path: Path) -> None:
        # A grant WIDENS the allowed root; it does NOT waive the
        # per-component symlink defense inside the granted tree.
        granted = tmp_path / "granted"
        granted.mkdir()
        real = tmp_path / "real"
        real.mkdir()
        (real / "s.txt").write_text("s")
        os.symlink(real, granted / "link")
        with pytest.raises(LocalArtifactError, match="symlink"):
            resolve_host_source(
                granted / "link" / "s.txt",
                base_dir=tmp_path / "base",
                grants=[SandboxPathGrant(path=str(granted))],
            )


class TestMaterializeLocalFile:
    async def test_copies_file_and_returns_record(self, tmp_path: Path) -> None:
        (tmp_path / "src.txt").write_bytes(b"hello-host")
        session = _recording_session()
        result = await materialize_local_file(
            session, "dst/src.txt", LocalFile(src=Path("src.txt")), base_dir=tmp_path, grants=[]
        )
        assert session.writes == {"dst/src.txt": b"hello-host"}
        assert result.path == "dst/src.txt"
        assert result.size_bytes == len(b"hello-host")
        assert result.is_directory is False
        assert session.runs == [("chmod", "644", "dst/src.txt")]

    async def test_missing_source_raises(self, tmp_path: Path) -> None:
        session = _recording_session()
        with pytest.raises(LocalArtifactError, match="could not be opened"):
            await materialize_local_file(
                session, "dst.txt", LocalFile(src=Path("nope.txt")), base_dir=tmp_path, grants=[]
            )

    async def test_symlinked_source_file_rejected(self, tmp_path: Path) -> None:
        # Exercises the deterministic pre-open component check (the
        # check->open TOCTOU window O_NOFOLLOW hardens on POSIX is
        # inherently racy and not deterministically testable).
        (tmp_path / "real.txt").write_bytes(b"r")
        os.symlink(tmp_path / "real.txt", tmp_path / "link.txt")
        session = _recording_session()
        with pytest.raises(LocalArtifactError, match="symlink"):
            await materialize_local_file(
                session, "dst.txt", LocalFile(src=Path("link.txt")), base_dir=tmp_path, grants=[]
            )


class TestMaterializeLocalDir:
    async def test_src_none_creates_empty_dir(self, tmp_path: Path) -> None:
        session = _recording_session()
        result = await materialize_local_dir(session, "out", LocalDir(), base_dir=tmp_path, grants=[])
        assert session.mkdirs == ["out"]
        assert session.writes == {}
        assert result.is_directory is True
        assert result.path == "out"
        # LocalDir entry: 0o644 read bits → traverse-augmented to 0o755.
        assert session.runs == [("chmod", "755", "out")]

    async def test_recursive_tree_copied(self, tmp_path: Path) -> None:
        src = tmp_path / "tree"
        (src / "sub").mkdir(parents=True)
        (src / "a.txt").write_bytes(b"A")
        (src / "sub" / "b.txt").write_bytes(b"B")
        session = _recording_session()
        result = await materialize_local_dir(session, "dst", LocalDir(src=Path("tree")), base_dir=tmp_path, grants=[])
        assert session.writes == {"dst/a.txt": b"A", "dst/sub/b.txt": b"B"}
        assert "dst" in session.mkdirs
        assert "dst/sub" in session.mkdirs
        assert session.mkdirs.index("dst") < session.mkdirs.index("dst/sub")
        assert result.is_directory is True

    async def test_follow_symlinks_false_skips_symlink(self, tmp_path: Path) -> None:
        src = tmp_path / "tree"
        src.mkdir()
        (src / "real.txt").write_bytes(b"R")
        os.symlink(src / "real.txt", src / "link.txt")
        session = _recording_session()
        await materialize_local_dir(
            session,
            "dst",
            LocalDir(src=Path("tree"), follow_symlinks=False),
            base_dir=tmp_path,
            grants=[],
        )
        assert session.writes == {"dst/real.txt": b"R"}
        assert "dst/link.txt" not in session.writes

    async def test_follow_symlinks_true_copies_contained_file_symlink(self, tmp_path: Path) -> None:
        src = tmp_path / "tree"
        src.mkdir()
        (src / "real.txt").write_bytes(b"R")
        os.symlink(src / "real.txt", src / "link.txt")
        session = _recording_session()
        await materialize_local_dir(
            session,
            "dst",
            LocalDir(src=Path("tree"), follow_symlinks=True),
            base_dir=tmp_path,
            grants=[],
        )
        assert session.writes == {"dst/real.txt": b"R", "dst/link.txt": b"R"}

    async def test_follow_symlinks_true_rejects_symlinked_directory(self, tmp_path: Path) -> None:
        src = tmp_path / "tree"
        src.mkdir()
        (src / "real.txt").write_bytes(b"R")
        realdir = tmp_path / "realdir"
        realdir.mkdir()
        os.symlink(realdir, src / "linkdir")
        session = _recording_session()
        with pytest.raises(LocalArtifactError, match="symlinked directory"):
            await materialize_local_dir(
                session,
                "dst",
                LocalDir(src=Path("tree"), follow_symlinks=True),
                base_dir=tmp_path,
                grants=[],
            )

    async def test_follow_symlinks_true_rejects_escaping_target(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        src = base / "tree"
        src.mkdir(parents=True)
        outside = tmp_path / "outside.txt"
        outside.write_bytes(b"SECRET")
        os.symlink(outside, src / "leak.txt")
        session = _recording_session()
        with pytest.raises(LocalArtifactError, match="escapes every allowed root"):
            await materialize_local_dir(
                session,
                "dst",
                LocalDir(src=Path("tree"), follow_symlinks=True),
                base_dir=base,
                grants=[],
            )

    async def test_src_outside_base_rejected(self, tmp_path: Path) -> None:
        base = tmp_path / "base"
        base.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        session = _recording_session()
        with pytest.raises(LocalArtifactError, match="outside the host"):
            await materialize_local_dir(session, "dst", LocalDir(src=other), base_dir=base, grants=[])
