"""Tests for materialize_manifest — the orchestration entry point.

Covers: declaration-order materialization, ``Dir`` tree flattening,
``Mount`` deferral to ``skipped_mounts`` (not file-materialized),
``only_ephemeral`` filtering, ancestor/descendant overlap still
materializing correctly, the ``UnsupportedManifestEntryError`` loud
guard for an unknown entry type, the concurrency bound, an explicit
``base_dir`` driving an end-to-end ``LocalFile``, and that an
entry-level failure aborts the whole manifest loudly (the
``gather_in_order`` fan-out does NOT swallow it).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest

from troopai.adk.exceptions.exceptions import ExecNonZeroError, UnsupportedManifestEntryError
from troopai.adk.sandbox.session.materialization import materialize_manifest
from troopai.adk.types.sandbox.entries import BaseEntry, Dir, File, LocalFile
from troopai.adk.types.sandbox.manifest import Manifest
from troopai.adk.types.sandbox.mounts import (
    InContainerMountStrategy,
    RcloneMountPattern,
    S3Mount,
)


class _UnknownEntry(BaseEntry):
    """All 5 standard entry types now have materializers; this synthetic
    type has none, so it still exercises the loud exhaustiveness guard."""

    type: Literal["_unknown_test_entry"] = "_unknown_test_entry"


def _recording_session(*, run_exit: int = 0) -> Any:
    class _Rec:
        def __init__(self) -> None:
            self.writes: dict[str, bytes] = {}
            self.mkdirs: list[str] = []

        async def write(self, path: object, data: Any, *, user: object = None) -> None:
            self.writes[str(path)] = data.read()

        async def mkdir(self, path: object, *, parents: bool = False, user: object = None) -> None:
            self.mkdirs.append(str(path))

        async def run(
            self, *command: object, timeout: float | None = None, shell: bool = True, user: object = None
        ) -> Any:
            return SimpleNamespace(exit_code=run_exit, stdout=b"", stderr=b"boom")

    return _Rec()


def _s3_mount() -> S3Mount:
    return S3Mount(
        bucket="data",
        mount_strategy=InContainerMountStrategy(pattern=RcloneMountPattern(remote_name="r")),
    )


class TestMaterializeManifest:
    async def test_file_and_dir_tree_in_declaration_order(self) -> None:
        session = _recording_session()
        manifest = Manifest(
            entries={
                "a.txt": File(content=b"A"),
                "d": Dir(children={"b.txt": File(content=b"B")}),
            },
        )
        result = await materialize_manifest(session, manifest)
        assert [f.path for f in result.files] == ["a.txt", "d", "d/b.txt"]
        assert [f.is_directory for f in result.files] == [False, True, False]
        assert session.writes == {"a.txt": b"A", "d/b.txt": b"B"}
        assert "d" in session.mkdirs
        assert result.skipped_mounts == []

    async def test_mount_is_deferred_not_materialized(self) -> None:
        session = _recording_session()
        manifest = Manifest(entries={"m": _s3_mount()})
        result = await materialize_manifest(session, manifest)
        assert result.skipped_mounts == ["m"]
        assert result.files == []
        assert session.writes == {}
        assert session.mkdirs == []

    async def test_only_ephemeral_filters_durable_entries(self) -> None:
        session = _recording_session()
        manifest = Manifest(
            entries={
                "durable.txt": File(content=b"keep"),
                "scratch.txt": File(content=b"tmp", ephemeral=True),
            },
        )
        result = await materialize_manifest(session, manifest, only_ephemeral=True)
        assert [f.path for f in result.files] == ["scratch.txt"]
        assert session.writes == {"scratch.txt": b"tmp"}

    async def test_overlapping_ancestor_and_descendant_both_materialize(self) -> None:
        session = _recording_session()
        manifest = Manifest(
            entries={
                "a/b.txt": File(content=b"1"),
                "a": Dir(),
            },
        )
        result = await materialize_manifest(session, manifest)
        assert {f.path for f in result.files} == {"a/b.txt", "a"}
        assert session.writes == {"a/b.txt": b"1"}
        assert "a" in session.mkdirs

    async def test_local_file_via_manifest_with_explicit_base_dir(self, tmp_path: Path) -> None:
        # base_dir is an explicit param so LocalFile resolution is
        # pinnable + testable end-to-end (no hidden Path.cwd() coupling).
        (tmp_path / "h.txt").write_bytes(b"host-bytes")
        session = _recording_session()
        manifest = Manifest(entries={"dst/h.txt": LocalFile(src=Path("h.txt"))})
        result = await materialize_manifest(session, manifest, base_dir=tmp_path)
        assert session.writes == {"dst/h.txt": b"host-bytes"}
        assert [f.path for f in result.files] == ["dst/h.txt"]

    async def test_unhandled_entry_type_raises_loudly(self) -> None:
        # All 5 standard entry types now dispatch; a genuinely-unknown
        # type still hits the loud exhaustiveness guard (never a silent
        # skip / missing workspace file).
        session = _recording_session()
        manifest = Manifest(entries={"x": _UnknownEntry()})
        with pytest.raises(UnsupportedManifestEntryError, match="_unknown_test_entry"):
            await materialize_manifest(session, manifest)

    async def test_entry_metadata_failure_aborts_materialize_manifest(self) -> None:
        # An ExecNonZeroError raised by apply_entry_metadata INSIDE the
        # gather_in_order fan-out must propagate OUT of materialize_manifest
        # (fail-fast) — never swallowed by the flush / cancellation drain.
        session = _recording_session(run_exit=1)
        manifest = Manifest(entries={"a.txt": File(content=b"x")})
        with pytest.raises(ExecNonZeroError, match="chmod 644 'a.txt' failed"):
            await materialize_manifest(session, manifest)

    async def test_concurrency_floor_enforced(self) -> None:
        session = _recording_session()
        with pytest.raises(ValueError, match="max_entry_concurrency must be >= 1"):
            await materialize_manifest(session, Manifest(), max_entry_concurrency=0)
