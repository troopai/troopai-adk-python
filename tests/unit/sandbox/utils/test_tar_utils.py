"""Tests for ``troopai.adk.sandbox.utils.tar_utils``."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from troopai.adk.sandbox.utils.tar_utils import (
    UnsafeTarMemberError,
    safe_extract_tarfile,
    safe_tar_member_rel_path,
    should_skip_tar_member,
    validate_tar_bytes,
    validate_tarfile,
)


def _make_tar(members: list[tuple[str, bytes | None, str]]) -> bytes:
    """Build a tar with (name, payload, kind) entries; kind ∈ {"file","dir","sym"}."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, payload, kind in members:
            info = tarfile.TarInfo(name=name)
            if kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            elif kind == "sym":
                info.type = tarfile.SYMTYPE
                assert payload is not None
                info.linkname = payload.decode("utf-8")
                tar.addfile(info)
            else:
                assert payload is not None
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def test_safe_tar_member_rel_path_accepts_normal_file() -> None:
    info = tarfile.TarInfo(name="a/b.txt")
    info.size = 0
    rel = safe_tar_member_rel_path(info)
    assert rel == Path("a/b.txt")


def test_safe_tar_member_rejects_absolute() -> None:
    info = tarfile.TarInfo(name="/etc/passwd")
    info.size = 0
    with pytest.raises(UnsafeTarMemberError, match="absolute path"):
        safe_tar_member_rel_path(info)


def test_safe_tar_member_rejects_parent_traversal() -> None:
    info = tarfile.TarInfo(name="a/../../etc")
    info.size = 0
    with pytest.raises(UnsafeTarMemberError, match="parent traversal"):
        safe_tar_member_rel_path(info)


def test_safe_tar_member_rejects_hardlink() -> None:
    info = tarfile.TarInfo(name="link")
    info.type = tarfile.LNKTYPE
    info.linkname = "target"
    with pytest.raises(UnsafeTarMemberError, match="hardlink"):
        safe_tar_member_rel_path(info)


def test_validate_tar_bytes_accepts_clean() -> None:
    raw = _make_tar(
        [
            ("dir/", None, "dir"),
            ("dir/file.txt", b"hello", "file"),
        ]
    )
    validate_tar_bytes(raw)


def test_validate_tar_bytes_rejects_traversal() -> None:
    raw = _make_tar(
        [
            ("../../etc/passwd", b"x", "file"),
        ]
    )
    with pytest.raises(UnsafeTarMemberError):
        validate_tar_bytes(raw)


def test_validate_tar_bytes_invalid_tar() -> None:
    with pytest.raises(UnsafeTarMemberError, match="invalid tar stream"):
        validate_tar_bytes(b"not a tar")


def test_should_skip_tar_member() -> None:
    assert should_skip_tar_member(
        ".sandbox/fingerprint.json",
        skip_rel_paths=[".sandbox"],
        root_name=None,
    )
    assert not should_skip_tar_member(
        "src/file.py",
        skip_rel_paths=[".sandbox"],
        root_name=None,
    )


def test_should_skip_strips_root_name() -> None:
    assert should_skip_tar_member(
        "workspace/.sandbox/x",
        skip_rel_paths=[".sandbox"],
        root_name="workspace",
    )


@pytest.mark.parametrize("degenerate", ["", ".", "./"])
def test_should_skip_ignores_empty_prefix(degenerate: str) -> None:
    # A skip prefix that normalizes to the empty path must match NOTHING,
    # not everything — otherwise it would silently exclude the whole archive.
    assert not should_skip_tar_member(
        "src/file.py",
        skip_rel_paths=[degenerate],
        root_name=None,
    )
    # A real prefix alongside a degenerate one still works.
    assert should_skip_tar_member(
        ".sandbox/x",
        skip_rel_paths=[degenerate, ".sandbox"],
        root_name=None,
    )


@pytest.mark.parametrize("degenerate", ["", ".", "./"])
def test_validate_tarfile_empty_skip_prefix_does_not_bypass_policy(degenerate: str) -> None:
    # An empty/'.' skip entry must not turn the security gate into a no-op:
    # an unsafe member still has to be rejected.
    raw = _make_tar([("nested/link", b"../../../outside", "sym")])
    with (
        pytest.raises(UnsafeTarMemberError, match="symlink target escapes archive root"),
        tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar,
    ):
        validate_tarfile(tar, skip_rel_paths=[degenerate])


def test_safe_extract_tarfile_writes_files(tmp_path: Path) -> None:
    raw = _make_tar(
        [
            ("dir/", None, "dir"),
            ("dir/hello.txt", b"hi", "file"),
        ]
    )
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
        safe_extract_tarfile(tar, root=tmp_path)
    assert (tmp_path / "dir" / "hello.txt").read_bytes() == b"hi"


def test_safe_extract_tarfile_writes_symlinks_last(tmp_path: Path) -> None:
    raw = _make_tar(
        [
            ("real.txt", b"hi", "file"),
            ("link.txt", b"real.txt", "sym"),
        ]
    )
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
        safe_extract_tarfile(tar, root=tmp_path)
    link = tmp_path / "link.txt"
    assert link.is_symlink()
    assert link.read_bytes() == b"hi"


def test_validate_tarfile_rejects_path_under_symlink() -> None:
    raw = _make_tar(
        [
            ("link", b"target", "sym"),
            ("link/child", b"x", "file"),
        ]
    )
    with (
        pytest.raises(UnsafeTarMemberError, match="descends through symlink"),
        tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar,
    ):
        validate_tarfile(tar)


def test_validate_tarfile_rejects_absolute_symlink_target_by_default() -> None:
    """The default policy (allow_external_symlink_targets=False) MUST reject /-rooted targets."""
    raw = _make_tar([("link", b"/etc/passwd", "sym")])
    with (
        pytest.raises(UnsafeTarMemberError, match="absolute symlink target"),
        tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar,
    ):
        validate_tarfile(tar)


def test_validate_tarfile_rejects_escaping_relative_symlink_target_by_default() -> None:
    """Relative targets that resolve outside the archive root are rejected by default."""
    raw = _make_tar([("nested/link", b"../../../outside", "sym")])
    with (
        pytest.raises(UnsafeTarMemberError, match="escapes archive root"),
        tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar,
    ):
        validate_tarfile(tar)


def test_validate_tarfile_accepts_inbound_relative_symlink_when_strict() -> None:
    """A relative symlink that stays inside the archive root passes the strict policy."""
    raw = _make_tar(
        [
            ("real.txt", b"hi", "file"),
            ("link.txt", b"real.txt", "sym"),
        ]
    )
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
        validate_tarfile(tar)  # no raise


def test_validate_tarfile_allows_external_target_when_opted_in() -> None:
    """Opt-in escape hatch: allow_external_symlink_targets=True permits /-rooted targets."""
    raw = _make_tar([("link", b"/etc/passwd", "sym")])
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
        validate_tarfile(tar, allow_external_symlink_targets=True)  # no raise


def test_validate_tar_bytes_default_rejects_absolute_symlink_target() -> None:
    raw = _make_tar([("link", b"/tmp/escape", "sym")])
    with pytest.raises(UnsafeTarMemberError, match="absolute symlink target"):
        validate_tar_bytes(raw)
