"""Tests for ``troopai.adk.sandbox.session.archive_extraction``.

Builds real in-memory tar / zip archives and runs them through the
validators so the actual escape / bomb / descent logic is
exercised, not mocked.
"""

from __future__ import annotations

import io
import pathlib
import tarfile
import zipfile
from typing import override

import pytest

from troopai.adk.sandbox.session import (
    ArchiveResourceLimitError,
    ArchiveStreamIntegrityError,
    SandboxArchiveLimits,
    UnsafeTarMemberError,
    UnsafeZipMemberError,
    safe_zip_member_rel_path,
    validate_tar_archive_for_extraction,
    validate_zipfile,
    zipfile_compatible_stream,
)
from troopai.adk.sandbox.session.archive_extraction import _zip_member_is_dir


def _tar_bytes(entries: list[tuple[str, bytes | None, str]]) -> bytes:
    """Build a tar; entry kind ∈ {"file","dir"}."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, payload, kind in entries:
            info = tarfile.TarInfo(name=name)
            if kind == "dir":
                info.type = tarfile.DIRTYPE
                tar.addfile(info)
            else:
                assert payload is not None
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def _zip_bytes(entries: list[tuple[str, bytes | None]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        for name, payload in entries:
            if payload is None:
                zf.writestr(zipfile.ZipInfo(name + "/"), b"")
            else:
                zf.writestr(name, payload)
    return buf.getvalue()


class TestTarValidation:
    def test_clean_tar_passes(self) -> None:
        raw = _tar_bytes([("dir/", None, "dir"), ("dir/a.txt", b"hi", "file")])
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
            validate_tar_archive_for_extraction(tar)

    def test_tar_traversal_rejected(self) -> None:
        raw = _tar_bytes([("../escape", b"x", "file")])
        with (
            pytest.raises(UnsafeTarMemberError),
            tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar,
        ):
            validate_tar_archive_for_extraction(tar)

    def test_tar_duplicate_path_rejected(self) -> None:
        raw = _tar_bytes([("a.txt", b"1", "file"), ("a.txt", b"2", "file")])
        with (
            pytest.raises(UnsafeTarMemberError, match="duplicate archive path"),
            tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar,
        ):
            validate_tar_archive_for_extraction(tar)

    def test_tar_descent_through_file_rejected(self) -> None:
        raw = _tar_bytes([("a", b"file-not-dir", "file"), ("a/b.txt", b"x", "file")])
        with (
            pytest.raises(UnsafeTarMemberError, match="descends through non-directory"),
            tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar,
        ):
            validate_tar_archive_for_extraction(tar)

    def test_tar_member_count_limit(self) -> None:
        raw = _tar_bytes([(f"f{i}.txt", b"x", "file") for i in range(5)])
        with (
            pytest.raises(ArchiveResourceLimitError, match="member count exceeds"),
            tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar,
        ):
            validate_tar_archive_for_extraction(tar, archive_limits=SandboxArchiveLimits(max_members=3))

    def test_tar_extracted_bytes_limit(self) -> None:
        raw = _tar_bytes([("big.bin", b"x" * 5000, "file")])
        with (
            pytest.raises(ArchiveResourceLimitError, match="extracted size exceeds"),
            tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar,
        ):
            validate_tar_archive_for_extraction(tar, archive_limits=SandboxArchiveLimits(max_extracted_bytes=1000))

    def test_tar_negative_declared_size_rejected(self) -> None:
        # Invariant: TarInfo.size is read verbatim from an
        # attacker-controlled header; a negative declared size must
        # fail loud. Flooring it to 0 would let a crafted member
        # contribute 0 to the extracted-byte total and defeat the
        # bomb guard, so a negative size is rejected outright.
        raw = _tar_bytes([("bomb.bin", b"x" * 16, "file")])
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
            # getmembers() loads + caches the TarInfo list; the
            # validator's `for member in tar` yields those same cached
            # objects, so mutating in place (no reassignment) is the
            # typed-safe way to forge a hostile declared size.
            tar.getmembers()[0].size = -999_999_999
            with pytest.raises(UnsafeTarMemberError, match="invalid \\(negative\\) declared member size"):
                validate_tar_archive_for_extraction(tar, archive_limits=SandboxArchiveLimits(max_extracted_bytes=1000))


class TestZipValidation:
    def test_clean_zip_passes(self) -> None:
        raw = _zip_bytes([("dir/", None), ("dir/a.txt", b"hi")])
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            validate_zipfile(zf)

    def test_zip_absolute_path_rejected(self) -> None:
        info = zipfile.ZipInfo("/etc/passwd")
        with pytest.raises(UnsafeZipMemberError, match="absolute path"):
            safe_zip_member_rel_path(info)

    def test_zip_parent_traversal_rejected(self) -> None:
        info = zipfile.ZipInfo("a/../../etc")
        with pytest.raises(UnsafeZipMemberError, match="parent traversal"):
            safe_zip_member_rel_path(info)

    def test_zip_windows_separator_rejected(self) -> None:
        info = zipfile.ZipInfo("a\\b")
        with pytest.raises(UnsafeZipMemberError, match="windows"):
            safe_zip_member_rel_path(info)

    @pytest.mark.parametrize(
        ("mode", "label"),
        [
            (0o120000, "symlink"),
            (0o020000, "char device"),
            (0o060000, "block device"),
            (0o010000, "fifo"),
            (0o140000, "socket"),
        ],
    )
    def test_zip_non_file_dir_member_types_rejected(self, mode: int, label: str) -> None:
        # Invariant: the member-type allow-list admits only regular
        # files (0o100000) and directories (0o040000). Every other
        # Unix type — symlink, char/block device, fifo, socket — is
        # rejected, so a device/socket/symlink member can never
        # materialize in the sandbox workspace.
        _ = label
        info = zipfile.ZipInfo("danger")
        info.external_attr = mode << 16
        with pytest.raises(UnsafeZipMemberError, match="non-file/non-directory member type not allowed"):
            safe_zip_member_rel_path(info)

    def test_zip_regular_file_and_dir_modes_pass(self) -> None:
        reg = zipfile.ZipInfo("ok.txt")
        reg.external_attr = 0o100644 << 16
        assert safe_zip_member_rel_path(reg) == pathlib.Path("ok.txt")
        d = zipfile.ZipInfo("d/")
        d.external_attr = 0o040755 << 16
        assert safe_zip_member_rel_path(d) == pathlib.Path("d")

    def test_zip_windows_authored_no_unix_metadata_passes(self) -> None:
        # external_attr == 0 → DOS/Windows-authored zip with no Unix
        # mode bits. Must still pass (name-based checks already ran);
        # rejecting it would break every Windows-created archive.
        info = zipfile.ZipInfo("ok.txt")
        info.external_attr = 0
        assert safe_zip_member_rel_path(info) == pathlib.Path("ok.txt")

    def test_zip_negative_declared_size_rejected(self) -> None:
        # Invariant: a zip member declaring a negative file_size must
        # fail loud — flooring it to 0 would let the member contribute
        # 0 to the extracted-byte total and defeat the bomb guard.
        raw = _zip_bytes([("bomb.bin", b"x" * 16)])
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            zf.infolist()[0].file_size = -1_000_000_000
            with pytest.raises(UnsafeZipMemberError, match="invalid \\(negative\\) declared member size"):
                validate_zipfile(zf, archive_limits=SandboxArchiveLimits(max_extracted_bytes=1000))

    def test_zip_root_member_returns_none(self) -> None:
        # All three canonical root spellings the guard tuple handles;
        # "." is the canonical Posix cwd member archives often include.
        assert safe_zip_member_rel_path(zipfile.ZipInfo("")) is None
        assert safe_zip_member_rel_path(zipfile.ZipInfo(".")) is None
        assert safe_zip_member_rel_path(zipfile.ZipInfo("./")) is None

    def test_zip_extracted_bytes_limit(self) -> None:
        # Pins the zip bomb-guard byte path (parallel to the tar
        # test) with a legitimate large-but-positive size, so a
        # future refactor dropping the zip byte accumulation is
        # caught — the negative-size test raises before the byte
        # check and cannot cover this.
        raw = _zip_bytes([("big.bin", b"x" * 5000)])
        with (
            pytest.raises(ArchiveResourceLimitError, match="extracted size exceeds"),
            zipfile.ZipFile(io.BytesIO(raw)) as zf,
        ):
            validate_zipfile(zf, archive_limits=SandboxArchiveLimits(max_extracted_bytes=1000))

    def test_zip_descent_through_file_rejected(self) -> None:
        raw = _zip_bytes([("a", b"file"), ("a/b.txt", b"x")])
        with (
            pytest.raises(UnsafeZipMemberError, match="descends through non-directory"),
            zipfile.ZipFile(io.BytesIO(raw)) as zf,
        ):
            validate_zipfile(zf)

    def test_zip_member_count_limit(self) -> None:
        raw = _zip_bytes([(f"f{i}.txt", b"x") for i in range(5)])
        with (
            pytest.raises(ArchiveResourceLimitError, match="member count exceeds"),
            zipfile.ZipFile(io.BytesIO(raw)) as zf,
        ):
            validate_zipfile(zf, archive_limits=SandboxArchiveLimits(max_members=2))


class TestArchiveLimitsConfig:
    def test_deny_by_default_bounds_present(self) -> None:
        limits = SandboxArchiveLimits()
        assert limits.max_members == 100_000
        assert limits.max_extracted_bytes == 10 * 1024 * 1024 * 1024

    def test_unbounded_opt_out_is_explicit(self) -> None:
        limits = SandboxArchiveLimits.unbounded()
        assert limits.max_members is None
        assert limits.max_extracted_bytes is None

    def test_frozen(self) -> None:
        limits = SandboxArchiveLimits()
        with pytest.raises((TypeError, ValueError)):
            limits.max_members = 5  # type: ignore[misc]  # frozen model — mutation must raise

    def test_none_archive_limits_applies_defaults_not_unbounded(self) -> None:
        # Passing archive_limits=None must NOT mean "unbounded" — it
        # falls back to the deny-by-default SandboxArchiveLimits().
        raw = _tar_bytes([(f"f{i}.txt", b"x", "file") for i in range(3)])
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
            validate_tar_archive_for_extraction(tar, archive_limits=None)  # under 100k, fine

    def test_resolve_limits_default_catches_bomb_without_explicit_config(self) -> None:
        # A "bomb" relative to a tightened default still trips with no
        # explicit limits passed.
        raw = _tar_bytes([("huge.bin", b"x" * 2000, "file")])
        with (
            pytest.raises(ArchiveResourceLimitError),
            tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar,
        ):
            validate_tar_archive_for_extraction(tar, archive_limits=SandboxArchiveLimits(max_extracted_bytes=500))

    def test_member_count_boundary_is_inclusive(self) -> None:
        # Pinned semantics: exactly max_members is ACCEPTED; one more
        # is rejected. A future >/>= flip must break this test.
        exactly = _zip_bytes([(f"f{i}.txt", b"x") for i in range(3)])
        with zipfile.ZipFile(io.BytesIO(exactly)) as zf:
            validate_zipfile(zf, archive_limits=SandboxArchiveLimits(max_members=3))
        one_over = _zip_bytes([(f"f{i}.txt", b"x") for i in range(4)])
        with (
            pytest.raises(ArchiveResourceLimitError, match="member count exceeds"),
            zipfile.ZipFile(io.BytesIO(one_over)) as zf,
        ):
            validate_zipfile(zf, archive_limits=SandboxArchiveLimits(max_members=3))

    def test_extracted_bytes_boundary_is_inclusive(self) -> None:
        # Exactly max_extracted_bytes is ACCEPTED; one byte over fails.
        exactly = _tar_bytes([("a.bin", b"x" * 100, "file")])
        with tarfile.open(fileobj=io.BytesIO(exactly), mode="r:") as tar:
            validate_tar_archive_for_extraction(tar, archive_limits=SandboxArchiveLimits(max_extracted_bytes=100))
        one_over = _tar_bytes([("a.bin", b"x" * 101, "file")])
        with (
            pytest.raises(ArchiveResourceLimitError, match="extracted size exceeds"),
            tarfile.open(fileobj=io.BytesIO(one_over), mode="r:") as tar,
        ):
            validate_tar_archive_for_extraction(tar, archive_limits=SandboxArchiveLimits(max_extracted_bytes=100))


class TestZipfileCompatibleStream:
    def test_seekable_stream_wrapped_directly(self) -> None:
        raw = _zip_bytes([("a.txt", b"hello")])
        src = io.BytesIO(raw)
        with zipfile_compatible_stream(src) as adapted, zipfile.ZipFile(adapted) as zf:
            assert zf.read("a.txt") == b"hello"

    def test_non_seekable_stream_spooled(self) -> None:
        raw = _zip_bytes([("a.txt", b"spooled")])

        class _NonSeekable(io.IOBase):
            def __init__(self, data: bytes) -> None:
                self._data = data
                self._pos = 0

            @override
            def readable(self) -> bool:
                return True

            @override
            def read(self, size: int = -1) -> bytes:
                if size < 0:
                    chunk = self._data[self._pos :]
                    self._pos = len(self._data)
                    return chunk
                chunk = self._data[self._pos : self._pos + size]
                self._pos += len(chunk)
                return chunk

            @override
            def seekable(self) -> bool:
                return False

            @override
            def tell(self) -> int:
                raise OSError("not seekable")

        # Intent: verify spooling round-trips. The stream is
        # non-seekable with no known length, so opt into the
        # unverifiable path explicitly (deny-by-default otherwise).
        with (
            zipfile_compatible_stream(_NonSeekable(raw), allow_unverified_length=True) as adapted,
            zipfile.ZipFile(adapted) as zf,
        ):
            assert zf.read("a.txt") == b"spooled"

    def test_premature_eof_source_fails_loud(self) -> None:
        # HIGH regression (reproduced by the gate): copyfileobj stops
        # at the first empty read(). A stream that yields data, then a
        # premature b"", then MORE data leaves the spool holding only a
        # prefix. The post-copy probe-read must detect the pending
        # bytes and raise rather than validate a truncated archive.
        full = _zip_bytes([("ok.txt", b"safe"), ("BOMB.bin", b"x" * 4096)])

        class _TruncatingStream(io.IOBase):
            def __init__(self, data: bytes, cut: int) -> None:
                self._head = data[:cut]
                self._tail = data[cut:]
                self._phase = 0

            @override
            def readable(self) -> bool:
                return True

            @override
            def seekable(self) -> bool:
                return False

            @override
            def tell(self) -> int:
                raise OSError("not seekable")

            @override
            def read(self, size: int = -1) -> bytes:
                # phase 0: emit head; phase 1: premature EOF (b"");
                # phase 2+: the bytes copyfileobj never copied.
                self._phase += 1
                if self._phase == 1:
                    return self._head
                if self._phase == 2:
                    return b""
                return self._tail

        # allow_unverified_length=True so the deny-by-default refusal
        # is bypassed and execution reaches the pending-tail probe —
        # the behaviour this test pins. match= is the pending-tail
        # message specifically, so it cannot pass on the
        # deny-by-default message by accident.
        with (
            pytest.raises(ArchiveStreamIntegrityError, match="data pending after copy terminated"),
            zipfile_compatible_stream(_TruncatingStream(full, cut=64), allow_unverified_length=True) as adapted,
        ):
            zipfile.ZipFile(adapted)

    def test_clean_boundary_truncation_deny_by_default_opt_out_and_oracle(self) -> None:
        # The reproduced bomb-guard bypass: a non-seekable stream
        # delivers a COMPLETE small zip prefix, then a *permanent*
        # b"" with no buffered tail (TCP half-close mid-stream).
        # copyfileobj stops and the probe read(1) ALSO returns b"" —
        # indistinguishable from a real EOF from the stream alone.
        prefix = _zip_bytes([("ok.txt", b"safe")])
        bomb = _zip_bytes([("BOMB.bin", b"x" * 4096)])
        full_len = len(prefix) + len(bomb)

        class _CleanTruncatingStream(io.IOBase):
            def __init__(self, payload: bytes) -> None:
                self._payload = payload
                self._done = False

            @override
            def readable(self) -> bool:
                return True

            @override
            def seekable(self) -> bool:
                return False

            @override
            def tell(self) -> int:
                raise OSError("not seekable")

            @override
            def read(self, size: int = -1) -> bytes:
                if self._done:
                    return b""  # permanent EOF, no buffered tail
                self._done = True
                return self._payload

        # Branch A — DENY-BY-DEFAULT: neither expected_size nor the
        # opt-out. A non-seekable stream of unknown length has no
        # truncation oracle, so it is REFUSED outright (not silently
        # spooled). This is the posture-hardening: the unsafe path
        # is never the default.
        with (
            pytest.raises(ArchiveStreamIntegrityError, match="non-seekable stream of unknown length"),
            zipfile_compatible_stream(_CleanTruncatingStream(prefix)) as adapted,
        ):
            zipfile.ZipFile(adapted)

        # Branch B — EXPLICIT OPT-OUT: allow_unverified_length=True.
        # The caller consciously accepts an unverifiable payload, so
        # the truncated prefix is spooled and validates. This pins
        # the documented residual gap as a greppable, explicit
        # decision — NOT a silent default.
        with (
            zipfile_compatible_stream(_CleanTruncatingStream(prefix), allow_unverified_length=True) as adapted,
            zipfile.ZipFile(adapted) as zf,
        ):
            assert zf.namelist() == ["ok.txt"]  # bomb absent; caller opted in

        # Branch C — ORACLE: expected_size supplied. The spooled byte
        # count (prefix only) != declared total → the reproduced
        # bomb-guard bypass is closed.
        with (
            pytest.raises(ArchiveStreamIntegrityError, match="truncated in transit"),
            zipfile_compatible_stream(_CleanTruncatingStream(prefix), expected_size=full_len) as adapted,
        ):
            zipfile.ZipFile(adapted)

    def test_malformed_tell_propagates_not_silently_spooled(self) -> None:
        # HIGH/MED converging finding: a stream advertising seekable()
        # but whose tell() raises AttributeError (a real defect, e.g. a
        # typo'd attribute) must propagate — NOT be swallowed and
        # silently routed to the slow spool path where the bug hides.
        class _BrokenSeekable(io.IOBase):
            @override
            def readable(self) -> bool:
                return True

            @override
            def seekable(self) -> bool:
                return True

            @override
            def tell(self) -> int:
                raise AttributeError("typo: self._psotion")

        with (
            pytest.raises(AttributeError, match="psotion"),
            zipfile_compatible_stream(_BrokenSeekable()) as adapted,
        ):
            _ = adapted

    def test_adapter_rejects_non_bytes_read(self) -> None:
        class _StrStream(io.IOBase):
            @override
            def read(self, size: int = -1) -> str:  # type: ignore[override]  # deliberately wrong return type to drive the fail-loud guard
                _ = size
                return "not bytes"

            @override
            def seekable(self) -> bool:
                return True

            @override
            def tell(self) -> int:
                return 0

            @override
            def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
                _ = offset, whence
                return 0

        with (
            zipfile_compatible_stream(_StrStream()) as adapted,
            pytest.raises(TypeError, match="expected bytes"),
        ):
            adapted.read(4)


class TestZipMemberIsDirHelper:
    """Regression tests for _zip_member_is_dir external_attr / trailing-slash unification."""

    def _make_member(self, filename: str, *, unix_mode: int | None = None) -> zipfile.ZipInfo:
        info = zipfile.ZipInfo(filename)
        if unix_mode is not None:
            info.external_attr = unix_mode << 16
        return info

    def test_trailing_slash_no_attr_is_dir(self) -> None:
        """No external_attr + trailing slash → directory (is_dir() fallback)."""
        member = self._make_member("somedir/")
        assert _zip_member_is_dir(member) is True

    def test_no_trailing_slash_no_attr_is_file(self) -> None:
        """No external_attr, no trailing slash → regular file."""
        member = self._make_member("file.txt")
        assert _zip_member_is_dir(member) is False

    def test_unix_dir_mode_no_trailing_slash_is_dir(self) -> None:
        """Unix dir mode (0o040755) without trailing slash must still be recognised as dir."""
        member = self._make_member("somedir", unix_mode=0o040755)
        assert _zip_member_is_dir(member) is True

    def test_unix_file_mode_is_not_dir(self) -> None:
        """Unix regular-file mode (0o100644) must not be classified as dir."""
        member = self._make_member("file.txt", unix_mode=0o100644)
        assert _zip_member_is_dir(member) is False

    def test_unix_dir_mode_with_trailing_slash_is_dir(self) -> None:
        member = self._make_member("d/", unix_mode=0o040755)
        assert _zip_member_is_dir(member) is True


class TestZipValidateDirWithoutTrailingSlash:
    """Regression: validate_zipfile must not false-positive on Unix-dir-mode entries without trailing slash."""

    def _make_zip_with_unix_dir_no_slash(self) -> zipfile.ZipFile:
        """Build a zip whose directory entry carries Unix mode but no trailing slash."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            # Directory entry: Unix dir mode, no trailing slash.
            dir_info = zipfile.ZipInfo("somedir")
            dir_info.external_attr = 0o040755 << 16
            zf.writestr(dir_info, b"")
            # Child file inside that directory.
            zf.writestr("somedir/file.txt", b"hello")
        buf.seek(0)
        return zipfile.ZipFile(buf)

    def test_unix_dir_without_trailing_slash_passes_validation(self) -> None:
        """A legitimate archive with Unix dir-mode entry (no trailing slash) must not raise."""
        archive = self._make_zip_with_unix_dir_no_slash()
        # Should NOT raise UnsafeZipMemberError — this is a valid archive.
        validate_zipfile(archive)
