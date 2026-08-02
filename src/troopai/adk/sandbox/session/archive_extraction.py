"""Archive-safety validation for workspace extraction (tar + zip).

Before any archive payload is written into a sandbox workspace it
is validated for three classes of attack:

* **Path traversal / escape** — members named ``../../etc/passwd``,
  absolute paths, Windows drive paths, or symlink/link members.
* **Descent through a non-directory** — a member ``a/b`` where an
  earlier member ``a`` is a regular file (would clobber/escape).
* **Archive bombs** — a small archive that expands to an enormous
  member count or extracted byte total (decompression bomb).

The tar path reuses the symlink-chaining-safe primitives from
``troopai.adk.sandbox.utils.tar_utils`` (shipped earlier); this
module adds the zip equivalents plus the resource-limit checks.

``SandboxArchiveLimits`` deliberately ships **deny-by-default**
bounds rather than the unbounded ``None`` of the upstream
reference: an archive-bomb guard that is off until you opt in is
the wrong default for a security control. Normal archives sit far
under the defaults; pass an explicit ``SandboxArchiveLimits`` (or
``SandboxArchiveLimits.unbounded()``) to widen them.
"""

from __future__ import annotations

import io
import shutil
import tarfile
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import ClassVar, cast, override

from pydantic import BaseModel, ConfigDict

from troopai.adk.sandbox.utils.tar_utils import (
    UnsafeTarMemberError,
    safe_tar_member_rel_path,
)

__all__ = [
    "ArchiveResourceLimitError",
    "ArchiveStreamIntegrityError",
    "SandboxArchiveLimits",
    # Re-exported from tar_utils: validate_tar_archive_for_extraction
    # raises it, so a caller catching archive failures finds every
    # archive error type on this module rather than two packages.
    "UnsafeTarMemberError",
    "UnsafeZipMemberError",
    "safe_zip_member_rel_path",
    "validate_tar_archive_for_extraction",
    "validate_zipfile",
    "zipfile_compatible_stream",
]

# Deny-by-default bounds. 100k members + 10 GiB extracted is far
# above any legitimate workspace seed yet decisively catches a
# zip/tar bomb. Tuned to be a non-event for real archives.
_DEFAULT_MAX_MEMBERS: int = 100_000
_DEFAULT_MAX_EXTRACTED_BYTES: int = 10 * 1024 * 1024 * 1024


class SandboxArchiveLimits(BaseModel):
    """Resource bounds applied while validating an archive for extraction.

    Boundary semantics are **inclusive**: an archive whose member
    count is exactly ``max_members`` (or whose summed extracted size
    is exactly ``max_extracted_bytes``) is ACCEPTED; rejection
    triggers only when the observed value strictly exceeds the cap.
    The defaults sit orders of magnitude above any real workspace
    seed, so the off-by-one at the boundary is immaterial in
    practice and the semantics are pinned here and by regression
    tests so a future tweak cannot silently flip them.

    Attributes:
        max_members: Inclusive hard cap on the number of archive
            members (exactly this many is allowed; one more is
            rejected). ``None`` disables the member-count check.
        max_extracted_bytes: Inclusive hard cap on the summed
            uncompressed size of all members. ``None`` disables the
            size check.

    Frozen: the limits are read during validation and must not be
    mutated mid-flight. Use ``model_copy(update=...)`` to derive a
    variant.
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    max_members: int | None = _DEFAULT_MAX_MEMBERS
    """Inclusive cap on archive member count (``None`` = unbounded)."""

    max_extracted_bytes: int | None = _DEFAULT_MAX_EXTRACTED_BYTES
    """Inclusive cap on total uncompressed bytes (``None`` = unbounded)."""

    @classmethod
    def unbounded(cls) -> SandboxArchiveLimits:
        """Explicit opt-out: no member-count or extracted-size limit.

        Naming the escape hatch (rather than defaulting to it)
        keeps the safe behaviour the default and the unsafe choice
        a visible, greppable decision.
        """
        return cls(max_members=None, max_extracted_bytes=None)


class UnsafeZipMemberError(ValueError):
    """Raised when a zip member would escape or violate extraction rules.

    Attributes:
        member: The offending member filename.
        reason: Human-readable explanation.
    """

    def __init__(self, *, member: str, reason: str) -> None:
        super().__init__(f"unsafe zip member {member!r}: {reason}")
        self.member = member
        self.reason = reason


class ArchiveResourceLimitError(ValueError):
    """Raised when an archive exceeds an extraction resource limit.

    Attributes:
        reason: Which limit tripped.
        limit: The configured ceiling.
        actual: The observed value that exceeded it.
        member: The member at which the limit tripped, if known.
    """

    def __init__(
        self,
        *,
        reason: str,
        limit: int,
        actual: int,
        member: str | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.limit = limit
        self.actual = actual
        self.member = member


class ArchiveStreamIntegrityError(Exception):
    """Raised when a spooled source stream is provably truncated.

    ``shutil.copyfileobj`` treats the first empty ``read()`` as
    end-of-stream. A dropped connection / partial flush / hostile
    short-read therefore yields a *prefix* of the archive that may
    itself parse as a small, under-limit zip — the resource-limit
    guard would then bless a truncated payload (the bomb members
    were simply never copied).

    Two truncation shapes exist, with different detectability:

    * **Pending-tail truncation** — the source stalls (premature
      ``b""``) but still has bytes buffered. The post-copy probe
      read detects the leftover and this error is raised.
    * **Clean-boundary truncation** — the source delivers a prefix
      then a *permanent* ``b""`` (e.g. a TCP half-close mid-stream
      with no buffered tail). This is information-theoretically
      indistinguishable from a genuine EOF *from the stream alone*;
      it is detectable ONLY when the caller supplies an
      out-of-band ``expected_size`` (HTTP ``Content-Length``,
      archive header total, ``stat`` size), against which the
      spooled byte count is asserted. Because the oracle is
      caller-supplied, a non-seekable stream with neither
      ``expected_size`` nor an explicit ``allow_unverified_length``
      opt-out is REFUSED with this error (deny-by-default) rather
      than spooled unverifiably.

    Deliberately NOT a ``ValueError``: the ``Unsafe*MemberError`` /
    ``ArchiveResourceLimitError`` family is content-policy
    rejection, whereas this signals *transport corruption* ("do
    not trust this payload"). Keeping it outside the ``ValueError``
    hierarchy stops a caller's ``except ValueError`` from silently
    conflating a corrupted-in-transit payload with a benign
    per-member shape rejection and retrying / continuing.
    """


def _supports_zip_random_access(stream: io.IOBase) -> bool:
    # A non-seekable stream is the legitimate "use the spool path"
    # signal: io.IOBase guarantees seekable(), and a stream that
    # advertises no seek support raises OSError /
    # io.UnsupportedOperation (an OSError subclass) on seek().
    # AttributeError / TypeError / ValueError out of tell()/seek()
    # mean a malformed stream object (missing/typo'd method, wrong
    # signature) — a genuine defect that MUST surface, NOT be
    # silently downgraded to the slower spool path where it would
    # hide forever (silent-failure: masking a real bug behind a
    # "works but slow" fallback).
    if not stream.seekable():
        return False
    try:
        position = stream.tell()
        stream.seek(position, io.SEEK_SET)
    except OSError:
        return False
    return True


class _ZipFileStreamAdapter(io.IOBase):
    """Random-access wrapper so ``zipfile`` accepts our stream.

    ``zipfile`` reads ``file.seekable`` directly on some CPython
    versions; this adapter guarantees the seekable/readable shape
    and fails loud (``TypeError``) if the wrapped stream yields a
    non-``bytes`` chunk rather than silently corrupting the read.
    """

    def __init__(self, stream: io.IOBase) -> None:
        self._stream = stream

    @override
    def seekable(self) -> bool:
        return True

    @override
    def readable(self) -> bool:
        return True

    @override
    def tell(self) -> int:
        return int(self._stream.tell())

    @override
    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return int(self._stream.seek(offset, whence))

    @override
    def read(self, size: int = -1) -> bytes:
        data = self._stream.read(size)
        if isinstance(data, bytes):
            return data
        raise TypeError(f"expected bytes from wrapped stream, got {type(data).__name__}")

    @override
    def close(self) -> None:
        # Intentional no-op: the caller owns the wrapped stream's
        # lifetime. For the spool path the SpooledTemporaryFile is
        # closed by its own `with` block in zipfile_compatible_stream;
        # for the direct-wrap path the original stream belongs to the
        # caller. Closing here would double-close the spool or yank
        # the caller's stream out from under it.
        return


@contextmanager
def zipfile_compatible_stream(
    stream: io.IOBase,
    *,
    expected_size: int | None = None,
    allow_unverified_length: bool = False,
) -> Iterator[io.IOBase]:
    """Yield a seekable, ``zipfile``-compatible view of ``stream``.

    Random-access streams are wrapped directly; non-seekable
    streams are spooled to a 16 MiB-threshold temp file (memory
    for small archives, disk for large) before wrapping.

    Truncation handling (spool path only). Two shapes exist:

    * **Pending-tail truncation** — a stalled source (premature
      ``b""`` with bytes still buffered). The post-copy probe read
      detects the leftover and raises ``ArchiveStreamIntegrityError``.
      Always detected, no caller cooperation needed.
    * **Clean-boundary truncation** — a prefix then a *permanent*
      ``b""`` (TCP half-close mid-stream, no buffered tail). This
      is indistinguishable from a real EOF *from the stream alone*.
      A non-seekable stream of unknown length therefore has no
      internal truncation oracle. Because this is a security
      control, that case is **deny-by-default**: a non-seekable
      stream is REFUSED unless the caller either supplies the
      authoritative ``expected_size`` (enabling detection) or
      explicitly opts out via ``allow_unverified_length=True``
      (accepting an unverifiable payload). The safe behaviour is
      the default; the unsafe choice is an explicit, greppable
      decision — the same posture as ``SandboxArchiveLimits``.

    Args:
        stream: The source byte stream.
        expected_size: Authoritative total byte length of the
            payload, known out-of-band (HTTP ``Content-Length``,
            an archive header, or ``stat``). When provided, the
            spooled byte count is asserted against it and any
            mismatch raises ``ArchiveStreamIntegrityError`` —
            closing the clean-boundary-truncation bomb-guard
            bypass. The oracle applies ONLY on the non-seekable
            spool path: a seekable random-access source is trusted
            by contract (it has a definite length and ``zipfile``
            reads its central directory directly), so
            ``expected_size`` is intentionally NOT enforced for a
            seekable ``stream`` and is silently unused there.
        allow_unverified_length: Explicit opt-out. When ``True``
            and ``expected_size`` is ``None``, a non-seekable
            stream is spooled WITHOUT clean-boundary-truncation
            detection — the caller takes responsibility for
            payload integrity (e.g. a verified digest). When
            ``False`` (default) and ``expected_size`` is ``None``
            and the stream is non-seekable, the stream is REFUSED
            with ``ArchiveStreamIntegrityError`` rather than
            silently spooling an unverifiable payload. When
            ``expected_size`` is provided this flag has NO effect —
            the length oracle always runs regardless, since a
            stronger integrity check is never suppressed by an
            opt-out meant only for the unknown-length case.

    Raises:
        ArchiveStreamIntegrityError: pending-tail truncation
            detected; or ``expected_size`` mismatch; or a
            non-seekable stream with neither ``expected_size`` nor
            ``allow_unverified_length`` (deny-by-default).
    """
    if _supports_zip_random_access(stream):
        yield _ZipFileStreamAdapter(stream)
        return

    # Deny-by-default: a non-seekable stream of unknown length has
    # no internal truncation oracle, so a clean-boundary truncation
    # would silently validate an under-limit prefix with the bomb
    # members never copied. Refuse unless the caller supplies the
    # authoritative length OR explicitly accepts the unverifiable
    # payload. Mirrors SandboxArchiveLimits: safe is the default,
    # the unsafe choice is explicit and greppable. Checked before
    # spooling — no point copying a payload we will refuse.
    if expected_size is None and not allow_unverified_length:
        raise ArchiveStreamIntegrityError(
            "non-seekable stream of unknown length: pass expected_size "
            "(the authoritative payload byte count) to enable "
            "clean-boundary-truncation detection, or set "
            "allow_unverified_length=True to explicitly accept an "
            "unverifiable payload (deny-by-default: refusing to spool "
            "a stream whose truncation cannot be detected)"
        )

    with tempfile.SpooledTemporaryFile(max_size=16 * 1024 * 1024, mode="w+b") as spool:
        shutil.copyfileobj(stream, spool)
        copied = spool.tell()
        # copyfileobj treats the FIRST empty read() as EOF. A clean
        # EOF leaves the source exhausted, so this probe returns
        # b"". A premature stop with a buffered tail (dropped conn /
        # partial flush / hostile short-read) leaves bytes pending —
        # proof the spool holds only a prefix. Fail loud.
        trailing = stream.read(1)
        if len(trailing) > 0:
            raise ArchiveStreamIntegrityError(
                "source stream had data pending after copy terminated — "
                "premature-EOF / short-read truncation; refusing to "
                "validate a truncated archive"
            )
        # Out-of-band length oracle: the ONLY way to catch a
        # clean-boundary truncation (prefix then permanent b"" with
        # no buffered tail) of a non-seekable stream. The probe
        # above cannot — that shape is indistinguishable from a real
        # EOF from the stream alone. A reproduced bomb-guard bypass
        # (valid_small_zip + bomb, cut at the first EOCD) is closed
        # here when the caller supplies the authoritative length.
        if expected_size is not None and copied != expected_size:
            raise ArchiveStreamIntegrityError(
                f"spooled {copied} bytes but caller declared "
                f"expected_size={expected_size} — payload truncated "
                f"in transit; refusing to validate"
            )
        spool.seek(0)
        # SpooledTemporaryFile satisfies the io.IOBase read/seek
        # protocol at runtime but is not a nominal io.IOBase
        # subclass in typeshed, so no isinstance branch or typed
        # local can narrow it — cast is the only available form.
        yield _ZipFileStreamAdapter(cast(io.IOBase, spool))


def _zip_member_is_dir(member: zipfile.ZipInfo) -> bool:
    """Determine whether a ZipInfo member represents a directory.

    Two representations exist in the wild:

    * Unix-authored zips store the Unix file type in the high word of
      ``external_attr``; mode ``0o040000`` unambiguously means directory.
    * DOS/Windows-authored zips (including some cross-platform tools)
      set ``external_attr`` mode to zero and instead indicate directories
      by appending a trailing ``/`` to the filename — what ``ZipInfo.is_dir()``
      checks.

    Using ``external_attr`` for the type allow-list check (in
    ``safe_zip_member_rel_path``) but ``is_dir()`` for the descent check
    causes false positives when a Unix tool omits the trailing slash from a
    directory entry.  This helper unifies the two representations: prefer the
    ``external_attr`` mode when it is non-zero, fall back to the trailing-slash
    convention for archives that carry no Unix metadata.
    """
    mode = (member.external_attr >> 16) & 0o170000
    if mode != 0:
        return mode == 0o040000
    return member.is_dir()


def safe_zip_member_rel_path(member: zipfile.ZipInfo) -> Path | None:
    """Validate one zip member; return its relative path or ``None`` for the root.

    Raises ``UnsafeZipMemberError`` for Windows drive/separator
    paths, absolute paths, ``..`` traversal, and any member whose
    Unix type (high word of ``external_attr``) is not a regular
    file or directory — symlink, hardlink, char/block device,
    FIFO, and socket members are all rejected (allow-list), since a
    device node or symlink must never materialize in the workspace.
    """
    if member.filename in ("", ".", "./"):
        return None

    windows_path = PureWindowsPath(member.filename)
    if len(windows_path.drive) > 0:
        raise UnsafeZipMemberError(member=member.filename, reason="windows drive path")
    if "\\" in member.filename:
        raise UnsafeZipMemberError(member=member.filename, reason="windows path separator")

    rel = PurePosixPath(member.filename)
    if rel.is_absolute():
        raise UnsafeZipMemberError(member=member.filename, reason="absolute path")
    if ".." in rel.parts:
        raise UnsafeZipMemberError(member=member.filename, reason="parent traversal")

    # Allow-list, mirroring the tar path (safe_tar_member_rel_path
    # permits only isdir / isreg). Only regular files (0o100000)
    # and directories (0o040000) may extract. mode == 0 is a zip
    # carrying no Unix metadata (e.g. a Windows-/DOS-authored
    # archive); the name-based traversal checks above already
    # constrained it, so treat it as a plain file/dir. Every other
    # type — symlink 0o120000, char device 0o020000, block device
    # 0o060000, FIFO 0o010000, socket 0o140000 — is rejected: such
    # a member must never materialize in the sandbox workspace.
    mode = (member.external_attr >> 16) & 0o170000
    if mode != 0 and mode not in (0o100000, 0o040000):
        raise UnsafeZipMemberError(
            member=member.filename,
            reason="non-file/non-directory member type not allowed",
        )

    return Path(*rel.parts)


def _check_archive_member_count(
    *,
    count: int,
    member: str,
    archive_limits: SandboxArchiveLimits | None,
) -> None:
    if archive_limits is None or archive_limits.max_members is None:
        return
    if count > archive_limits.max_members:
        raise ArchiveResourceLimitError(
            reason="archive member count exceeds limit",
            limit=archive_limits.max_members,
            actual=count,
            member=member,
        )


def _check_archive_extracted_bytes(
    *,
    total: int,
    member: str,
    archive_limits: SandboxArchiveLimits | None,
) -> None:
    if archive_limits is None or archive_limits.max_extracted_bytes is None:
        return
    if total > archive_limits.max_extracted_bytes:
        raise ArchiveResourceLimitError(
            reason="archive extracted size exceeds limit",
            limit=archive_limits.max_extracted_bytes,
            actual=total,
            member=member,
        )


def _resolve_limits(archive_limits: SandboxArchiveLimits | None) -> SandboxArchiveLimits:
    # None means "caller did not pass limits" — apply the deny-by-default
    # bounds rather than running unbounded. Explicit unbounded is
    # SandboxArchiveLimits.unbounded().
    return archive_limits if archive_limits is not None else SandboxArchiveLimits()


def _validated_member_size(
    declared: int,
    *,
    member: str,
    unsafe_error: type[UnsafeTarMemberError] | type[UnsafeZipMemberError],
) -> int:
    """Return the declared uncompressed size, or raise if it is hostile.

    ``TarInfo.size`` / ``ZipInfo.file_size`` are read verbatim from
    attacker-controlled member headers and are NOT cross-checked by
    ``tarfile`` / ``zipfile`` against the actual data region. A
    negative declared size is a corrupt / hostile header, NOT a
    zero-byte file. Silently flooring it with ``max(size, 0)`` lets
    a bomb member opt itself out of the extracted-byte guard (a
    crafted member declaring ``size = -1_000_000_000`` contributes
    0 to the running total — empirically a full bomb-guard bypass).
    Reject it loud so the limit cannot be defeated by a lying
    header.
    """
    if declared < 0:
        raise unsafe_error(
            member=member,
            reason=f"invalid (negative) declared member size: {declared}",
        )
    return declared


def validate_tar_archive_for_extraction(
    archive: tarfile.TarFile,
    *,
    archive_limits: SandboxArchiveLimits | None = None,
    allow_symlinks: bool = False,
) -> None:
    """Validate a tar archive for safe workspace extraction.

    Rejects: path traversal / absolute / windows / link members
    (via ``safe_tar_member_rel_path``), duplicate paths, descent
    through a non-directory member, and archives exceeding the
    member-count or extracted-byte limits.

    ``allow_symlinks`` defaults to ``False`` so externally-supplied
    archives (``extract``) reject symlink members. Workspace snapshots
    produced and re-consumed by the framework set it ``True`` — dev
    workspaces legitimately ship symlinks (e.g. ``venv``) and the
    ``filter="data"`` extraction pass still enforces symlink-target
    safety at write time.
    """
    limits = _resolve_limits(archive_limits)
    members_by_rel_path: dict[Path, tarfile.TarInfo] = {}
    descendant_by_parent_path: dict[Path, tarfile.TarInfo] = {}
    member_count = 0
    extracted_bytes = 0

    for member in archive:
        rel_path = safe_tar_member_rel_path(member, allow_symlinks=allow_symlinks)
        if rel_path is None:
            continue

        member_count += 1
        _check_archive_member_count(count=member_count, member=member.name, archive_limits=limits)
        extracted_bytes += _validated_member_size(member.size, member=member.name, unsafe_error=UnsafeTarMemberError)
        _check_archive_extracted_bytes(total=extracted_bytes, member=member.name, archive_limits=limits)

        previous = members_by_rel_path.get(rel_path)
        if previous is not None and not (previous.isdir() and member.isdir()):
            raise UnsafeTarMemberError(
                member=member.name,
                reason=f"duplicate archive path: {rel_path.as_posix()}",
            )

        for parent in rel_path.parents:
            if parent == Path():
                break
            parent_member = members_by_rel_path.get(parent)
            if parent_member is not None and not parent_member.isdir():
                raise UnsafeTarMemberError(
                    member=member.name,
                    reason=f"archive path descends through non-directory: {parent.as_posix()}",
                )

        if not member.isdir():
            descendant = descendant_by_parent_path.get(rel_path)
            if descendant is not None:
                raise UnsafeTarMemberError(
                    member=descendant.name,
                    reason=f"archive path descends through non-directory: {rel_path.as_posix()}",
                )

        members_by_rel_path[rel_path] = member
        for parent in rel_path.parents:
            if parent == Path():
                break
            descendant_by_parent_path.setdefault(parent, member)


def validate_zipfile(
    archive: zipfile.ZipFile,
    *,
    archive_limits: SandboxArchiveLimits | None = None,
) -> None:
    """Validate a zip archive for safe workspace extraction.

    Same guarantees as the tar validator: traversal/escape/link
    rejection, duplicate-path rejection, descent-through-non-dir
    rejection, and member-count / extracted-byte limits.
    """
    limits = _resolve_limits(archive_limits)
    members_by_rel_path: dict[Path, zipfile.ZipInfo] = {}
    members: list[tuple[zipfile.ZipInfo, Path]] = []
    extracted_bytes = 0

    for member in archive.infolist():
        rel_path = safe_zip_member_rel_path(member)
        if rel_path is None:
            continue

        _check_archive_member_count(count=len(members) + 1, member=member.filename, archive_limits=limits)
        extracted_bytes += _validated_member_size(
            member.file_size, member=member.filename, unsafe_error=UnsafeZipMemberError
        )
        _check_archive_extracted_bytes(total=extracted_bytes, member=member.filename, archive_limits=limits)

        previous = members_by_rel_path.get(rel_path)
        if previous is not None and not (_zip_member_is_dir(previous) and _zip_member_is_dir(member)):
            raise UnsafeZipMemberError(
                member=member.filename,
                reason=f"duplicate archive path: {rel_path.as_posix()}",
            )
        members_by_rel_path[rel_path] = member
        members.append((member, rel_path))

    for member, rel_path in members:
        for parent in rel_path.parents:
            if parent == Path():
                break
            parent_member = members_by_rel_path.get(parent)
            if parent_member is not None and not _zip_member_is_dir(parent_member):
                raise UnsafeZipMemberError(
                    member=member.filename,
                    reason=f"archive path descends through non-directory: {parent.as_posix()}",
                )
