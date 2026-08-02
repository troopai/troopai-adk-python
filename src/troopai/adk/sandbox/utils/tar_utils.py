"""Safe tar validation and extraction.

Two attack surfaces this module closes:

* **Path traversal.** A tar member named ``../../../etc/passwd``
  must not escape the extraction root.
* **Symlink chaining.** A tar member that is a symlink whose
  target points outside the root must not become a write
  primitive once a later member is extracted under that name.

Both are handled before any data lands on disk:
``validate_tarfile`` raises ``UnsafeTarMemberError`` on rejection;
``safe_extract_tarfile`` skips symlink members until directories
and files are placed, then materializes symlinks last (so a later
member can't slip through a symlinked parent).
"""

from __future__ import annotations

import contextlib
import copy
import io
import logging
import os
import shutil
import tarfile
import tempfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath

logger = logging.getLogger(__name__)

__all__ = [
    "UnsafeTarMemberError",
    "safe_extract_tarfile",
    "safe_tar_member_rel_path",
    "should_skip_tar_member",
    "strip_tar_member_prefix",
    "validate_tar_bytes",
    "validate_tarfile",
]


class UnsafeTarMemberError(ValueError):
    """Raised when a tar member violates the safe-extraction policy.

    Attributes:
        member: The offending member name (verbatim from the archive).
        reason: Human-readable explanation of which rule fired.
    """

    def __init__(self, *, member: str, reason: str) -> None:
        super().__init__(f"unsafe tar member {member!r}: {reason}")
        self.member = member
        self.reason = reason


def _validate_archive_root_member(member: tarfile.TarInfo) -> None:
    if member.isdir():
        return
    if member.issym():
        raise UnsafeTarMemberError(member=member.name, reason="archive root symlink")
    if member.islnk():
        raise UnsafeTarMemberError(member=member.name, reason="archive root hardlink")
    raise UnsafeTarMemberError(member=member.name, reason="archive root member must be directory")


def _raise_if_windows_member_path(member_name: str) -> None:
    windows_path = PureWindowsPath(member_name)
    if len(windows_path.drive) > 0:
        raise UnsafeTarMemberError(member=member_name, reason="windows drive path")
    if "\\" in member_name:
        raise UnsafeTarMemberError(member=member_name, reason="windows path separator")


def _normalize_posix_path_without_root(path: PurePosixPath) -> tuple[str, ...] | None:
    normalized: list[str] = []
    for part in path.parts:
        if part in ("", ".", "/"):
            continue
        if part == "..":
            if len(normalized) == 0:
                return None
            normalized.pop()
            continue
        normalized.append(part)
    return tuple(normalized)


def _validate_symlink_target(
    member: tarfile.TarInfo,
    *,
    rel_path: Path,
    allow_external_symlink_targets: bool,
) -> None:
    if not member.issym() or allow_external_symlink_targets:
        return

    target = PurePosixPath(member.linkname)
    if target.is_absolute():
        raise UnsafeTarMemberError(
            member=member.name,
            reason=f"absolute symlink target not allowed: {member.linkname}",
        )

    member_parent = PurePosixPath(rel_path.as_posix()).parent
    normalized = _normalize_posix_path_without_root(member_parent / target)
    if normalized is None:
        raise UnsafeTarMemberError(
            member=member.name,
            reason=f"symlink target escapes archive root: {member.linkname}",
        )


def safe_tar_member_rel_path(
    member: tarfile.TarInfo,
    *,
    allow_symlinks: bool = False,
) -> Path | None:
    """Validate one tar member and return its relative path within the archive.

    Returns ``None`` for the archive root (members named ``""``,
    ``"."``, or ``"./"`` — those are accepted as long as they're
    directories). Raises ``UnsafeTarMemberError`` for absolute paths,
    parent traversals, Windows-style paths, hardlinks, non-regular
    /non-directory members, and disallowed symlinks.
    """
    if member.name in ("", ".", "./"):
        _validate_archive_root_member(member)
        return None
    _raise_if_windows_member_path(member.name)
    rel = PurePosixPath(member.name)
    if rel.is_absolute():
        raise UnsafeTarMemberError(member=member.name, reason="absolute path")
    if ".." in rel.parts:
        raise UnsafeTarMemberError(member=member.name, reason="parent traversal")
    if member.issym() and not allow_symlinks:
        raise UnsafeTarMemberError(member=member.name, reason="symlink member not allowed")
    if member.islnk():
        raise UnsafeTarMemberError(member=member.name, reason="hardlink member not allowed")
    if not (member.isdir() or member.isreg() or (allow_symlinks and member.issym())):
        raise UnsafeTarMemberError(member=member.name, reason="unsupported member type")
    return Path(*rel.parts)


def _normalize_rel(prefix: str | Path) -> Path:
    rel = prefix if isinstance(prefix, Path) else Path(prefix)
    posix = rel.as_posix()
    parts = [p for p in Path(posix).parts if p not in ("", ".")]
    if parts[:1] == ["/"]:
        parts = parts[1:]
    return Path(*parts)


def _is_within(path: Path, prefix: Path) -> bool:
    if prefix == Path():
        return True
    if path == prefix:
        return True
    return path.parts[: len(prefix.parts)] == prefix.parts


def strip_tar_member_prefix(data: io.IOBase, *, prefix: str | Path) -> io.IOBase:
    """Rewrite a tar stream so every member's leading ``prefix`` is replaced with ``.``.

    Used when normalizing backend-specific workspace tar layouts.
    Docker archives ``/tmp/stage/workspace`` as ``workspace/...``;
    portable workspace snapshots should be ``.``-rooted so a
    different backend can hydrate the same payload.

    Returns a SEEKABLE ``TemporaryFile``; the caller owns lifetime.
    Raises ``UnsafeTarMemberError`` if any member sits outside the
    prefix or if the stripped archive fails ``validate_tarfile``.
    """
    prefix_rel = _normalize_rel(prefix)
    if prefix_rel == Path():
        raise ValueError("strip_tar_member_prefix: prefix must not be empty")

    out = tempfile.TemporaryFile()  # noqa: SIM115 — returned to the caller; close is their responsibility
    try:
        with data, tarfile.open(fileobj=data, mode="r|*") as src, tarfile.open(fileobj=out, mode="w|") as dst:
            for member in src:
                rel_path = safe_tar_member_rel_path(member, allow_symlinks=True)
                if rel_path is None or rel_path == prefix_rel:
                    stripped_name = "."
                elif rel_path.parts[: len(prefix_rel.parts)] == prefix_rel.parts:
                    stripped_name = Path(*rel_path.parts[len(prefix_rel.parts) :]).as_posix()
                else:
                    raise UnsafeTarMemberError(
                        member=member.name,
                        reason=f"member does not start with prefix: {prefix_rel.as_posix()}",
                    )

                rewritten = copy.copy(member)
                rewritten.name = stripped_name
                rewritten.pax_headers = dict(member.pax_headers)
                rewritten.pax_headers.pop("path", None)
                if member.isreg():
                    fileobj = src.extractfile(member)
                    if fileobj is None:
                        raise UnsafeTarMemberError(
                            member=member.name,
                            reason="missing file payload",
                        )
                    try:
                        dst.addfile(rewritten, fileobj)
                    finally:
                        fileobj.close()
                else:
                    dst.addfile(rewritten)

        out.seek(0)
        with tarfile.open(fileobj=out, mode="r:*") as tar:
            validate_tarfile(tar)
        out.seek(0)
        return out
    except Exception:
        out.close()
        raise


def should_skip_tar_member(
    member_name: str,
    *,
    skip_rel_paths: Iterable[str | Path],
    root_name: str | None,
) -> bool:
    """Return True iff ``member_name`` matches any workspace-relative skip prefix.

    Used to filter out paths that should not be persisted into a
    workspace snapshot (e.g. the snapshot fingerprint cache itself).
    """
    raw_parts = [p for p in Path(member_name).parts if p not in ("", ".")]
    if raw_parts[:1] == ["/"]:
        raw_parts = raw_parts[1:]
    if len(raw_parts) == 0:
        rel_variants = [Path()]
    else:
        rel_variants = [Path(*raw_parts)]
        if root_name is not None and len(raw_parts) > 0 and raw_parts[0] == root_name:
            rel_variants.append(Path(*raw_parts[1:]))

    # Drop prefixes that normalize to the empty path ("", ".", "./"): an
    # empty prefix would make _is_within match every member, silently
    # skipping the entire archive past all safe-extraction checks.
    prefixes = [p for p in (_normalize_rel(p) for p in skip_rel_paths) if p != Path()]
    return any(_is_within(rel, prefix) for rel in rel_variants for prefix in prefixes)


def _ensure_no_symlink_parents(*, root: Path, dest: Path, check_leaf: bool = True) -> None:
    root_resolved = root.resolve()
    path_to_resolve = dest if check_leaf else dest.parent
    dest_resolved = path_to_resolve.resolve()
    if not (dest_resolved == root_resolved or dest_resolved.is_relative_to(root_resolved)):
        raise UnsafeTarMemberError(
            member=dest.as_posix(),
            reason="path escapes root after resolution",
        )

    rel = dest.relative_to(root)
    cur = root
    for part in rel.parts[:-1]:
        cur = cur / part
        if cur.exists() and cur.is_symlink():
            raise UnsafeTarMemberError(
                member=str(rel.as_posix()),
                reason="symlink in parent path",
            )


def validate_tarfile(
    tar: tarfile.TarFile,
    *,
    reject_symlink_rel_paths: Iterable[str | Path] = (),
    skip_rel_paths: Iterable[str | Path] = (),
    root_name: str | None = None,
    allow_symlinks: bool = True,
    allow_external_symlink_targets: bool = False,
) -> None:
    """Validate every member in ``tar`` against the safe-extraction policy.

    Symlinks ARE allowed by default (normal dev workspaces ship
    them — e.g. ``venv``); the rule is that nothing in the archive
    may sit *underneath* a symlink member, so a chained
    write-through-symlink attack is impossible.
    """
    rejected_symlink_rel_paths = {_normalize_rel(path) for path in reject_symlink_rel_paths}
    members_by_rel_path: dict[Path, tarfile.TarInfo] = {}
    symlink_rel_paths: set[Path] = set()
    members: list[tuple[tarfile.TarInfo, Path]] = []

    for member in tar.getmembers():
        if should_skip_tar_member(
            member.name,
            skip_rel_paths=skip_rel_paths,
            root_name=root_name,
        ):
            continue
        rel_path = safe_tar_member_rel_path(member, allow_symlinks=allow_symlinks)
        if rel_path is None:
            continue

        previous = members_by_rel_path.get(rel_path)
        if previous is not None and not (previous.isdir() and member.isdir()):
            raise UnsafeTarMemberError(
                member=member.name,
                reason=f"duplicate archive path: {rel_path.as_posix()}",
            )
        members_by_rel_path[rel_path] = member

        if member.issym():
            _validate_symlink_target(
                member,
                rel_path=rel_path,
                allow_external_symlink_targets=allow_external_symlink_targets,
            )
            if rel_path in rejected_symlink_rel_paths:
                raise UnsafeTarMemberError(
                    member=member.name,
                    reason=f"symlink member not allowed: {rel_path.as_posix()}",
                )
            symlink_rel_paths.add(rel_path)
        members.append((member, rel_path))

    for member, rel_path in members:
        for parent in rel_path.parents:
            if parent == Path():
                break
            if parent in symlink_rel_paths:
                raise UnsafeTarMemberError(
                    member=member.name,
                    reason=f"archive path descends through symlink: {parent.as_posix()}",
                )
            parent_member = members_by_rel_path.get(parent)
            if parent_member is not None and not parent_member.isdir():
                raise UnsafeTarMemberError(
                    member=member.name,
                    reason=f"archive path descends through non-directory: {parent.as_posix()}",
                )


def validate_tar_bytes(
    raw: bytes,
    *,
    reject_symlink_rel_paths: Iterable[str | Path] = (),
    skip_rel_paths: Iterable[str | Path] = (),
    root_name: str | None = None,
    allow_external_symlink_targets: bool = False,
) -> None:
    """Validate raw tar bytes with the same policy as ``validate_tarfile``."""
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as tar:
            validate_tarfile(
                tar,
                reject_symlink_rel_paths=reject_symlink_rel_paths,
                skip_rel_paths=skip_rel_paths,
                root_name=root_name,
                allow_external_symlink_targets=allow_external_symlink_targets,
            )
    except UnsafeTarMemberError:
        raise
    except tarfile.TarError as e:
        # Surface which tarfile variant fired so callers can distinguish
        # truncated payload (retryable) from corrupted header (data-bug)
        # without inspecting __cause__ by hand.
        raise UnsafeTarMemberError(
            member="<tar>",
            reason=f"invalid tar stream: {type(e).__name__}: {e}",
        ) from e


def safe_extract_tarfile(
    tar: tarfile.TarFile,
    *,
    root: Path,
    allow_external_symlink_targets: bool = False,
) -> None:
    """Extract ``tar`` into ``root`` after running the safe-extraction policy.

    Materializes directories and files first; symlinks are written
    last so that a member nested under a symlink can never be
    redirected — the rejection happens in ``validate_tarfile``
    before any disk write, but the deferred-symlink order is the
    defense-in-depth.
    """
    root.mkdir(parents=True, exist_ok=True)
    root_resolved = root.resolve()

    members = tar.getmembers()
    validate_tarfile(
        tar,
        allow_external_symlink_targets=allow_external_symlink_targets,
    )

    def _prepare_replaceable_leaf(*, dest: Path, rel_path: Path, name: str) -> None:
        _ensure_no_symlink_parents(root=root_resolved, dest=dest, check_leaf=False)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_dir() and not dest.is_symlink():
            raise UnsafeTarMemberError(
                member=name,
                reason=f"destination directory already exists: {rel_path.as_posix()}",
            )
        with contextlib.suppress(FileNotFoundError):
            dest.unlink()

    def _prepare_directory_leaf(*, dest: Path) -> None:
        _ensure_no_symlink_parents(root=root_resolved, dest=dest, check_leaf=False)
        if dest.is_symlink() or (dest.exists() and not dest.is_dir()):
            dest.unlink()

    def _write_file(member: tarfile.TarInfo, *, dest: Path, rel_path: Path, name: str) -> None:
        fileobj = tar.extractfile(member)
        if fileobj is None:
            raise UnsafeTarMemberError(member=name, reason="missing file payload")

        _prepare_replaceable_leaf(dest=dest, rel_path=rel_path, name=name)

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(dest, flags, 0o600)
        try:
            with os.fdopen(fd, "wb") as out:
                shutil.copyfileobj(fileobj, out)
        finally:
            # Narrow suppression: only the documented close()-on-closed-archive
            # OSError is silenced; everything else (incl. tarfile.StreamError)
            # propagates so corrupt archives don't cascade silently.
            try:
                fileobj.close()
            except OSError as exc:
                logger.warning("safe_extract_tarfile: fileobj.close() failed for %s: %s", name, exc)

    for member in members:
        name = member.name
        rel_path = safe_tar_member_rel_path(member, allow_symlinks=True)
        if rel_path is None:
            continue
        if member.issym():
            continue

        dest = root_resolved / rel_path

        if member.isdir():
            _prepare_directory_leaf(dest=dest)
            dest.mkdir(parents=True, exist_ok=True)
            continue

        _write_file(member, dest=dest, rel_path=rel_path, name=name)

    for member in members:
        if not member.issym():
            continue
        rel_path = safe_tar_member_rel_path(member, allow_symlinks=True)
        if rel_path is None:
            continue
        dest = root_resolved / rel_path
        _prepare_replaceable_leaf(dest=dest, rel_path=rel_path, name=member.name)
        # Defense-in-depth: resolve the FULL destination path before writing
        # the symlink so a concurrent extraction that turned a parent
        # component into a symlink can't redirect this write outside the
        # workspace root. The pre-extraction validate_tarfile pass catches
        # in-archive descents through symlinks; this check protects against
        # filesystem state mutations between the two extraction passes.
        _ensure_no_symlink_parents(root=root_resolved, dest=dest, check_leaf=True)
        os.symlink(member.linkname, dest)
