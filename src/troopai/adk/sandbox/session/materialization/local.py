"""Materializers for ``LocalFile`` / ``LocalDir`` — host paths copied in.

OpenAI attaches this to ``LocalFile.apply`` / ``LocalDir.apply``; our
entries are behaviorless data, so the host read, symlink defense, and
path-grant enforcement live here and drive the typed ``session.write``.

``src`` resolves against the host process working directory
(``base_dir``) and MUST stay under it or a ``SandboxPathGrant`` root.
Resolution uses ``os.path.abspath`` — NOT ``Path.resolve`` — so a
symlinked path component cannot canonicalize the real target past the
boundary check; every component from the allowed root down to ``src``
is then explicitly rejected if it is a symlink (platform-independent —
this is the primary defense). On POSIX the final ``open`` adds
``O_NOFOLLOW`` to harden the check→open TOCTOU window; on a platform
without ``O_NOFOLLOW`` that hardening is unavailable and the pre-open
component check is the sole symlink defense.

``LocalDir`` walks the source tree with ``os.walk(followlinks=False)``
(bounded — symlinked directories are never descended, so no symlink
cycle). ``follow_symlinks=False`` skips symlinks; ``follow_symlinks=
True`` follows only file symlinks whose REAL target stays within an
allowed root, and rejects symlinked directories loudly (a followed
directory symlink is an unbounded cycle / escape vector — strictly
safer than blindly following, and a superset of OpenAI's
reject-all-symlinks behavior that still honors the typed field).
"""

from __future__ import annotations

import logging
import os
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from troopai.adk.exceptions.exceptions import LocalArtifactError
from troopai.adk.sandbox.session.concurrency import gather_in_order
from troopai.adk.sandbox.session.materialization.metadata import apply_entry_metadata
from troopai.adk.sandbox.session.materialization.paths import normalize_workspace_key
from troopai.adk.types.sandbox.entries import MaterializedFile

if TYPE_CHECKING:
    from collections.abc import Sequence

    from troopai.adk.sandbox.clients.session import BaseSandboxSession
    from troopai.adk.types.sandbox.entries import LocalDir, LocalFile
    from troopai.adk.types.sandbox.workspace_paths import SandboxPathGrant

logger = logging.getLogger(__name__)

__all__ = ["materialize_local_dir", "materialize_local_file", "resolve_host_source"]

DEFAULT_MAX_LOCAL_DIR_FILE_CONCURRENCY = 4
"""Cost-conservative bound: at most 4 files within one ``LocalDir`` copy
concurrently (bounded, not unbounded ``asyncio.gather``)."""


def _allowed_roots(base_dir: Path, grants: Sequence[SandboxPathGrant]) -> list[Path]:
    """Absolute (lexical, no symlink resolution) roots a source may sit under."""
    return [Path(os.path.abspath(base_dir)), *(Path(os.path.abspath(g.path)) for g in grants)]


def _is_within(path: Path, roots: list[Path]) -> bool:
    """True iff ``path`` equals or is contained by an allowed root."""
    return any(path == root or path.is_relative_to(root) for root in roots)


def resolve_host_source(src: Path, *, base_dir: Path, grants: Sequence[SandboxPathGrant]) -> Path:
    """Resolve ``src`` to an absolute host path under an allowed root.

    ``src`` is made absolute WITHOUT resolving symlinks; it must then
    sit under ``base_dir`` or a granted root, and no component from
    that root down to ``src`` may be a symlink.

    Args:
        src: ``LocalFile.src`` (absolute, or relative to ``base_dir``).
        base_dir: Host process working directory.
        grants: Absolute-path grants permitting sources outside base.

    Raises:
        LocalArtifactError: ``src`` escapes every allowed root, or a
            path component on the chain is a symlink.
    """
    raw = src if src.is_absolute() else base_dir / src
    # abspath (NOT Path.resolve / os.path.realpath): lexically collapse
    # ``..`` / ``.`` WITHOUT resolving symlinks, so a symlinked
    # component cannot canonicalize the real target past the
    # containment + per-component symlink checks below. The two
    # sibling normalizers differ deliberately — do not "unify" to
    # normpath/realpath without re-deriving this property.
    absolute = Path(os.path.abspath(raw))
    roots = _allowed_roots(base_dir, grants)
    if not _is_within(absolute, roots):
        raise LocalArtifactError(
            f"host source {str(src)!r} -> {str(absolute)!r} is outside the host "
            f"working directory and every SandboxPathGrant"
        )
    for component in [absolute, *absolute.parents]:
        if component.is_symlink():
            raise LocalArtifactError(f"host source component {str(component)!r} is a symlink; refusing to follow")
        if any(component == root for root in roots):
            break
    return absolute


async def _copy_host_file(session: BaseSandboxSession, host: Path, dest_key: str) -> int:
    """Stream an already-validated host file into the workspace at ``dest_key``.

    ``host`` MUST already be containment- and symlink-validated (by
    ``resolve_host_source`` or the ``LocalDir`` walk). Returns the host
    file size from ``os.fstat`` at open — ``session.write`` does not
    report the bytes it actually wrote.

    Raises:
        LocalArtifactError: a host-side ``OSError`` on open / fstat /
            read. A backend ``session.write`` failure that is NOT an
            ``OSError`` (e.g. a transport / ``SandboxError`` from a
            remote backend) propagates UNTRANSLATED by design — it is
            a sandbox-side write fault, not a "from the host" artifact
            fault, so masking it as ``LocalArtifactError`` would hide
            the operator's real diagnostic.
    """
    flags = os.O_RDONLY | (os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0)
    try:
        fd = os.open(host, flags)
    except OSError as exc:
        raise LocalArtifactError(f"host source {str(host)!r} could not be opened: {exc}") from exc
    try:
        size = os.fstat(fd).st_size
        handle = os.fdopen(fd, "rb")
    except OSError as exc:
        os.close(fd)
        raise LocalArtifactError(f"host source {str(host)!r} stat/open failed: {exc}") from exc
    try:
        with handle:
            await session.write(dest_key, handle)
    except OSError as exc:
        raise LocalArtifactError(f"host source {str(host)!r} copy failed: {exc}") from exc
    return size


async def materialize_local_file(
    session: BaseSandboxSession,
    key: str,
    entry: LocalFile,
    *,
    base_dir: Path,
    grants: Sequence[SandboxPathGrant],
) -> MaterializedFile:
    """Copy ``entry.src`` from the host into the workspace at ``key``.

    Args:
        session: Live backend session.
        key: Workspace-relative destination path.
        entry: The ``LocalFile`` entry.
        base_dir: Host process working directory ``src`` resolves against.
        grants: Absolute-path grants for sources outside ``base_dir``.

    Note:
        ``MaterializedFile.size_bytes`` is the host file size at open
        (``os.fstat``); ``session.write`` does not report the bytes it
        actually wrote, so a backend transform/truncation (rare; e.g.
        a volatile mount) would not be reflected in the audit record.

    Raises:
        LocalArtifactError: source escapes the boundary, is a symlink,
            or cannot be read.
    """
    safe = normalize_workspace_key(key)
    host = resolve_host_source(entry.src, base_dir=base_dir, grants=grants)
    size = await _copy_host_file(session, host, safe)
    await apply_entry_metadata(session, safe, entry)
    logger.debug("materialized LocalFile %s -> %s (%d bytes)", host, safe, size)
    return MaterializedFile(
        path=safe,
        size_bytes=size,
        permissions=entry.permissions,
        is_directory=False,
    )


def _resolve_followed_file_symlink(link: Path, roots: list[Path]) -> Path:
    """Resolve a followed symlink; its real target MUST be a contained regular file.

    Raises:
        LocalArtifactError: the resolved target escapes every allowed
            root, or is not a regular file (e.g. a symlinked directory
            — never followed: unbounded cycle / escape vector).
    """
    target = Path(os.path.realpath(link))
    if not _is_within(target, roots):
        raise LocalArtifactError(f"LocalDir symlink {str(link)!r} -> {str(target)!r} escapes every allowed root")
    if not target.is_file() or target.is_symlink():
        raise LocalArtifactError(
            f"LocalDir symlink {str(link)!r} -> {str(target)!r} is not a regular file (not followed)"
        )
    return target


def _walk_local_dir(
    src_root: Path,
    *,
    roots: list[Path],
    follow_symlinks: bool,
) -> tuple[list[str], list[tuple[Path, str]]]:
    """Walk ``src_root`` (no symlink descent) → (relative subdir keys, (host_file, rel) pairs).

    ``os.walk(followlinks=False)`` is bounded (a symlinked directory is
    never descended, so no symlink cycle). Symlinked directories are
    skipped (``follow_symlinks=False``) or rejected loudly
    (``follow_symlinks=True``); file symlinks are skipped or followed
    to a contained regular target; sockets/fifos/devices are skipped.

    Raises:
        LocalArtifactError: a symlinked directory under
            ``follow_symlinks=True``, or a followed file symlink whose
            target escapes the allowed roots.
    """
    rel_dirs: list[str] = []
    files: list[tuple[Path, str]] = []
    for dirpath, dirnames, filenames in os.walk(src_root, followlinks=False):
        base = Path(dirpath)
        kept: list[str] = []
        for name in dirnames:
            child = base / name
            if child.is_symlink():
                if follow_symlinks:
                    raise LocalArtifactError(
                        f"LocalDir symlinked directory {str(child)!r} is not followed (cycle/escape risk)"
                    )
                logger.debug("LocalDir: skipping symlinked directory %s", child)
                continue
            kept.append(name)
            rel_dirs.append(child.relative_to(src_root).as_posix())
        dirnames[:] = kept
        for name in filenames:
            child = base / name
            rel = child.relative_to(src_root).as_posix()
            if child.is_symlink():
                if not follow_symlinks:
                    logger.debug("LocalDir: skipping symlink %s (follow_symlinks=False)", child)
                    continue
                files.append((_resolve_followed_file_symlink(child, roots), rel))
            elif child.is_file():
                files.append((child, rel))
            else:
                logger.debug("LocalDir: skipping non-regular entry %s", child)
    return rel_dirs, files


async def materialize_local_dir(
    session: BaseSandboxSession,
    key: str,
    entry: LocalDir,
    *,
    base_dir: Path,
    grants: Sequence[SandboxPathGrant],
    max_file_concurrency: int = DEFAULT_MAX_LOCAL_DIR_FILE_CONCURRENCY,
) -> MaterializedFile:
    """Recursively copy ``entry.src`` into the workspace at ``key``.

    ``src=None`` creates an empty directory. See the module docstring
    for the symlink policy.

    Raises:
        LocalArtifactError: source escapes the boundary, or a symlink
            policy violation (see ``_walk_local_dir``).
    """
    safe = normalize_workspace_key(key)
    if entry.src is None:
        await session.mkdir(safe, parents=True)
        await apply_entry_metadata(session, safe, entry)
        logger.debug("materialized empty LocalDir at %s", safe)
        return MaterializedFile(path=safe, size_bytes=0, permissions=entry.permissions, is_directory=True)
    src_root = resolve_host_source(entry.src, base_dir=base_dir, grants=grants)
    # _allowed_roots is recomputed here even though resolve_host_source
    # computed its own internally: both calls take the SAME
    # base_dir/grants locals, so the lists are identical by
    # construction — no divergence hazard. resolve_host_source stays
    # self-contained public API rather than threading roots through
    # its signature for an internal micro-optimization.
    roots = _allowed_roots(base_dir, grants)
    rel_dirs, files = _walk_local_dir(src_root, roots=roots, follow_symlinks=entry.follow_symlinks)
    await session.mkdir(safe, parents=True)
    for rel in rel_dirs:
        await session.mkdir(f"{safe}/{rel}", parents=True)
    factories = [partial(_copy_host_file, session, host, f"{safe}/{rel}") for host, rel in files]
    # gather_in_order's list[int] result (per-file host sizes) is
    # intentionally discarded: a directory has no single size, so the
    # dir record reports size_bytes=0 (matching inline.materialize_dir).
    # The await is RETAINED because gather_in_order is fail-fast — a
    # _copy_host_file LocalArtifactError aborts the whole dir copy
    # loudly rather than silently dropping a child.
    await gather_in_order(factories, max_concurrency=max_file_concurrency)
    await apply_entry_metadata(session, safe, entry)
    logger.debug("materialized LocalDir %s -> %s (%d files)", src_root, safe, len(files))
    return MaterializedFile(path=safe, size_bytes=0, permissions=entry.permissions, is_directory=True)
