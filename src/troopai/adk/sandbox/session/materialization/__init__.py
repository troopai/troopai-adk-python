"""Manifest-entry materialization — external typed dispatch.

OpenAI's sandbox places an ``apply()`` method on each entry class;
this project's manifest entries are Layer-1 behaviorless data, so
materialization is an EXTERNAL dispatch keyed on the concrete entry
type (``isinstance`` over the closed entry union). ``materialize_manifest``
is the single entry point a backend calls from ``apply_manifest``.

Concurrency mirrors the upstream applier: entries materialize in
declaration order, fanned out up to ``max_entry_concurrency`` through
the order-preserving ``gather_in_order`` helper, with a batch flush
whenever an entry's destination overlaps an already-queued entry
(so an ancestor ``mkdir`` cannot race a descendant write) or the
entry is a ``Mount``. Mounts are NOT file-materialized here — they
are attached natively by the backend at create time; the materializer
only records their keys in ``skipped_mounts``.

An entry whose concrete type has no materializer raises
``UnsupportedManifestEntryError`` — a registered entry type without a
dispatch arm is a framework gap that must surface loudly, never a
silently-skipped (and therefore missing) workspace file.

``only_ephemeral=True`` assumes durable ancestors already exist on
the (resumed) workspace: a skipped non-ephemeral ``Dir`` is NOT
re-created, so a child entry still materializes but relies on the
backend's ``write`` auto-creating parents (the in-tree backends do).
``ephemeral`` is strictly per-entry — marking a ``Dir`` ephemeral
does NOT propagate to its children, because ``Manifest.iter_entries``
flattens each child into its own entry filtered on its own flag.

``LocalFile``/``LocalDir`` host sources resolve against
``materialize_manifest``'s ``base_dir``: the default is the host
process working directory (``Path.cwd()`` sampled at call time) —
pass ``base_dir`` explicitly to pin resolution when the caller may
``os.chdir`` between manifest construction and materialization, and
so tests can fix it deterministically. Sources outside that root
require a ``SandboxPathGrant`` on the manifest (``extra_path_grants``).
The pre-open symlink defense in ``resolve_host_source`` — rejecting a
symlink on every component from the allowed root down to the source —
is platform-independent and always runs; ``O_NOFOLLOW`` adds
POSIX-only hardening of the check→open TOCTOU window at the final
``open`` (absent on non-POSIX, where the pre-open check is the sole
symlink defense). ``GitRepo`` clones run inside the sandbox image.
"""

from __future__ import annotations

import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from troopai.adk.exceptions.exceptions import UnsupportedManifestEntryError
from troopai.adk.sandbox.clients.session import MaterializationResult
from troopai.adk.sandbox.session.concurrency import gather_in_order
from troopai.adk.sandbox.session.materialization.git import materialize_git_repo
from troopai.adk.sandbox.session.materialization.inline import materialize_dir, materialize_file
from troopai.adk.sandbox.session.materialization.local import materialize_local_dir, materialize_local_file
from troopai.adk.sandbox.session.materialization.paths import normalize_workspace_key, paths_overlap
from troopai.adk.types.sandbox.entries import Dir, File, GitRepo, LocalDir, LocalFile, MaterializedFile
from troopai.adk.types.sandbox.mounts import Mount

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from troopai.adk.sandbox.clients.session import BaseSandboxSession
    from troopai.adk.types.sandbox.entries import BaseEntry
    from troopai.adk.types.sandbox.manifest import Manifest
    from troopai.adk.types.sandbox.workspace_paths import SandboxPathGrant

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_MAX_ENTRY_CONCURRENCY", "materialize_entry", "materialize_manifest"]

DEFAULT_MAX_ENTRY_CONCURRENCY = 4
"""Cost-conservative default: at most 4 entries materialize concurrently.

Bounded (not unbounded ``asyncio.gather``) so a large manifest cannot
saturate the event loop or a backend's exec channel; the developer may
raise it explicitly via ``materialize_manifest(max_entry_concurrency=)``.
"""

_SUPPORTED_TYPES = ("dir", "file", "git_repo", "local_dir", "local_file")
"""Entry ``type`` discriminators the dispatcher currently materializes."""


async def materialize_entry(
    session: BaseSandboxSession,
    key: str,
    entry: BaseEntry,
    *,
    base_dir: Path,
    grants: Sequence[SandboxPathGrant],
) -> MaterializedFile:
    """Dispatch one manifest entry to its concrete materializer.

    Args:
        session: Live backend session.
        key: Workspace-relative destination path.
        entry: The manifest entry to materialize (NOT a ``Mount`` —
            mounts are filtered by ``materialize_manifest`` first).
        base_dir: Host directory ``LocalFile``/``LocalDir`` sources
            resolve against (see the module docstring).
        grants: Absolute-path grants permitting sources outside
            ``base_dir``.

    Raises:
        UnsupportedManifestEntryError: ``entry``'s concrete type has
            no materializer arm.
    """
    safe = normalize_workspace_key(key)
    if isinstance(entry, File):
        return await materialize_file(session, safe, entry)
    if isinstance(entry, Dir):
        return await materialize_dir(session, safe, entry)
    if isinstance(entry, LocalFile):
        return await materialize_local_file(session, safe, entry, base_dir=base_dir, grants=grants)
    if isinstance(entry, LocalDir):
        return await materialize_local_dir(session, safe, entry, base_dir=base_dir, grants=grants)
    if isinstance(entry, GitRepo):
        return await materialize_git_repo(session, safe, entry)
    raise UnsupportedManifestEntryError(entry.type, supported_types=_SUPPORTED_TYPES)


async def materialize_manifest(
    session: BaseSandboxSession,
    manifest: Manifest,
    *,
    only_ephemeral: bool = False,
    base_dir: Path | None = None,
    max_entry_concurrency: int = DEFAULT_MAX_ENTRY_CONCURRENCY,
) -> MaterializationResult:
    """Materialize every manifest entry into the session's workspace.

    Parameter semantics — the ``only_ephemeral`` durable-ancestor
    assumption, per-entry ``ephemeral``, and the ``base_dir``
    host-resolution root (default ``Path.cwd()`` at call time) — are
    documented in the module docstring.

    Raises:
        ValueError: ``max_entry_concurrency`` < 1.
        UnsupportedManifestEntryError: an entry type has no materializer.
    """
    if max_entry_concurrency < 1:
        raise ValueError(f"max_entry_concurrency must be >= 1, got {max_entry_concurrency}")
    effective_base = base_dir if base_dir is not None else Path.cwd()
    grants = manifest.extra_path_grants
    files: list[MaterializedFile] = []
    skipped_mounts: list[str] = []
    batch: list[tuple[str, BaseEntry]] = []

    async def _flush() -> None:
        if len(batch) == 0:
            return
        factories: list[Callable[[], Awaitable[MaterializedFile]]] = [
            partial(materialize_entry, session, key, entry, base_dir=effective_base, grants=grants)
            for key, entry in batch
        ]
        files.extend(await gather_in_order(factories, max_concurrency=max_entry_concurrency))
        batch.clear()

    for key, entry in manifest.iter_entries():
        if isinstance(entry, Mount):
            await _flush()
            skipped_mounts.append(key)
            continue
        if only_ephemeral and not entry.ephemeral:
            continue
        if any(paths_overlap(key, queued_key) for queued_key, _ in batch):
            await _flush()
        batch.append((key, entry))
    await _flush()
    logger.info(
        "materialized %d manifest entries; %d mounts deferred to native attach",
        len(files),
        len(skipped_mounts),
    )
    return MaterializationResult(files=files, skipped_mounts=skipped_mounts)
