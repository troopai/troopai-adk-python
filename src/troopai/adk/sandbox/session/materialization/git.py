"""Materializer for ``GitRepo`` — a repo cloned INSIDE the sandbox.

OpenAI attaches this to ``GitRepo.apply``; our entries are behaviorless
data, so the clone orchestration lives here and drives ``session.run``.
git runs inside the sandbox image (per the ``GitRepo`` contract — the
host is never used, so Docker / K8s / remote backends clone
in-container).

Every git invocation is prefixed with ``env GIT_TERMINAL_PROMPT=0
GIT_ASKPASS=true`` so a private / auth-required repo fails fast
instead of hanging on a credential prompt (OpenAI's upstream does NOT
set this and can hang indefinitely), and every step carries a bounded
``timeout`` (R2 — a clone must never be unbounded). A ``ref`` shaped
like a commit SHA uses init+fetch+checkout; a named branch/tag uses a
shallow ``git clone --branch``, with a commit→named fallback
(mirroring upstream so a hex-looking branch name still resolves). The
clone temp dir is removed in a ``finally`` — best-effort: a cleanup
failure is logged at WARNING and never masks the real error.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import TYPE_CHECKING

from troopai.adk.exceptions.exceptions import GitArtifactError
from troopai.adk.sandbox.session.materialization.metadata import apply_entry_metadata
from troopai.adk.sandbox.session.materialization.paths import normalize_workspace_key
from troopai.adk.types.sandbox.entries import MaterializedFile

if TYPE_CHECKING:
    from troopai.adk.sandbox.clients.session import BaseSandboxSession
    from troopai.adk.types.sandbox.entries import GitRepo

logger = logging.getLogger(__name__)

__all__ = ["materialize_git_repo"]

DEFAULT_GIT_CLONE_TIMEOUT = 300.0
"""Bounded per-clone wall-clock cap (R2 — a clone is never unbounded);
cost-conservative and a backstop if the no-prompt env somehow fails."""

_GIT_OP_TIMEOUT = 60.0
"""Bounded cap for the fast git / cp / rm ops (probe, copy, cleanup)."""

_GIT_ENV = ("env", "GIT_TERMINAL_PROMPT=0", "GIT_ASKPASS=true")
"""Prefix making every git call non-interactive fail-fast (no hang)."""

_COMMIT_SHA = re.compile(r"[0-9a-fA-F]{7,40}")
_STDERR_CAP = 300


def _looks_like_commit(ref: str) -> bool:
    """True iff ``ref`` is a 7–40 char hex string (a commit-SHA shape)."""
    return _COMMIT_SHA.fullmatch(ref) is not None


def _depth_args(depth: int | None) -> tuple[str, ...]:
    """``--depth N`` for a shallow clone; empty for a full clone."""
    return () if depth is None else ("--depth", str(depth))


async def _run_checked(
    session: BaseSandboxSession,
    argv: tuple[str, ...],
    *,
    timeout: float,
    what: str,
) -> None:
    """Run ``argv`` (shell=False); raise ``GitArtifactError`` on non-zero exit."""
    result = await session.run(*argv, shell=False, timeout=timeout)
    if result.exit_code != 0:
        detail = result.stderr.decode("utf-8", errors="replace")[:_STDERR_CAP]
        raise GitArtifactError(f"{what} failed (exit={result.exit_code}): {detail}")


async def _clone_named_ref(
    session: BaseSandboxSession, *, url: str, ref: str, depth: int | None, tmp: str, timeout: float
) -> None:
    """Shallow ``git clone --single-branch --branch <ref>`` of a branch/tag."""
    await _run_checked(
        session,
        (*_GIT_ENV, "git", "clone", *_depth_args(depth), "--no-tags", "--single-branch", "--branch", ref, url, tmp),
        timeout=timeout,
        what=f"git clone {url}@{ref}",
    )


async def _fetch_commit_ref(
    session: BaseSandboxSession, *, url: str, ref: str, depth: int | None, tmp: str, timeout: float
) -> None:
    """init + remote add + fetch + detached checkout for a commit-SHA ref."""
    await _run_checked(session, (*_GIT_ENV, "git", "init", tmp), timeout=timeout, what="git init")
    await _run_checked(
        session,
        (*_GIT_ENV, "git", "-C", tmp, "remote", "add", "origin", url),
        timeout=timeout,
        what="git remote add",
    )
    await _run_checked(
        session,
        (*_GIT_ENV, "git", "-C", tmp, "fetch", *_depth_args(depth), "--no-tags", "origin", ref),
        timeout=timeout,
        what=f"git fetch {ref}",
    )
    await _run_checked(
        session,
        (*_GIT_ENV, "git", "-C", tmp, "checkout", "--detach", "FETCH_HEAD"),
        timeout=timeout,
        what="git checkout",
    )


async def _clone(
    session: BaseSandboxSession, *, url: str, ref: str, depth: int | None, tmp: str, timeout: float
) -> None:
    """Clone ``url@ref`` into ``tmp``.

    A commit-SHA-shaped ref tries fetch first; on failure the tmp dir
    is reset and a named-ref clone is attempted (so a hex-looking
    BRANCH name still resolves — mirrors upstream). A final failure
    propagates ``GitArtifactError`` loudly.
    """
    if _looks_like_commit(ref):
        try:
            await _fetch_commit_ref(session, url=url, ref=ref, depth=depth, tmp=tmp, timeout=timeout)
            return
        except GitArtifactError as exc:
            logger.debug("commit-ref fetch for %s failed (%s); falling back to named-ref clone", ref, exc)
            await _run_checked(session, ("rm", "-rf", "--", tmp), timeout=_GIT_OP_TIMEOUT, what="git tmp reset")
    await _clone_named_ref(session, url=url, ref=ref, depth=depth, tmp=tmp, timeout=timeout)


async def materialize_git_repo(
    session: BaseSandboxSession,
    key: str,
    entry: GitRepo,
    *,
    clone_timeout: float = DEFAULT_GIT_CLONE_TIMEOUT,
) -> MaterializedFile:
    """Clone ``entry`` into the workspace at ``key`` (git runs in-sandbox).

    Raises:
        GitArtifactError: ``ref`` starts with ``-`` (argument-injection
            defense), git missing in the image, a clone / fetch /
            checkout step failed, or the (sub)path copy failed.
    """
    safe = normalize_workspace_key(key)
    # GitRepo.ref has no entry-level validator and flows into argv as
    # `git ... --branch <ref>` / `git fetch origin <ref>`. shell=False
    # blocks shell injection, but a leading `-` is a git
    # argument-injection vector (e.g. `--upload-pack=...`). Reject it.
    if entry.ref.startswith("-"):
        raise GitArtifactError(f"GitRepo.ref {entry.ref!r} may not start with '-' (argument-injection defense)")
    probe = await session.run(*_GIT_ENV, "git", "--version", shell=False, timeout=_GIT_OP_TIMEOUT)
    if probe.exit_code != 0:
        raise GitArtifactError(f"git is not available in the sandbox image (git --version exit={probe.exit_code})")
    url = f"https://{entry.host}/{entry.repo}.git"
    tmp = f"/tmp/troopai-git-{uuid.uuid4().hex}"
    try:
        await _clone(session, url=url, ref=entry.ref, depth=entry.depth, tmp=tmp, timeout=clone_timeout)
        git_src = tmp if entry.subpath is None else f"{tmp}/{entry.subpath}"
        await session.mkdir(safe, parents=True)
        await _run_checked(
            session,
            ("cp", "-R", "--", f"{git_src}/.", f"{safe}/"),
            timeout=_GIT_OP_TIMEOUT,
            what="git (sub)path copy",
        )
    finally:
        try:
            cleanup = await session.run("rm", "-rf", "--", tmp, shell=False, timeout=_GIT_OP_TIMEOUT)
            if cleanup.exit_code != 0:
                logger.warning("git tmp cleanup non-zero for %s (exit=%d)", tmp, cleanup.exit_code)
        except Exception as exc:
            # Best-effort cleanup: a failure here MUST NOT mask the
            # primary clone/copy error nor crash materialization
            # because a `/tmp` rm hiccupped. CancelledError is
            # BaseException, so cancellation still propagates.
            logger.warning("git tmp cleanup error for %s: %s", tmp, exc)
    await apply_entry_metadata(session, safe, entry)
    logger.debug("materialized GitRepo %s@%s -> %s", entry.repo, entry.ref, safe)
    return MaterializedFile(path=safe, size_bytes=0, permissions=entry.permissions, is_directory=True)
