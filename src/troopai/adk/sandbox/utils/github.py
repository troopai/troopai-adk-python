"""GitHub repo clone helpers.

``GitRepo`` manifest entries need a shallow checkout of the
declared ``ref`` (branch / tag / commit SHA). ``clone_repo``
tries the fast shallow path first and falls back to a full
clone + explicit checkout when ``ref`` is a SHA (which
``--depth 1 --branch SHA`` rejects).

Failure modes carry git's stderr through to the raised exception
so the operator can debug auth, network, and missing-ref errors
without re-running with extra verbosity. The fallback path
scrubs the partial clone state left by a failed shallow attempt
so the full clone doesn't trip on a non-empty destination.

Every git invocation is non-interactive (``GIT_TERMINAL_PROMPT=0``,
``GIT_ASKPASS=true``, stdin sealed to ``/dev/null``) and bounded by
a wall-clock timeout, so a private / auth-required / missing repo
fails fast with stderr instead of hanging on a credential prompt.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["clone_repo", "ensure_git_available"]

_CLONE_TIMEOUT_SECONDS = 300.0
"""Bounded wall-clock cap on a ``git clone`` so a hung network or
silently-blocking auth challenge never wedges the caller."""

_OP_TIMEOUT_SECONDS = 60.0
"""Bounded cap on the fast post-clone ops (the explicit checkout)."""


def ensure_git_available() -> None:
    """Raise ``RuntimeError`` if ``git`` is not on ``PATH``."""
    if shutil.which("git") is None:
        raise RuntimeError("git is required to use GitHub repo artifacts; install git and re-run.")


def _run_git(args: list[str], *, timeout: float = _OP_TIMEOUT_SECONDS) -> subprocess.CompletedProcess[str]:
    """Run ``git`` with captured stdout/stderr; do NOT raise on non-zero.

    Runs non-interactively (credential prompts disabled, stdin sealed)
    and bounded by ``timeout`` so an auth-required / missing repo fails
    fast rather than blocking on a credential prompt. Raises
    ``subprocess.TimeoutExpired`` if the call exceeds ``timeout``.
    """
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "true"}
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        check=False,
        stdin=subprocess.DEVNULL,
        env=env,
        timeout=timeout,
    )


def _run_git_step(
    args: list[str], *, timeout: float, repo: str, ref: str, what: str
) -> subprocess.CompletedProcess[str]:
    """Run one git step, converting a timeout into a clear ``RuntimeError``."""
    try:
        return _run_git(args, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"clone_repo({repo}@{ref}): {what} timed out after {timeout:.0f}s "
            "(the repo may be private / missing — credential prompts are disabled, so it fails rather than hangs)."
        ) from exc


def clone_repo(*, repo: str, ref: str, dest: Path) -> None:
    """Clone ``github.com/<repo>`` at ``ref`` into ``dest``.

    First attempts a shallow ``--depth 1 --no-tags --branch <ref>``
    clone (fast path for branches and tags). If that fails (most
    commonly because ``ref`` is a SHA), wipes the partial
    destination and falls back to ``--no-checkout`` followed by an
    explicit ``git checkout <ref>``.

    Args:
        repo: ``owner/name`` slug — must NOT include the URL prefix.
        ref: Branch, tag, or commit SHA to check out.
        dest: Target directory. Parent dirs are created if absent.

    Raises:
        RuntimeError: ``git`` binary not found on ``PATH``, a clone /
            checkout step exceeded its wall-clock timeout, or every
            clone strategy failed. The message includes git's stderr
            from each attempted step so the operator can debug
            auth, network, and missing-ref errors.
    """
    ensure_git_available()
    url = f"https://github.com/{repo}.git"
    dest.parent.mkdir(parents=True, exist_ok=True)

    shallow = _run_git_step(
        ["git", "clone", "--depth", "1", "--no-tags", "--branch", ref, url, str(dest)],
        timeout=_CLONE_TIMEOUT_SECONDS,
        repo=repo,
        ref=ref,
        what="shallow clone",
    )
    if shallow.returncode == 0:
        logger.info("clone_repo(%s @ %s) shallow OK -> %s", repo, ref, dest)
        return

    logger.warning(
        "clone_repo(%s @ %s) shallow failed (rc=%d); stderr=%s; falling back to full clone",
        repo,
        ref,
        shallow.returncode,
        shallow.stderr.strip(),
    )
    # Scrub the partial clone so the fallback doesn't trip on a
    # non-empty destination. ignore_errors=True because the path may
    # not exist at all (very early shallow failure).
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)

    full = _run_git_step(
        ["git", "clone", "--no-checkout", url, str(dest)],
        timeout=_CLONE_TIMEOUT_SECONDS,
        repo=repo,
        ref=ref,
        what="full clone",
    )
    if full.returncode != 0:
        raise RuntimeError(
            f"clone_repo({repo}@{ref}): both shallow and full clone failed. "
            f"Shallow stderr: {shallow.stderr.strip()!r}. "
            f"Full stderr: {full.stderr.strip()!r}."
        )

    checkout = _run_git_step(
        ["git", "-C", str(dest), "checkout", ref],
        timeout=_OP_TIMEOUT_SECONDS,
        repo=repo,
        ref=ref,
        what="checkout",
    )
    if checkout.returncode != 0:
        raise RuntimeError(
            f"clone_repo({repo}@{ref}): checkout failed after successful full clone. "
            f"Checkout stderr: {checkout.stderr.strip()!r}."
        )
    logger.info("clone_repo(%s @ %s) fallback checkout OK -> %s", repo, ref, dest)
