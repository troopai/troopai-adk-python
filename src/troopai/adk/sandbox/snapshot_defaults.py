"""Defaults for ADK-managed local snapshot storage.

When the developer doesn't specify a ``SnapshotSpec``, the
framework picks a per-OS-conventional path under the user's
state directory (XDG on Linux, Application Support on macOS,
``LOCALAPPDATA`` on Windows). A best-effort TTL sweep deletes
stale tar files so the directory doesn't grow forever.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import time
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath

from troopai.adk.types.sandbox.snapshot import LocalSnapshotSpec

__all__ = [
    "cleanup_stale_default_local_snapshots",
    "default_local_snapshot_base_dir",
    "resolve_default_local_snapshot_spec",
]

logger = logging.getLogger(__name__)

_DEFAULT_LOCAL_SNAPSHOT_TTL_SECONDS: int = 60 * 60 * 24 * 30  # 30 days
_DEFAULT_LOCAL_SNAPSHOT_SUBDIR: Path = Path("troopai-adk-python") / "sandbox" / "snapshots"


def _first_absolute_windows_env_path(env: Mapping[str, str], *names: str) -> Path | None:
    """Return the first env var whose value is a Windows-absolute path."""
    for name in names:
        value = env.get(name)
        if value is None or len(value) == 0:
            continue
        if PureWindowsPath(value).is_absolute():
            return Path(value)
    return None


def default_local_snapshot_base_dir(
    *,
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
    platform: str | None = None,
    os_name: str | None = None,
) -> Path:
    """Return the default snapshot directory for the current OS.

    Args:
        home: Override for ``Path.home()`` (test seam).
        env: Override for ``os.environ`` (test seam).
        platform: Override for ``sys.platform`` (test seam).
        os_name: Override for ``os.name`` (test seam).

    Returns:
        Base path under the OS-conventional state directory.
    """
    resolved_home = home if home is not None else Path.home()
    resolved_env = env if env is not None else os.environ
    resolved_platform = platform if platform is not None else sys.platform
    resolved_os_name = os_name if os_name is not None else os.name

    if resolved_platform == "darwin":
        base = resolved_home / "Library" / "Application Support"
    elif resolved_os_name == "nt":
        env_base = _first_absolute_windows_env_path(
            resolved_env,
            "LOCALAPPDATA",
            "APPDATA",
        )
        base = env_base if env_base is not None else resolved_home / "AppData" / "Local"
    else:
        xdg_state_home = resolved_env.get("XDG_STATE_HOME")
        base = (
            Path(xdg_state_home)
            if xdg_state_home is not None and len(xdg_state_home) > 0
            else resolved_home / ".local" / "state"
        )

    return base / _DEFAULT_LOCAL_SNAPSHOT_SUBDIR


def cleanup_stale_default_local_snapshots(
    base_path: Path,
    *,
    now: float | None = None,
    max_age_seconds: int = _DEFAULT_LOCAL_SNAPSHOT_TTL_SECONDS,
) -> None:
    """Delete ``*.tar`` files in ``base_path`` older than ``max_age_seconds``.

    This is intentionally scoped to the ADK-managed default directory.
    Snapshots created by explicit developer configuration are NOT
    swept — pause/resume flows may still need them. If your code
    needs broader cleanup, write it as a separate opt-in path that
    can also account for backend-specific remote artifacts.
    """
    if max_age_seconds < 0 or not base_path.exists():
        return

    cutoff = (time.time() if now is None else now) - max_age_seconds
    try:
        candidates = list(base_path.glob("*.tar"))
    except OSError as exc:
        logger.debug("cleanup_stale_default_local_snapshots: glob failed for %s: %s", base_path, exc)
        return

    for candidate in candidates:
        try:
            if not candidate.is_file():
                continue
            if candidate.stat().st_mtime >= cutoff:
                continue
            candidate.unlink(missing_ok=True)
            logger.debug("cleanup_stale_default_local_snapshots: removed stale %s", candidate)
        except OSError as exc:
            logger.debug(
                "cleanup_stale_default_local_snapshots: skip %s due to OS error: %s",
                candidate,
                exc,
            )
            continue


def resolve_default_local_snapshot_spec(
    *,
    home: Path | None = None,
    env: Mapping[str, str] | None = None,
    platform: str | None = None,
    os_name: str | None = None,
) -> LocalSnapshotSpec:
    """Build a ``LocalSnapshotSpec`` rooted at the default directory.

    Creates the directory if missing and tightens its mode to
    ``0o700`` on POSIX so other local users can't read snapshot
    payloads.
    """
    base_path = default_local_snapshot_base_dir(
        home=home,
        env=env,
        platform=platform,
        os_name=os_name,
    )
    base_path.mkdir(parents=True, exist_ok=True, mode=0o700)
    resolved_os_name = os_name if os_name is not None else os.name
    if resolved_os_name != "nt":
        with contextlib.suppress(OSError):
            base_path.chmod(0o700)
    return LocalSnapshotSpec(base_path=base_path)
