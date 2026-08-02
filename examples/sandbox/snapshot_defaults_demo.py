"""Inspect the ADK-managed default snapshot directory + stale-payload sweep.

When the developer doesn't configure a ``SnapshotSpec``, the
framework writes snapshots under an OS-conventional state path
(``XDG_STATE_HOME`` on Linux, ``~/Library/Application Support`` on
macOS, ``%LOCALAPPDATA%`` on Windows). A best-effort 30-day TTL
sweep prunes stale ``*.tar`` payloads so the directory doesn't
grow forever.

This script:
1. Shows what path the defaults resolve to on the current host.
2. Materializes the directory.
3. Demonstrates the cleanup sweep on a synthetic old payload.

No external API key required.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import logging
import os
import tempfile
import time
from pathlib import Path

from troopai.adk.sandbox.snapshot_defaults import (
    cleanup_stale_default_local_snapshots,
    default_local_snapshot_base_dir,
    resolve_default_local_snapshot_spec,
)

logger = logging.getLogger(__name__)


def show_host_default() -> None:
    """Print the path the defaults resolve to on THIS host."""
    out = default_local_snapshot_base_dir()
    logger.info("=== Default snapshot directory ===")
    logger.info("Resolved path: %s", out)
    logger.info("Exists?         %s", out.exists())


def show_resolved_spec(tmp_dir: Path) -> None:
    """Materialize the directory and inspect the resulting LocalSnapshotSpec."""
    spec = resolve_default_local_snapshot_spec(
        home=tmp_dir,
        env={"XDG_STATE_HOME": str(tmp_dir / "state")},
        platform="linux",
        os_name="posix",
    )
    logger.info("=== resolve_default_local_snapshot_spec (sandboxed home) ===")
    logger.info("Base path:    %s", spec.base_path)
    logger.info("Exists?       %s", spec.base_path.exists())
    assert spec.base_path.exists()


def demo_cleanup_sweep(tmp_dir: Path) -> None:
    """Drop a synthetic 'old' tar and watch the sweep remove it."""
    base = tmp_dir / "snapshots"
    base.mkdir(parents=True, exist_ok=True)
    old = base / "old.tar"
    old.write_bytes(b"stale payload")
    past = time.time() - 60 * 60 * 24 * 60  # 60 days ago
    os.utime(old, (past, past))

    recent = base / "recent.tar"
    recent.write_bytes(b"fresh payload")

    logger.info("=== Cleanup sweep (max_age = 30 days) ===")
    logger.info("Before: %s", sorted(p.name for p in base.glob("*.tar")))
    cleanup_stale_default_local_snapshots(base, max_age_seconds=60 * 60 * 24 * 30)
    logger.info("After:  %s", sorted(p.name for p in base.glob("*.tar")))
    assert not old.exists()
    assert recent.exists()


def main() -> None:
    show_host_default()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        show_resolved_spec(tmp_path)
        demo_cleanup_sweep(tmp_path)
    logger.info("All snapshot-defaults demos passed.")


if __name__ == "__main__":
    main()
