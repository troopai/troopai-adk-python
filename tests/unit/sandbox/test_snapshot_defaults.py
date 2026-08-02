"""Tests for ``troopai.adk.sandbox.snapshot_defaults``."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from troopai.adk.sandbox.snapshot_defaults import (
    cleanup_stale_default_local_snapshots,
    default_local_snapshot_base_dir,
    resolve_default_local_snapshot_spec,
)


class TestDefaultLocalSnapshotBaseDir:
    def test_linux_xdg_state_home_honored(self, tmp_path: Path) -> None:
        out = default_local_snapshot_base_dir(
            home=tmp_path,
            env={"XDG_STATE_HOME": str(tmp_path / "custom_state")},
            platform="linux",
            os_name="posix",
        )
        assert out.is_relative_to(tmp_path / "custom_state")
        assert "troopai-adk-python" in str(out)

    def test_linux_falls_back_to_home_local_state(self, tmp_path: Path) -> None:
        out = default_local_snapshot_base_dir(
            home=tmp_path,
            env={},
            platform="linux",
            os_name="posix",
        )
        assert out.is_relative_to(tmp_path / ".local" / "state")

    def test_darwin_uses_application_support(self, tmp_path: Path) -> None:
        out = default_local_snapshot_base_dir(
            home=tmp_path,
            env={},
            platform="darwin",
            os_name="posix",
        )
        assert out.is_relative_to(tmp_path / "Library" / "Application Support")

    def test_windows_localappdata(self, tmp_path: Path) -> None:
        out = default_local_snapshot_base_dir(
            home=tmp_path,
            env={"LOCALAPPDATA": "C:\\Users\\me\\AppData\\Local"},
            platform="win32",
            os_name="nt",
        )
        assert str(out).startswith("C:")

    def test_windows_fallback_when_env_missing(self, tmp_path: Path) -> None:
        out = default_local_snapshot_base_dir(
            home=tmp_path,
            env={},
            platform="win32",
            os_name="nt",
        )
        assert out.is_relative_to(tmp_path / "AppData" / "Local")

    def test_empty_env_does_not_read_real_process_environment_linux(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # An empty mapping means "empty environment" and is a distinct,
        # valid value from None (the "not provided" sentinel). A real
        # ``XDG_STATE_HOME`` in the process environment MUST NOT leak in.
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "leaked_state"))
        out = default_local_snapshot_base_dir(
            home=tmp_path,
            env={},
            platform="linux",
            os_name="posix",
        )
        assert out.is_relative_to(tmp_path / ".local" / "state")
        assert not out.is_relative_to(tmp_path / "leaked_state")

    def test_empty_env_does_not_read_real_process_environment_windows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # On Windows a real ``LOCALAPPDATA`` is almost always present;
        # an explicit empty env must still produce the home-based fallback.
        monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\real\\AppData\\Local")
        out = default_local_snapshot_base_dir(
            home=tmp_path,
            env={},
            platform="win32",
            os_name="nt",
        )
        assert out.is_relative_to(tmp_path / "AppData" / "Local")
        assert not str(out).startswith("C:")


class TestCleanupStaleDefaultLocalSnapshots:
    def test_removes_old_tar(self, tmp_path: Path) -> None:
        old = tmp_path / "old.tar"
        old.write_bytes(b"x")
        past = time.time() - 365 * 24 * 60 * 60  # 1 year ago
        import os as _os

        _os.utime(old, (past, past))
        cleanup_stale_default_local_snapshots(tmp_path, max_age_seconds=24 * 60 * 60)
        assert not old.exists()

    def test_keeps_recent_tar(self, tmp_path: Path) -> None:
        recent = tmp_path / "recent.tar"
        recent.write_bytes(b"x")
        cleanup_stale_default_local_snapshots(tmp_path, max_age_seconds=365 * 24 * 60 * 60)
        assert recent.exists()

    def test_handles_missing_directory(self, tmp_path: Path) -> None:
        # MUST not raise.
        cleanup_stale_default_local_snapshots(tmp_path / "doesnotexist")

    def test_ignores_non_tar_files(self, tmp_path: Path) -> None:
        other = tmp_path / "old.log"
        other.write_bytes(b"x")
        past = time.time() - 365 * 24 * 60 * 60
        import os as _os

        _os.utime(other, (past, past))
        cleanup_stale_default_local_snapshots(tmp_path, max_age_seconds=24 * 60 * 60)
        assert other.exists()

    def test_skips_when_max_age_negative(self, tmp_path: Path) -> None:
        old = tmp_path / "old.tar"
        old.write_bytes(b"x")
        cleanup_stale_default_local_snapshots(tmp_path, max_age_seconds=-1)
        assert old.exists()


class TestResolveDefaultLocalSnapshotSpec:
    def test_creates_directory(self, tmp_path: Path) -> None:
        spec = resolve_default_local_snapshot_spec(
            home=tmp_path,
            env={"XDG_STATE_HOME": str(tmp_path / "state")},
            platform="linux",
            os_name="posix",
        )
        assert spec.base_path.exists()
        assert spec.base_path.is_dir()
