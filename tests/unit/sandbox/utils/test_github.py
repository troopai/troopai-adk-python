"""Tests for ``troopai.adk.sandbox.utils.github``."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from troopai.adk.sandbox.utils.github import _run_git, clone_repo, ensure_git_available


def _completed(returncode: int, *, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["git"], returncode=returncode, stdout="", stderr=stderr)


class TestEnsureGitAvailable:
    def test_passes_when_git_present(self) -> None:
        with patch("troopai.adk.sandbox.utils.github.shutil.which", return_value="/usr/bin/git"):
            ensure_git_available()  # no raise

    def test_raises_when_git_absent(self) -> None:
        with (
            patch("troopai.adk.sandbox.utils.github.shutil.which", return_value=None),
            pytest.raises(RuntimeError, match="git is required"),
        ):
            ensure_git_available()


class TestCloneRepo:
    def test_shallow_clone_path(self, tmp_path: Path) -> None:
        dest = tmp_path / "repo"
        with (
            patch("troopai.adk.sandbox.utils.github.shutil.which", return_value="/usr/bin/git"),
            patch(
                "troopai.adk.sandbox.utils.github._run_git",
                return_value=_completed(0),
            ) as run,
        ):
            clone_repo(repo="o/r", ref="main", dest=dest)
        assert run.call_count == 1
        cmd = run.call_args[0][0]
        assert "--depth" in cmd
        assert "1" in cmd
        assert "--branch" in cmd
        assert "main" in cmd

    def test_falls_back_to_full_clone_on_shallow_failure(self, tmp_path: Path) -> None:
        dest = tmp_path / "repo"
        rtmpath = MagicMock()
        with (
            patch("troopai.adk.sandbox.utils.github.shutil.which", return_value="/usr/bin/git"),
            patch("troopai.adk.sandbox.utils.github.shutil.rmtree", rtmpath),
            patch(
                "troopai.adk.sandbox.utils.github._run_git",
                side_effect=[
                    _completed(128, stderr="fatal: branch not a ref"),  # shallow fail
                    _completed(0),  # full clone OK
                    _completed(0),  # checkout OK
                ],
            ) as run,
        ):
            clone_repo(repo="o/r", ref="abc123sha", dest=dest)
        assert run.call_count == 3
        full_clone_cmd = run.call_args_list[1][0][0]
        assert "--no-checkout" in full_clone_cmd
        checkout_cmd = run.call_args_list[2][0][0]
        assert "checkout" in checkout_cmd

    def test_raises_when_full_clone_fails(self, tmp_path: Path) -> None:
        dest = tmp_path / "repo"
        with (
            patch("troopai.adk.sandbox.utils.github.shutil.which", return_value="/usr/bin/git"),
            patch("troopai.adk.sandbox.utils.github.shutil.rmtree"),
            patch(
                "troopai.adk.sandbox.utils.github._run_git",
                side_effect=[
                    _completed(128, stderr="shallow no"),
                    _completed(128, stderr="full no either"),
                ],
            ),
            pytest.raises(RuntimeError) as exc_info,
        ):
            clone_repo(repo="o/r", ref="abc", dest=dest)
        msg = str(exc_info.value)
        assert "shallow no" in msg
        assert "full no either" in msg

    def test_raises_when_checkout_fails(self, tmp_path: Path) -> None:
        dest = tmp_path / "repo"
        with (
            patch("troopai.adk.sandbox.utils.github.shutil.which", return_value="/usr/bin/git"),
            patch("troopai.adk.sandbox.utils.github.shutil.rmtree"),
            patch(
                "troopai.adk.sandbox.utils.github._run_git",
                side_effect=[
                    _completed(128, stderr="shallow no"),
                    _completed(0),
                    _completed(1, stderr="unknown ref"),
                ],
            ),
            pytest.raises(RuntimeError, match="unknown ref"),
        ):
            clone_repo(repo="o/r", ref="abc", dest=dest)

    def test_propagates_git_missing(self, tmp_path: Path) -> None:
        with (
            patch("troopai.adk.sandbox.utils.github.shutil.which", return_value=None),
            pytest.raises(RuntimeError, match="git is required"),
        ):
            clone_repo(repo="o/r", ref="main", dest=tmp_path / "x")


class TestRunGitNonInteractive:
    """``_run_git`` must fail fast (no credential-prompt hang)."""

    def test_disables_prompts_seals_stdin_and_bounds_timeout(self) -> None:
        captured: dict[str, object] = {}

        def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            captured.update(kwargs)
            return _completed(0)

        with patch("troopai.adk.sandbox.utils.github.subprocess.run", side_effect=fake_run):
            _run_git(["git", "--version"], timeout=12.0)

        # A missing/private HTTPS clone hangs on a credential prompt unless
        # stdin is sealed, prompts are disabled, and a wall-clock cap applies.
        assert captured["stdin"] is subprocess.DEVNULL
        assert captured["timeout"] == 12.0
        env = captured["env"]
        assert isinstance(env, dict)
        assert env["GIT_TERMINAL_PROMPT"] == "0"
        assert env["GIT_ASKPASS"] == "true"

    def test_clone_passes_timeout_through(self, tmp_path: Path) -> None:
        captured: dict[str, object] = {}

        def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            captured.update(kwargs)
            return _completed(0)

        with (
            patch("troopai.adk.sandbox.utils.github.shutil.which", return_value="/usr/bin/git"),
            patch("troopai.adk.sandbox.utils.github.subprocess.run", side_effect=fake_run),
        ):
            clone_repo(repo="o/r", ref="main", dest=tmp_path / "repo")

        # The shallow clone (only call on the happy path) must be bounded.
        assert isinstance(captured["timeout"], float)
        assert captured["timeout"] > 0


class TestCloneRepoTimeout:
    """A hung git step must surface as ``RuntimeError``, never propagate raw."""

    def test_shallow_clone_timeout_becomes_runtimeerror(self, tmp_path: Path) -> None:
        with (
            patch("troopai.adk.sandbox.utils.github.shutil.which", return_value="/usr/bin/git"),
            patch(
                "troopai.adk.sandbox.utils.github._run_git",
                side_effect=subprocess.TimeoutExpired(cmd="git clone", timeout=300.0),
            ),
            pytest.raises(RuntimeError, match="shallow clone timed out"),
        ):
            clone_repo(repo="o/r", ref="main", dest=tmp_path / "repo")

    def test_checkout_timeout_becomes_runtimeerror(self, tmp_path: Path) -> None:
        with (
            patch("troopai.adk.sandbox.utils.github.shutil.which", return_value="/usr/bin/git"),
            patch("troopai.adk.sandbox.utils.github.shutil.rmtree"),
            patch(
                "troopai.adk.sandbox.utils.github._run_git",
                side_effect=[
                    _completed(128, stderr="shallow no"),  # shallow fail -> fallback
                    _completed(0),  # full clone OK
                    subprocess.TimeoutExpired(cmd="git checkout", timeout=60.0),  # checkout hangs
                ],
            ),
            pytest.raises(RuntimeError, match="checkout timed out"),
        ):
            clone_repo(repo="o/r", ref="abc", dest=tmp_path / "repo")
