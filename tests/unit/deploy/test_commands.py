"""Tests for the command-runner seam."""

from __future__ import annotations

import sys

import pytest

from troopai.adk.deploy.commands import (
    CommandResult,
    DeployCommandFailed,
    DeployToolMissing,
    RecordingRunner,
    SubprocessRunner,
    require_tool,
    run_checked,
)


def test_recording_runner_records_calls() -> None:
    runner = RecordingRunner()
    runner.run(["docker", "build", "."])
    assert runner.calls == [["docker", "build", "."]]


def test_recording_runner_which() -> None:
    assert RecordingRunner().which("docker") is True
    assert RecordingRunner(available={"git"}).which("docker") is False


def test_require_tool_raises_when_missing() -> None:
    with pytest.raises(DeployToolMissing):
        require_tool(RecordingRunner(available={"git"}), "docker")


def test_run_checked_raises_on_nonzero() -> None:
    runner = RecordingRunner(results=[CommandResult(returncode=1, stdout="", stderr="boom")])
    with pytest.raises(DeployCommandFailed):
        run_checked(runner, ["docker", "build", "."])


def test_subprocess_runner_which_false_for_unknown() -> None:
    assert SubprocessRunner().which("this-tool-does-not-exist-zzz") is False


def test_subprocess_runner_runs_real_process() -> None:
    result = SubprocessRunner().run([sys.executable, "-c", "import sys; sys.stdout.write('ok')"])
    assert result.returncode == 0
    assert result.stdout == "ok"


def test_subprocess_runner_rejects_empty_args() -> None:
    with pytest.raises(ValueError):
        SubprocessRunner().run([])
