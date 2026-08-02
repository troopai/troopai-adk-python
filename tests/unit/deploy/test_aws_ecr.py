"""Tests for ECR login + push helpers."""

from __future__ import annotations

from pathlib import Path

from troopai.adk.deploy.aws_ecr import build_and_push_to_ecr, ecr_login, ecr_registry
from troopai.adk.deploy.commands import CommandResult, RecordingRunner
from troopai.adk.deploy.context import DeployContext


def _ctx() -> DeployContext:
    return DeployContext(
        agent_ref="app:agent", image="acct.dkr.ecr.us-east-1.amazonaws.com/my-agent:1", app_name="my-agent"
    )


def test_ecr_registry_extracts_host() -> None:
    assert ecr_registry("acct.dkr.ecr.r.amazonaws.com/ns/img:1") == "acct.dkr.ecr.r.amazonaws.com"


def test_ecr_login_pipes_password_via_stdin_not_argv() -> None:
    runner = RecordingRunner(results=[CommandResult(returncode=0, stdout="secret-pw\n", stderr="")])
    ecr_login(runner, "acct.dkr.ecr.us-east-1.amazonaws.com/my-agent:1", region="us-east-1")
    assert runner.calls[0] == ["aws", "ecr", "get-login-password", "--region", "us-east-1"]
    assert runner.calls[1] == [
        "docker",
        "login",
        "--username",
        "AWS",
        "--password-stdin",
        "acct.dkr.ecr.us-east-1.amazonaws.com",
    ]
    # The password is fed to stdin, never placed in argv.
    assert runner.inputs[1] == "secret-pw"
    assert "secret-pw" not in runner.calls[1]


def test_build_and_push_logs_in_then_builds_and_pushes() -> None:
    runner = RecordingRunner(results=[CommandResult(returncode=0, stdout="pw", stderr="")])
    build_and_push_to_ecr(_ctx(), runner, region="us-east-1", context_dir=Path("/wk"))
    assert runner.calls[0][:3] == ["aws", "ecr", "get-login-password"]
    assert runner.calls[1][:2] == ["docker", "login"]
    assert runner.calls[2][:2] == ["docker", "build"]
    assert runner.calls[3][:2] == ["docker", "push"]
