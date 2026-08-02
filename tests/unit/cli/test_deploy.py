"""Tests for ``troopai deploy`` (init + build; build never invokes docker)."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from troopai.adk.cli import main


def test_init_writes_container_artifacts(tmp_path: Path) -> None:
    result = CliRunner().invoke(main, ["deploy", "init", "--agent", "app:agent", "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "Dockerfile").exists()
    assert (tmp_path / ".dockerignore").exists()
    assert (tmp_path / "requirements.txt").exists()
    assert "0.0.0.0" in (tmp_path / "Dockerfile").read_text(encoding="utf-8")


def test_init_skips_existing_without_force(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text("KEEP", encoding="utf-8")
    result = CliRunner().invoke(main, ["deploy", "init", "--agent", "app:agent", "--dir", str(tmp_path)])
    assert result.exit_code == 0
    assert (tmp_path / "Dockerfile").read_text(encoding="utf-8") == "KEEP"
    assert "skipped" in result.output


def test_init_force_overwrites(tmp_path: Path) -> None:
    (tmp_path / "Dockerfile").write_text("KEEP", encoding="utf-8")
    result = CliRunner().invoke(main, ["deploy", "init", "--agent", "app:agent", "--dir", str(tmp_path), "--force"])
    assert result.exit_code == 0
    assert "KEEP" not in (tmp_path / "Dockerfile").read_text(encoding="utf-8")


def test_init_derives_app_name_from_registry_port_image(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main,
        [
            "deploy",
            "init",
            "--target",
            "helm",
            "--agent",
            "app:agent",
            "--image",
            "registry.io:5000/team/my-agent:v2",
            "--dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    # The app name is the final repository segment ('my-agent'), never the
    # registry host — the registry ':5000' port must not be read as the tag.
    assert (tmp_path / "deploy" / "helm" / "my-agent" / "Chart.yaml").exists()
    assert not (tmp_path / "deploy" / "helm" / "registry.io").exists()


def test_init_invalid_app_name_errors(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main, ["deploy", "init", "--agent", "app:agent", "--image", "Bad_Name:1", "--dir", str(tmp_path)]
    )
    assert result.exit_code == 2
    assert "RFC 1123" in result.output


def test_build_invokes_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from troopai.adk.deploy.commands import RecordingRunner

    rec = RecordingRunner()
    # The `deploy` group shadows the cli.deploy submodule attribute; reach the
    # real module via sys.modules to patch its SubprocessRunner binding.
    monkeypatch.setattr(sys.modules["troopai.adk.cli.deploy"], "SubprocessRunner", lambda: rec)
    result = CliRunner().invoke(
        main, ["deploy", "build", "--agent", "app:agent", "--image", "my-agent:1", "--dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert rec.calls[0][:3] == ["docker", "build", "-t"]
    assert (tmp_path / "Dockerfile").exists()


def test_build_missing_docker_guides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from troopai.adk.deploy.commands import RecordingRunner

    monkeypatch.setattr(
        sys.modules["troopai.adk.cli.deploy"], "SubprocessRunner", lambda: RecordingRunner(available={"git"})
    )
    result = CliRunner().invoke(
        main, ["deploy", "build", "--agent", "app:agent", "--image", "my-agent:1", "--dir", str(tmp_path)]
    )
    assert result.exit_code == 2
    assert "docker" in result.output


def test_deploy_k8s_applies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from troopai.adk.deploy.commands import RecordingRunner

    rec = RecordingRunner()
    monkeypatch.setattr(sys.modules["troopai.adk.cli.deploy"], "SubprocessRunner", lambda: rec)
    result = CliRunner().invoke(
        main, ["deploy", "k8s", "--agent", "app:agent", "--image", "my-agent:1", "--dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "deploy" / "k8s" / "deployment.yaml").exists()
    assert rec.calls[-1][:2] == ["kubectl", "apply"]


def test_deploy_helm_installs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from troopai.adk.deploy.commands import RecordingRunner

    rec = RecordingRunner()
    monkeypatch.setattr(sys.modules["troopai.adk.cli.deploy"], "SubprocessRunner", lambda: rec)
    result = CliRunner().invoke(
        main, ["deploy", "helm", "--agent", "app:agent", "--image", "my-agent:1", "--dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "deploy" / "helm" / "my-agent" / "Chart.yaml").exists()
    assert rec.calls[-1][:3] == ["helm", "upgrade", "--install"]


def test_deploy_gke_requires_cluster(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        main, ["deploy", "gke", "--agent", "app:agent", "--image", "my-agent:1", "--dir", str(tmp_path)]
    )
    assert result.exit_code == 2  # missing required --project/--region/--cluster


def test_deploy_cloud_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from troopai.adk.deploy.commands import RecordingRunner

    rec = RecordingRunner()
    monkeypatch.setattr(sys.modules["troopai.adk.cli.deploy"], "SubprocessRunner", lambda: rec)
    result = CliRunner().invoke(
        main,
        [
            "deploy",
            "cloud-run",
            "--agent",
            "app:agent",
            "--image",
            "gcr.io/p/my-agent:1",
            "--project",
            "p",
            "--region",
            "us-central1",
            "--dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "deploy" / "cloudrun" / "service.yaml").exists()
    assert rec.calls[0][:3] == ["gcloud", "run", "deploy"]


def test_deploy_ecs_registers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from troopai.adk.deploy.commands import RecordingRunner

    rec = RecordingRunner()
    monkeypatch.setattr(sys.modules["troopai.adk.cli.deploy"], "SubprocessRunner", lambda: rec)
    result = CliRunner().invoke(
        main,
        [
            "deploy",
            "ecs",
            "--agent",
            "app:agent",
            "--image",
            "acct/my-agent:1",
            "--region",
            "us-east-1",
            "--execution-role-arn",
            "arn:role",
            "--dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "deploy" / "aws-ecs" / "task-definition.json").exists()
    assert rec.calls[0][:3] == ["aws", "ecs", "register-task-definition"]


def test_deploy_ecs_push_logs_into_ecr(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from troopai.adk.deploy.commands import CommandResult, RecordingRunner

    rec = RecordingRunner(results=[CommandResult(returncode=0, stdout="pw", stderr="")])
    monkeypatch.setattr(sys.modules["troopai.adk.cli.deploy"], "SubprocessRunner", lambda: rec)
    result = CliRunner().invoke(
        main,
        [
            "deploy",
            "ecs",
            "--agent",
            "app:agent",
            "--image",
            "acct.dkr.ecr.us-east-1.amazonaws.com/my-agent:1",
            "--region",
            "us-east-1",
            "--execution-role-arn",
            "arn:role",
            "--push",
            "--dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert rec.calls[0][:3] == ["aws", "ecr", "get-login-password"]


def test_deploy_lambda_updates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from troopai.adk.deploy.commands import RecordingRunner

    rec = RecordingRunner()
    monkeypatch.setattr(sys.modules["troopai.adk.cli.deploy"], "SubprocessRunner", lambda: rec)
    result = CliRunner().invoke(
        main,
        [
            "deploy",
            "lambda",
            "--agent",
            "app:agent",
            "--image",
            "acct/my-agent:1",
            "--region",
            "us-east-1",
            "--dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "deploy" / "aws-lambda" / "Dockerfile").exists()
    assert rec.calls[0][:3] == ["aws", "lambda", "update-function-code"]
