"""Tests for the Cloud Run target."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("yaml")
import yaml

from troopai.adk.deploy.commands import RecordingRunner
from troopai.adk.deploy.context import DeployContext
from troopai.adk.deploy.targets.cloudrun import CloudRunTarget


def _ctx(env_keys: tuple[str, ...] = ()) -> DeployContext:
    return DeployContext(
        agent_ref="app:agent", image="gcr.io/p/my-agent:1", app_name="my-agent", port=8080, env_keys=env_keys
    )


def test_service_yaml_parses_and_sets_port() -> None:
    files = CloudRunTarget().generate(_ctx())
    container = yaml.safe_load(files["deploy/cloudrun/service.yaml"])["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == "gcr.io/p/my-agent:1"
    assert container["ports"][0]["containerPort"] == 8080


def test_generate_includes_dockerfile() -> None:
    assert "Dockerfile" in CloudRunTarget().generate(_ctx())


def test_deploy_runs_gcloud_run_deploy() -> None:
    runner = RecordingRunner()
    CloudRunTarget().deploy(_ctx(), runner, project="p", region="us-central1", source_dir=Path("/wk"))
    call = runner.calls[0]
    assert call[:4] == ["gcloud", "run", "deploy", "my-agent"]
    assert "--source" in call
    assert "--no-allow-unauthenticated" in call
    assert "--port" in call


def test_deploy_allow_unauth_secrets_and_min_instances() -> None:
    runner = RecordingRunner()
    CloudRunTarget().deploy(
        _ctx(env_keys=("OPENAI_API_KEY",)),
        runner,
        project="p",
        region="r",
        source_dir=Path("/wk"),
        allow_unauthenticated=True,
        min_instances=2,
    )
    call = runner.calls[0]
    assert "--allow-unauthenticated" in call
    assert "--set-secrets" in call
    assert "OPENAI_API_KEY=OPENAI_API_KEY:latest" in call
    assert "--min-instances" in call
    assert "2" in call
