"""Tests for the Docker deploy target."""

from __future__ import annotations

from pathlib import Path

import pytest

from troopai.adk.deploy.commands import (
    CommandResult,
    DeployCommandFailed,
    DeployToolMissing,
    RecordingRunner,
)
from troopai.adk.deploy.context import DeployContext
from troopai.adk.deploy.targets.docker import DockerTarget
from troopai.adk.deploy.targets.gke import GKETarget
from troopai.adk.deploy.targets.helm import HelmTarget
from troopai.adk.deploy.targets.k8s import K8sTarget


def _ctx() -> DeployContext:
    return DeployContext(agent_ref="app:agent", image="reg/my-agent:1", app_name="my-agent")


def test_generate_returns_three_artifacts() -> None:
    assert set(DockerTarget().generate(_ctx())) == {"Dockerfile", ".dockerignore", "requirements.txt"}


def test_build_runs_docker_build() -> None:
    runner = RecordingRunner()
    DockerTarget().build(_ctx(), runner, context_dir=Path("/tmp/ctx"))
    assert runner.calls == [["docker", "build", "-t", "reg/my-agent:1", "/tmp/ctx"]]


def test_build_push_adds_push_command() -> None:
    runner = RecordingRunner()
    DockerTarget().build(_ctx(), runner, context_dir=Path("/tmp/ctx"), push=True)
    assert runner.calls[-1] == ["docker", "push", "reg/my-agent:1"]


def test_build_requires_docker() -> None:
    with pytest.raises(DeployToolMissing):
        DockerTarget().build(_ctx(), RecordingRunner(available={"git"}), context_dir=Path("/tmp/ctx"))


def test_build_raises_on_failed_command() -> None:
    runner = RecordingRunner(results=[CommandResult(returncode=1, stdout="", stderr="no daemon")])
    with pytest.raises(DeployCommandFailed):
        DockerTarget().build(_ctx(), runner, context_dir=Path("/tmp/ctx"))


def test_k8s_generate_includes_docker_and_manifests() -> None:
    files = K8sTarget().generate(_ctx())
    assert "Dockerfile" in files
    assert "deploy/k8s/deployment.yaml" in files


def test_k8s_apply_runs_kubectl() -> None:
    runner = RecordingRunner()
    K8sTarget().apply(runner, context_dir=Path("/wk"))
    assert runner.calls == [["kubectl", "apply", "-k", "/wk/deploy/k8s"]]


def test_k8s_apply_passes_context() -> None:
    runner = RecordingRunner()
    K8sTarget().apply(runner, context_dir=Path("/wk"), kube_context="prod")
    assert runner.calls[0][:3] == ["kubectl", "--context", "prod"]


def test_gke_deploy_sequence() -> None:
    runner = RecordingRunner()
    GKETarget().deploy(_ctx(), runner, project="p", region="r", cluster="c", context_dir=Path("/wk"))
    assert [call[0] for call in runner.calls] == ["docker", "docker", "gcloud", "kubectl"]
    assert any("get-credentials" in call for call in runner.calls)


def test_helm_install_runs_upgrade() -> None:
    runner = RecordingRunner()
    HelmTarget().install(_ctx(), runner, context_dir=Path("/wk"))
    assert runner.calls[0][:4] == ["helm", "upgrade", "--install", "my-agent"]


def test_helm_install_with_namespace() -> None:
    runner = RecordingRunner()
    HelmTarget().install(_ctx(), runner, context_dir=Path("/wk"), namespace="agents")
    assert "--namespace" in runner.calls[0]
    assert "agents" in runner.calls[0]
