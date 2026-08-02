"""Tests for the rendered Helm chart."""

from __future__ import annotations

import pytest

pytest.importorskip("yaml")
import yaml

from troopai.adk.deploy.context import DeployContext
from troopai.adk.deploy.helm_chart import render_helm_chart, split_image_reference


def _ctx(env_keys: tuple[str, ...] = ()) -> DeployContext:
    return DeployContext(agent_ref="app:agent", image="reg/my-agent:1.2", app_name="my-agent", env_keys=env_keys)


def test_chart_files_present() -> None:
    paths = set(render_helm_chart(_ctx()))
    assert "deploy/helm/my-agent/Chart.yaml" in paths
    assert "deploy/helm/my-agent/values.yaml" in paths
    assert "deploy/helm/my-agent/templates/deployment.yaml" in paths


def test_chart_yaml_names_chart() -> None:
    chart = yaml.safe_load(render_helm_chart(_ctx())["deploy/helm/my-agent/Chart.yaml"])
    assert chart["name"] == "my-agent"


def test_values_split_image_and_secret_env() -> None:
    values = yaml.safe_load(render_helm_chart(_ctx(env_keys=("OPENAI_API_KEY",)))["deploy/helm/my-agent/values.yaml"])
    assert values["image"]["repository"] == "reg/my-agent"
    assert values["image"]["tag"] == "1.2"
    assert values["containerPort"] == 8080
    assert values["secretEnv"] == ["OPENAI_API_KEY"]


def test_deployment_template_has_probes_and_secret_block() -> None:
    tpl = render_helm_chart(_ctx())["deploy/helm/my-agent/templates/deployment.yaml"]
    assert "/healthz" in tpl
    assert "/readyz" in tpl
    assert "secretKeyRef" in tpl


@pytest.mark.parametrize(
    ("image", "expected"),
    [
        ("my-agent", ("my-agent", "latest")),
        ("my-agent:1.2", ("my-agent", "1.2")),
        ("my-agent:", ("my-agent", "latest")),
        ("reg/ns/my-agent", ("reg/ns/my-agent", "latest")),
        ("reg/ns/my-agent:v2", ("reg/ns/my-agent", "v2")),
        ("registry:5000/ns/my-agent", ("registry:5000/ns/my-agent", "latest")),
        ("registry:5000/ns/my-agent:v2", ("registry:5000/ns/my-agent", "v2")),
        ("gcr.io/proj/my-agent:latest", ("gcr.io/proj/my-agent", "latest")),
    ],
)
def test_split_image_reference(image: str, expected: tuple[str, str]) -> None:
    assert split_image_reference(image) == expected


def test_registry_port_image_keeps_repo_and_tag() -> None:
    # A registry port (:5000) precedes the repo path, so the tag is only the
    # segment after the final colon that follows the last '/'.
    ctx = DeployContext(agent_ref="app:agent", image="registry:5000/team/my-agent:v2", app_name="my-agent")
    values = yaml.safe_load(render_helm_chart(ctx)["deploy/helm/my-agent/values.yaml"])
    assert values["image"]["repository"] == "registry:5000/team/my-agent"
    assert values["image"]["tag"] == "v2"
