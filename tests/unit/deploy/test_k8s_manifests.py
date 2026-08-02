"""Tests that the rendered Kubernetes manifests are valid and complete."""

from __future__ import annotations

import pytest

pytest.importorskip("yaml")
import yaml

from troopai.adk.deploy.context import DeployContext
from troopai.adk.deploy.k8s_manifests import (
    render_deployment,
    render_k8s_manifests,
    render_secret_example,
)


def _ctx(env_keys: tuple[str, ...] = ()) -> DeployContext:
    return DeployContext(agent_ref="app:agent", image="reg/my-agent:1", app_name="my-agent", env_keys=env_keys)


def test_all_manifests_parse_as_yaml() -> None:
    for content in render_k8s_manifests(_ctx(env_keys=("OPENAI_API_KEY",))).values():
        yaml.safe_load(content)


def test_deployment_has_three_probes_and_grace() -> None:
    spec = yaml.safe_load(render_deployment(_ctx()))["spec"]["template"]["spec"]
    container = spec["containers"][0]
    assert container["startupProbe"]["httpGet"]["path"] == "/healthz"
    assert container["readinessProbe"]["httpGet"]["path"] == "/readyz"
    assert container["livenessProbe"]["httpGet"]["path"] == "/healthz"
    assert spec["terminationGracePeriodSeconds"] == 45


def test_deployment_env_includes_secret_refs() -> None:
    container = yaml.safe_load(render_deployment(_ctx(env_keys=("OPENAI_API_KEY",))))["spec"]["template"]["spec"][
        "containers"
    ][0]
    names = {entry["name"] for entry in container["env"]}
    assert {"PORT", "AGENT_REF", "OPENAI_API_KEY"} <= names
    secret_entry = next(entry for entry in container["env"] if entry["name"] == "OPENAI_API_KEY")
    assert secret_entry["valueFrom"]["secretKeyRef"]["name"] == "my-agent-secrets"


def test_secret_example_lists_env_keys() -> None:
    doc = yaml.safe_load(render_secret_example(_ctx(env_keys=("OPENAI_API_KEY", "ANTHROPIC_API_KEY"))))
    assert set(doc["stringData"]) == {"OPENAI_API_KEY", "ANTHROPIC_API_KEY"}
