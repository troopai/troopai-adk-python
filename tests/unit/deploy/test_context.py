"""Tests for ``DeployContext`` validation."""

from __future__ import annotations

import pytest

from troopai.adk.deploy.context import DeployContext


def test_valid_context_defaults() -> None:
    ctx = DeployContext(agent_ref="app:agent", image="my-agent:latest", app_name="my-agent")
    assert ctx.port == 8080
    assert ctx.extras == "serve,a2a"
    assert ctx.python_version == "3.12"
    assert ctx.env_keys == ()


def test_empty_agent_ref_rejected() -> None:
    with pytest.raises(ValueError):
        DeployContext(agent_ref="", image="x", app_name="x")


def test_empty_image_rejected() -> None:
    with pytest.raises(ValueError):
        DeployContext(agent_ref="a:b", image="", app_name="x")


def test_port_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        DeployContext(agent_ref="a:b", image="x", app_name="x", port=70000)


def test_non_dns_app_name_rejected() -> None:
    with pytest.raises(ValueError):
        DeployContext(agent_ref="a:b", image="x", app_name="My_Agent")
