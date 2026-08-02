"""Tests that rendered artifacts satisfy the container contract."""

from __future__ import annotations

from troopai.adk.deploy.context import DeployContext
from troopai.adk.deploy.templates import (
    render_dockerfile,
    render_dockerignore,
    render_requirements,
)


def _ctx() -> DeployContext:
    return DeployContext(
        agent_ref="app:agent",
        image="my-agent:latest",
        app_name="my-agent",
        port=9000,
        python_version="3.13",
        extras="serve",
    )


def test_dockerfile_satisfies_container_contract() -> None:
    text = render_dockerfile(_ctx())
    assert "FROM python:3.13-slim" in text
    assert "--host 0.0.0.0" in text
    assert "$PORT" in text  # shell var preserved by safe_substitute
    assert "EXPOSE 9000" in text
    assert "AGENT_REF=app:agent" in text
    assert "USER appuser" in text


def test_dockerfile_references_extras() -> None:
    assert "troopai-adk-python[serve]" in render_dockerfile(_ctx())


def test_dockerignore_excludes_vcs_and_secrets() -> None:
    text = render_dockerignore()
    assert ".git" in text
    assert ".env" in text


def test_requirements_pins_extras() -> None:
    assert "troopai-adk-python[serve]" in render_requirements(_ctx())
