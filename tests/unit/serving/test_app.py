"""Tests for ``build_app`` surface composition."""

from __future__ import annotations

import pytest

pytest.importorskip("starlette")
pytest.importorskip("sse_starlette")

from troopai.adk.agents.agent import Agent
from troopai.adk.serving import build_app


def _route_paths(app: object) -> set[str | None]:
    return {getattr(route, "path", None) for route in app.routes}  # type: ignore[attr-defined]


def test_build_app_requires_a_surface(scripted_agent: Agent[None]) -> None:
    with pytest.raises(ValueError):
        build_app(scripted_agent)


def test_health_only_has_no_run_route(scripted_agent: Agent[None]) -> None:
    paths = _route_paths(build_app(scripted_agent, health=True))
    assert "/healthz" in paths
    assert "/run" not in paths


def test_rest_and_health_compose(scripted_agent: Agent[None]) -> None:
    paths = _route_paths(build_app(scripted_agent, rest=True, health=True))
    assert {"/run", "/run_sse", "/healthz", "/readyz"} <= paths


def test_a2a_surface_mounts_discovery_route(scripted_agent: Agent[None]) -> None:
    pytest.importorskip("a2a.types")
    from a2a.types import AgentCapabilities, AgentCard, AgentInterface

    from troopai.adk.a2a import A2AServer

    card = AgentCard(
        name="support",
        description="Test agent.",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(url="http://localhost:8080", protocol_binding="JSONRPC", protocol_version="1.0"),
        ],
        capabilities=AgentCapabilities(streaming=True),
    )
    server = A2AServer(agent=scripted_agent, agent_card=card)
    paths = _route_paths(build_app(scripted_agent, a2a_server=server))
    assert "/.well-known/agent-card.json" in paths
