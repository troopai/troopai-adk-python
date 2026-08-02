"""Tests for the health/readiness routes."""

from __future__ import annotations

import pytest

pytest.importorskip("starlette")
pytest.importorskip("sse_starlette")
pytest.importorskip("httpx")

from starlette.testclient import TestClient

from troopai.adk.agents.agent import Agent
from troopai.adk.serving import build_app, health_routes


def test_healthz_reports_alive(scripted_agent: Agent[None]) -> None:
    client = TestClient(build_app(scripted_agent, health=True))
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "alive"}


def test_readyz_default_reports_ready(scripted_agent: Agent[None]) -> None:
    client = TestClient(build_app(scripted_agent, health=True))
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ready"}


def test_readyz_probe_false_returns_503(scripted_agent: Agent[None]) -> None:
    async def probe() -> bool:
        return False

    client = TestClient(build_app(scripted_agent, health=True, readiness_probe=probe))
    resp = client.get("/readyz")
    assert resp.status_code == 503
    assert resp.json() == {"status": "not_ready"}


def test_health_routes_returns_two_routes() -> None:
    assert len(health_routes()) == 2


def test_health_routes_rejects_empty_path() -> None:
    with pytest.raises(ValueError):
        health_routes(liveness_path="")
