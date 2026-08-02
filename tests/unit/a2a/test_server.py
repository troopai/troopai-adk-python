"""Tests for ``A2AServer`` config object + ``build_starlette_app`` factory."""

import dataclasses
import logging

import pytest

# Skip module if optional `a2a` extra missing.
pytest.importorskip("a2a.types")
pytest.importorskip("starlette")

from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface
from starlette.applications import Starlette

from troopai.adk.a2a import A2AServer, build_starlette_app
from troopai.adk.agents import Agent


@pytest.fixture
def basic_agent() -> Agent[None]:
    return Agent(name="test_agent", system_prompt="Be helpful.")


@pytest.fixture
def basic_card() -> AgentCard:
    return AgentCard(
        name="test",
        description="Test agent.",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(
                url="http://localhost:8080",
                protocol_binding="JSONRPC",
                protocol_version="1.0",
            ),
        ],
        capabilities=AgentCapabilities(streaming=True),
    )


class TestA2AServerConfig:
    def test_construction(self, basic_agent: Agent[None], basic_card: AgentCard) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card)
        assert server.agent is basic_agent
        assert server.agent_card is basic_card
        assert server.task_store is None
        assert server.max_turns == 10
        assert server.run_config is None
        assert server.rpc_url == "/"

    def test_is_frozen(self, basic_agent: Agent[None], basic_card: AgentCard) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card)
        with pytest.raises(dataclasses.FrozenInstanceError):
            server.max_turns = 99  # type: ignore[misc]

    def test_max_turns_override(self, basic_agent: Agent[None], basic_card: AgentCard) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card, max_turns=42)
        assert server.max_turns == 42

    def test_rpc_url_override(self, basic_agent: Agent[None], basic_card: AgentCard) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card, rpc_url="/a2a")
        assert server.rpc_url == "/a2a"

    def test_custom_task_store_honoured(self, basic_agent: Agent[None], basic_card: AgentCard) -> None:
        custom_store = InMemoryTaskStore()
        server = A2AServer(agent=basic_agent, agent_card=basic_card, task_store=custom_store)
        assert server.task_store is custom_store


class TestBuildStarletteApp:
    def test_returns_starlette_instance(self, basic_agent: Agent[None], basic_card: AgentCard) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card)
        app = build_starlette_app(server)
        assert isinstance(app, Starlette)

    def test_card_route_at_well_known_url(self, basic_agent: Agent[None], basic_card: AgentCard) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card)
        app = build_starlette_app(server)
        # Per A2A spec, the card MUST be at /.well-known/agent-card.json.
        card_paths = [getattr(r, "path", None) for r in app.routes]
        assert "/.well-known/agent-card.json" in card_paths

    def test_jsonrpc_route_at_default_root(self, basic_agent: Agent[None], basic_card: AgentCard) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card)
        app = build_starlette_app(server)
        rpc_paths = [getattr(r, "path", None) for r in app.routes]
        assert "/" in rpc_paths

    def test_jsonrpc_route_honours_custom_rpc_url(self, basic_agent: Agent[None], basic_card: AgentCard) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card, rpc_url="/a2a")
        app = build_starlette_app(server)
        rpc_paths = [getattr(r, "path", None) for r in app.routes]
        assert "/a2a" in rpc_paths
        # Default "/" still NOT present when custom rpc_url is set.
        assert "/" not in rpc_paths

    def test_warns_when_default_inmemory_task_store(
        self,
        basic_agent: Agent[None],
        basic_card: AgentCard,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Production callers MUST supply a persistent TaskStore — the
        # factory logs a WARNING when falling back to InMemoryTaskStore
        # so the choice is visible in deployment logs.
        server = A2AServer(agent=basic_agent, agent_card=basic_card)
        with caplog.at_level(logging.WARNING, logger="troopai.adk.a2a.app_factory"):
            build_starlette_app(server)
        assert any("InMemoryTaskStore" in r.getMessage() for r in caplog.records)

    def test_no_warning_when_custom_task_store(
        self,
        basic_agent: Agent[None],
        basic_card: AgentCard,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        custom = InMemoryTaskStore()
        server = A2AServer(agent=basic_agent, agent_card=basic_card, task_store=custom)
        with caplog.at_level(logging.WARNING, logger="troopai.adk.a2a.app_factory"):
            build_starlette_app(server)
        assert not any("InMemoryTaskStore" in r.getMessage() for r in caplog.records)
