"""Tests for Features 1 & 2: compat_earlier_protocol and extended-card fields.

Feature 1: ``A2AServer.compat_earlier_protocol`` threads ``enable_v0_3_compat``
into ``create_jsonrpc_routes``.

Feature 2: ``A2AServer.extended_agent_card``, ``extended_card_modifier``, and
``card_modifier`` thread into ``DefaultRequestHandler`` and
``create_agent_card_routes``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("a2a.types")
pytest.importorskip("starlette")

from a2a.types import AgentCapabilities, AgentCard, AgentInterface

from troopai.adk.a2a import A2AServer, build_starlette_app
from troopai.adk.agents import Agent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


@pytest.fixture
def extended_card(basic_card: AgentCard) -> AgentCard:
    return AgentCard(
        name="test-extended",
        description="Test agent (extended).",
        version="1.0.0",
        supported_interfaces=basic_card.supported_interfaces,
        capabilities=AgentCapabilities(streaming=True),
    )


# ---------------------------------------------------------------------------
# Feature 1: compat_earlier_protocol defaults
# ---------------------------------------------------------------------------


class TestCompatEarlierProtocolDefault:
    def test_default_is_false(self, basic_agent: Agent[None], basic_card: AgentCard) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card)
        assert server.compat_earlier_protocol is False

    def test_explicit_true(self, basic_agent: Agent[None], basic_card: AgentCard) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card, compat_earlier_protocol=True)
        assert server.compat_earlier_protocol is True

    def test_frozen_cannot_mutate(self, basic_agent: Agent[None], basic_card: AgentCard) -> None:
        import dataclasses

        server = A2AServer(agent=basic_agent, agent_card=basic_card)
        with pytest.raises(dataclasses.FrozenInstanceError):
            server.compat_earlier_protocol = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Feature 1: compat_earlier_protocol reaches create_jsonrpc_routes
# ---------------------------------------------------------------------------


class TestCompatEarlierProtocolPassthrough:
    def test_false_passes_enable_v0_3_compat_false(
        self,
        basic_agent: Agent[None],
        basic_card: AgentCard,
    ) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card, compat_earlier_protocol=False)
        with patch("troopai.adk.a2a.app_factory.create_jsonrpc_routes") as mock_routes:
            mock_routes.return_value = []
            build_starlette_app(server)
        mock_routes.assert_called_once()
        _, kwargs = mock_routes.call_args
        assert kwargs.get("enable_v0_3_compat") is False

    def test_true_passes_enable_v0_3_compat_true(
        self,
        basic_agent: Agent[None],
        basic_card: AgentCard,
    ) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card, compat_earlier_protocol=True)
        with patch("troopai.adk.a2a.app_factory.create_jsonrpc_routes") as mock_routes:
            mock_routes.return_value = []
            build_starlette_app(server)
        mock_routes.assert_called_once()
        _, kwargs = mock_routes.call_args
        assert kwargs.get("enable_v0_3_compat") is True


# ---------------------------------------------------------------------------
# Feature 2: extended_agent_card and extended_card_modifier defaults
# ---------------------------------------------------------------------------


class TestExtendedCardDefaults:
    def test_extended_agent_card_default_none(self, basic_agent: Agent[None], basic_card: AgentCard) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card)
        assert server.extended_agent_card is None

    def test_extended_card_modifier_default_none(self, basic_agent: Agent[None], basic_card: AgentCard) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card)
        assert server.extended_card_modifier is None

    def test_card_modifier_default_none(self, basic_agent: Agent[None], basic_card: AgentCard) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card)
        assert server.card_modifier is None


# ---------------------------------------------------------------------------
# Feature 2: extended fields reach DefaultRequestHandler
# ---------------------------------------------------------------------------


class TestExtendedCardPassthrough:
    def test_extended_agent_card_reaches_handler(
        self,
        basic_agent: Agent[None],
        basic_card: AgentCard,
        extended_card: AgentCard,
    ) -> None:
        server = A2AServer(
            agent=basic_agent,
            agent_card=basic_card,
            extended_agent_card=extended_card,
        )
        with patch("troopai.adk.a2a.app_factory.DefaultRequestHandler") as mock_handler:
            mock_handler.return_value = MagicMock()
            build_starlette_app(server)
        mock_handler.assert_called_once()
        _, kwargs = mock_handler.call_args
        assert kwargs.get("extended_agent_card") is extended_card

    def test_extended_card_modifier_reaches_handler(
        self,
        basic_agent: Agent[None],
        basic_card: AgentCard,
    ) -> None:
        async def my_modifier(card: AgentCard, ctx: object) -> AgentCard:
            return card

        server = A2AServer(
            agent=basic_agent,
            agent_card=basic_card,
            extended_card_modifier=my_modifier,
        )
        with patch("troopai.adk.a2a.app_factory.DefaultRequestHandler") as mock_handler:
            mock_handler.return_value = MagicMock()
            build_starlette_app(server)
        mock_handler.assert_called_once()
        _, kwargs = mock_handler.call_args
        assert kwargs.get("extended_card_modifier") is my_modifier

    def test_none_extended_card_reaches_handler_as_none(
        self,
        basic_agent: Agent[None],
        basic_card: AgentCard,
    ) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card)
        with patch("troopai.adk.a2a.app_factory.DefaultRequestHandler") as mock_handler:
            mock_handler.return_value = MagicMock()
            build_starlette_app(server)
        mock_handler.assert_called_once()
        _, kwargs = mock_handler.call_args
        assert kwargs.get("extended_agent_card") is None
        assert kwargs.get("extended_card_modifier") is None

    def test_card_modifier_reaches_agent_card_routes(
        self,
        basic_agent: Agent[None],
        basic_card: AgentCard,
    ) -> None:
        async def my_card_modifier(card: AgentCard) -> AgentCard:
            return card

        server = A2AServer(
            agent=basic_agent,
            agent_card=basic_card,
            card_modifier=my_card_modifier,
        )
        with patch("troopai.adk.a2a.app_factory.create_agent_card_routes") as mock_card_routes:
            mock_card_routes.return_value = []
            build_starlette_app(server)
        mock_card_routes.assert_called_once()
        _, kwargs = mock_card_routes.call_args
        assert kwargs.get("card_modifier") is my_card_modifier

    def test_none_card_modifier_reaches_agent_card_routes_as_none(
        self,
        basic_agent: Agent[None],
        basic_card: AgentCard,
    ) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card)
        with patch("troopai.adk.a2a.app_factory.create_agent_card_routes") as mock_card_routes:
            mock_card_routes.return_value = []
            build_starlette_app(server)
        mock_card_routes.assert_called_once()
        _, kwargs = mock_card_routes.call_args
        assert kwargs.get("card_modifier") is None
