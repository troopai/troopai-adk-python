"""Tests for ``build_starlette_app`` executor / task-store wiring.

Covers the contract that ``A2AServer.executor_task_store`` is forwarded to
the :class:`A2AExecutor` the factory builds, so a developer can inject a
framework ``TaskStore`` (e.g. ``SQLiteTaskStore``) for executor-level
restart recovery. Without forwarding, the executor always falls back to an
in-memory store and ``recover_on_startup`` is unreachable.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("a2a.types")
pytest.importorskip("starlette")

from a2a.types import AgentCapabilities, AgentCard, AgentInterface

from troopai.adk.a2a import A2AServer, build_starlette_app
from troopai.adk.a2a.executor import A2AExecutor
from troopai.adk.a2a.task_store import InMemoryTaskStore
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


# ---------------------------------------------------------------------------
# executor_task_store default
# ---------------------------------------------------------------------------


class TestExecutorTaskStoreDefault:
    def test_default_is_none(self, basic_agent: Agent[None], basic_card: AgentCard) -> None:
        server = A2AServer(agent=basic_agent, agent_card=basic_card)
        assert server.executor_task_store is None

    def test_default_executor_uses_in_memory_store(
        self,
        basic_agent: Agent[None],
        basic_card: AgentCard,
    ) -> None:
        """When unset, the built executor falls back to an in-memory store."""
        server = A2AServer(agent=basic_agent, agent_card=basic_card)
        with patch("troopai.adk.a2a.app_factory.A2AExecutor") as mock_executor:
            mock_executor.return_value = MagicMock()
            build_starlette_app(server)
        mock_executor.assert_called_once()
        _, kwargs = mock_executor.call_args
        assert kwargs.get("task_store") is None


# ---------------------------------------------------------------------------
# executor_task_store forwarding (the regression guard)
# ---------------------------------------------------------------------------


class TestExecutorTaskStorePassthrough:
    def test_executor_task_store_reaches_executor(
        self,
        basic_agent: Agent[None],
        basic_card: AgentCard,
    ) -> None:
        """A framework TaskStore set on the server must reach the executor.

        Fails before the fix: the factory never forwarded
        ``executor_task_store`` so the executor always got ``None`` and a
        fresh InMemoryTaskStore.
        """
        store = InMemoryTaskStore()
        server = A2AServer(
            agent=basic_agent,
            agent_card=basic_card,
            executor_task_store=store,
        )
        with patch("troopai.adk.a2a.app_factory.A2AExecutor") as mock_executor:
            mock_executor.return_value = MagicMock()
            build_starlette_app(server)
        mock_executor.assert_called_once()
        _, kwargs = mock_executor.call_args
        assert kwargs.get("task_store") is store

    def test_executor_receives_real_store_instance(
        self,
        basic_agent: Agent[None],
        basic_card: AgentCard,
    ) -> None:
        """End-to-end: the real A2AExecutor adopts the injected store.

        Builds the app without mocking A2AExecutor and reaches in to the
        handler-owned executor to confirm it adopted the injected store
        rather than minting its own InMemoryTaskStore.
        """
        store = InMemoryTaskStore()
        server = A2AServer(
            agent=basic_agent,
            agent_card=basic_card,
            executor_task_store=store,
        )
        captured: dict[str, A2AExecutor] = {}

        def _capture(**kwargs: object) -> A2AExecutor:
            executor = A2AExecutor(**kwargs)  # type: ignore[arg-type]
            captured["executor"] = executor
            return executor

        with patch("troopai.adk.a2a.app_factory.A2AExecutor", side_effect=_capture):
            build_starlette_app(server)
        executor = captured["executor"]
        assert executor._task_store is store
