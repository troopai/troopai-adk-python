"""Tests for ``A2AExecutableAdapter`` — graph-node wrapper for ``A2AAgent``."""

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip module if the optional `a2a` extra is missing.
pytest.importorskip("a2a.client")

from troopai.adk.a2a import A2AAgent, A2AClient, A2AExecutableAdapter, A2ARunResult
from troopai.adk.graphs.adapters import to_executable
from troopai.adk.orchestration.executable import (
    Executable,
    ExecutableInput,
    NodeResult,
)
from troopai.adk.run.config import DEFAULT_RUN_CONFIG
from troopai.adk.run.context import RunContext


@pytest.fixture
def remote() -> A2AAgent:
    return A2AAgent(name="Remote", url="http://example.com")


@pytest.fixture
def adapter(remote: A2AAgent) -> A2AExecutableAdapter:
    return A2AExecutableAdapter(agent=remote)


@pytest.fixture
def mock_client() -> MagicMock:
    """Build a stub :class:`A2AClient` whose ``send_message`` is mocked.

    The adapter dispatches via ``A2ARunner.arun`` →
    ``agent.get_client()`` → ``client.send_message``. Mocking at
    the get_client boundary keeps the test against the real
    runner code path while avoiding any network IO.
    """
    client = MagicMock(spec=A2AClient)
    client.send_message = AsyncMock(return_value=A2ARunResult(text="ok", task_id="t1", context_id="c1"))
    return client


def _patched_get_client(remote: A2AAgent, client: MagicMock) -> AsyncMock:
    """Return an ``AsyncMock`` patched onto ``remote.get_client``."""
    return AsyncMock(return_value=client)


def _input(text: str = "hello") -> ExecutableInput:
    """Build an ExecutableInput carrying a single text Layer 1 item."""
    return ExecutableInput(content=[{"type": "input_text", "text": text}])


def _ctx() -> RunContext[None]:
    return RunContext.make(None)


class TestAdapterConstruction:
    def test_is_executable(self, adapter: A2AExecutableAdapter) -> None:
        assert isinstance(adapter, Executable)

    def test_holds_agent_reference(self, adapter: A2AExecutableAdapter, remote: A2AAgent) -> None:
        assert adapter.agent is remote


class TestAdapterInvoke:
    """Adapter's ``invoke`` dispatches via ``A2ARunner.arun``.

    Tests mock ``agent.get_client()`` (the public accessor) so the
    real ``A2ARunner`` code path runs — including its
    ``isinstance(agent, A2AAgent)`` guard, the mutex checks, and
    the flag dispatch. This catches any future regression where
    the runner's contract drifts away from the adapter's expectations.
    """

    async def test_packages_text_into_node_result(self, adapter: A2AExecutableAdapter, remote: A2AAgent) -> None:
        client = MagicMock(spec=A2AClient)
        client.send_message = AsyncMock(
            return_value=A2ARunResult(text="remote answer", task_id="t-abc", context_id="c-xyz")
        )
        with patch.object(remote, "get_client", _patched_get_client(remote, client)):
            result = await adapter.invoke(_input("ask me"), _ctx(), DEFAULT_RUN_CONFIG)
        assert isinstance(result, NodeResult)
        assert result.output == "remote answer"
        assert result.final_text == "remote answer"
        assert result.new_items == []
        assert result.metadata["adapter"] == "a2a"
        assert result.metadata["agent_name"] == "Remote"
        assert result.metadata["task_id"] == "t-abc"
        assert result.metadata["context_id"] == "c-xyz"
        assert result.metadata["remote_url"] == "http://example.com"

    async def test_extracts_prompt_from_input_content(
        self,
        adapter: A2AExecutableAdapter,
        remote: A2AAgent,
        mock_client: MagicMock,
    ) -> None:
        with patch.object(remote, "get_client", _patched_get_client(remote, mock_client)):
            await adapter.invoke(_input("specific prompt"), _ctx(), DEFAULT_RUN_CONFIG)
        mock_client.send_message.assert_awaited_once_with("specific prompt", context_id=None, continuation_token=None)

    async def test_concatenates_multiple_text_items(
        self,
        adapter: A2AExecutableAdapter,
        remote: A2AAgent,
        mock_client: MagicMock,
    ) -> None:
        multi = ExecutableInput(
            content=[
                {"type": "input_text", "text": "first "},
                {"type": "input_text", "text": "second"},
            ],
        )
        with patch.object(remote, "get_client", _patched_get_client(remote, mock_client)):
            await adapter.invoke(multi, _ctx(), DEFAULT_RUN_CONFIG)
        mock_client.send_message.assert_awaited_once_with("first second", context_id=None, continuation_token=None)

    async def test_extracts_prompt_from_easy_message_str_content(
        self,
        adapter: A2AExecutableAdapter,
        remote: A2AAgent,
        mock_client: MagicMock,
    ) -> None:
        # Graph entry node receives a normalised easy-message dict.
        easy = ExecutableInput(content=[{"role": "user", "content": "user prompt"}])
        with patch.object(remote, "get_client", _patched_get_client(remote, mock_client)):
            await adapter.invoke(easy, _ctx(), DEFAULT_RUN_CONFIG)
        mock_client.send_message.assert_awaited_once_with("user prompt", context_id=None, continuation_token=None)

    async def test_extracts_prompt_from_easy_message_list_content(
        self,
        adapter: A2AExecutableAdapter,
        remote: A2AAgent,
        mock_client: MagicMock,
    ) -> None:
        easy = ExecutableInput(
            content=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "first "},
                        {"type": "input_text", "text": "second"},
                    ],
                },
            ],
        )
        with patch.object(remote, "get_client", _patched_get_client(remote, mock_client)):
            await adapter.invoke(easy, _ctx(), DEFAULT_RUN_CONFIG)
        mock_client.send_message.assert_awaited_once_with("first second", context_id=None, continuation_token=None)

    async def test_skips_non_text_items_with_log(
        self,
        adapter: A2AExecutableAdapter,
        remote: A2AAgent,
        mock_client: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        mixed = ExecutableInput(
            content=[
                {"type": "input_text", "text": "text item "},
                {"type": "input_image", "image_url": "http://x/img.png"},
                {"type": "input_text", "text": "more text"},
            ],
        )
        with (
            patch.object(remote, "get_client", _patched_get_client(remote, mock_client)),
            caplog.at_level(logging.DEBUG, logger="troopai.adk.a2a.adapters"),
        ):
            await adapter.invoke(mixed, _ctx(), DEFAULT_RUN_CONFIG)
        mock_client.send_message.assert_awaited_once_with(
            "text item more text", context_id=None, continuation_token=None
        )

    async def test_empty_prompt_raises_value_error(
        self,
        adapter: A2AExecutableAdapter,
        remote: A2AAgent,
        mock_client: MagicMock,
    ) -> None:
        # Regression (finding 9): empty prompt must fail-closed rather than
        # silently calling remote with an empty string. Empty prompts are a
        # graph wiring bug; they must not trigger a remote network call.
        empty = ExecutableInput(content=[])
        with (
            patch.object(remote, "get_client", _patched_get_client(remote, mock_client)),
            pytest.raises(ValueError, match="empty prompt"),
        ):
            await adapter.invoke(empty, _ctx(), DEFAULT_RUN_CONFIG)
        mock_client.send_message.assert_not_awaited()


class TestToExecutableDispatch:
    """Verify ``to_executable()`` wraps an ``A2AAgent`` automatically."""

    def test_wraps_a2a_agent(self, remote: A2AAgent) -> None:
        wrapped = to_executable(remote)
        assert isinstance(wrapped, A2AExecutableAdapter)
        assert wrapped.agent is remote

    def test_returns_existing_executable_unchanged(self, adapter: A2AExecutableAdapter) -> None:
        wrapped = to_executable(adapter)
        assert wrapped is adapter
