"""Tests for ``A2ARunner`` — execution entry point for ``A2AAgent``.

Verifies:
* :meth:`A2ARunner.arun` accepts ONLY ``A2AAgent`` (raises
  ``TypeError`` for ``Agent``, ``Swarm``, plain object, ``None``).
* Same A2AAgent-only constraint on :meth:`poll_task` and
  :meth:`cancel_task`.
* The ``stream=True`` × ``background=True`` mutex.
* The ``continuation_token`` × ``context_id`` mutex.
* Each flag combination dispatches to the correct underlying
  :class:`A2AClient` method.
* ``TypeError`` message names :class:`Runner` so the developer
  knows where to go for local agents.
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Skip module if the optional `a2a` extra is missing.
pytest.importorskip("a2a.client")

from troopai.adk.a2a import (
    A2AAgent,
    A2AClient,
    A2AContinuationToken,
    A2ARunner,
    A2ARunResult,
    A2AStreamEvent,
    A2ATaskStatus,
)
from troopai.adk.agents import Agent


@pytest.fixture
def mock_client() -> MagicMock:
    """An A2AClient stub with every public method mocked."""
    client = MagicMock(spec=A2AClient)
    client.send_message = AsyncMock()
    client.submit_background = AsyncMock()
    # ``stream_message`` is an async generator — NOT AsyncMock
    # (returns iterator directly, not a coroutine).
    client.stream_message = MagicMock()
    client.poll_task = AsyncMock()
    client.cancel_task = AsyncMock()
    return client


@pytest.fixture
def remote(mock_client: MagicMock) -> A2AAgent:
    """An A2AAgent with its lazy ``_client`` slot pre-filled with the mock."""
    agent = A2AAgent(name="remote", url="http://example.com")
    object.__setattr__(agent, "_client", mock_client)
    return agent


# ---------------------------------------------------------------------------
# A2AAgent-only contract — the load-bearing rule. Documented in the class
# docstring, every method docstring, the runtime TypeError message, the
# package navigation guide, and the user-facing docs page.
# ---------------------------------------------------------------------------


class TestA2AAgentOnlyContract:
    """``A2ARunner`` accepts only :class:`A2AAgent`; everything else raises."""

    async def test_arun_rejects_local_agent(self) -> None:
        local = Agent(name="local", system_prompt="Be helpful.")
        with pytest.raises(TypeError, match=r"only A2AAgent instances"):
            await A2ARunner.arun(local, "hi")  # type: ignore[arg-type]

    async def test_arun_typeerror_points_at_runner(self) -> None:
        local = Agent(name="local", system_prompt="Be helpful.")
        with pytest.raises(TypeError, match=r"troopai\.adk\.run\.runner\.Runner"):
            await A2ARunner.arun(local, "hi")  # type: ignore[arg-type]

    async def test_arun_rejects_plain_object(self) -> None:
        with pytest.raises(TypeError, match=r"only A2AAgent instances"):
            await A2ARunner.arun(object(), "hi")  # type: ignore[arg-type]

    async def test_arun_rejects_none(self) -> None:
        with pytest.raises(TypeError, match=r"only A2AAgent instances"):
            await A2ARunner.arun(None, "hi")  # type: ignore[arg-type]

    async def test_arun_rejects_str(self) -> None:
        # An ``Any``-typed boundary case — JSON-loaded config etc.
        with pytest.raises(TypeError, match=r"only A2AAgent instances"):
            await A2ARunner.arun("not an agent", "hi")  # type: ignore[arg-type]

    async def test_poll_task_rejects_local_agent(self) -> None:
        local = Agent(name="local", system_prompt="Be helpful.")
        token = A2AContinuationToken(task_id="t", context_id="c", remote_url="http://x")
        with pytest.raises(TypeError, match=r"only A2AAgent instances"):
            await A2ARunner.poll_task(local, token)  # type: ignore[arg-type]

    async def test_cancel_task_rejects_local_agent(self) -> None:
        local = Agent(name="local", system_prompt="Be helpful.")
        token = A2AContinuationToken(task_id="t", context_id="c", remote_url="http://x")
        with pytest.raises(TypeError, match=r"only A2AAgent instances"):
            await A2ARunner.cancel_task(local, token)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Mutex constraints
# ---------------------------------------------------------------------------


class TestMutexConstraints:
    async def test_stream_and_background_mutually_exclusive(self, remote: A2AAgent, mock_client: MagicMock) -> None:
        with pytest.raises(ValueError, match=r"cannot combine stream=True with background=True"):
            await A2ARunner.arun(remote, "hi", stream=True, background=True)  # type: ignore[call-overload]
        # Neither client method called.
        mock_client.send_message.assert_not_awaited()
        mock_client.submit_background.assert_not_awaited()
        mock_client.stream_message.assert_not_called()

    async def test_continuation_token_and_context_id_mutually_exclusive(
        self, remote: A2AAgent, mock_client: MagicMock
    ) -> None:
        token = A2AContinuationToken(task_id="t", context_id="c", remote_url="http://x")
        with pytest.raises(ValueError, match=r"cannot accept both continuation_token and context_id"):
            await A2ARunner.arun(remote, "hi", continuation_token=token, context_id="c-other")
        mock_client.send_message.assert_not_awaited()

    async def test_stream_and_continuation_token_mutually_exclusive(
        self, remote: A2AAgent, mock_client: MagicMock
    ) -> None:
        # Regression: stream=True + continuation_token silently dropped the
        # token and routed to a fresh task instead of the resumed one.
        # The fix rejects the combination with a ValueError.
        token = A2AContinuationToken(task_id="t-123", context_id="c-1", remote_url="http://x")
        with pytest.raises(ValueError, match=r"stream=True.*continuation_token|continuation_token.*stream=True"):
            await A2ARunner.arun(
                remote,
                "hi",
                stream=True,
                continuation_token=token,  # type: ignore[call-overload]
            )
        mock_client.stream_message.assert_not_called()
        mock_client.send_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# Flag dispatch — verify each combination routes to the right A2AClient method
# ---------------------------------------------------------------------------


class TestFlagDispatch:
    async def test_blocking_default_routes_to_send_message(self, remote: A2AAgent, mock_client: MagicMock) -> None:
        mock_client.send_message.return_value = A2ARunResult(text="answer", task_id="t1", context_id="c1")
        result = await A2ARunner.arun(remote, "hi")
        assert isinstance(result, A2ARunResult)
        assert result.text == "answer"
        mock_client.send_message.assert_awaited_once_with("hi", context_id=None, continuation_token=None)

    async def test_blocking_with_context_id(self, remote: A2AAgent, mock_client: MagicMock) -> None:
        mock_client.send_message.return_value = A2ARunResult(text="x", task_id="t2", context_id="c1")
        await A2ARunner.arun(remote, "follow-up", context_id="c1")
        mock_client.send_message.assert_awaited_once_with("follow-up", context_id="c1", continuation_token=None)

    async def test_blocking_with_continuation_token(self, remote: A2AAgent, mock_client: MagicMock) -> None:
        token = A2AContinuationToken(task_id="t1", context_id="c1", remote_url="http://x")
        mock_client.send_message.return_value = A2ARunResult(text="resumed", task_id="t1", context_id="c1")
        await A2ARunner.arun(remote, "more please", continuation_token=token)
        mock_client.send_message.assert_awaited_once_with("more please", context_id=None, continuation_token=token)

    async def test_background_routes_to_submit_background(self, remote: A2AAgent, mock_client: MagicMock) -> None:
        token = A2AContinuationToken(task_id="t1", context_id="c1", remote_url="http://x")
        mock_client.submit_background.return_value = token
        result = await A2ARunner.arun(remote, "long task", background=True)  # type: ignore[call-overload]
        assert result == token
        mock_client.submit_background.assert_awaited_once_with("long task", context_id=None)

    async def test_stream_routes_to_stream_message(self, remote: A2AAgent, mock_client: MagicMock) -> None:
        async def fake_stream() -> AsyncIterator[A2AStreamEvent]:
            yield {"type": "text_delta", "text_delta": "hel"}
            yield {"type": "text_delta", "text_delta": "lo"}
            yield {"type": "completed", "result_text": "hello"}

        mock_client.stream_message.return_value = fake_stream()
        events: list[A2AStreamEvent] = []
        async for ev in await A2ARunner.arun(remote, "stream me", stream=True):  # type: ignore[call-overload]
            events.append(ev)
        mock_client.stream_message.assert_called_once_with("stream me", context_id=None)
        assert len(events) == 3
        assert events[-1]["type"] == "completed"


# ---------------------------------------------------------------------------
# poll_task / cancel_task — A2AAgent-only contract + forwarding
# ---------------------------------------------------------------------------


class TestPollAndCancel:
    async def test_poll_task_forwards_to_client(self, remote: A2AAgent, mock_client: MagicMock) -> None:
        token = A2AContinuationToken(task_id="t1", context_id="c1", remote_url="http://x")
        mock_client.poll_task.return_value = A2ATaskStatus(task_id="t1", context_id="c1", state="working")
        status = await A2ARunner.poll_task(remote, token)
        assert status.state == "working"
        mock_client.poll_task.assert_awaited_once_with(token)

    async def test_cancel_task_forwards_to_client(self, remote: A2AAgent, mock_client: MagicMock) -> None:
        token = A2AContinuationToken(task_id="t1", context_id="c1", remote_url="http://x")
        await A2ARunner.cancel_task(remote, token)
        mock_client.cancel_task.assert_awaited_once_with(token)


# ---------------------------------------------------------------------------
# A2AExecutableAdapter — confirm the internal switch to A2ARunner works
# ---------------------------------------------------------------------------


class TestAdapterUsesRunner:
    """The graph adapter now dispatches via ``A2ARunner.arun``."""

    async def test_adapter_invokes_runner(self, remote: A2AAgent, mock_client: MagicMock) -> None:
        from troopai.adk.a2a import A2AExecutableAdapter
        from troopai.adk.orchestration.executable import ExecutableInput
        from troopai.adk.run.config import DEFAULT_RUN_CONFIG
        from troopai.adk.run.context import RunContext

        mock_client.send_message.return_value = A2ARunResult(text="adapter answer", task_id="t1", context_id="c1")
        adapter = A2AExecutableAdapter(agent=remote)
        ctx = RunContext.make(None)
        node_input = ExecutableInput(content=[{"role": "user", "content": "ask the remote"}])
        result = await adapter.invoke(node_input, ctx, DEFAULT_RUN_CONFIG)
        assert result.output == "adapter answer"
        mock_client.send_message.assert_awaited_once_with("ask the remote", context_id=None, continuation_token=None)


# ---------------------------------------------------------------------------
# A2AAgent.as_tool — the closure now dispatches via A2ARunner too
# ---------------------------------------------------------------------------


class TestAsToolUsesRunner:
    async def test_as_tool_closure_invokes_runner(self, remote: A2AAgent, mock_client: MagicMock) -> None:
        mock_client.send_message.return_value = A2ARunResult(text="tool answer", task_id="t1", context_id="c1")
        tool = remote.as_tool()
        assert tool.on_invoke is not None
        result = await tool.on_invoke(MagicMock(), '{"input": "ask via tool"}')
        assert result == "tool answer"
        mock_client.send_message.assert_awaited_once_with("ask via tool", context_id=None, continuation_token=None)


# ---------------------------------------------------------------------------
# Lazy client init through A2ARunner — verifies the runner uses get_client()
# ---------------------------------------------------------------------------


class TestLazyClientInitViaRunner:
    async def test_arun_constructs_client_on_first_call(self) -> None:
        agent = A2AAgent(name="remote", url="http://example.com")
        with patch("troopai.adk.a2a.a2a_agent.A2AClient") as mock_cls:
            mock_inst = MagicMock(spec=A2AClient)
            mock_inst.send_message = AsyncMock(return_value=A2ARunResult(text="ok", task_id="t1", context_id="c1"))
            mock_cls.return_value = mock_inst
            await A2ARunner.arun(agent, "hi")
            assert mock_cls.call_count == 1
            # Second call reuses the cached client.
            await A2ARunner.arun(agent, "hi")
            assert mock_cls.call_count == 1
