"""Tests for session lifecycle hooks."""

from unittest.mock import patch

import pytest

from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.session import SQLiteMultiSessions
from troopai.adk.session.session_event import SessionEvent, create_session_event
from troopai.adk.types.items import ItemHelpers


def _user_event(content: str) -> SessionEvent:
    return create_session_event(author="user", content={"role": "user", "content": content})


class SessionTrackingHooks(RunHooks):
    """Test hooks that record session lifecycle calls."""

    def __init__(self):
        self.load_calls: list[tuple] = []
        self.save_calls: list[tuple] = []

    async def on_session_load(self, _context, session, events):
        self.load_calls.append((session.id, len(events)))

    async def on_session_save(self, _context, session, events):
        self.save_calls.append((session.id, len(events)))


class TestSessionLoadHook:
    @pytest.mark.asyncio
    async def test_on_session_load_called_with_events(self):
        """Hook fires when session has history to load."""
        from troopai.adk.agents.agent import Agent
        from troopai.adk.run.runner import Runner

        store = SQLiteMultiSessions()
        session = await store.create("test")
        await session.add([_user_event("Hello"), _user_event("World")])

        agent = Agent(name="test", system_prompt="You are helpful.")
        hooks = SessionTrackingHooks()

        async def mock_run_agent_loop(*, agent, user_prompt, **kwargs):
            from troopai.adk.types.run import RunResult

            return RunResult(
                final_output="Done",
                user_prompt=user_prompt,
                new_items=[],
                context=kwargs["context"],
                last_agent=agent,
            )

        with (
            patch("troopai.adk.run.runner.run_agent_loop", side_effect=mock_run_agent_loop),
            patch("troopai.adk.run.runner.run_blocking_input_guardrails", return_value=[]),
            patch("troopai.adk.run.runner.run_parallel_input_guardrails", return_value=[]),
            patch("troopai.adk.run.runner.run_output_guardrails", return_value=[]),
        ):
            await Runner.arun(agent, "New message", hooks=hooks, session=session)

        assert len(hooks.load_calls) == 1
        assert hooks.load_calls[0] == ("test", 2)

    @pytest.mark.asyncio
    async def test_on_session_load_not_called_when_empty(self):
        """Hook does NOT fire when session is empty."""
        from troopai.adk.agents.agent import Agent
        from troopai.adk.run.runner import Runner

        store = SQLiteMultiSessions()
        session = await store.create("test")

        agent = Agent(name="test", system_prompt="You are helpful.")
        hooks = SessionTrackingHooks()

        async def mock_run_agent_loop(*, agent, user_prompt, **kwargs):
            from troopai.adk.types.run import RunResult

            return RunResult(
                final_output="Done",
                user_prompt=user_prompt,
                new_items=[],
                context=kwargs["context"],
                last_agent=agent,
            )

        with (
            patch("troopai.adk.run.runner.run_agent_loop", side_effect=mock_run_agent_loop),
            patch("troopai.adk.run.runner.run_blocking_input_guardrails", return_value=[]),
            patch("troopai.adk.run.runner.run_parallel_input_guardrails", return_value=[]),
            patch("troopai.adk.run.runner.run_output_guardrails", return_value=[]),
        ):
            await Runner.arun(agent, "Hello", hooks=hooks, session=session)

        assert len(hooks.load_calls) == 0


class TestSessionSaveHook:
    @pytest.mark.asyncio
    async def test_on_session_save_called_with_events(self):
        """Hook fires when events are saved to session."""
        from troopai.adk.agents.agent import Agent
        from troopai.adk.run.runner import Runner

        store = SQLiteMultiSessions()
        session = await store.create("test")

        agent = Agent(name="test", system_prompt="You are helpful.")
        hooks = SessionTrackingHooks()

        async def mock_run_agent_loop(*, agent, user_prompt, **kwargs):
            from troopai.adk.types.run import RunResult

            return RunResult(
                final_output="Done",
                user_prompt=user_prompt,
                new_items=ItemHelpers.messages_to_run_items([{"role": "assistant", "content": "Hello!"}]),
                context=kwargs["context"],
                last_agent=agent,
            )

        with (
            patch("troopai.adk.run.runner.run_agent_loop", side_effect=mock_run_agent_loop),
            patch("troopai.adk.run.runner.run_blocking_input_guardrails", return_value=[]),
            patch("troopai.adk.run.runner.run_parallel_input_guardrails", return_value=[]),
            patch("troopai.adk.run.runner.run_output_guardrails", return_value=[]),
        ):
            await Runner.arun(agent, "Hi", hooks=hooks, session=session)

        assert len(hooks.save_calls) == 1
        # 2 events: user message + assistant response
        assert hooks.save_calls[0] == ("test", 2)


class TestDefaultHooksAreNoop:
    @pytest.mark.asyncio
    async def test_default_hooks_dont_error(self):
        """Default RunHooks session methods are no-ops."""
        hooks = RunHooks()
        # Should not raise
        await hooks.on_session_load(None, None, [])
        await hooks.on_session_save(None, None, [])
        await hooks.on_state_save(None, None, None)
