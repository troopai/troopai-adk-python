"""Tests for the SessionStore protocol.

Covers:
- SQLiteSession satisfies SessionStore structurally (no inheritance).
- InMemorySessionStore (test double) satisfies SessionStore structurally.
- Runner session integration tests pass with InMemorySessionStore.
- @runtime_checkable isinstance checks work.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from troopai.adk.session import SessionSettings, SQLiteMultiSessions
from troopai.adk.session.session_event import SessionEvent, create_session_event
from troopai.adk.types.session import SessionStore

# ── Test double ─────────────────────────────────────────────────────────────


class InMemorySessionStore:
    """Minimal in-memory session store that satisfies SessionStore structurally.

    Intended for use in tests and as a reference implementation.  Does not
    inherit from Session or SessionStore — satisfaction is purely structural.

    Attributes:
        _session_id: Unique identifier for this session handle.
        _settings: Per-session configuration, or None to use defaults.
        _events: Ordered list of stored events.
        _state: Mutable dict-like state for this session.
        _closed: Whether close() has been called.
    """

    def __init__(
        self,
        session_id: str = "mem-session",
        settings: SessionSettings | None = None,
    ) -> None:
        """Initialise an in-memory session store.

        Args:
            session_id: Unique identifier for this session handle.
            settings: Per-session configuration, or None to use defaults.
        """
        self._session_id = session_id
        self._settings = settings
        self._events: list[SessionEvent] = []
        self._state: dict[str, Any] = {}
        self._closed = False

    @property
    def id(self) -> str:
        """Unique identifier for this session handle."""
        return self._session_id

    @property
    def settings(self) -> SessionSettings | None:
        """Per-session configuration, or None to use defaults."""
        return self._settings

    async def get(self, limit: int | None = None) -> list[SessionEvent]:
        """Return stored events, honouring settings.limit when limit is None.

        Args:
            limit: Maximum number of events to return.  Falls back to
                ``settings.limit`` when ``None``; returns all events when
                neither is set.

        Returns:
            Events in chronological order (oldest first).
        """
        effective_limit = limit
        if effective_limit is None and self._settings is not None:
            effective_limit = self._settings.limit
        if effective_limit is not None:
            return self._events[-effective_limit:]
        return list(self._events)

    async def add(self, events: list[SessionEvent]) -> None:
        """Append events to the session.

        Args:
            events: Events to persist.  An empty list is a no-op.
        """
        self._events.extend(events)

    async def save_state(self) -> None:
        """No-op — state is already in memory."""

    async def close(self) -> None:
        """Mark this handle as closed."""
        self._closed = True


# ── Helpers ──────────────────────────────────────────────────────────────────


def _user_event(content: str) -> SessionEvent:
    return create_session_event(author="user", content={"role": "user", "content": content})


def _assistant_event(content: str) -> SessionEvent:
    return create_session_event(author="assistant", content={"role": "assistant", "content": content})


# ── Typed assertion: SQLiteSession satisfies SessionStore ─────────────────────


@pytest.mark.asyncio
async def test_sqlite_session_satisfies_protocol() -> None:
    """SQLiteSession satisfies SessionStore without inheriting from it."""
    store = SQLiteMultiSessions()
    try:
        session = await store.create("protocol-check")
        # Typed assertion — no isinstance needed; this line would fail mypy
        # if SQLiteSession were missing a required attribute.
        _: SessionStore = session
        assert isinstance(session, SessionStore), "SQLiteSession must satisfy SessionStore via @runtime_checkable"
    finally:
        await store.close()


# ── Typed assertion: InMemorySessionStore satisfies SessionStore ─────────────


def test_in_memory_session_store_satisfies_protocol() -> None:
    """InMemorySessionStore satisfies SessionStore without inheriting from it."""
    mem = InMemorySessionStore("mem-test")
    _: SessionStore = mem
    assert isinstance(mem, SessionStore), "InMemorySessionStore must satisfy SessionStore via @runtime_checkable"


# ── InMemorySessionStore functional tests ────────────────────────────────────


@pytest.mark.asyncio
async def test_in_memory_get_empty() -> None:
    """get() on a fresh store returns an empty list."""
    mem = InMemorySessionStore()
    assert await mem.get() == []


@pytest.mark.asyncio
async def test_in_memory_add_and_get() -> None:
    """Events added via add() are returned by get() in order."""
    mem = InMemorySessionStore()
    await mem.add([_user_event("Hello"), _assistant_event("Hi!")])
    events = await mem.get()
    assert len(events) == 2
    assert events[0].author == "user"
    assert events[1].author == "assistant"


@pytest.mark.asyncio
async def test_in_memory_get_with_explicit_limit() -> None:
    """get(limit=N) returns the N most-recent events."""
    mem = InMemorySessionStore()
    await mem.add([_user_event(f"msg{i}") for i in range(5)])
    events = await mem.get(limit=2)
    assert len(events) == 2
    assert events[0].content == {"role": "user", "content": "msg3"}
    assert events[1].content == {"role": "user", "content": "msg4"}


@pytest.mark.asyncio
async def test_in_memory_get_respects_settings_limit() -> None:
    """get() with no explicit limit falls back to settings.limit."""
    mem = InMemorySessionStore(settings=SessionSettings(limit=3))
    await mem.add([_user_event(f"msg{i}") for i in range(5)])
    events = await mem.get()
    assert len(events) == 3


@pytest.mark.asyncio
async def test_in_memory_add_empty_is_noop() -> None:
    """add([]) does not change stored events."""
    mem = InMemorySessionStore()
    await mem.add([])
    assert await mem.get() == []


@pytest.mark.asyncio
async def test_in_memory_save_state_is_noop() -> None:
    """save_state() completes without error."""
    mem = InMemorySessionStore()
    await mem.save_state()  # must not raise


@pytest.mark.asyncio
async def test_in_memory_close_marks_closed() -> None:
    """close() marks the handle as closed."""
    mem = InMemorySessionStore()
    assert not mem._closed
    await mem.close()
    assert mem._closed


@pytest.mark.asyncio
async def test_in_memory_multi_add_accumulates() -> None:
    """Multiple add() calls accumulate events in order."""
    mem = InMemorySessionStore()
    await mem.add([_user_event("Q1"), _assistant_event("A1")])
    await mem.add([_user_event("Q2"), _assistant_event("A2")])
    events = await mem.get()
    assert len(events) == 4


# ── Runner session integration tests against InMemorySessionStore ─────────────


@pytest.mark.asyncio
async def test_runner_loads_history_from_in_memory_store() -> None:
    """Runner loads session history from InMemorySessionStore identically to SQLiteSession."""
    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.runner import Runner
    from troopai.adk.types.run import RunResult

    mem = InMemorySessionStore("runner-load-test")
    await mem.add([_user_event("First"), _assistant_event("Response")])

    agent = Agent(name="test", system_prompt="You are helpful.")

    captured: dict[str, Any] = {}

    async def mock_run_agent_loop(*, agent, user_prompt, **kwargs):
        captured["user_prompt"] = user_prompt
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
        await Runner.arun(agent, "Second message", session=mem)

    # Runner should have prepended the stored history before "Second message"
    prompt = captured["user_prompt"]
    assert isinstance(prompt, list), "History prepend must produce a list"
    assert len(prompt) == 3  # "First", "Response", "Second message"
    assert prompt[0]["content"] == "First"
    assert prompt[1]["content"] == "Response"
    assert prompt[2]["content"] == "Second message"


@pytest.mark.asyncio
async def test_runner_saves_events_to_in_memory_store() -> None:
    """Runner appends new events to InMemorySessionStore after the LLM turn."""
    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.runner import Runner
    from troopai.adk.types.items import ItemHelpers
    from troopai.adk.types.run import RunResult

    mem = InMemorySessionStore("runner-save-test")

    agent = Agent(name="test", system_prompt="You are helpful.")

    new_items = ItemHelpers.messages_to_run_items([{"role": "assistant", "content": "Hello!"}])

    async def mock_run_agent_loop(*, agent, user_prompt, **kwargs):
        return RunResult(
            final_output="Hello!",
            user_prompt=user_prompt,
            new_items=new_items,
            context=kwargs["context"],
            last_agent=agent,
        )

    with (
        patch("troopai.adk.run.runner.run_agent_loop", side_effect=mock_run_agent_loop),
        patch("troopai.adk.run.runner.run_blocking_input_guardrails", return_value=[]),
        patch("troopai.adk.run.runner.run_parallel_input_guardrails", return_value=[]),
        patch("troopai.adk.run.runner.run_output_guardrails", return_value=[]),
    ):
        await Runner.arun(agent, "Hi!", session=mem)

    saved = await mem.get()
    assert len(saved) >= 2, "At least user event + assistant event must be saved"
    authors = [e.author for e in saved]
    assert "user" in authors
    assert "assistant" in authors


@pytest.mark.asyncio
async def test_runner_empty_in_memory_store_keeps_string_input() -> None:
    """Empty InMemorySessionStore leaves the prompt as a plain string."""
    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.runner import Runner
    from troopai.adk.types.run import RunResult

    mem = InMemorySessionStore("runner-empty-test")

    agent = Agent(name="test", system_prompt="You are helpful.")
    captured: dict[str, Any] = {}

    async def mock_run_agent_loop(*, agent, user_prompt, **kwargs):
        captured["user_prompt"] = user_prompt
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
        await Runner.arun(agent, "Hello there", session=mem)

    # No history → prompt must remain a plain string
    assert captured["user_prompt"] == "Hello there"


@pytest.mark.asyncio
async def test_runner_settings_limit_honoured_on_in_memory_store() -> None:
    """Runner respects settings.limit when loading history from InMemorySessionStore."""
    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.runner import Runner
    from troopai.adk.types.run import RunResult

    mem = InMemorySessionStore("runner-limit-test", settings=SessionSettings(limit=2))
    for i in range(5):
        await mem.add([_user_event(f"msg{i}")])

    agent = Agent(name="test", system_prompt="You are helpful.")
    captured: dict[str, Any] = {}

    async def mock_run_agent_loop(*, agent, user_prompt, **kwargs):
        captured["user_prompt"] = user_prompt
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
        await Runner.arun(agent, "New", session=mem)

    # 2 history events (limit=2) + "New" user message
    prompt = captured["user_prompt"]
    assert isinstance(prompt, list)
    assert len(prompt) == 3
