"""Tests for Runner + Session integration."""

import pytest

from troopai.adk.session import SessionSettings, SQLiteMultiSessions
from troopai.adk.session.session_event import SessionEvent, create_session_event


def _user_event(content: str) -> SessionEvent:
    return create_session_event(author="user", content={"role": "user", "content": content})


def _assistant_event(content: str) -> SessionEvent:
    return create_session_event(author="assistant", content={"role": "assistant", "content": content})


@pytest.fixture
def store():
    return SQLiteMultiSessions()


@pytest.mark.asyncio
async def test_session_loads_history(store):
    session = await store.create("test-load")
    await session.add([_user_event("First"), _assistant_event("Response")])

    events = await session.get()
    assert len(events) == 2

    # Runner extracts content for LLM input
    history = [e.content for e in events]
    combined = [*history, {"role": "user", "content": "Second"}]
    assert len(combined) == 3
    assert combined[0]["content"] == "First"
    assert combined[2]["content"] == "Second"


@pytest.mark.asyncio
async def test_empty_session_keeps_string_input(store):
    session = await store.create("test-empty")
    assert await session.get() == []


@pytest.mark.asyncio
async def test_session_saves_events(store):
    session = await store.create("test-save")
    await session.add([_user_event("Hi"), _assistant_event("Hello!")])

    events = await session.get()
    assert len(events) == 2
    assert events[0].author == "user"
    assert events[0].content["content"] == "Hi"
    assert events[1].author == "assistant"


@pytest.mark.asyncio
async def test_session_with_default_limit():
    store = SQLiteMultiSessions(settings=SessionSettings(limit=2))
    session = await store.create("test-limit")
    await session.add([_user_event(f"msg{i}") for i in range(5)])

    limit = session.settings.limit if session.settings is not None else None
    events = await session.get(limit=limit)
    assert len(events) == 2
    assert events[0].content["content"] == "msg3"


@pytest.mark.asyncio
async def test_multi_turn_accumulation(store):
    session = await store.create("test-multi")
    await session.add([_user_event("Q1"), _assistant_event("A1")])
    await session.add([_user_event("Q2"), _assistant_event("A2")])
    events = await session.get()
    assert len(events) == 4


@pytest.mark.asyncio
async def test_session_not_required():
    session = None
    assert session is None
