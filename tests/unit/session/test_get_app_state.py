"""Tests for the get_app_state() read-only helper on SQLiteMultiSessions (Feature 4).

Verifies that app-scoped state can be read back without constructing a Session.
"""

from __future__ import annotations

import pytest

from troopai.adk.session import SQLiteMultiSessions


@pytest.fixture
async def store():
    """In-memory SQLiteMultiSessions with deterministic teardown."""
    s = SQLiteMultiSessions(app_name="testapp")
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_get_app_state_empty_initially(store: SQLiteMultiSessions) -> None:
    """Returns an empty dict when no app-scoped state has been written."""
    result = await store.get_app_state()
    assert result == {}


@pytest.mark.asyncio
async def test_get_app_state_reflects_written_keys(store: SQLiteMultiSessions) -> None:
    """App-scoped writes via save_state() are visible through get_app_state()."""
    session = await store.create("s6", user_id="u6")
    session.state["app:theme"] = "dark"
    session.state["app:lang"] = "fr"
    await session.save_state()

    result = await store.get_app_state()
    assert result.get("app:theme") == "dark"
    assert result.get("app:lang") == "fr"


@pytest.mark.asyncio
async def test_get_app_state_does_not_include_session_keys(store: SQLiteMultiSessions) -> None:
    """Session-scoped state does not leak into get_app_state()."""
    session = await store.create("s7", user_id="u7")
    session.state["local_key"] = "value"
    session.state["app:shared"] = "yes"
    await session.save_state()

    result = await store.get_app_state()
    assert "local_key" not in result
    assert result.get("app:shared") == "yes"


@pytest.mark.asyncio
async def test_get_app_state_across_sessions(store: SQLiteMultiSessions) -> None:
    """App-scoped state written in one session is accessible via get_app_state()."""
    s1 = await store.create("s8", user_id="u8a")
    s1.state["app:counter"] = 1
    await s1.save_state()

    s2 = await store.create("s9", user_id="u8b")
    s2.state["app:counter"] = 2
    await s2.save_state()

    result = await store.get_app_state()
    # Last write wins on the same key.
    assert result.get("app:counter") == 2


@pytest.mark.asyncio
async def test_get_app_state_without_constructing_session() -> None:
    """get_app_state() works on a fresh store without any get()/create() call."""
    store = SQLiteMultiSessions(app_name="standalone")
    try:
        session = await store.create("sx", user_id="ux")
        session.state["app:val"] = 42
        await session.save_state()

        # Read it back without constructing another Session object.
        result = await store.get_app_state()
        assert result.get("app:val") == 42
    finally:
        await store.close()
