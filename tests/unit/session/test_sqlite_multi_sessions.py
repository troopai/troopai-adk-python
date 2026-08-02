"""Tests for SQLiteMultiSessions — session manager."""

import sqlite3
from unittest.mock import patch

import pytest

from troopai.adk.session import Session, SessionSettings, SQLiteMultiSessions, sqlite_multi_sessions as sms
from troopai.adk.session.session_event import create_session_event
from troopai.adk.session.sqlite_session import build_create_schema_sql


def _user_event(content: str):
    return create_session_event(author="user", content={"role": "user", "content": content})


@pytest.fixture
async def store():
    s = SQLiteMultiSessions()
    yield s
    # Deterministic teardown: the in-memory store holds a persistent
    # aiosqlite connection whose non-daemon worker thread only stops on
    # close() (or GC, which exception-chain cycles can defer past
    # interpreter shutdown, deadlocking the final thread join).
    await s.close()


# ── Create ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_returns_session(store):
    session = await store.create("conv-1")
    assert isinstance(session, Session)
    assert session.id == "conv-1"


@pytest.mark.asyncio
async def test_create_raises_if_exists(store):
    await store.create("conv-1")
    with pytest.raises(ValueError, match="already exists"):
        await store.create("conv-1")


@pytest.mark.asyncio
async def test_create_with_user_id(store):
    session = await store.create("conv-1", user_id="user-1")
    assert session.user_id == "user-1"


@pytest.mark.asyncio
async def test_create_same_id_different_user(store):
    s1 = await store.create("conv-1", user_id="user-1")
    s2 = await store.create("conv-1", user_id="user-2")
    assert s1.user_id == "user-1"
    assert s2.user_id == "user-2"


@pytest.mark.asyncio
async def test_create_with_initial_state(store):
    session = await store.create("test", state={"theme": "dark"})
    assert session.state["theme"] == "dark"


@pytest.mark.asyncio
async def test_create_settings_override_default():
    store = SQLiteMultiSessions(settings=SessionSettings(limit=10))
    try:
        session = await store.create("test", settings=SessionSettings(limit=2))
        assert session.settings.limit == 2
    finally:
        await store.close()


# ── Get ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_existing(store):
    await store.create("conv-1")
    session = await store.get("conv-1")
    assert session is not None
    assert session.id == "conv-1"


@pytest.mark.asyncio
async def test_get_missing(store):
    assert await store.get("nonexistent") is None


@pytest.mark.asyncio
async def test_get_with_user_id(store):
    await store.create("conv-1", user_id="user-1")
    assert await store.get("conv-1", user_id="user-1") is not None
    assert await store.get("conv-1", user_id="user-2") is None


# ── Get or Create ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_or_create_new(store):
    session = await store.get_or_create("conv-1")
    assert session.id == "conv-1"
    assert await store.count() == 1


@pytest.mark.asyncio
async def test_get_or_create_existing(store):
    await store.create("conv-1")
    session = await store.get_or_create("conv-1")
    assert session.id == "conv-1"
    assert await store.count() == 1


# ── List ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_all(store):
    await store.create("a")
    await store.create("b")
    sessions = await store.list()
    assert len(sessions) == 2
    assert {s.session_id for s in sessions} == {"a", "b"}


@pytest.mark.asyncio
async def test_list_by_user(store):
    await store.create("a", user_id="u1")
    await store.create("b", user_id="u2")
    sessions = await store.list(user_id="u1")
    assert len(sessions) == 1
    assert sessions[0].session_id == "a"


@pytest.mark.asyncio
async def test_list_session_info_fields(store):
    await store.create("conv-1", user_id="u1")
    infos = await store.list()
    info = infos[0]
    assert info.session_id == "conv-1"
    assert info.app_name == ""
    assert info.user_id == "u1"
    assert info.created_at
    assert info.updated_at


@pytest.mark.asyncio
async def test_list_with_app_name():
    store = SQLiteMultiSessions(app_name="myapp")
    try:
        await store.create("conv-1")
        infos = await store.list()
        assert infos[0].app_name == "myapp"
    finally:
        await store.close()


# ── Delete ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_existing(store):
    await store.create("conv-1")
    assert await store.delete("conv-1") is True
    assert await store.count() == 0


@pytest.mark.asyncio
async def test_delete_missing(store):
    assert await store.delete("nonexistent") is False


@pytest.mark.asyncio
async def test_delete_removes_messages(store):
    session = await store.create("conv-1")
    await session.add([_user_event("hi")])
    await store.delete("conv-1")
    session2 = await store.create("conv-1")
    assert await session2.get() == []


# ── Count ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_count(store):
    assert await store.count() == 0
    await store.create("a")
    await store.create("b")
    assert await store.count() == 2


@pytest.mark.asyncio
async def test_count_by_user(store):
    await store.create("a", user_id="u1")
    await store.create("b", user_id="u2")
    assert await store.count(user_id="u1") == 1


# ── Isolation ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_sessions_isolated(store):
    s1 = await store.create("a")
    s2 = await store.create("b")
    await s1.add([_user_event("from a")])
    await s2.add([_user_event("from b")])
    assert (await s1.get())[0].content["content"] == "from a"
    assert (await s2.get())[0].content["content"] == "from b"


# ── Persistence ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_file_persistence(tmp_path):
    db_path = tmp_path / "persist.db"
    store1 = SQLiteMultiSessions(path=db_path)
    session = await store1.create("conv-1")
    await session.add([_user_event("persisted")])
    await store1.close()

    store2 = SQLiteMultiSessions(path=db_path)
    session2 = await store2.get("conv-1")
    events = await session2.get()
    assert events[0].content["content"] == "persisted"
    await store2.close()


# ── Manager properties ───────────────────────────────────────────────


def test_app_name_property():
    store = SQLiteMultiSessions(app_name="myapp")
    assert store.app_name == "myapp"


def test_manager_is_not_session():
    assert not isinstance(SQLiteMultiSessions(), Session)


# ── Close ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close():
    store = SQLiteMultiSessions()
    await store.create("test")
    await store.close()


# ── TOCTOU race: concurrent create() raises ValueError not IntegrityError ────


@pytest.mark.asyncio
async def test_create_concurrent_raises_value_error(tmp_path) -> None:
    """Concurrent create() with same session_id must raise ValueError, not IntegrityError."""
    import asyncio

    store = SQLiteMultiSessions(path=tmp_path / "sessions.db")
    try:
        errors: list[Exception] = []

        async def create() -> None:
            try:
                await store.create("race-session", user_id="user-1")
            except Exception as exc:
                errors.append(exc)

        # Fire two concurrent creates for the same session — at least one must fail
        await asyncio.gather(create(), create(), return_exceptions=True)

        # One succeeded, one failed — the failure must be ValueError (not IntegrityError)
        assert len(errors) == 1
        exc = errors[0]
        assert isinstance(exc, ValueError), f"Expected ValueError but got {type(exc).__name__}: {exc}"
        assert "already exists" in str(exc)
    finally:
        await store.close()


# ── Migration probe does not leak the inspection connection ──────────────


def test_is_migration_needed_closes_connection(tmp_path) -> None:
    """``_is_migration_needed`` must close the sqlite3 connection it opens.

    ``sqlite3.Connection.__exit__`` only commits/rolls back — it does NOT
    close the connection — so a bare ``with sqlite3.connect(...)`` leaks the
    file handle until GC.  Capture the connection the probe opens and assert
    it is closed once the probe returns.
    """
    # A pre-existing DB file with the expected (schema-complete) table so the
    # probe reaches the inspection branch and returns False.
    db_path = tmp_path / "probe.db"
    seed = sqlite3.connect(str(db_path))
    try:
        seed.executescript(build_create_schema_sql())
        seed.commit()
    finally:
        seed.close()

    captured: list[sqlite3.Connection] = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        captured.append(conn)
        return conn

    with patch.object(sms.sqlite3, "connect", side_effect=tracking_connect):
        result = sms._is_migration_needed(str(db_path))

    # The probe opened exactly one connection for inspection and found the
    # schema complete (no migration needed).
    assert len(captured) == 1
    assert result is False

    # The leaked-handle bug: a bare ``with`` leaves the connection open, so
    # ``execute`` still works.  With the fix the connection is closed and
    # operating on it raises ``ProgrammingError``.
    with pytest.raises(sqlite3.ProgrammingError):
        captured[0].execute("SELECT 1")
