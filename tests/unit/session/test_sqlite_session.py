"""Tests for SQLiteSession — bound session implementation."""

import asyncio
import json

import pytest

from troopai.adk.exceptions import SessionAppendConflictError
from troopai.adk.session import SessionSettings, SQLiteMultiSessions
from troopai.adk.session.session_event import SessionEvent, create_session_event
from troopai.adk.session.sqlite_session import (
    DEFAULT_MESSAGES_TABLE,
    DEFAULT_SESSIONS_TABLE,
    SQLiteSession,
    get_session_updated_at,
)
from troopai.adk.session.state import State


def _user_event(content: str) -> SessionEvent:
    return create_session_event(author="user", content={"role": "user", "content": content})


def _assistant_event(content: str) -> SessionEvent:
    return create_session_event(author="assistant", content={"role": "assistant", "content": content})


def _tool_call_event(call_id: str, name: str = "t") -> SessionEvent:
    return create_session_event(
        author="assistant",
        content={"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"},
    )


def _tool_result_event(call_id: str, output: str = "result") -> SessionEvent:
    return create_session_event(
        author="tool",
        content={"type": "function_call_output", "call_id": call_id, "output": output},
    )


@pytest.fixture
def store():
    return SQLiteMultiSessions()


# ── Basic CRUD ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_empty_session_returns_empty_list(store):
    session = await store.create("test")
    assert await session.get() == []


@pytest.mark.asyncio
async def test_add_and_get(store):
    session = await store.create("test")
    await session.add([_user_event("Hello"), _assistant_event("Hi there!")])
    events = await session.get()
    assert len(events) == 2
    assert isinstance(events[0], SessionEvent)
    assert events[0].author == "user"
    assert events[0].content["content"] == "Hello"
    assert events[1].author == "assistant"


@pytest.mark.asyncio
async def test_add_empty_list_is_noop(store):
    session = await store.create("test")
    await session.add([])
    assert await session.get() == []


@pytest.mark.asyncio
async def test_multiple_adds(store):
    session = await store.create("test")
    await session.add([_user_event("msg1")])
    await session.add([_assistant_event("msg2")])
    events = await session.get()
    assert len(events) == 2
    assert events[0].content["content"] == "msg1"


# ── Limits ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_with_explicit_limit(store):
    session = await store.create("test")
    await session.add([_user_event(f"msg{i}") for i in range(10)])
    events = await session.get(limit=3)
    assert len(events) == 3
    assert events[0].content["content"] == "msg7"


@pytest.mark.asyncio
async def test_default_limit_from_settings():
    store = SQLiteMultiSessions(settings=SessionSettings(limit=3))
    session = await store.create("test")
    await session.add([_user_event(f"msg{i}") for i in range(10)])
    events = await session.get()
    assert len(events) == 3


# ── Windowed limit must not orphan a tool result (finding: get:307) ────


@pytest.mark.asyncio
async def test_limit_drops_leading_orphan_tool_result(store):
    """A windowed get() that cuts between a tool call and its result must not
    return the orphaned result — Anthropic/Gemini reject a tool_result with no
    preceding tool_use with a 400.

    History: [user, call(c1), result(c1), user]. get(limit=2) → [result, user],
    where the tool call fell outside the window. The orphan result must be
    dropped, leaving [user].
    """
    session = await store.create("test")
    await session.add(
        [
            _user_event("q"),
            _tool_call_event("c1"),
            _tool_result_event("c1"),
            _user_event("next"),
        ]
    )
    events = await session.get(limit=2)
    assert all(e.content.get("type") != "function_call_output" for e in events)
    assert [e.content.get("content") for e in events] == ["next"]


@pytest.mark.asyncio
async def test_limit_keeps_intact_tool_pair(store):
    """When the window contains a complete call/result pair, both are kept."""
    session = await store.create("test")
    await session.add([_user_event("q"), _tool_call_event("c1"), _tool_result_event("c1")])
    events = await session.get(limit=2)
    types = [e.content.get("type") for e in events]
    assert types == ["function_call", "function_call_output"]


@pytest.mark.asyncio
async def test_limit_drops_parallel_orphan_tool_result(store):
    """A parallel-call window can orphan a non-leading tool result; it too must
    be dropped (identity reuse of remove_orphaned_tool_results, not a leading-
    edge-only heuristic)."""
    session = await store.create("test")
    # [call_a, call_b, result_a, result_b]; limit=3 → [call_b, result_a, result_b].
    # result_a's call_a is outside the window → orphan, though not leading.
    await session.add(
        [
            _tool_call_event("a"),
            _tool_call_event("b"),
            _tool_result_event("a"),
            _tool_result_event("b"),
        ]
    )
    events = await session.get(limit=3)
    result_ids = [e.content.get("call_id") for e in events if e.content.get("type") == "function_call_output"]
    # Only result_b survives (its call_b is in the window); orphan result_a is gone.
    assert result_ids == ["b"]


@pytest.mark.asyncio
async def test_no_limit_returns_everything_unfiltered(store):
    """An unbounded get() returns the full stored history verbatim (no orphan
    stripping — nothing was windowed away)."""
    session = await store.create("test")
    await session.add([_tool_result_event("dangling"), _user_event("q")])
    events = await session.get()
    assert len(events) == 2


# ── pop_last ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pop_last(store):
    session = await store.create("test")
    await session.add([_user_event("first"), _assistant_event("second")])
    popped = await session.pop_last()
    assert isinstance(popped, SessionEvent)
    assert popped.content["content"] == "second"
    assert len(await session.get()) == 1


@pytest.mark.asyncio
async def test_pop_last_empty(store):
    session = await store.create("test")
    assert await session.pop_last() is None


# ── clear ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_clear(store):
    session = await store.create("test")
    await session.add([_user_event("msg")])
    await session.clear()
    assert await session.get() == []


# ── Properties ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_id_property(store):
    session = await store.create("conv-42")
    assert session.id == "conv-42"


@pytest.mark.asyncio
async def test_app_name_property():
    store = SQLiteMultiSessions(app_name="myapp")
    session = await store.create("test")
    assert session.app_name == "myapp"


@pytest.mark.asyncio
async def test_user_id_property(store):
    session = await store.create("test", user_id="user-1")
    assert session.user_id == "user-1"


@pytest.mark.asyncio
async def test_settings_property():
    store = SQLiteMultiSessions(settings=SessionSettings(limit=5))
    session = await store.create("test")
    assert session.settings is not None
    assert session.settings.limit == 5


@pytest.mark.asyncio
async def test_state_property(store):
    session = await store.create("test", state={"key": "value"})
    assert session.state["key"] == "value"


# ── Event persistence roundtrip ──────────────────────────────────────


@pytest.mark.asyncio
async def test_event_persistence_roundtrip(tmp_path):
    db_path = tmp_path / "roundtrip.db"
    store = SQLiteMultiSessions(path=db_path)
    session = await store.create("test")

    original = create_session_event(
        author="user",
        content={"role": "user", "content": "hello"},
        state_delta={"mood": "happy"},
    )
    await session.add([original])
    await store.close()

    # Reload
    store2 = SQLiteMultiSessions(path=db_path)
    session2 = await store2.get("test")
    events = await session2.get()
    assert len(events) == 1
    assert events[0].id == original.id
    assert events[0].author == "user"
    assert events[0].content == {"role": "user", "content": "hello"}
    assert events[0].state_delta == {"mood": "happy"}
    assert events[0].timestamp == pytest.approx(original.timestamp, abs=0.01)
    await store2.close()


# ── State persistence ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_state(tmp_path):
    db_path = tmp_path / "state.db"
    store = SQLiteMultiSessions(path=db_path)
    session = await store.create("test")
    session.state["color"] = "blue"
    await session.save_state()
    await store.close()

    store2 = SQLiteMultiSessions(path=db_path)
    session2 = await store2.get("test")
    assert session2.state["color"] == "blue"
    await store2.close()


@pytest.mark.asyncio
async def test_temp_state_not_persisted(tmp_path):
    db_path = tmp_path / "temp.db"
    store = SQLiteMultiSessions(path=db_path)
    session = await store.create("test")
    session.state["temp:scratch"] = "wip"
    session.state["persistent"] = "kept"
    await session.save_state()
    await store.close()

    store2 = SQLiteMultiSessions(path=db_path)
    session2 = await store2.get("test")
    assert session2.state.get("persistent") == "kept"
    assert session2.state.get("temp:scratch") is None
    await store2.close()


@pytest.mark.asyncio
async def test_app_state_shared(tmp_path):
    db_path = tmp_path / "app.db"
    store = SQLiteMultiSessions(path=db_path, app_name="myapp")
    s1 = await store.create("conv-1", user_id="u1")
    s1.state["app:config"] = "v2"
    await s1.save_state()

    s2 = await store.create("conv-2", user_id="u2")
    assert s2.state.get("app:config") == "v2"
    await store.close()


# ── Initial app-scoped state routing (finding: create:280) ────────────


@pytest.mark.asyncio
async def test_initial_app_state_shared_and_survives_first_save(tmp_path):
    """App-scoped keys passed to create() must be routed to the app-state table:
    shared with sibling sessions immediately, and NOT lost on the first save.

    Pre-fix: initial state was dumped verbatim into the session column, so the
    app: key was never written to the app-state table (invisible to siblings)
    and to_persist() dropped it on the first save_state().
    """
    db_path = tmp_path / "initapp.db"
    store = SQLiteMultiSessions(path=db_path, app_name="app1")
    s1 = await store.create("c1", user_id="u1", state={"app:theme": "dark", "local": "x"})

    # Routed to the shared app-state table, not stranded in the session column.
    assert (await store.get_app_state()).get("app:theme") == "dark"
    # Immediately visible to a sibling session of the same app.
    s2 = await store.create("c2", user_id="u2")
    assert s2.state.get("app:theme") == "dark"

    # A first save of a session-scoped change must NOT drop the app: key.
    s1.state["local"] = "y"
    await s1.save_state()
    await store.close()

    store2 = SQLiteMultiSessions(path=db_path, app_name="app1")
    reloaded = await store2.get("c1", user_id="u1")
    assert reloaded is not None
    assert reloaded.state.get("app:theme") == "dark"
    assert reloaded.state.get("local") == "y"
    await store2.close()


@pytest.mark.asyncio
async def test_get_or_create_seeds_app_state_only_on_create(store):
    """get_or_create seeds app-scoped defaults only when it creates the row; an
    existing session keeps the shared app state rather than being overwritten."""
    await store.get_or_create("c1", user_id="u1", state={"app:flag": "on"})
    assert (await store.get_app_state()).get("app:flag") == "on"

    # Second call hits the existing row (INSERT OR IGNORE) — must not clobber.
    await store.get_or_create("c1", user_id="u1", state={"app:flag": "off"})
    assert (await store.get_app_state()).get("app:flag") == "on"


# ── Close ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_close_is_noop(store):
    session = await store.create("test")
    await session.close()
    session2 = await store.create("test2")
    assert session2.id == "test2"


# ── Data-loss guards ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pop_last_raises_on_corrupt_data(store):
    """A corrupt event row must cause pop_last to raise JSONDecodeError.

    The atomic DELETE…RETURNING statement removes and returns the row in one
    round-trip, eliminating the race where two concurrent callers could both
    receive the same event (the higher-severity concurrency bug).  As a
    known trade-off of this atomicity fix, a parse failure on a corrupt
    ``data`` column raises *after* the row is already deleted.  This is
    acceptable: a corrupt row is unreadable regardless, and the error
    surfaces rather than being silently swallowed.
    """
    session = await store.create("test")
    await session.add([_user_event("hello")])

    # Corrupt the stored JSON in the single message row.
    async with session._db.connect() as db:
        await db.execute(f"UPDATE {DEFAULT_MESSAGES_TABLE} SET data = 'NOT_JSON'")
        await db.commit()

    with pytest.raises(json.JSONDecodeError):
        await session.pop_last()


@pytest.mark.asyncio
async def test_save_state_raises_when_session_row_missing(store):
    """save_state must surface a missing session row, not silently drop state.

    Regression: the UPDATE never checked ``rowcount``, so a missing session
    row was a 0-row no-op, yet ``_state.commit()`` marked the state
    persisted — silent state loss. Now raises, and the in-memory delta is
    preserved (not falsely marked persisted).
    """
    session = await store.create("test")
    session.state["k"] = "v"
    assert session.state.has_changes()

    # Remove the underlying session row out from under the bound session.
    async with session._db.connect() as db:
        await db.execute(f"DELETE FROM {DEFAULT_SESSIONS_TABLE}")
        await db.commit()

    with pytest.raises(RuntimeError):
        await session.save_state()

    # The pending delta must NOT have been marked persisted.
    assert session.state.has_changes()


# ── Messages table index ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_messages_table_has_session_index(store):
    """The messages table must carry a composite index on (app_name, user_id, session_id)."""
    session = await store.create("indexed-test")
    # Index must exist after table creation
    async with session._db.connect() as db:
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=:tbl",
            {"tbl": DEFAULT_MESSAGES_TABLE},
        )
        rows = await cursor.fetchall()
        index_names = [row[0] for row in rows]

    assert any("session" in name.lower() for name in index_names), (
        f"Expected a session index on {DEFAULT_MESSAGES_TABLE}, found: {index_names}"
    )


# ── App-state lost-update race (finding: sqlite_session.py:502) ────────


@pytest.mark.asyncio
async def test_concurrent_app_state_writes_preserve_disjoint_keys():
    """Two sessions of the same app writing disjoint app-scoped keys must both
    survive when their save_state() cycles run concurrently.

    The shared in-memory connection serialises access (one transaction at a
    time), so the two save cycles cannot interleave and corrupt each other;
    each writer's per-key SQL merge then applies its key against the live row.
    Disjoint writers never lose each other's keys.

    (This previously forced s1 to pause mid-transaction while s2 completed a
    full cycle on the same connection — a shared-transaction interleave that
    the connection's serialization lock now makes impossible. Per-key merge
    deletion / nested-value correctness is covered by
    ``test_app_state_delete_and_nested_values_via_merge``.)
    """
    store = SQLiteMultiSessions(app_name="app1")
    try:
        s1 = await store.create("c1", user_id="u1")
        s2 = await store.create("c2", user_id="u2")
        s1.state["app:a"] = 1
        s2.state["app:b"] = 2

        await asyncio.gather(s1.save_state(), s2.save_state())

        final = await store.get_app_state()
        assert final.get("app:a") == 1
        assert final.get("app:b") == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_app_state_delete_and_nested_values_via_merge(tmp_path):
    """Per-key app-state merge handles deletion and nested values, and a
    deletion of one key does not drop a sibling key."""
    db_path = tmp_path / "appstate.db"
    store = SQLiteMultiSessions(path=db_path, app_name="app1")
    session = await store.create("c1", user_id="u1")

    session.state["app:keep"] = {"nested": [1, 2]}
    session.state["app:drop"] = "gone-soon"
    await session.save_state()
    assert (await store.get_app_state()) == {
        "app:keep": {"nested": [1, 2]},
        "app:drop": "gone-soon",
    }

    del session.state["app:drop"]
    await session.save_state()
    assert (await store.get_app_state()) == {"app:keep": {"nested": [1, 2]}}
    await store.close()


# ── pop_last / clear bump updated_at (finding: sqlite_session.py:397) ──


@pytest.mark.asyncio
async def test_pop_last_bumps_updated_at(store):
    """pop_last() mutates the session, so SessionInfo.updated_at must advance."""
    session = await store.create("s", user_id="u")
    await session.add([_user_event("a"), _user_event("b")])
    before = (await store.list(user_id="u"))[0].updated_at

    await asyncio.sleep(0.01)
    popped = await session.pop_last()
    assert popped is not None

    after = (await store.list(user_id="u"))[0].updated_at
    assert after > before, "pop_last() left updated_at stale"


@pytest.mark.asyncio
async def test_clear_bumps_updated_at(store):
    """clear() removes events, so SessionInfo.updated_at must advance."""
    session = await store.create("s", user_id="u")
    await session.add([_user_event("a")])
    before = (await store.list(user_id="u"))[0].updated_at

    await asyncio.sleep(0.01)
    await session.clear()

    after = (await store.list(user_id="u"))[0].updated_at
    assert after > before, "clear() left updated_at stale"


@pytest.mark.asyncio
async def test_pop_last_empty_does_not_bump_updated_at(store):
    """Popping an empty session changes nothing, so updated_at must not move."""
    session = await store.create("s", user_id="u")
    before = (await store.list(user_id="u"))[0].updated_at

    await asyncio.sleep(0.01)
    assert await session.pop_last() is None

    after = (await store.list(user_id="u"))[0].updated_at
    assert after == before


@pytest.mark.asyncio
async def test_clear_empty_does_not_bump_updated_at(store):
    """Clearing an already-empty session must not falsify updated_at."""
    session = await store.create("s", user_id="u")
    before = (await store.list(user_id="u"))[0].updated_at

    await asyncio.sleep(0.01)
    await session.clear()

    after = (await store.list(user_id="u"))[0].updated_at
    assert after == before


@pytest.mark.asyncio
async def test_strict_handle_pop_then_add_no_false_positive(store):
    """A strict handle that pops then adds on itself must not raise: pop_last
    refreshes the handle's own watermark to the bump it just caused."""
    await store.create("s", user_id="u")
    async with store._db.connect() as db:
        watermark = await get_session_updated_at(db, "", "u", "s")

    handle = SQLiteSession(
        session_id="s",
        app_name="",
        user_id="u",
        db=store._db,
        state=State(),
        strict_concurrency=True,
        updated_at_watermark=watermark,
    )
    await handle.add([_user_event("1"), _user_event("2")])
    await handle.pop_last()
    # Must NOT raise SessionAppendConflictError on the handle's own pop.
    await handle.add([_user_event("3")])


@pytest.mark.asyncio
async def test_strict_add_detects_concurrent_pop(store):
    """A concurrent pop_last() by another writer advances updated_at, so a
    strict handle's next add() detects the conflict."""
    session = await store.create("s", user_id="u")
    await session.add([_user_event("1"), _user_event("2")])

    async with store._db.connect() as db:
        watermark = await get_session_updated_at(db, "", "u", "s")

    strict = SQLiteSession(
        session_id="s",
        app_name="",
        user_id="u",
        db=store._db,
        state=State(),
        strict_concurrency=True,
        updated_at_watermark=watermark,
    )

    await asyncio.sleep(0.01)
    await session.pop_last()  # another writer mutates the session

    with pytest.raises(SessionAppendConflictError):
        await strict.add([_user_event("3")])
