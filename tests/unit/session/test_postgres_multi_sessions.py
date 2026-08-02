"""Real-database tests for ``PostgresMultiSessions`` via pytest-postgresql.

Requires PostgreSQL + psycopg[binary,pool] — both present in the test
environment. Tests do NOT skip; they fail hard if the infra is absent.
"""

from __future__ import annotations

import pytest
from pytest_postgresql.factories import postgresql, postgresql_proc

from troopai.adk.session.postgres_multi_sessions import PostgresMultiSessions
from troopai.adk.session.session_event import create_session_event

pytestmark = pytest.mark.postgres

postgresql_my_proc = postgresql_proc()
postgresql_my = postgresql("postgresql_my_proc")


@pytest.fixture
def conninfo(postgresql_my) -> str:
    info = postgresql_my.info
    parts = [f"dbname={info.dbname}", f"user={info.user}", f"host={info.host}", f"port={info.port}"]
    if info.password is not None and len(info.password) > 0:
        parts.append(f"password={info.password}")
    return " ".join(parts)


async def test_create_and_get_round_trip(conninfo: str) -> None:
    manager = PostgresMultiSessions(conninfo, app_name="app")
    try:
        created = await manager.create("c1", user_id="u1", state={"k": "v"})
        assert created.id == "c1"
        fetched = await manager.get("c1", user_id="u1")
        assert fetched is not None
        assert fetched.state["k"] == "v"
    finally:
        await manager.close()


async def test_create_duplicate_raises(conninfo: str) -> None:
    manager = PostgresMultiSessions(conninfo, app_name="app")
    try:
        await manager.create("c1", user_id="u1")
        with pytest.raises(ValueError):
            await manager.create("c1", user_id="u1")
    finally:
        await manager.close()


async def test_get_missing_returns_none(conninfo: str) -> None:
    manager = PostgresMultiSessions(conninfo, app_name="app")
    try:
        assert await manager.get("nope", user_id="u1") is None
    finally:
        await manager.close()


async def test_get_or_create_both_branches(conninfo: str) -> None:
    manager = PostgresMultiSessions(conninfo, app_name="app")
    try:
        first = await manager.get_or_create("c1", user_id="u1", state={"k": 1})
        first.state["k"] = 2
        await first.save_state()
        # Second call must return the existing row, not re-create it.
        second = await manager.get_or_create("c1", user_id="u1", state={"k": 99})
        assert second.state["k"] == 2
        assert await manager.count(user_id="u1") == 1
    finally:
        await manager.close()


async def test_add_and_get_events_in_order(conninfo: str) -> None:
    manager = PostgresMultiSessions(conninfo, app_name="app")
    try:
        session = await manager.get_or_create("c1", user_id="u1")
        ev1 = create_session_event("user", {"role": "user", "content": "hi"})
        ev2 = create_session_event("assistant", {"role": "assistant", "content": "hello"})
        await session.add([ev1, ev2])
        events = await session.get()
        assert [e.author for e in events] == ["user", "assistant"]
        assert events[0].id == ev1.id
        assert (await session.get(limit=1))[0].author == "assistant"
    finally:
        await manager.close()


async def test_state_persists_across_reload(conninfo: str) -> None:
    manager = PostgresMultiSessions(conninfo, app_name="app")
    try:
        session = await manager.create("c1", user_id="u1")
        session.state["score"] = 42
        await session.save_state()
        reloaded = await manager.get("c1", user_id="u1")
        assert reloaded is not None
        assert reloaded.state["score"] == 42
    finally:
        await manager.close()


async def test_app_state_shared_across_sessions(conninfo: str) -> None:
    manager = PostgresMultiSessions(conninfo, app_name="app")
    try:
        first = await manager.get_or_create("c1", user_id="u1")
        first.state["app:theme"] = "dark"
        await first.save_state()
        second = await manager.get_or_create("c2", user_id="u1")
        assert second.state["app:theme"] == "dark"
    finally:
        await manager.close()


async def test_pop_last_and_clear(conninfo: str) -> None:
    manager = PostgresMultiSessions(conninfo, app_name="app")
    try:
        session = await manager.get_or_create("c1", user_id="u1")
        await session.add([create_session_event("user", {"role": "user", "content": "a"})])
        popped = await session.pop_last()
        assert popped is not None
        assert popped.author == "user"
        assert await session.pop_last() is None
        await session.add([create_session_event("user", {"role": "user", "content": "b"})])
        await session.clear()
        assert len(await session.get()) == 0
    finally:
        await manager.close()


async def test_list_count_delete(conninfo: str) -> None:
    manager = PostgresMultiSessions(conninfo, app_name="app")
    try:
        await manager.create("c1", user_id="u1")
        await manager.create("c2", user_id="u1")
        assert await manager.count(user_id="u1") == 2
        assert {info.session_id for info in await manager.list(user_id="u1")} == {"c1", "c2"}
        assert await manager.delete("c1", user_id="u1") is True
        assert await manager.delete("c1", user_id="u1") is False
        assert await manager.count(user_id="u1") == 1
    finally:
        await manager.close()


async def test_close_is_idempotent(conninfo: str) -> None:
    manager = PostgresMultiSessions(conninfo, app_name="app")
    await manager.create("c1", user_id="u1")
    await manager.close()
    await manager.close()


def test_empty_conninfo_rejected() -> None:
    with pytest.raises(ValueError):
        PostgresMultiSessions("")


async def test_limit_drops_leading_orphan_tool_result(conninfo: str) -> None:
    """A windowed get() that slices between a tool call and its result must not
    return the orphaned result (Anthropic/Gemini reject it with 400)."""
    manager = PostgresMultiSessions(conninfo, app_name="app")
    try:
        session = await manager.get_or_create("c1", user_id="u1")
        await session.add(
            [
                create_session_event("user", {"role": "user", "content": "q"}),
                create_session_event(
                    "assistant", {"type": "function_call", "call_id": "c1", "name": "t", "arguments": "{}"}
                ),
                create_session_event("tool", {"type": "function_call_output", "call_id": "c1", "output": "r"}),
                create_session_event("user", {"role": "user", "content": "next"}),
            ]
        )
        events = await session.get(limit=2)
        assert all(e.content.get("type") != "function_call_output" for e in events)
        assert [e.content.get("content") for e in events] == ["next"]
    finally:
        await manager.close()


async def test_initial_app_state_routed_to_app_table(conninfo: str) -> None:
    """App-scoped keys passed to create() must land in the app-state table so
    they are shared with siblings and survive the first save_state().

    Pre-fix: initial state was dumped verbatim into the session column, so the
    app: key never reached the app-state table and was dropped by to_persist().
    """
    manager = PostgresMultiSessions(conninfo, app_name="app")
    try:
        s1 = await manager.create("c1", user_id="u1", state={"app:theme": "dark", "local": "x"})
        assert (await manager.get_app_state()).get("app:theme") == "dark"

        s2 = await manager.create("c2", user_id="u2")
        assert s2.state.get("app:theme") == "dark"

        # First save of a session-scoped change must not drop the app: key.
        s1.state["local"] = "y"
        await s1.save_state()
        reloaded = await manager.get("c1", user_id="u1")
        assert reloaded is not None
        assert reloaded.state.get("app:theme") == "dark"
        assert reloaded.state.get("local") == "y"
    finally:
        await manager.close()


async def test_get_or_create_seeds_app_state_only_on_create(conninfo: str) -> None:
    """get_or_create seeds app-scoped defaults only when it creates the row."""
    manager = PostgresMultiSessions(conninfo, app_name="app")
    try:
        await manager.get_or_create("c1", user_id="u1", state={"app:flag": "on"})
        assert (await manager.get_app_state()).get("app:flag") == "on"
        # Second call hits the existing row — must not clobber shared app state.
        await manager.get_or_create("c1", user_id="u1", state={"app:flag": "off"})
        assert (await manager.get_app_state()).get("app:flag") == "on"
    finally:
        await manager.close()
