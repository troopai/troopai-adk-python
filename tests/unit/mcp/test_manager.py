"""Tests for ``troopai.adk.mcp.manager.MCPServerManager``."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.mcp.exceptions import MCPConnectionError
from troopai.adk.mcp.manager import MCPServerManager


def _server(name: str) -> MagicMock:
    s = MagicMock()
    s.name = name
    s.connect = AsyncMock()
    s.cleanup = AsyncMock()
    return s


async def test_connect_all_calls_connect_on_every_server() -> None:
    a, b = _server("a"), _server("b")
    manager = MCPServerManager(servers=[a, b])

    await manager.connect_all()

    a.connect.assert_awaited_once()
    b.connect.assert_awaited_once()


async def test_connect_all_idempotent() -> None:
    a = _server("a")
    manager = MCPServerManager(servers=[a])

    await manager.connect_all()
    await manager.connect_all()

    a.connect.assert_awaited_once()


async def test_connect_all_rolls_back_on_failure() -> None:
    a, b = _server("a"), _server("b")
    b.connect.side_effect = RuntimeError("nope")
    manager = MCPServerManager(servers=[a, b])

    with pytest.raises(MCPConnectionError):
        await manager.connect_all(parallel=False)

    # ``a`` connected then was rolled back
    a.connect.assert_awaited_once()
    a.cleanup.assert_awaited_once()


async def test_cleanup_all_swallows_individual_errors() -> None:
    a, b = _server("a"), _server("b")
    a.cleanup.side_effect = RuntimeError("a-cleanup")
    manager = MCPServerManager(servers=[a, b])
    await manager.connect_all()

    await manager.cleanup_all()  # MUST NOT raise

    a.cleanup.assert_awaited_once()
    b.cleanup.assert_awaited_once()


async def test_async_with_connects_and_cleans_up() -> None:
    a = _server("a")
    manager = MCPServerManager(servers=[a])

    async with manager:
        a.connect.assert_awaited_once()
        a.cleanup.assert_not_called()
    a.cleanup.assert_awaited_once()


async def test_ensure_connected_rejects_unknown_server() -> None:
    a, b = _server("a"), _server("b")
    manager = MCPServerManager(servers=[a])

    with pytest.raises(MCPConnectionError):
        await manager.ensure_connected(b)


async def test_cleanup_all_runs_in_reverse_registration_order() -> None:
    """LIFO disposal is required because anyio cancel scopes stack
    per task — see ``MCPServerManager.cleanup_all`` docstring.
    """
    order: list[str] = []
    a, b, c = _server("a"), _server("b"), _server("c")
    a.cleanup.side_effect = lambda: order.append("a")
    b.cleanup.side_effect = lambda: order.append("b")
    c.cleanup.side_effect = lambda: order.append("c")
    manager = MCPServerManager(servers=[a, b, c])
    await manager.connect_all(parallel=False)

    await manager.cleanup_all()

    assert order == ["c", "b", "a"], (
        f"cleanup_all MUST iterate in reverse to honour anyio's LIFO cancel-scope invariant; got {order}"
    )


async def test_connect_all_rollback_runs_in_reverse_order() -> None:
    """On a sequential connect failure, already-connected servers are
    rolled back in reverse — the last successful connect is cleaned
    up first to honour anyio LIFO scope nesting.
    """
    order: list[str] = []
    a, b, c = _server("a"), _server("b"), _server("c")
    a.cleanup.side_effect = lambda: order.append("a")
    b.cleanup.side_effect = lambda: order.append("b")
    c.connect.side_effect = RuntimeError("third one explodes")
    manager = MCPServerManager(servers=[a, b, c])

    with pytest.raises(MCPConnectionError):
        await manager.connect_all(parallel=False)

    assert order == ["b", "a"], (
        "Rollback MUST iterate the already-connected servers in reverse to honour "
        f"the anyio LIFO invariant; got {order}"
    )


# --------------------------------- ensure_connected ref-count regression


async def test_ensure_connected_increments_ref_count() -> None:
    """Regression: ensure_connected was connecting but never incrementing
    _ref_counts, so a subsequent release() became a no-op and the server
    was never cleaned up (connection/subprocess leak).
    """
    s = _server("a")
    manager = MCPServerManager(servers=[s])

    await manager.ensure_connected(s)

    assert manager.get_ref_count(s) == 1, "ensure_connected MUST increment _ref_counts so release() can clean up"
    s.connect.assert_awaited_once()


async def test_ensure_connected_then_release_cleans_up() -> None:
    """After ensure_connected, release() MUST decrement to zero and call cleanup."""
    s = _server("a")
    manager = MCPServerManager(servers=[s])

    await manager.ensure_connected(s)
    await manager.release(s)

    s.cleanup.assert_awaited_once()
    assert manager.get_ref_count(s) == 0


async def test_ensure_connected_is_idempotent_when_already_acquired() -> None:
    """ensure_connected is a no-op when the server already has a ref."""
    s = _server("a")
    manager = MCPServerManager(servers=[s])

    await manager.acquire(s)
    await manager.ensure_connected(s)  # Should not connect again

    s.connect.assert_awaited_once()  # Still only one connect call
    assert manager.get_ref_count(s) == 1  # ref count unchanged by ensure_connected


# --------------------------------- parallel=True same-task invariant


async def test_connect_all_parallel_true_connects_all_servers() -> None:
    """parallel=True must still connect every server (same-task sequential path)."""
    a, b, c = _server("a"), _server("b"), _server("c")
    manager = MCPServerManager(servers=[a, b, c])

    await manager.connect_all(parallel=True)

    a.connect.assert_awaited_once()
    b.connect.assert_awaited_once()
    c.connect.assert_awaited_once()
    assert manager.is_active is True


async def test_connect_all_parallel_true_rolls_back_in_reverse_order() -> None:
    """parallel=True rollback MUST run cleanup in reverse order (anyio LIFO invariant).

    Before the fix, connect_all(parallel=True) used asyncio.create_task() to spawn
    server.connect() in separate tasks. cleanup_all() then ran cleanup() from the
    original (main) task — crossing the anyio cancel-scope task boundary and
    silently leaking the transport resources. Both the cross-task spawning and
    the rollback order are verified here.
    """
    import asyncio

    order: list[str] = []
    connect_tasks: list[object] = []
    a, b, c = _server("a"), _server("b"), _server("c")

    original_connect_a = a.connect
    original_connect_b = b.connect

    async def tracked_connect_a() -> None:
        connect_tasks.append(asyncio.current_task())
        await original_connect_a()

    async def tracked_connect_b() -> None:
        connect_tasks.append(asyncio.current_task())
        await original_connect_b()

    a.connect = tracked_connect_a
    b.connect = tracked_connect_b

    a.cleanup.side_effect = lambda: order.append("a")
    b.cleanup.side_effect = lambda: order.append("b")
    c.connect.side_effect = RuntimeError("c explodes")

    manager = MCPServerManager(servers=[a, b, c])
    caller_task = asyncio.current_task()

    with pytest.raises(MCPConnectionError):
        await manager.connect_all(parallel=True)

    # All connects MUST run in the calling task, not in spawned child tasks.
    for t in connect_tasks:
        assert t is caller_task, (
            "connect_all MUST NOT spawn server.connect() in a separate task — "
            "anyio cancel scopes require cleanup to run in the same task as connect"
        )

    # Rollback must be LIFO: b connected after a, so b is cleaned up first.
    assert order == ["b", "a"], f"Rollback must be LIFO to honour anyio cancel-scope stacking; got {order}"


# --------------------------------- cancellation propagation regression


async def test_connect_all_propagates_cancelled_error() -> None:
    """Regression: a CancelledError raised mid-connect was caught by the
    ``except BaseException`` arm, accumulated into ``errors``, and re-raised
    as ``MCPConnectionError`` after attempting the rest of the fleet. The
    CancelledError type was erased and cancellation was delayed. It MUST now
    propagate immediately.
    """
    a, b, c = _server("a"), _server("b"), _server("c")
    b.connect.side_effect = asyncio.CancelledError()
    manager = MCPServerManager(servers=[a, b, c])

    with pytest.raises(asyncio.CancelledError):
        await manager.connect_all(parallel=False)

    # ``c`` MUST NOT be attempted — cancellation is honoured immediately.
    c.connect.assert_not_called()
    # ``a`` connected before the cancel; it MUST be rolled back.
    a.connect.assert_awaited_once()
    a.cleanup.assert_awaited_once()
    # Manager stays inactive (never reached the success path).
    assert manager.is_active is False


async def test_connect_all_cancelled_rolls_back_in_reverse_order() -> None:
    """On cancellation mid-connect, already-connected servers are rolled
    back in reverse (anyio LIFO cancel-scope invariant).
    """
    order: list[str] = []
    a, b, c = _server("a"), _server("b"), _server("c")
    a.cleanup.side_effect = lambda: order.append("a")
    b.cleanup.side_effect = lambda: order.append("b")
    c.connect.side_effect = asyncio.CancelledError()
    manager = MCPServerManager(servers=[a, b, c])

    with pytest.raises(asyncio.CancelledError):
        await manager.connect_all(parallel=False)

    assert order == ["b", "a"], (
        "Cancellation rollback MUST iterate already-connected servers in reverse "
        f"to honour the anyio LIFO invariant; got {order}"
    )


# --------------------------------- cleanup_all drains ref-counted servers


async def test_cleanup_all_cleans_up_ensure_connected_server() -> None:
    """Regression: cleanup_all early-returned when ``_connected`` was False,
    silently leaking servers connected via ``ensure_connected`` (which records
    a ref count but never sets ``_connected``). cleanup_all MUST drain them.
    """
    s = _server("s")
    manager = MCPServerManager(servers=[s])

    await manager.ensure_connected(s)
    assert manager.is_active is False  # ref-counted path, not connect_all

    await manager.cleanup_all()

    s.cleanup.assert_awaited_once()
    assert manager.get_ref_count(s) == 0


async def test_cleanup_all_cleans_up_acquired_server() -> None:
    """cleanup_all also drains servers held via ``acquire`` ref counting."""
    s = _server("s")
    manager = MCPServerManager(servers=[s])

    await manager.acquire(s)

    await manager.cleanup_all()

    s.cleanup.assert_awaited_once()
    assert manager.get_ref_count(s) == 0


async def test_cleanup_all_skips_unconnected_servers() -> None:
    """cleanup_all only cleans up servers that are actually live: a registered
    but never-connected server (no ref count, not in connect_all) is skipped.
    """
    a, b = _server("a"), _server("b")
    manager = MCPServerManager(servers=[a, b])

    await manager.ensure_connected(a)  # only ``a`` is live

    await manager.cleanup_all()

    a.cleanup.assert_awaited_once()
    b.cleanup.assert_not_called()


async def test_cleanup_all_drains_ref_counted_in_reverse_order() -> None:
    """Ref-counted teardown via cleanup_all is LIFO over registration order."""
    order: list[str] = []
    a, b, c = _server("a"), _server("b"), _server("c")
    a.cleanup.side_effect = lambda: order.append("a")
    b.cleanup.side_effect = lambda: order.append("b")
    c.cleanup.side_effect = lambda: order.append("c")
    manager = MCPServerManager(servers=[a, b, c])

    await manager.ensure_connected(a)
    await manager.ensure_connected(b)
    await manager.ensure_connected(c)

    await manager.cleanup_all()

    assert order == ["c", "b", "a"], f"ref-counted cleanup_all MUST be LIFO; got {order}"


async def test_cleanup_all_idempotent_with_ref_counts() -> None:
    """A second cleanup_all after ref-counted teardown is a clean no-op."""
    s = _server("s")
    manager = MCPServerManager(servers=[s])

    await manager.ensure_connected(s)
    await manager.cleanup_all()
    await manager.cleanup_all()  # MUST NOT raise or re-clean

    s.cleanup.assert_awaited_once()


async def test_cleanup_all_still_cleans_connect_all_fleet() -> None:
    """connect_all lifecycle teardown is unchanged: every server is cleaned up."""
    a, b = _server("a"), _server("b")
    manager = MCPServerManager(servers=[a, b])
    await manager.connect_all(parallel=False)

    await manager.cleanup_all()

    a.cleanup.assert_awaited_once()
    b.cleanup.assert_awaited_once()
    assert manager.is_active is False
