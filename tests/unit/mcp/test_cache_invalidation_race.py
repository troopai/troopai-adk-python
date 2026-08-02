"""An invalidation arriving mid-fetch must not be clobbered by the fetch.

Regression: ``list_tools`` snapshotted nothing and unconditionally set
``_cache_dirty = False`` after awaiting ``ClientSession.list_tools()``. A
``tools/list_changed`` notification (or an explicit ``invalidate_tools_cache``)
landing *during* that await flipped the flag to ``True``, only for the
post-fetch line to immediately clobber it back to ``False`` — so the next
``list_tools`` served the stale, pre-change list. ``list_tools`` now snapshots
a monotonic invalidation counter before the await and keeps the cache dirty
when the counter moved underneath it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from troopai.adk.mcp import MCPServerStdio, MCPServerStdioParams
from troopai.adk.mcp.mcp_server import MCPServerWithClientSession


class _InvalidatingSession:
    """Fake ``ClientSession`` whose first fetch fires an invalidation mid-flight.

    The mid-fetch invalidation models a ``tools/list_changed`` notification
    handled on a concurrent task while ``list_tools`` awaits the round-trip.
    """

    def __init__(self, server: MCPServerWithClientSession) -> None:
        self._server = server
        self.call_count = 0

    async def list_tools(self) -> Any:
        self.call_count += 1
        if self.call_count == 1:
            # The change notification lands WHILE this fetch is in flight.
            self._server.invalidate_tools_cache()
        return SimpleNamespace(tools=[MagicMock()])


async def test_invalidation_during_fetch_is_not_clobbered() -> None:
    """A fetch racing an invalidation leaves the cache dirty, forcing a re-fetch."""
    server = MCPServerStdio(name="x", params=MCPServerStdioParams(command="echo"))
    fake = _InvalidatingSession(server)
    # White-box: inject the fake session the cache reads through.
    server._session = fake  # type: ignore[assignment]

    await server.list_tools()
    assert fake.call_count == 1

    # The invalidation arrived mid-fetch, so this MUST re-fetch rather than
    # serve the now-stale cached list.
    await server.list_tools()
    assert fake.call_count == 2

    # Fetch #2 raced no invalidation, so the cache is warm again: a third call
    # serves it without another round-trip (normal caching still works).
    await server.list_tools()
    assert fake.call_count == 2
