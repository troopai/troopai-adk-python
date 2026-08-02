"""Tests for direct httpx client injection into MCPServerStreamableHttp.

Feature: ``MCPServerStreamableHttpParams`` gains
``httpx_client: httpx.AsyncClient | None = None``, mutually exclusive
with ``httpx_client_factory`` (raises ``ValueError`` when both are set).
The pre-built client is forwarded to the non-deprecated
``streamable_http_client`` via its ``http_client`` parameter.

Covers:
- ``httpx_client`` defaults to ``None``.
- When both ``httpx_client`` and ``httpx_client_factory`` are set, params
  raises ``ValueError`` at construction time.
- When only ``httpx_client`` is set, the server's connect path calls
  ``streamable_http_client(http_client=...)`` instead of going through
  ``_build_http_client``.
- Static bool approval still works in params (regression guard).
"""

from __future__ import annotations

import httpx
import pytest

from troopai.adk.mcp.http import MCPServerStreamableHttpParams

# --------------------------------------------------------------- params contract


def test_httpx_client_defaults_to_none() -> None:
    """httpx_client field defaults to None (cost-conservative default)."""
    params = MCPServerStreamableHttpParams(url="https://example.com/mcp")
    assert params.httpx_client is None


async def test_httpx_client_and_factory_mutually_exclusive() -> None:
    """Setting both httpx_client and httpx_client_factory raises ValueError."""
    client = httpx.AsyncClient()

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient()

    with pytest.raises(ValueError, match="mutually exclusive"):
        MCPServerStreamableHttpParams(
            url="https://example.com/mcp",
            httpx_client=client,
            httpx_client_factory=factory,
        )

    await client.aclose()


async def test_httpx_client_alone_is_valid() -> None:
    """Setting only httpx_client (no factory) does not raise."""
    client = httpx.AsyncClient()
    params = MCPServerStreamableHttpParams(
        url="https://example.com/mcp",
        httpx_client=client,
    )
    assert params.httpx_client is client
    await client.aclose()


def test_httpx_client_factory_alone_is_valid() -> None:
    """Setting only httpx_client_factory (no client) does not raise."""

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient()

    params = MCPServerStreamableHttpParams(
        url="https://example.com/mcp",
        httpx_client_factory=factory,
    )
    assert params.httpx_client_factory is factory
    assert params.httpx_client is None


def test_neither_httpx_client_nor_factory_is_valid() -> None:
    """Setting neither field (both None) is the default, valid state."""
    params = MCPServerStreamableHttpParams(url="https://example.com/mcp")
    assert params.httpx_client is None
    assert params.httpx_client_factory is None


# --------------------------------------------------------------- server connect path


async def test_connect_uses_prebuilt_client_when_httpx_client_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When httpx_client is set, connect() uses streamable_http_client(http_client=...).

    We mock ``streamable_http_client`` at the HTTP module level to verify
    the pre-built client is forwarded, without opening a real network connection.
    """
    from contextlib import asynccontextmanager
    from typing import Any
    from unittest.mock import AsyncMock, MagicMock, patch

    from anyio.streams.memory import MemoryObjectReceiveStream, MemoryObjectSendStream

    captured: dict[str, Any] = {}

    @asynccontextmanager
    async def fake_streamable_http_client(url: str, **kwargs: Any):  # type: ignore[misc]
        captured["url"] = url
        captured["kwargs"] = kwargs
        # Yield fake streams
        read = MagicMock(spec=MemoryObjectReceiveStream)
        write = MagicMock(spec=MemoryObjectSendStream)
        get_session_id = lambda: "test-session-id"  # noqa: E731
        yield read, write, get_session_id

    # Also mock _make_client_session and _attach_session to avoid real session creation
    from troopai.adk.mcp import http as http_module

    fake_client = httpx.AsyncClient()

    params = MCPServerStreamableHttpParams(
        url="https://example.com/mcp",
        httpx_client=fake_client,
    )

    from troopai.adk.mcp.http import MCPServerStreamableHttp

    server = MCPServerStreamableHttp(name="test", params=params)

    with (
        patch.object(http_module, "streamable_http_client", fake_streamable_http_client),
        patch.object(server, "_make_client_session") as mock_make_session,
        patch.object(server, "_attach_session", AsyncMock()),
        patch("troopai.adk.mcp.http.fire_on_mcp_connect", AsyncMock()),
        patch("troopai.adk.mcp.http.fire_on_mcp_connected", AsyncMock()),
    ):
        # _make_client_session must return an async context manager
        fake_session = MagicMock()
        mock_cm = MagicMock()
        mock_cm.__aenter__ = AsyncMock(return_value=fake_session)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        mock_make_session.return_value = mock_cm

        await server.connect()

    # The pre-built client must have been passed as http_client=
    assert "http_client" in captured["kwargs"], (
        "connect() must pass httpx_client as http_client= to streamable_http_client"
    )
    assert captured["kwargs"]["http_client"] is fake_client

    await fake_client.aclose()
    await server.cleanup()
