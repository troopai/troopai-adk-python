"""Vercel hosted-bridge client tests — team_id query-scope wiring.

The Vercel REST API scopes team resources via the ``teamId`` query
parameter. These tests pin that a configured ``team_id`` is actually
sent on the create POST (and that the documented project_id invariant
is enforced), rather than silently dropped.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.exceptions.exceptions import SandboxStartFailed
from troopai.adk.sandbox.clients.hosted.vercel import (
    VercelSandboxClient,
    VercelSandboxClientOptions,
)


def _make_http_client(*, status_code: int = 200, body: dict[str, Any] | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body or {}
    response.text = "" if body is None else str(body)
    http = MagicMock()
    http.post = AsyncMock(return_value=response)
    return http


async def test_create_sends_team_id_query_param() -> None:
    http = _make_http_client(body={"sandboxId": "sbx-1"})
    client = VercelSandboxClient(http_client=http)
    await client.create(
        options=VercelSandboxClientOptions(
            api_key="key-1",
            project_id="prj-1",
            team_id="team-1",
        ),
    )
    http.post.assert_called_once()
    params = http.post.call_args.kwargs["params"]
    assert params == {"teamId": "team-1"}


async def test_create_without_team_id_sends_no_team_param() -> None:
    http = _make_http_client(body={"sandboxId": "sbx-1"})
    client = VercelSandboxClient(http_client=http)
    await client.create(
        options=VercelSandboxClientOptions(api_key="key-1", project_id="prj-1"),
    )
    http.post.assert_called_once()
    params = http.post.call_args.kwargs["params"]
    assert "teamId" not in params


async def test_create_empty_team_id_sends_no_team_param() -> None:
    http = _make_http_client(body={"sandboxId": "sbx-1"})
    client = VercelSandboxClient(http_client=http)
    await client.create(
        options=VercelSandboxClientOptions(api_key="key-1", project_id="prj-1", team_id=""),
    )
    http.post.assert_called_once()
    params = http.post.call_args.kwargs["params"]
    assert "teamId" not in params


async def test_create_team_id_without_project_id_raises() -> None:
    http = _make_http_client(body={"sandboxId": "sbx-1"})
    client = VercelSandboxClient(http_client=http)
    with pytest.raises(SandboxStartFailed, match="project_id is required"):
        await client.create(
            options=VercelSandboxClientOptions(api_key="key-1", team_id="team-1"),
        )
    http.post.assert_not_called()
