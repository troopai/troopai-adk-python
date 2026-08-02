"""Deep tests for hosted-bridge providers (E2B + 6 others).

Each provider gets a parametrized check:
- create() raises SandboxStartFailed when api_key is None.
- create() POSTs the right body to the right endpoint when api_key set.
- create() returns a RemoteVMSandboxSession bound to the returned id.
- resume() requires provider_payload['sandbox_id'].
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.exceptions.exceptions import SandboxStartFailed
from troopai.adk.types.sandbox.manifest import Manifest
from troopai.adk.types.sandbox.session_state import SandboxSessionState

# (module_path, client_cls, options_cls, expected_endpoint, id_field).
PROVIDERS: list[tuple[str, str, str, str, str]] = [
    ("e2b", "E2bSandboxClient", "E2bSandboxClientOptions", "/sandboxes", "sandboxID"),
    ("vercel", "VercelSandboxClient", "VercelSandboxClientOptions", "/v1/sandboxes", "sandboxId"),
    ("modal", "ModalSandboxClient", "ModalSandboxClientOptions", "/v1/sandboxes", "sandbox_id"),
    ("daytona", "DaytonaSandboxClient", "DaytonaSandboxClientOptions", "/sandboxes", "sandboxId"),
    ("cloudflare", "CloudflareSandboxClient", "CloudflareSandboxClientOptions", "/sandboxes", "sandbox_id"),
    ("blaxel", "BlaxelSandboxClient", "BlaxelSandboxClientOptions", "/sandboxes", "sandbox_id"),
    ("runloop", "RunloopSandboxClient", "RunloopSandboxClientOptions", "/v1/devboxes", "id"),
]


def _import(module_name: str, *names: str) -> tuple[Any, ...]:
    import importlib

    module = importlib.import_module(f"troopai.adk.sandbox.clients.hosted.{module_name}")
    return tuple(getattr(module, n) for n in names)


def _make_http_client(*, status_code: int = 200, body: dict[str, Any] | None = None) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = body or {}
    response.text = "" if body is None else str(body)
    http = MagicMock()
    http.post = AsyncMock(return_value=response)
    return http


@pytest.mark.parametrize(
    "provider,client_name,options_name,endpoint,id_field",
    PROVIDERS,
)
class TestHostedReal:
    @pytest.mark.asyncio
    async def test_create_without_api_key_raises(
        self,
        provider: str,
        client_name: str,
        options_name: str,
        endpoint: str,
        id_field: str,
    ) -> None:
        del endpoint, id_field
        client_cls, options_cls = _import(provider, client_name, options_name)
        client = client_cls(http_client=_make_http_client())
        with pytest.raises(SandboxStartFailed, match="api_key"):
            await client.create(options=options_cls())

    @pytest.mark.asyncio
    async def test_create_posts_to_endpoint(
        self,
        provider: str,
        client_name: str,
        options_name: str,
        endpoint: str,
        id_field: str,
    ) -> None:
        client_cls, options_cls = _import(provider, client_name, options_name)
        http = _make_http_client(body={id_field: "sandbox-xyz"})
        client = client_cls(http_client=http)
        session = await client.create(options=options_cls(api_key="key-1"))
        assert session is not None
        http.post.assert_called_once()
        # First positional arg is the endpoint path.
        called_endpoint = http.post.call_args.args[0]
        assert called_endpoint == endpoint

    @pytest.mark.asyncio
    async def test_create_uses_id_from_response(
        self,
        provider: str,
        client_name: str,
        options_name: str,
        endpoint: str,
        id_field: str,
    ) -> None:
        del endpoint
        client_cls, options_cls = _import(provider, client_name, options_name)
        http = _make_http_client(body={id_field: "sandbox-abc"})
        client = client_cls(http_client=http)
        session = await client.create(options=options_cls(api_key="key-1"))
        assert session.session_id == "sandbox-abc"

    @pytest.mark.asyncio
    async def test_create_missing_id_raises(
        self,
        provider: str,
        client_name: str,
        options_name: str,
        endpoint: str,
        id_field: str,
    ) -> None:
        del endpoint, id_field
        client_cls, options_cls = _import(provider, client_name, options_name)
        http = _make_http_client(body={})  # no id field
        client = client_cls(http_client=http)
        with pytest.raises(SandboxStartFailed, match="did not return a sandbox ID"):
            await client.create(options=options_cls(api_key="key-1"))

    @pytest.mark.asyncio
    async def test_create_with_manifest_warns_discard(
        self,
        provider: str,
        client_name: str,
        options_name: str,
        endpoint: str,
        id_field: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A hosted backend cannot materialize a manifest remotely, so it must
        WARN that the configured manifest is discarded — not drop it silently."""
        del endpoint
        client_cls, options_cls = _import(provider, client_name, options_name)
        http = _make_http_client(body={id_field: "sandbox-xyz"})
        client = client_cls(http_client=http)
        with caplog.at_level(logging.WARNING):
            session = await client.create(options=options_cls(api_key="key-1"), manifest=Manifest())
        assert session is not None
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("manifest is discarded" in m and provider in m for m in messages)

    @pytest.mark.asyncio
    async def test_create_without_manifest_no_discard_warning(
        self,
        provider: str,
        client_name: str,
        options_name: str,
        endpoint: str,
        id_field: str,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The manifest-discard warning must be conditional: no manifest, no warning."""
        del endpoint
        client_cls, options_cls = _import(provider, client_name, options_name)
        http = _make_http_client(body={id_field: "sandbox-xyz"})
        client = client_cls(http_client=http)
        with caplog.at_level(logging.WARNING):
            await client.create(options=options_cls(api_key="key-1"))
        assert not any("manifest is discarded" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_resume_requires_sandbox_id(
        self,
        provider: str,
        client_name: str,
        options_name: str,
        endpoint: str,
        id_field: str,
    ) -> None:
        del options_name, endpoint, id_field
        (client_cls,) = _import(provider, client_name)
        client = client_cls(http_client=_make_http_client())
        state = SandboxSessionState(backend_id=provider)
        with pytest.raises(SandboxStartFailed, match="sandbox_id"):
            await client.resume(state)

    @pytest.mark.asyncio
    async def test_resume_finds_sandbox(
        self,
        provider: str,
        client_name: str,
        options_name: str,
        endpoint: str,
        id_field: str,
    ) -> None:
        del options_name, endpoint, id_field
        (client_cls,) = _import(provider, client_name)
        client = client_cls(http_client=_make_http_client())
        state = SandboxSessionState(
            backend_id=provider,
            provider_payload={"sandbox_id": "saved-id", "api_key": "k"},
        )
        session = await client.resume(state)
        assert session.session_id == "saved-id"
