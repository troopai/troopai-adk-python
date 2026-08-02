"""Tests for RemoteVMSandboxSession (TRV.5+TRV.6)."""

from __future__ import annotations

import base64
import json
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.sandbox.clients.hosted.remote_vm.remote_vm_session import (
    RemoteVMSandboxSession,
)


def _mock_http(*, response_json: Any = None, status: int = 200) -> Any:
    resp = MagicMock()
    resp.status_code = status
    resp.json = MagicMock(return_value=response_json or {})
    resp.text = json.dumps(response_json or {})
    resp.request = MagicMock()
    http = MagicMock()
    http.post = AsyncMock(return_value=resp)
    http.get = AsyncMock(return_value=resp)
    http.aclose = AsyncMock()
    return http


class TestLifecycle:
    @pytest.mark.asyncio
    async def test_start_sets_started(self) -> None:
        s = RemoteVMSandboxSession(
            http_client=_mock_http(),
            session_endpoint="https://api/sessions/abc",
        )
        await s.start()
        assert s._started is True  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_aclose_closes_http(self) -> None:
        http = _mock_http()
        s = RemoteVMSandboxSession(
            http_client=http,
            session_endpoint="https://api/sessions/abc",
        )
        await s.start()
        await s.aclose()
        http.aclose.assert_awaited_once()


class TestRunCommand:
    @pytest.mark.asyncio
    async def test_run_posts_to_exec(self) -> None:
        http = _mock_http(
            response_json={
                "stdout": "hello",
                "stderr": "",
                "exit_code": 0,
                "duration_ms": 5,
            },
        )
        s = RemoteVMSandboxSession(
            http_client=http,
            session_endpoint="https://api/sessions/abc",
        )
        result = await s.run("echo", "hello", shell=True)
        http.post.assert_awaited_once()
        call_url = http.post.call_args.args[0]
        assert call_url == "https://api/sessions/abc/exec"
        body = http.post.call_args.kwargs["json"]
        assert body["command"] == "echo hello"
        assert body["shell"] is True
        assert result.exit_code == 0
        assert result.stdout == b"hello"
        assert result.duration_ms == 5

    @pytest.mark.asyncio
    async def test_run_argv_when_shell_false(self) -> None:
        http = _mock_http(
            response_json={"stdout": "", "stderr": "", "exit_code": 0, "duration_ms": 1},
        )
        s = RemoteVMSandboxSession(
            http_client=http,
            session_endpoint="https://api/sessions/abc",
        )
        await s.run("echo", "hello", shell=False)
        body = http.post.call_args.kwargs["json"]
        assert body["command"] == ["echo", "hello"]
        assert body["shell"] is False

    @pytest.mark.asyncio
    async def test_run_forwards_custom_shell_list(self) -> None:
        # Regression: a custom shell list (e.g. ["/bin/bash", "-lc"]) was
        # collapsed to ``shell: True`` and dropped from the wire body, so the
        # provider ran with its default shell instead of the requested one —
        # diverging from the local subprocess backend, which honors the list.
        http = _mock_http(
            response_json={"stdout": "", "stderr": "", "exit_code": 0, "duration_ms": 1},
        )
        s = RemoteVMSandboxSession(
            http_client=http,
            session_endpoint="https://api/sessions/abc",
        )
        await s.run("echo", "hello", shell=["/bin/bash", "-lc"])
        body = http.post.call_args.kwargs["json"]
        # The custom shell must survive verbatim, not be collapsed to True.
        assert body["shell"] == ["/bin/bash", "-lc"]
        # argv is forwarded as a list so the shell prefix can be prepended,
        # matching ``create_subprocess_exec(*shell, *argv)`` semantics.
        assert body["command"] == ["echo", "hello"]


class TestFileOps:
    @pytest.mark.asyncio
    async def test_write_then_read(self) -> None:
        payload = b"hello"
        encoded = base64.b64encode(payload).decode()
        http = _mock_http(response_json={"data_b64": encoded})
        s = RemoteVMSandboxSession(
            http_client=http,
            session_endpoint="https://api/sessions/abc",
        )
        await s.write("foo.txt", BytesIO(payload))
        stream = await s.read("foo.txt")
        assert stream.read() == payload

    @pytest.mark.asyncio
    async def test_ls_parses_entries(self) -> None:
        http = _mock_http(
            response_json={
                "entries": [
                    {"name": "a.txt", "is_directory": False, "size_bytes": 42},
                    {"name": "sub", "is_directory": True},
                ],
            },
        )
        s = RemoteVMSandboxSession(
            http_client=http,
            session_endpoint="https://api/sessions/abc",
        )
        entries = await s.ls(".")
        assert len(entries) == 2
        assert entries[0].name == "a.txt"
        assert entries[0].is_directory is False
        assert entries[1].is_directory is True


class TestPortResolution:
    @pytest.mark.asyncio
    async def test_resolve_exposed_port(self) -> None:
        http = _mock_http(
            response_json={"host": "preview.example.com", "port": 443, "tls": True},
        )
        s = RemoteVMSandboxSession(
            http_client=http,
            session_endpoint="https://api/sessions/abc",
        )
        endpoint = await s.resolve_exposed_port(8080)
        assert endpoint.host == "preview.example.com"
        assert endpoint.port == 443
        assert endpoint.tls is True


class TestErrorMapping:
    @pytest.mark.asyncio
    async def test_403_raises_config_error(self) -> None:
        from troopai.adk.exceptions.exceptions import SandboxConfigurationError

        http = _mock_http(response_json={"error": "forbidden"}, status=403)
        s = RemoteVMSandboxSession(
            http_client=http,
            session_endpoint="https://api/sessions/abc",
        )
        with pytest.raises(SandboxConfigurationError):
            await s.run("ls")


class TestMalformedPayloadRaises:
    """A 200-OK response missing its required payload key must not decode to empty.

    Regression: ``read``/``persist_workspace`` did ``b64decode(payload.get(key, ""))``,
    so a missing/renamed key silently produced empty bytes — ``read`` returned an
    empty stream and ``persist_workspace`` wrote a 0-byte tar the snapshot store
    accepted as durable, with no error or log.
    """

    @pytest.mark.asyncio
    async def test_read_raises_when_no_data_key(self) -> None:
        from troopai.adk.exceptions.exceptions import ExecTransportError

        http = _mock_http(response_json={"unexpected": "shape"})
        s = RemoteVMSandboxSession(http_client=http, session_endpoint="https://api/sessions/abc")
        with pytest.raises(ExecTransportError, match="malformed"):
            await s.read("foo.txt")

    @pytest.mark.asyncio
    async def test_read_allows_present_but_empty_file(self) -> None:
        # A present-but-empty data_b64 is a legitimately empty file, NOT an error.
        http = _mock_http(response_json={"data_b64": ""})
        s = RemoteVMSandboxSession(http_client=http, session_endpoint="https://api/sessions/abc")
        stream = await s.read("empty.txt")
        assert stream.read() == b""

    @pytest.mark.asyncio
    async def test_persist_workspace_raises_on_missing_archive(self) -> None:
        from troopai.adk.exceptions.exceptions import ExecTransportError

        http = _mock_http(response_json={})  # no archive_b64 key
        s = RemoteVMSandboxSession(http_client=http, session_endpoint="https://api/sessions/abc")
        with pytest.raises(ExecTransportError, match="empty workspace snapshot"):
            await s.persist_workspace()

    @pytest.mark.asyncio
    async def test_persist_workspace_returns_archive(self) -> None:
        payload = b"tar-bytes"
        http = _mock_http(response_json={"archive_b64": base64.b64encode(payload).decode()})
        s = RemoteVMSandboxSession(http_client=http, session_endpoint="https://api/sessions/abc")
        stream = await s.persist_workspace()
        assert stream.read() == payload
