"""Tests for the RemoteVMSandboxClient helper functions (TRV.1-TRV.4)."""

from __future__ import annotations

import base64

import pytest

from troopai.adk.exceptions.exceptions import (
    ExecTimeoutError,
    ExecTransportError,
    SandboxConfigurationError,
    SandboxRuntimeError,
)
from troopai.adk.sandbox.clients.hosted.remote_vm.remote_vm_client import (
    build_httpx_client,
    map_http_error_to_sandbox_error,
    parse_exec_result,
    retry_async_call,
)


class TestBuildHttpxClient:
    def test_returns_async_client(self) -> None:
        client = build_httpx_client(base_url="https://example.com")
        # Just verify the type's interface — don't actually open
        # a connection.
        assert hasattr(client, "get")
        assert hasattr(client, "post")
        assert hasattr(client, "aclose")

    def test_headers_propagated(self) -> None:
        client = build_httpx_client(
            base_url="https://e.com",
            headers={"X-Token": "secret"},
        )
        assert client.headers["X-Token"] == "secret"


class TestRetryAsyncCall:
    @pytest.mark.asyncio
    async def test_returns_on_first_success(self) -> None:
        calls: list[int] = []

        async def fn() -> str:
            calls.append(1)
            return "ok"

        result = await retry_async_call(fn, max_retries=3)
        assert result == "ok"
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_retries_on_transport_error(self) -> None:
        import httpx

        calls: list[int] = []

        async def fn() -> str:
            calls.append(1)
            if len(calls) < 3:
                raise httpx.ConnectError("net flap")
            return "recovered"

        result = await retry_async_call(
            fn,
            max_retries=3,
            initial_wait=0.01,
            max_wait=0.05,
        )
        assert result == "recovered"
        assert len(calls) == 3

    @pytest.mark.asyncio
    async def test_exhausts_budget_raises(self) -> None:
        import httpx

        async def fn() -> str:
            raise httpx.ConnectError("always fails")

        with pytest.raises(ExecTransportError, match="transport error"):
            await retry_async_call(fn, max_retries=2, initial_wait=0.01)


class TestMapHttpErrorToSandboxError:
    @pytest.mark.parametrize(
        "code,expected_type",
        [
            (401, SandboxConfigurationError),
            (403, SandboxConfigurationError),
            (404, SandboxConfigurationError),
            (408, ExecTimeoutError),
            (409, SandboxConfigurationError),
            (422, SandboxConfigurationError),
            (429, ExecTransportError),
            (400, SandboxConfigurationError),
            (500, SandboxRuntimeError),
            (502, SandboxRuntimeError),
            (503, SandboxRuntimeError),
        ],
    )
    def test_status_code_mapping(
        self,
        code: int,
        expected_type: type[Exception],
    ) -> None:
        err = map_http_error_to_sandbox_error(code, "boom")
        assert isinstance(err, expected_type)
        assert "boom" in str(err)


class TestParseExecResult:
    def test_basic_text(self) -> None:
        r = parse_exec_result(
            {"stdout": "hello", "stderr": "", "exit_code": 0},
        )
        assert r.stdout == b"hello"
        assert r.stderr == b""
        assert r.exit_code == 0

    def test_base64_decoded(self) -> None:
        encoded = base64.b64encode(b"binary payload").decode("ascii")
        r = parse_exec_result(
            {"stdout": encoded, "exit_code": 0},
            encoding="base64",
        )
        assert r.stdout == b"binary payload"

    def test_camel_case_exit_code(self) -> None:
        r = parse_exec_result(
            {"stdout": "", "stderr": "", "exitCode": 1},
        )
        assert r.exit_code == 1

    def test_missing_exit_code_defaults_to_minus_one(self) -> None:
        r = parse_exec_result({"stdout": "", "stderr": ""})
        assert r.exit_code == -1

    def test_duration_ms_camel_case(self) -> None:
        r = parse_exec_result(
            {"stdout": "", "stderr": "", "exit_code": 0, "durationMs": 42},
        )
        assert r.duration_ms == 42

    def test_custom_keys(self) -> None:
        r = parse_exec_result(
            {"out": "x", "err": "y", "code": 7},
            stdout_key="out",
            stderr_key="err",
            exit_code_key="code",
        )
        assert r.stdout == b"x"
        assert r.stderr == b"y"
        assert r.exit_code == 7
