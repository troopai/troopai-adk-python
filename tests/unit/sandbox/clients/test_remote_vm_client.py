"""Regression tests for ``remote_vm_client`` confirmed-bug fixes.

Covers:
- ``parse_exec_result`` no longer guesses base64 vs text (data corruption).
- ``retry_async_call`` honours its documented retry-on-returned-5xx/429
  contract.
- Cost-conservative ``max_retries`` defaults.
"""

from __future__ import annotations

import base64

import httpx
import pytest

from troopai.adk.sandbox.clients.hosted.remote_vm import RemoteVMSandboxClientOptions
from troopai.adk.sandbox.clients.hosted.remote_vm.remote_vm_client import (
    parse_exec_result,
    retry_async_call,
)


class TestParseExecResultEncoding:
    """``parse_exec_result`` must not silently base64-decode plain text."""

    @pytest.mark.parametrize(
        "token",
        ["test", "true", "ABCD", "PASS", "FAIL", "DONE", "dead", "cafe", "beef"],
    )
    def test_short_base64_alphabet_text_preserved(self, token: str) -> None:
        # These tokens are valid base64 (length % 4 == 0, alphabet match),
        # so the old "try base64 first" heuristic decoded them to binary
        # garbage. The default encoding is now text, so they round-trip.
        r = parse_exec_result({"stdout": token, "exit_code": 0})
        assert r.stdout == token.encode("utf-8")

    def test_default_encoding_is_text(self) -> None:
        encoded = base64.b64encode(b"binary payload").decode("ascii")
        r = parse_exec_result({"stdout": encoded, "exit_code": 0})
        # Default text mode keeps the wire string verbatim — no decode.
        assert r.stdout == encoded.encode("utf-8")

    def test_explicit_base64_decodes(self) -> None:
        encoded = base64.b64encode(b"binary payload").decode("ascii")
        r = parse_exec_result(
            {"stdout": encoded, "exit_code": 0},
            encoding="base64",
        )
        assert r.stdout == b"binary payload"

    def test_text_with_newline_preserved(self) -> None:
        r = parse_exec_result({"stdout": "test\n", "exit_code": 0})
        assert r.stdout == b"test\n"


def _response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://example.com/sandboxes")
    return httpx.Response(status_code, request=request, json={})


class TestRetryAsyncCallReturnedStatus:
    """``retry_async_call`` retries returned (non-raising) 5xx/429."""

    async def test_retries_returned_503_then_succeeds(self) -> None:
        calls: list[int] = []

        async def fn() -> httpx.Response:
            calls.append(1)
            if len(calls) < 3:
                return _response(503)
            return _response(200)

        result = await retry_async_call(
            fn,
            max_retries=3,
            initial_wait=0.01,
            max_wait=0.05,
        )
        assert result.status_code == 200
        assert len(calls) == 3

    async def test_retries_returned_429(self) -> None:
        calls: list[int] = []

        async def fn() -> httpx.Response:
            calls.append(1)
            if len(calls) < 2:
                return _response(429)
            return _response(200)

        result = await retry_async_call(
            fn,
            max_retries=3,
            initial_wait=0.01,
            max_wait=0.05,
        )
        assert result.status_code == 200
        assert len(calls) == 2

    async def test_exhausted_budget_returns_last_response(self) -> None:
        calls: list[int] = []

        async def fn() -> httpx.Response:
            calls.append(1)
            return _response(503)

        # On exhaustion the last response is returned so the caller can
        # map it via map_http_error_to_sandbox_error.
        result = await retry_async_call(
            fn,
            max_retries=2,
            initial_wait=0.01,
            max_wait=0.05,
        )
        assert result.status_code == 503
        assert len(calls) == 3

    async def test_non_retriable_4xx_returned_immediately(self) -> None:
        calls: list[int] = []

        async def fn() -> httpx.Response:
            calls.append(1)
            return _response(404)

        result = await retry_async_call(
            fn,
            max_retries=3,
            initial_wait=0.01,
            max_wait=0.05,
        )
        assert result.status_code == 404
        assert len(calls) == 1

    async def test_zero_budget_does_not_retry_returned_503(self) -> None:
        calls: list[int] = []

        async def fn() -> httpx.Response:
            calls.append(1)
            return _response(503)

        result = await retry_async_call(fn, max_retries=0)
        assert result.status_code == 503
        assert len(calls) == 1


class TestCostConservativeDefaults:
    def test_options_default_max_retries_is_zero(self) -> None:
        opts = RemoteVMSandboxClientOptions()
        assert opts.max_retries == 0

    async def test_retry_async_call_default_no_retry(self) -> None:
        calls: list[int] = []

        async def fn() -> str:
            calls.append(1)
            raise httpx.ConnectError("net flap")

        with pytest.raises(Exception):
            await retry_async_call(fn, initial_wait=0.01, max_wait=0.05)
        # Default budget is 0 -> exactly one attempt, no retry.
        assert len(calls) == 1
