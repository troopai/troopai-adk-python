"""Tests for troopai.adk.llms.openai.openai_retry.

Covers:
- ``openai_exception_to_kind`` mapping for every SDK exception class.
- ``APIStatusError`` status-code branches (429 / 408 / 500+ / 401 / 400).
- The thin ``call_with_retry`` wrapper delegates through to
  ``llms/retry.py`` with the OpenAI classifier.
"""

from __future__ import annotations

import httpx
import pytest
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)

from troopai.adk.llms.openai.openai_retry import (
    call_with_retry,
    openai_exception_to_kind,
)
from troopai.adk.types.llms import LLMRetryPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/responses")


def _make_response(status: int) -> httpx.Response:
    return httpx.Response(status_code=status, request=_make_request())


def _make_api_status_error(status: int, message: str = "status error") -> APIStatusError:
    return APIStatusError(message, response=_make_response(status), body=None)


# ---------------------------------------------------------------------------
# Direct classifier tests — one assertion per branch
# ---------------------------------------------------------------------------


class TestOpenAIExceptionToKind:
    def test_rate_limit_error_maps_to_rate_limit(self) -> None:
        exc = RateLimitError("too many", response=_make_response(429), body=None)
        assert openai_exception_to_kind(exc) == "rate_limit"

    def test_api_timeout_error_maps_to_timeout(self) -> None:
        exc = APITimeoutError(request=_make_request())
        assert openai_exception_to_kind(exc) == "timeout"

    def test_api_connection_error_maps_to_server_error(self) -> None:
        exc = APIConnectionError(message="network blip", request=_make_request())
        assert openai_exception_to_kind(exc) == "server_error"

    @pytest.mark.parametrize("status", [500, 502, 503, 504, 529])
    def test_status_5xx_maps_to_server_error(self, status: int) -> None:
        exc = _make_api_status_error(status)
        assert openai_exception_to_kind(exc) == "server_error"

    def test_status_429_maps_to_rate_limit(self) -> None:
        exc = _make_api_status_error(429)
        assert openai_exception_to_kind(exc) == "rate_limit"

    def test_status_408_maps_to_timeout(self) -> None:
        exc = _make_api_status_error(408)
        assert openai_exception_to_kind(exc) == "timeout"

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_other_4xx_maps_to_none(self, status: int) -> None:
        exc = _make_api_status_error(status)
        assert openai_exception_to_kind(exc) is None

    def test_non_openai_exception_maps_to_none(self) -> None:
        assert openai_exception_to_kind(ValueError("oops")) is None

    def test_internal_server_error_subclass_maps_to_server_error(self) -> None:
        """SDK exposes ``InternalServerError`` as a 500-subclass of APIStatusError."""
        exc = InternalServerError(
            "boom",
            response=_make_response(500),
            body=None,
        )
        assert openai_exception_to_kind(exc) == "server_error"

    def test_authentication_error_maps_to_none(self) -> None:
        """Auth failures are permanent; the classifier must NOT retry them."""
        exc = AuthenticationError(
            "bad key",
            response=_make_response(401),
            body=None,
        )
        assert openai_exception_to_kind(exc) is None

    def test_bad_request_error_maps_to_none(self) -> None:
        exc = BadRequestError("malformed", response=_make_response(400), body=None)
        assert openai_exception_to_kind(exc) is None

    def test_permission_denied_error_maps_to_none(self) -> None:
        exc = PermissionDeniedError(
            "forbidden",
            response=_make_response(403),
            body=None,
        )
        assert openai_exception_to_kind(exc) is None

    def test_not_found_error_maps_to_none(self) -> None:
        exc = NotFoundError("missing", response=_make_response(404), body=None)
        assert openai_exception_to_kind(exc) is None


# ---------------------------------------------------------------------------
# Integration with the generic retry loop
# ---------------------------------------------------------------------------


class TestCallWithRetryIntegration:
    async def test_retries_on_rate_limit_until_success(self) -> None:
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RateLimitError(
                    "slow down",
                    response=_make_response(429),
                    body=None,
                )
            return "ok"

        policy = LLMRetryPolicy(max_retries=5, initial_delay=0.001, jitter=False)
        result = await call_with_retry(op, policy, model="gpt-5.1")
        assert result == "ok"
        assert calls == 3

    async def test_permanent_error_propagates_immediately(self) -> None:
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise BadRequestError(
                "bad",
                response=_make_response(400),
                body=None,
            )

        policy = LLMRetryPolicy(max_retries=5, initial_delay=0.001, jitter=False)
        with pytest.raises(BadRequestError):
            await call_with_retry(op, policy, model="gpt-5.1")
        assert calls == 1

    async def test_budget_exhaustion_raises_rate_limit(self) -> None:
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise RateLimitError(
                "still too many",
                response=_make_response(429),
                body=None,
            )

        policy = LLMRetryPolicy(max_retries=2, initial_delay=0.001, jitter=False)
        with pytest.raises(RateLimitError):
            await call_with_retry(op, policy, model="gpt-5.1")
        # 1 initial + 2 retries = 3 attempts.
        assert calls == 3
