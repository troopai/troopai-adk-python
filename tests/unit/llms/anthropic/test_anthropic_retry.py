"""Tests for the Anthropic retry classifier.

Pins the exception → retry-kind mapping defined in
``llms/anthropic/anthropic_retry.py``.  Mirrors the OpenAI retry tests
to keep the two providers' classifier matrices in lockstep.
"""

from __future__ import annotations

import httpx
import pytest
from anthropic import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    PermissionDeniedError,
    RateLimitError,
)

from troopai.adk.llms.anthropic.anthropic_retry import anthropic_exception_to_kind


def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _response(status: int) -> httpx.Response:
    return httpx.Response(status_code=status, request=_request())


class TestAnthropicExceptionToKind:
    def test_rate_limit_error_maps_to_rate_limit(self) -> None:
        exc = RateLimitError(message="429", response=_response(429), body=None)
        assert anthropic_exception_to_kind(exc) == "rate_limit"

    def test_api_timeout_error_maps_to_timeout(self) -> None:
        assert anthropic_exception_to_kind(APITimeoutError(_request())) == "timeout"

    def test_api_connection_error_maps_to_server_error(self) -> None:
        assert anthropic_exception_to_kind(APIConnectionError(request=_request())) == "server_error"

    def test_internal_server_error_maps_to_server_error(self) -> None:
        exc = InternalServerError(message="500", response=_response(500), body=None)
        assert anthropic_exception_to_kind(exc) == "server_error"

    def test_status_429_maps_to_rate_limit(self) -> None:
        exc = APIStatusError(message="429", response=_response(429), body=None)
        assert anthropic_exception_to_kind(exc) == "rate_limit"

    def test_status_529_maps_to_rate_limit(self) -> None:
        # Anthropic's overload signal — OverloadedError is a subclass
        # of APIStatusError.
        exc = APIStatusError(message="529", response=_response(529), body=None)
        assert anthropic_exception_to_kind(exc) == "rate_limit"

    def test_status_408_maps_to_timeout(self) -> None:
        exc = APIStatusError(message="408", response=_response(408), body=None)
        assert anthropic_exception_to_kind(exc) == "timeout"

    def test_status_504_maps_to_timeout(self) -> None:
        exc = APIStatusError(message="504", response=_response(504), body=None)
        assert anthropic_exception_to_kind(exc) == "timeout"

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_status_5xx_maps_to_server_error(self, status: int) -> None:
        exc = APIStatusError(message=str(status), response=_response(status), body=None)
        assert anthropic_exception_to_kind(exc) == "server_error"

    def test_authentication_error_is_permanent(self) -> None:
        exc = AuthenticationError(message="401", response=_response(401), body=None)
        assert anthropic_exception_to_kind(exc) is None

    def test_permission_denied_is_permanent(self) -> None:
        exc = PermissionDeniedError(message="403", response=_response(403), body=None)
        assert anthropic_exception_to_kind(exc) is None

    def test_bad_request_is_permanent(self) -> None:
        exc = BadRequestError(message="400", response=_response(400), body=None)
        assert anthropic_exception_to_kind(exc) is None

    def test_unknown_exception_is_permanent(self) -> None:
        assert anthropic_exception_to_kind(ValueError("nope")) is None
