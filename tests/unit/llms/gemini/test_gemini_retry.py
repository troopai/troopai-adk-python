"""Tests for the Gemini retry classifier."""

from __future__ import annotations

import pytest
from google.genai.errors import APIError, ClientError, ServerError

from troopai.adk.llms.gemini.gemini_retry import gemini_exception_to_kind


def _client_error(status: int) -> ClientError:
    return ClientError(status, response_json={"error": {"message": "test"}}, response=None)


def _server_error(status: int) -> ServerError:
    return ServerError(status, response_json={"error": {"message": "test"}}, response=None)


class TestGeminiExceptionToKind:
    def test_rate_limit_429(self) -> None:
        assert gemini_exception_to_kind(_client_error(429)) == "rate_limit"

    @pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
    def test_other_4xx_permanent(self, status: int) -> None:
        assert gemini_exception_to_kind(_client_error(status)) is None

    def test_timeout_504(self) -> None:
        assert gemini_exception_to_kind(_server_error(504)) == "timeout"

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_other_5xx_server_error(self, status: int) -> None:
        assert gemini_exception_to_kind(_server_error(status)) == "server_error"

    def test_unknown_exception(self) -> None:
        assert gemini_exception_to_kind(ValueError("nope")) is None

    def test_apierror_base_not_classified(self) -> None:
        # Plain APIError without a 429/5xx status: classifier returns None
        # because only ClientError / ServerError variants are handled.
        # APIError is the abstract base; the SDK always raises a
        # subclass. This pinpoints that we don't accidentally retry
        # on the abstract base class.
        # ``APIError`` constructor needs a code+response; we synth.
        exc = APIError(418, response_json={"error": {"message": "teapot"}}, response=None)
        # 418 isn't 429 nor 504 nor 5xx — so even if we hit a branch,
        # it returns None. The point: APIError is not a ClientError
        # or ServerError, so neither branch fires.
        if isinstance(exc, (ClientError, ServerError)):
            # If the SDK reorganised so APIError IS a Client/Server,
            # update this test.
            pytest.skip("APIError became a typed subclass; test skipped.")
        assert gemini_exception_to_kind(exc) is None
