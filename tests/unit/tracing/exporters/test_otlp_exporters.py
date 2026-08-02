from unittest.mock import patch

import pytest

from troopai.adk.tracing.exporters.langsmith import langsmith_headers, setup_langsmith
from troopai.adk.tracing.exporters.logfire import logfire_headers, setup_logfire
from troopai.adk.tracing.exporters.phoenix import setup_phoenix
from troopai.adk.types.tracing.convention import TracingConvention


def test_langsmith_headers_include_api_key_and_project():
    headers = langsmith_headers(api_key="ls-secret", project="my-proj")
    assert headers["x-api-key"] == "ls-secret"
    assert headers["Langsmith-Project"] == "my-proj"


def test_langsmith_headers_reject_empty_project():
    with pytest.raises(ValueError, match="project must be non-empty"):
        langsmith_headers(api_key="ls-secret", project="")


def test_logfire_headers_require_token():
    with pytest.raises(ValueError):
        logfire_headers(token="")
    assert "Authorization" in logfire_headers(token="tok")
    assert logfire_headers(token="raw-tok")["Authorization"] == "raw-tok"


def test_langsmith_headers_omits_project_when_none():
    headers = langsmith_headers(api_key="k")
    assert "Langsmith-Project" not in headers


def test_setup_phoenix_forwards_to_setup_otel_without_live_provider():
    # patch setup_otel so no real provider is installed (no network/global state)
    with patch("troopai.adk.tracing.exporters.phoenix.setup_otel") as mock_setup:
        setup_phoenix(endpoint="http://localhost:6006/v1/traces")
        mock_setup.assert_called_once()
        _, kwargs = mock_setup.call_args
        assert kwargs["convention"] is TracingConvention.OPENINFERENCE


def test_setup_langsmith_forwards_headers_without_live_provider():
    with patch("troopai.adk.tracing.exporters.langsmith.setup_otel") as mock_setup:
        setup_langsmith(api_key="ls-secret", project="p")
        mock_setup.assert_called_once()
        _, kwargs = mock_setup.call_args
        assert kwargs["headers"]["x-api-key"] == "ls-secret"


def test_setup_logfire_forwards_headers_without_live_provider():
    with patch("troopai.adk.tracing.exporters.logfire.setup_otel") as mock_setup:
        setup_logfire(token="tok")
        mock_setup.assert_called_once()
        _, kwargs = mock_setup.call_args
        assert kwargs["headers"]["Authorization"] == "tok"
