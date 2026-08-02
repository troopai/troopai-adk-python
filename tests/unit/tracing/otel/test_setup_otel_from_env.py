"""Tests for setup_otel_from_env().

The function must:
- Raise ValueError when OTEL_EXPORTER_OTLP_ENDPOINT is unset or empty.
- Delegate to setup_otel (no endpoint kwarg) so the SDK reads the env var
  natively when the endpoint is present.
- Return an OTelTracer on success.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("opentelemetry")

from troopai.adk.tracing.otel import OTelTracer, setup_otel_from_env


class TestSetupOtelFromEnvMissingEndpoint:
    def test_raises_valueerror_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        with pytest.raises(ValueError, match="OTEL_EXPORTER_OTLP_ENDPOINT"):
            setup_otel_from_env()

    def test_raises_valueerror_when_env_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        with pytest.raises(ValueError, match="OTEL_EXPORTER_OTLP_ENDPOINT"):
            setup_otel_from_env()

    def test_raises_valueerror_when_env_whitespace_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "   ")
        with pytest.raises(ValueError, match="OTEL_EXPORTER_OTLP_ENDPOINT"):
            setup_otel_from_env()


class TestSetupOtelFromEnvDelegatesToSetupOtel:
    def test_calls_setup_otel_without_endpoint_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """setup_otel_from_env must not pass endpoint= so the SDK reads the env var directly."""
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        fake_tracer = MagicMock(spec=OTelTracer)
        with patch("troopai.adk.tracing.otel.setup.setup_otel", return_value=fake_tracer) as mock_setup:
            result = setup_otel_from_env()
        mock_setup.assert_called_once()
        call_kwargs = mock_setup.call_args[1]
        assert "endpoint" not in call_kwargs
        assert result is fake_tracer

    def test_passes_console_kwarg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        fake_tracer = MagicMock(spec=OTelTracer)
        with patch("troopai.adk.tracing.otel.setup.setup_otel", return_value=fake_tracer) as mock_setup:
            setup_otel_from_env(console=True)
        assert mock_setup.call_args[1].get("console") is True

    def test_passes_additional_processors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        fake_tracer = MagicMock(spec=OTelTracer)
        fake_processor = MagicMock()
        with patch("troopai.adk.tracing.otel.setup.setup_otel", return_value=fake_tracer) as mock_setup:
            setup_otel_from_env(additional_processors=[fake_processor])
        assert mock_setup.call_args[1].get("additional_processors") == [fake_processor]

    def test_returns_oteltracer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        fake_tracer = MagicMock(spec=OTelTracer)
        with patch("troopai.adk.tracing.otel.setup.setup_otel", return_value=fake_tracer):
            result = setup_otel_from_env()
        assert result is fake_tracer


class TestSetupOtelFromEnvServiceName:
    """OTEL_SERVICE_NAME must reach the service.name resource attribute.

    setup_otel always sets service.name explicitly, which shadows the SDK's
    env-driven OTELResourceDetector. setup_otel_from_env must therefore read
    OTEL_SERVICE_NAME itself and thread it through.
    """

    def test_passes_env_service_name_to_setup_otel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "my-prod-service")
        fake_tracer = MagicMock(spec=OTelTracer)
        with patch("troopai.adk.tracing.otel.setup.setup_otel", return_value=fake_tracer) as mock_setup:
            setup_otel_from_env()
        assert mock_setup.call_args[1].get("service_name") == "my-prod-service"

    def test_defaults_service_name_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
        fake_tracer = MagicMock(spec=OTelTracer)
        with patch("troopai.adk.tracing.otel.setup.setup_otel", return_value=fake_tracer) as mock_setup:
            setup_otel_from_env()
        assert mock_setup.call_args[1].get("service_name") == "troopai-adk-python"

    def test_resource_attribute_reflects_env_service_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """End-to-end: the installed provider's resource carries the env value.

        Without the explicit OTEL_SERVICE_NAME passthrough, setup_otel's
        hard-coded service.name="troopai-adk-python" shadows the env value and this
        assertion fails.
        """
        from opentelemetry.sdk.trace import TracerProvider

        monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        monkeypatch.setenv("OTEL_SERVICE_NAME", "from-env-name")
        with patch("opentelemetry.trace.set_tracer_provider") as mock_set:
            setup_otel_from_env()
        provider = mock_set.call_args[0][0]
        assert isinstance(provider, TracerProvider)
        assert provider.resource.attributes.get("service.name") == "from-env-name"


class TestSetupOtelFromEnvPublicExport:
    def test_importable_from_tracing_otel(self) -> None:
        from troopai.adk.tracing.otel import setup_otel_from_env as fn

        assert callable(fn)

    def test_importable_from_tracing_top_level(self) -> None:
        from troopai.adk import tracing

        assert hasattr(tracing, "setup_otel_from_env")
        assert tracing.setup_otel_from_env is not None


def test_signal_specific_endpoint_variable_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    """OTEL_EXPORTER_OTLP_TRACES_ENDPOINT alone is a valid SDK config."""
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "http://collector:4317")
    tracer = setup_otel_from_env()
    assert tracer is not None
