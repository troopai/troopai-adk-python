"""Tests pinning S1 — OTel span attributes MUST redact credential shapes
and truncate oversize tool I/O by default.

The threat model: ``FunctionSpanData.input`` and ``.output`` carry raw
tool arguments and return values. Without the redact+truncate pass a
single tool call can push multi-MB payloads — or worse, user-pasted
``Bearer <jwt>`` / ``sk-...`` / ``AIza...`` / ``api_key=...`` strings —
to every configured OTel backend. These tests pin:

1. Unit-level redaction coverage for the main credential shapes.
2. Char-cap truncation preserving a "... [truncated; N chars total]"
   suffix so operators can see the blob was clipped.
3. The default-safe / opt-out-verbatim contract of ``OTelTracer``.

The tests go through the real OpenTelemetry SDK via ``InMemorySpanExporter``
so the assertions observe what actually lands on the span, not what we
intended to put there.
"""

from __future__ import annotations

from typing import Any

import pytest

from troopai.adk.types.tracing import FunctionSpanData

otel_sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
otel_sdk_export = pytest.importorskip("opentelemetry.sdk.trace.export")
otel_in_memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from troopai.adk.tracing.otel import OTelTracer
from troopai.adk.tracing.otel.otel_tracer import (
    _DEFAULT_TOOL_IO_MAX_CHARS,
    _redact,
    _redact_and_truncate,
)

# --------------------------------------------------------------------------
# Unit tests — the pure helpers
# --------------------------------------------------------------------------


class TestRedactPatterns:
    """``_redact`` catches the credential shapes documented in the pattern
    tuple. Each test exercises one shape in isolation so a regression to
    a single pattern is instantly identifiable."""

    def test_bearer_token_is_redacted(self) -> None:
        value = "Authorization: Bearer abc.DEF-ghi_jkl+mno/pqr=123"
        redacted = _redact(value)
        assert "abc.DEF-ghi" not in redacted
        assert "Bearer ***" in redacted

    def test_lowercase_bearer_token_is_redacted(self) -> None:
        value = "authorization: bearer ey.JWT.Payload.Signature"
        redacted = _redact(value)
        assert "ey.JWT" not in redacted
        assert "bearer ***" in redacted.lower()

    def test_openai_key_is_redacted(self) -> None:
        value = "key is sk-proj-abcdefghijklmnopqrstuv"
        redacted = _redact(value)
        assert "sk-proj-abcde" not in redacted
        assert "sk-***" in redacted

    def test_anthropic_key_is_redacted(self) -> None:
        value = "using sk-ant-api03-abcdefghijklmnopqrstuv"
        redacted = _redact(value)
        assert "sk-ant-api03" not in redacted
        assert "sk-ant-***" in redacted

    def test_google_key_is_redacted(self) -> None:
        value = "token=AIzaSyAbCdEfGhIjKlMnOpQrStUvWx"
        redacted = _redact(value)
        assert "AIzaSyAbCd" not in redacted
        assert "AIza***" in redacted

    def test_json_embedded_api_key_is_redacted(self) -> None:
        value = '{"api_key": "secret-value-12345"}'
        redacted = _redact(value)
        assert "secret-value-12345" not in redacted
        assert "***" in redacted

    def test_json_embedded_password_is_redacted(self) -> None:
        value = '{"password": "hunter2-supersecret"}'
        redacted = _redact(value)
        assert "hunter2-supersecret" not in redacted

    def test_json_embedded_token_is_redacted(self) -> None:
        value = '{"token": "abcdef0123456789"}'
        redacted = _redact(value)
        assert "abcdef0123456789" not in redacted

    def test_non_secret_content_passes_through(self) -> None:
        value = "Just a regular message with no credentials at all."
        assert _redact(value) == value

    def test_aws_access_key_id_is_redacted(self) -> None:
        value = "cfg = {AccessKeyId: AKIAIOSFODNN7EXAMPLE}"
        redacted = _redact(value)
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "AKIA***" in redacted

    def test_aws_session_key_prefix_is_redacted(self) -> None:
        value = "key ASIAIOSFODNN7EXAMPLE rotated"
        redacted = _redact(value)
        assert "ASIAIOSFODNN7EXAMPLE" not in redacted
        assert "AKIA***" in redacted

    def test_github_personal_token_is_redacted(self) -> None:
        # ghp_ + 36 alphanumeric chars — GitHub's documented shape.
        value = "auth: ghp_abcdefghijklmnopqrstuvwxyz0123456789"
        redacted = _redact(value)
        assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in redacted
        assert "gh_***" in redacted

    def test_github_oauth_token_is_redacted(self) -> None:
        value = "x-gh: gho_abcdefghijklmnopqrstuvwxyz0123456789"
        redacted = _redact(value)
        assert "gho_abcdefghijklmnopqrstuvwxyz0123456789" not in redacted
        assert "gh_***" in redacted

    def test_slack_bot_token_is_redacted(self) -> None:
        value = "slack: xoxb-1234567890-abcdefghij"
        redacted = _redact(value)
        assert "1234567890-abcdefghij" not in redacted
        assert "xox-***" in redacted

    def test_pem_private_key_block_is_redacted(self) -> None:
        pem = (
            "-----BEGIN PRIVATE KEY-----\n"
            "MIIBVgIBADANBgkqhkiG9w0BAQEFAASCAT4wggE6AgEAAkEAoMPB9X...\n"
            "AAAA\n"
            "-----END PRIVATE KEY-----"
        )
        redacted = _redact(pem)
        assert "MIIBVgIBADAN" not in redacted
        assert "BEGIN PRIVATE KEY" in redacted
        assert "***" in redacted

    def test_camelcase_access_token_field_is_redacted(self) -> None:
        """Many SDKs emit camelCase field names — the generic JSON pattern
        must cover them, not only snake_case ``access_token``."""
        value = '{"accessToken": "sensitive-oauth-value-xyz"}'
        redacted = _redact(value)
        assert "sensitive-oauth-value-xyz" not in redacted

    def test_camelcase_client_secret_is_redacted(self) -> None:
        value = '{"clientSecret": "abc123def456ghi789"}'
        redacted = _redact(value)
        assert "abc123def456ghi789" not in redacted

    def test_redacted_output_is_not_double_redacted(self) -> None:
        """Negative lookaheads block the generic JSON pattern from
        re-matching values the prefix patterns already replaced."""
        value = "token: sk-abcdefghijklmnopqrstuv"
        first = _redact(value)
        second = _redact(first)
        assert first == second


class TestTruncation:
    def test_short_value_is_not_truncated(self) -> None:
        value = "hello world"
        result = _redact_and_truncate(value, max_chars=64)
        assert result == "hello world"

    def test_value_over_cap_is_truncated_with_suffix(self) -> None:
        value = "a" * 5000
        result = _redact_and_truncate(value, max_chars=100)
        assert len(result) < 5000
        assert result.startswith("a" * 100)
        assert "truncated" in result
        assert "5000" in result

    def test_redaction_runs_before_truncation(self) -> None:
        """A secret at the tail of a long string must still be redacted —
        not allowed to escape by sitting past the cut point."""
        prefix = "x" * 3000
        value = f"{prefix} Bearer abcdefghijklmnop"
        result = _redact_and_truncate(value, max_chars=50)
        # Bearer token must not survive anywhere, even though it was
        # past the truncation boundary in the original string.
        assert "abcdefghijklmnop" not in result

    def test_truncation_suffix_uses_redacted_length_not_original(self) -> None:
        """The truncation suffix must reflect len(redacted), not len(value).

        A short secret in a large payload causes len(redacted) < len(value).
        Reporting the original length leaks the pre-redaction character count,
        which can reveal the secret's context. The suffix must only ever
        describe what the redacted string looks like.
        """
        # Build a payload where a secret shortens the string after redaction.
        # api_key field with a long value → redacted to "***" (much shorter).
        secret = "very-long-secret-value-" + "x" * 100
        # Pad the prefix to exceed max_chars so truncation fires.
        prefix = "a" * 200
        value = f'{prefix} "api_key": "{secret}"'
        result = _redact_and_truncate(value, max_chars=100)

        assert "truncated" in result
        # The reported total must NOT be len(value) (which is much larger
        # than the redacted string).
        original_length_str = str(len(value))
        assert original_length_str not in result, (
            f"Truncation suffix exposed original pre-redaction length {original_length_str}"
        )


# --------------------------------------------------------------------------
# Integration — redaction + truncation applied through real OTel spans
# --------------------------------------------------------------------------


@pytest.fixture
def exporter() -> InMemorySpanExporter:
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter: InMemorySpanExporter) -> TracerProvider:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider


def _tool_span_attrs(exporter: InMemorySpanExporter, tool_name: str) -> dict[str, Any]:
    matches = [s for s in exporter.get_finished_spans() if s.name == f"tool.{tool_name}"]
    assert len(matches) == 1
    return dict(matches[0].attributes or {})


class TestFunctionSpanDefaultRedaction:
    """Default ``OTelTracer`` redacts and truncates tool I/O."""

    def test_tool_input_with_bearer_token_is_redacted(
        self, exporter: InMemorySpanExporter, provider: TracerProvider
    ) -> None:
        tracer = OTelTracer(provider=provider, service_name="test")
        span = tracer.function_span(
            FunctionSpanData(
                name="http_get",
                input='{"headers":{"Authorization":"Bearer abcdef.ghijkl.mnopqr"}}',
                output="200 OK",
            )
        )
        with span:
            pass

        attrs = _tool_span_attrs(exporter, "http_get")
        raw_input = str(attrs["troopai.tool.input"])
        assert "abcdef.ghijkl.mnopqr" not in raw_input
        assert "Bearer ***" in raw_input

    def test_tool_output_with_api_key_is_redacted(
        self, exporter: InMemorySpanExporter, provider: TracerProvider
    ) -> None:
        tracer = OTelTracer(provider=provider, service_name="test")
        span = tracer.function_span(
            FunctionSpanData(
                name="fetch_creds",
                input="{}",
                output='{"api_key": "real-secret-value-123456"}',
            )
        )
        with span:
            pass

        attrs = _tool_span_attrs(exporter, "fetch_creds")
        raw_output = str(attrs["troopai.tool.output"])
        assert "real-secret-value-123456" not in raw_output

    def test_oversize_tool_output_is_truncated(self, exporter: InMemorySpanExporter, provider: TracerProvider) -> None:
        tracer = OTelTracer(provider=provider, service_name="test")
        huge = "PAYLOAD" * 1000  # 7000 chars
        span = tracer.function_span(FunctionSpanData(name="big_tool", input="{}", output=huge))
        with span:
            pass

        attrs = _tool_span_attrs(exporter, "big_tool")
        raw_output = str(attrs["troopai.tool.output"])
        assert len(raw_output) <= _DEFAULT_TOOL_IO_MAX_CHARS + 64
        assert "truncated" in raw_output
        assert "7000" in raw_output

    def test_custom_max_chars_is_honoured(self, exporter: InMemorySpanExporter, provider: TracerProvider) -> None:
        tracer = OTelTracer(provider=provider, service_name="test", tool_io_max_chars=32)
        span = tracer.function_span(FunctionSpanData(name="tiny_cap", input="a" * 200, output="b" * 200))
        with span:
            pass

        attrs = _tool_span_attrs(exporter, "tiny_cap")
        raw_input = str(attrs["troopai.tool.input"])
        raw_output = str(attrs["troopai.tool.output"])
        # Cap is tight — both sides must be clipped under 32 + suffix.
        assert raw_input.startswith("a" * 32)
        assert raw_output.startswith("b" * 32)
        assert "truncated" in raw_input
        assert "truncated" in raw_output


class TestRecordFullOptOut:
    """``record_tool_io_full=True`` preserves verbatim — documented escape
    hatch for trusted environments and debugging."""

    def test_verbatim_preserves_secret_shapes(self, exporter: InMemorySpanExporter, provider: TracerProvider) -> None:
        tracer = OTelTracer(provider=provider, service_name="test", record_tool_io_full=True)
        verbatim = "Bearer abcdef.ghijkl.mnopqr"
        span = tracer.function_span(FunctionSpanData(name="raw_tool", input=verbatim, output=verbatim))
        with span:
            pass

        attrs = _tool_span_attrs(exporter, "raw_tool")
        assert attrs["troopai.tool.input"] == verbatim
        assert attrs["troopai.tool.output"] == verbatim

    def test_verbatim_preserves_oversize_payload(
        self, exporter: InMemorySpanExporter, provider: TracerProvider
    ) -> None:
        tracer = OTelTracer(provider=provider, service_name="test", record_tool_io_full=True)
        huge = "X" * 10_000
        span = tracer.function_span(FunctionSpanData(name="raw_big", input=huge, output=huge))
        with span:
            pass

        attrs = _tool_span_attrs(exporter, "raw_big")
        assert attrs["troopai.tool.input"] == huge
        assert attrs["troopai.tool.output"] == huge
