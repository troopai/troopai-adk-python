"""Tests for the A2A tracing integration.

Verifies the changes to ``FunctionSpanData`` and the OTel bridge:

* ``FunctionSpanData.a2a_data`` is exported in ``export()``.
* ``function_span(a2a_data=...)`` factory propagates the field.
* When OTel is installed, ``OTelTracer.function_span`` switches the
  span name prefix to ``a2a.`` when ``a2a_data`` is set, falls back
  to ``mcp.`` for ``mcp_data`` only, and to ``tool.`` otherwise.
* ``a2a_data`` keys are flattened under the ``troopai.a2a`` attribute
  prefix on emitted OTel attributes.

The OTel-specific tests are skipped if the optional ``otel`` extra
is not installed.
"""

from typing import Any

import pytest

from troopai.adk.tracing import function_span
from troopai.adk.types.tracing.span_data import FunctionSpanData


class TestFunctionSpanDataExport:
    def test_a2a_data_field_default_none(self) -> None:
        data = FunctionSpanData(name="x")
        assert data.a2a_data is None

    def test_a2a_data_field_set(self) -> None:
        data = FunctionSpanData(
            name="x",
            a2a_data={"task_id": "t1", "context_id": "c1", "remote_url": "http://x"},
        )
        assert data.a2a_data == {
            "task_id": "t1",
            "context_id": "c1",
            "remote_url": "http://x",
        }

    def test_export_includes_a2a_data(self) -> None:
        data = FunctionSpanData(name="x", a2a_data={"task_id": "t1"})
        exported = data.export()
        assert "a2a_data" in exported
        assert exported["a2a_data"] == {"task_id": "t1"}

    def test_export_a2a_data_none_when_unset(self) -> None:
        data = FunctionSpanData(name="x")
        exported = data.export()
        assert exported["a2a_data"] is None


class TestFunctionSpanFactory:
    def test_factory_propagates_a2a_data(self) -> None:
        with function_span(
            name="remote_call",
            a2a_data={"task_id": "abc", "remote_url": "http://x"},
            disabled=True,
        ) as span:
            assert span.data.a2a_data == {"task_id": "abc", "remote_url": "http://x"}

    def test_factory_propagates_mcp_data_independently(self) -> None:
        # Both fields can be set on the same span (e.g. an A2A call
        # that fetched its tool list via an MCP server). The OTel
        # bridge prefers a2a_data for the prefix; both still surface
        # as attributes.
        with function_span(
            name="hybrid",
            mcp_data={"server_name": "fs"},
            a2a_data={"task_id": "abc"},
            disabled=True,
        ) as span:
            assert span.data.mcp_data == {"server_name": "fs"}
            assert span.data.a2a_data == {"task_id": "abc"}


class TestOTelPrefixSwitching:
    """OTel bridge prefix-switching tests — skipped without [otel] extra."""

    def setup_method(self) -> None:
        # Skip the whole class if opentelemetry isn't importable.
        pytest.importorskip("opentelemetry")

    def test_a2a_prefix_when_a2a_data_set(self) -> None:
        from troopai.adk.tracing.otel.otel_tracer import OTelTracer

        tracer = OTelTracer()
        data = FunctionSpanData(name="my_call", a2a_data={"task_id": "t1"})
        span = tracer.function_span(data)
        # OTelSpan stores the name on ``_name`` for later use; verify
        # via the constructor argument we passed in the bridge logic
        # by reflecting on the OTelSpan instance.
        assert _get_otel_span_name(span) == "a2a.my_call"

    def test_mcp_prefix_when_only_mcp_data_set(self) -> None:
        from troopai.adk.tracing.otel.otel_tracer import OTelTracer

        tracer = OTelTracer()
        data = FunctionSpanData(name="my_call", mcp_data={"server_name": "x"})
        span = tracer.function_span(data)
        assert _get_otel_span_name(span) == "mcp.my_call"

    def test_tool_prefix_when_neither_set(self) -> None:
        from troopai.adk.tracing.otel.otel_tracer import OTelTracer

        tracer = OTelTracer()
        data = FunctionSpanData(name="my_call")
        span = tracer.function_span(data)
        assert _get_otel_span_name(span) == "tool.my_call"

    def test_a2a_takes_precedence_over_mcp(self) -> None:
        from troopai.adk.tracing.otel.otel_tracer import OTelTracer

        tracer = OTelTracer()
        data = FunctionSpanData(
            name="my_call",
            mcp_data={"server_name": "x"},
            a2a_data={"task_id": "t1"},
        )
        span = tracer.function_span(data)
        # A2A is the wider boundary; MCP-via-A2A still surfaces as a2a.
        assert _get_otel_span_name(span) == "a2a.my_call"

    def test_a2a_attributes_use_troopai_a2a_prefix(self) -> None:
        from troopai.adk.tracing.otel.otel_tracer import _function_attrs

        data = FunctionSpanData(
            name="t",
            a2a_data={"task_id": "abc", "remote_url": "http://x"},
        )
        attrs = _function_attrs(data, record_full=True, max_chars=10_000)
        # Attribute keys flattened under troopai.a2a.<key>.
        assert attrs["troopai.a2a.task_id"] == "abc"
        assert attrs["troopai.a2a.remote_url"] == "http://x"


def _get_otel_span_name(span: Any) -> str:
    """Reflect on an ``OTelSpan`` to read its constructor-time name.

    The OTelSpan ctor stores the name as ``self._otel_name`` today; if
    that attribute name changes the test fails fast and this helper is
    the single update point.
    """
    return getattr(span, "_otel_name", "<unknown>")


class TestA2ASpanNamesNotDoublePrefix:
    """Span names in a2a_client / executor must NOT carry an 'a2a.' prefix.

    The OTelTracer already prepends 'a2a.' when a2a_data is present.
    If the name= argument already starts with 'a2a.', the final span
    name becomes 'a2a.a2a.<rest>' — a double-prefix bug.
    """

    def test_a2a_client_span_names_do_not_start_with_a2a_dot(self) -> None:
        import inspect

        import troopai.adk.a2a.a2a_client as mod

        src = inspect.getsource(mod)
        # Find all function_span(name=...) calls; collect the name strings.
        import re

        # Match name=f"..." or name="..." patterns inside function_span calls.
        names = re.findall(r'function_span\s*\(\s*\n?\s*name=f?"([^"]+)"', src)
        bad = [n for n in names if n.startswith("a2a.")]
        assert len(bad) == 0, (
            f"a2a_client.py span names must NOT start with 'a2a.' (OTelTracer already prepends it). Bad names: {bad}"
        )

    def test_executor_span_names_do_not_start_with_a2a_dot(self) -> None:
        import inspect
        import re

        import troopai.adk.a2a.executor as mod

        src = inspect.getsource(mod)
        names = re.findall(r'function_span\s*\(\s*\n?\s*name=f?"([^"]+)"', src)
        bad = [n for n in names if n.startswith("a2a.")]
        assert len(bad) == 0, (
            f"executor.py span names must NOT start with 'a2a.' (OTelTracer already prepends it). Bad names: {bad}"
        )
