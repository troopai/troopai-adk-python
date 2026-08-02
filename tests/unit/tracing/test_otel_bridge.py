"""Tests for :mod:`troopai.adk.tracing.otel`.

Exercises the OpenTelemetry bridge against OTel's own
:class:`~opentelemetry.sdk.trace.export.in_memory_span_exporter.InMemorySpanExporter`
so we get true-to-runtime assertions about span names, attributes, and
parent-child relationships.

Also verifies the graceful-degradation contract: when the
``opentelemetry`` packages are not installed, constructing an
:class:`OTelTracer` MUST raise
:class:`~troopai.adk.exceptions.TracingDependencyError` with the install
command — not a confusing low-level :class:`ImportError`.
"""

from __future__ import annotations

import sys
from typing import Any

import pytest

from troopai.adk.exceptions import TracingDependencyError
from troopai.adk.tracing import (
    custom_span,
    set_tracer,
)
from troopai.adk.types.tracing import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
)

otel_sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
otel_sdk_export = pytest.importorskip("opentelemetry.sdk.trace.export")
otel_in_memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from troopai.adk.tracing.otel import OTelTracer


@pytest.fixture
def exporter_and_tracer() -> Any:
    """Fresh :class:`InMemorySpanExporter` + :class:`OTelTracer` per test.

    The ``TracerProvider`` is passed explicitly so the test never
    touches the global OTel provider (other tests in the suite could
    race with a shared provider)."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OTelTracer(provider=provider, service_name="test-svc")
    yield exporter, tracer
    exporter.clear()
    set_tracer(None)


def _finished_span_by_name(exporter: InMemorySpanExporter, name: str) -> Any:
    matches = [s for s in exporter.get_finished_spans() if s.name == name]
    assert len(matches) == 1, (
        f"expected exactly one span named {name!r}; got {[s.name for s in exporter.get_finished_spans()]}"
    )
    return matches[0]


def test_finish_ends_span_even_when_flatten_raises() -> None:
    """finish() must end the OTel span even if attribute flattening raises.

    Regression: finish() wrapped ``_flatten`` + ``set_attributes`` +
    ``set_status`` + ``end()`` in ONE try/except. If ``_flatten`` raised,
    ``end()`` was skipped while the OTel context had already been detached
    — the span was started-but-never-ended, leaking as a dangling open
    span in the exporter's buffer. ``end()`` now runs in a ``finally``, so
    the span is always closed even when attribute flushing fails.
    """
    from troopai.adk.tracing.otel.otel_span import OTelSpan
    from troopai.adk.types.tracing.span_data import CustomSpanData

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_tracer = provider.get_tracer("leak-test")

    def boom_flatten(_exported: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("flatten exploded")

    span = OTelSpan(
        CustomSpanData(name="leaky", data={}),
        otel_tracer=otel_tracer,
        name="leaky",
        attribute_flattener=boom_flatten,
    )
    span.start()
    span.finish()  # must NOT raise, and must still end the span

    finished = [s.name for s in exporter.get_finished_spans()]
    assert "leaky" in finished, f"span leaked (never ended); finished={finished}"


def test_agent_span_emits_otel_span_with_attributes(
    exporter_and_tracer: Any,
) -> None:
    exporter, tracer = exporter_and_tracer
    span = tracer.agent_span(
        AgentSpanData(
            name="planner",
            handoffs=["writer", "researcher"],
            tools=["search"],
            output_type="PlanResult",
            metadata={"tenant": "acme"},
        )
    )
    with span:
        pass

    otel_span = _finished_span_by_name(exporter, "agent.planner")
    attrs = dict(otel_span.attributes or {})
    assert attrs["troopai.agent.name"] == "planner"
    assert list(attrs["troopai.agent.handoffs"]) == ["writer", "researcher"]
    assert list(attrs["troopai.agent.tools"]) == ["search"]
    assert attrs["troopai.agent.output_type"] == "PlanResult"
    assert attrs["troopai.metadata.tenant"] == "acme"


def test_generation_span_uses_genai_semconv(
    exporter_and_tracer: Any,
) -> None:
    """GenAI semantic-convention keys must be emitted for ingestion by
    Phoenix/Langwatch/Honeycomb without an adapter."""
    exporter, tracer = exporter_and_tracer
    span = tracer.generation_span(
        GenerationSpanData(
            model="claude-opus-4-7",
            usage={"prompt_tokens": 123, "completion_tokens": 45},
        )
    )
    with span:
        pass

    otel_span = _finished_span_by_name(exporter, "llm.generation")
    attrs = dict(otel_span.attributes or {})
    assert attrs["gen_ai.system"] == "troopai"
    assert attrs["gen_ai.request.model"] == "claude-opus-4-7"
    assert attrs["gen_ai.usage.input_tokens"] == 123
    assert attrs["gen_ai.usage.output_tokens"] == 45


def test_function_span_becomes_mcp_prefix_when_mcp_data_present(
    exporter_and_tracer: Any,
) -> None:
    """Decision #6: MCP spans reuse ``FunctionSpanData.mcp_data``; the
    bridge must name-switch ``tool.*`` → ``mcp.*``."""
    exporter, tracer = exporter_and_tracer
    regular = tracer.function_span(FunctionSpanData(name="compute", input="{}", output="42"))
    with regular:
        pass
    mcp = tracer.function_span(
        FunctionSpanData(
            name="fetch_url",
            input='{"url":"https://ex.com"}',
            mcp_data={"server_name": "browser", "tool_name": "fetch_url"},
        )
    )
    with mcp:
        pass

    names = {s.name for s in exporter.get_finished_spans()}
    assert "tool.compute" in names
    assert "mcp.fetch_url" in names

    mcp_span = _finished_span_by_name(exporter, "mcp.fetch_url")
    attrs = dict(mcp_span.attributes or {})
    assert attrs["troopai.mcp.server_name"] == "browser"
    assert attrs["troopai.mcp.tool_name"] == "fetch_url"


def test_handoff_span_records_from_and_to(exporter_and_tracer: Any) -> None:
    exporter, tracer = exporter_and_tracer
    span = tracer.handoff_span(HandoffSpanData(from_agent="triage", to_agent="billing"))
    with span:
        pass

    otel_span = _finished_span_by_name(exporter, "agent.handoff")
    attrs = dict(otel_span.attributes or {})
    assert attrs["troopai.handoff.from"] == "triage"
    assert attrs["troopai.handoff.to"] == "billing"


def test_guardrail_span_flags_trigger(exporter_and_tracer: Any) -> None:
    exporter, tracer = exporter_and_tracer
    span = tracer.guardrail_span(GuardrailSpanData(name="pii_check", triggered=True))
    with span:
        pass

    otel_span = _finished_span_by_name(exporter, "guardrail.pii_check")
    attrs = dict(otel_span.attributes or {})
    assert attrs["troopai.guardrail.name"] == "pii_check"
    assert attrs["troopai.guardrail.triggered"] is True


def test_response_span_records_response_id(exporter_and_tracer: Any) -> None:
    exporter, tracer = exporter_and_tracer
    span = tracer.response_span(ResponseSpanData(response_id="resp_xyz"))
    with span:
        pass

    otel_span = _finished_span_by_name(exporter, "llm.response")
    attrs = dict(otel_span.attributes or {})
    assert attrs["gen_ai.system"] == "troopai"
    assert attrs["gen_ai.response.id"] == "resp_xyz"


def test_custom_span_uses_caller_name_and_data(
    exporter_and_tracer: Any,
) -> None:
    exporter, tracer = exporter_and_tracer
    set_tracer(tracer)
    with custom_span("rank_results", data={"n": 10, "strategy": "bm25"}):
        pass

    otel_span = _finished_span_by_name(exporter, "rank_results")
    attrs = dict(otel_span.attributes or {})
    assert attrs["troopai.span.name"] == "rank_results"
    assert attrs["troopai.custom.n"] == 10
    assert attrs["troopai.custom.strategy"] == "bm25"


def test_nested_spans_form_parent_chain(exporter_and_tracer: Any) -> None:
    """OTel's own context propagation must auto-parent children started
    inside the outer span's body."""
    exporter, tracer = exporter_and_tracer
    outer = tracer.agent_span(AgentSpanData(name="root"))
    with outer:
        inner = tracer.function_span(FunctionSpanData(name="child"))
        with inner:
            pass

    outer_span = _finished_span_by_name(exporter, "agent.root")
    inner_span = _finished_span_by_name(exporter, "tool.child")
    assert inner_span.parent is not None
    assert inner_span.parent.span_id == outer_span.context.span_id


def test_span_records_error_as_otel_status_and_exception(
    exporter_and_tracer: Any,
) -> None:
    """An exception in the ``with`` body must surface as both
    ``Status(ERROR)`` and an ``exception`` event on the OTel span."""
    from opentelemetry.trace import StatusCode

    exporter, tracer = exporter_and_tracer
    span = tracer.custom_span(CustomSpanData(name="boom", data={}))
    with pytest.raises(ValueError), span:
        raise ValueError("kaboom")

    otel_span = _finished_span_by_name(exporter, "boom")
    assert otel_span.status.status_code == StatusCode.ERROR
    event_names = [e.name for e in (otel_span.events or [])]
    assert "exception" in event_names


def test_construction_fails_with_helpful_error_when_otel_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulate a fresh Python process where ``opentelemetry`` is not
    installed: :class:`OTelTracer` MUST raise
    :class:`TracingDependencyError`, not ``ImportError``.

    Implemented by clearing every cached ``opentelemetry.*`` module from
    :data:`sys.modules` and wiring an import hook that raises
    ``ModuleNotFoundError`` for the package root. Inline the
    :class:`OTelTracer` constructor below re-imports
    :mod:`opentelemetry.trace` — the hook intercepts it.
    """

    class _BlockOTel:
        def find_spec(
            self,
            name: str,
            path: Any | None = None,
            target: Any | None = None,
        ) -> Any:
            del path, target
            if name == "opentelemetry" or name.startswith("opentelemetry."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    to_remove = [m for m in sys.modules if m.startswith("opentelemetry")]
    for m in to_remove:
        monkeypatch.delitem(sys.modules, m, raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockOTel(), *sys.meta_path])

    with pytest.raises(TracingDependencyError) as exc_info:
        OTelTracer()

    message = str(exc_info.value)
    assert "opentelemetry" in message
    assert "troopai-adk-python[otel]" in message
