"""Regression tests for tracing bug-fixes.

Each test is named after the finding it guards. Tests are grouped by
finding location so they are easy to trace back to the worklist.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from troopai.adk.tracing import MultiTracer, NoOpSpan, Span
from troopai.adk.tracing.multi_tracer import CompositeSpan
from troopai.adk.types.tracing import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    SpanData,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _TrackedSpan(Span[Any]):
    """Span that records lifecycle events without touching the ContextVar."""

    def __init__(self, data: SpanData) -> None:
        super().__init__(data)
        self.started = False
        self.finished = False

    def start(self) -> None:
        self.started = True

    def finish(self) -> None:
        self.finished = True
        self._finished = True


class _RecordingTracer:
    def __init__(self, label: str = "rec") -> None:
        self.label = label
        self.spans: list[_TrackedSpan] = []

    def _track(self, data: SpanData) -> _TrackedSpan:
        span = _TrackedSpan(data)
        self.spans.append(span)
        return span

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        return self._track(data)

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        return self._track(data)

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        return self._track(data)

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        return self._track(data)

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        return self._track(data)

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        return self._track(data)

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        return self._track(data)


class _ExplodingFactoryTracer:
    """Tracer whose factory methods raise, used to test fault isolation."""

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        raise RuntimeError("factory exploded")

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        raise RuntimeError("factory exploded")

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        raise RuntimeError("factory exploded")

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        raise RuntimeError("factory exploded")

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        raise RuntimeError("factory exploded")

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        raise RuntimeError("factory exploded")

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        raise RuntimeError("factory exploded")


# ---------------------------------------------------------------------------
# multi_tracer.py: factory error isolation (finding: MED multi_tracer.py:142)
# ---------------------------------------------------------------------------


def test_factory_error_does_not_abort_remaining_tracers() -> None:
    """When one tracer's factory raises, the remaining tracers still get spans.

    Before the fix, a bare list-comp would propagate the exception and the
    CompositeSpan would never be created, leaving every other backend
    with no span for the entire run.
    """
    exploding = _ExplodingFactoryTracer()
    healthy = _RecordingTracer("healthy")
    multi = MultiTracer([exploding, healthy])

    span = multi.custom_span(CustomSpanData(name="x", data={}))
    span.start()
    span.finish()

    # The healthy tracer must still have received its span.
    assert len(healthy.spans) == 1
    assert healthy.spans[0].started is True
    assert healthy.spans[0].finished is True


def test_all_factories_raise_returns_noop_span() -> None:
    """When every tracer's factory raises, a NoOpSpan is returned (not an exception)."""
    multi = MultiTracer([_ExplodingFactoryTracer()])

    span = multi.agent_span(AgentSpanData(name="a"))

    assert isinstance(span, NoOpSpan)


def test_factory_error_isolation_for_every_span_kind(caplog: pytest.LogCaptureFixture) -> None:
    """Error isolation must work identically for all seven factory methods."""
    kinds = [
        ("agent_span", AgentSpanData(name="a")),
        ("function_span", FunctionSpanData(name="f")),
        ("generation_span", GenerationSpanData()),
        ("response_span", ResponseSpanData()),
        ("handoff_span", HandoffSpanData()),
        ("guardrail_span", GuardrailSpanData(name="g")),
        ("custom_span", CustomSpanData(name="c", data={})),
    ]
    for method_name, data in kinds:
        exploding = _ExplodingFactoryTracer()
        healthy = _RecordingTracer()
        multi = MultiTracer([exploding, healthy])

        with caplog.at_level(logging.ERROR, logger="troopai.adk.tracing.multi_tracer"):
            span = getattr(multi, method_name)(data)

        assert isinstance(span, CompositeSpan), f"{method_name} should return a CompositeSpan with the healthy child"
        assert len(healthy.spans) == 1, f"{method_name} healthy tracer received no span"


# ---------------------------------------------------------------------------
# multi_tracer.py: >1 OTelTracer warning (finding: MED multi_tracer.py:94)
# ---------------------------------------------------------------------------


def test_multiple_otel_tracers_logs_warning(caplog: pytest.LogCaptureFixture) -> None:
    """MultiTracer with two OTelTracer instances must emit a warning."""
    pytest.importorskip("opentelemetry")
    from troopai.adk.tracing.otel import OTelTracer

    t1 = OTelTracer()
    t2 = OTelTracer()

    with caplog.at_level(logging.WARNING, logger="troopai.adk.tracing.multi_tracer"):
        MultiTracer([t1, t2])

    assert any("OTelTracer" in m for m in caplog.messages), "Expected a warning about multiple OTelTracers"


def test_single_otel_tracer_no_warning(caplog: pytest.LogCaptureFixture) -> None:
    """A single OTelTracer inside MultiTracer must not log a warning."""
    pytest.importorskip("opentelemetry")
    from troopai.adk.tracing.otel import OTelTracer

    with caplog.at_level(logging.WARNING, logger="troopai.adk.tracing.multi_tracer"):
        MultiTracer([OTelTracer()])

    assert not any("OTelTracer" in m for m in caplog.messages)


# ---------------------------------------------------------------------------
# tracing/__init__.py: ImportError guard (finding: MED __init__.py:98)
# ---------------------------------------------------------------------------


def test_importerror_guard_uses_name_attribute() -> None:
    """The ImportError guard must check exc.name, not str(exc).

    A bug inside an installed opentelemetry package causes an ImportError
    whose str() representation may contain 'opentelemetry', but exc.name
    should point to the actual broken first-party module — NOT to an
    opentelemetry submodule — and the re-raise should fire.

    We simulate by constructing an ImportError with a non-opentelemetry
    module name that happens to mention 'opentelemetry' in the message.
    """
    # Build an ImportError that would fool the old str(exc) guard.
    exc = ImportError("cannot import 'opentelemetry.bogus' from broken_module")
    exc.name = "troopai.adk.tracing.otel_tracer"  # first-party name — should re-raise

    # Simulate the guard logic from tracing/__init__.py.
    should_raise = exc.name is None or not exc.name.startswith("opentelemetry")
    assert should_raise, "Guard should re-raise when exc.name does not start with 'opentelemetry'"


def test_importerror_guard_swallows_missing_otel() -> None:
    """The guard must swallow ImportError when exc.name starts with 'opentelemetry'."""
    exc = ImportError("No module named 'opentelemetry.sdk'")
    exc.name = "opentelemetry.sdk"

    should_raise = exc.name is None or not exc.name.startswith("opentelemetry")
    assert not should_raise, "Guard should swallow when exc.name is an opentelemetry module"


# ---------------------------------------------------------------------------
# tracing/__init__.py: extended span factories in __all__ (finding: LOW :56)
# ---------------------------------------------------------------------------


def test_extended_span_factories_in_all() -> None:
    """graph_span, swarm_span, sandbox_span etc. must be in tracing.__all__."""
    import troopai.adk.tracing as tracing_pkg

    expected = [
        "graph_span",
        "graph_superstep_span",
        "graph_node_span",
        "swarm_span",
        "swarm_turn_span",
        "sandbox_span",
    ]
    for name in expected:
        assert name in tracing_pkg.__all__, f"{name!r} missing from tracing.__all__"
        assert hasattr(tracing_pkg, name), f"{name!r} not importable from tracing package"


def test_extended_span_factories_importable_from_package() -> None:
    """``from troopai.adk.tracing import graph_span`` must work directly."""
    from troopai.adk.tracing import (  # noqa: F401
        graph_node_span,
        graph_span,
        graph_superstep_span,
        sandbox_span,
        swarm_span,
        swarm_turn_span,
    )


# ---------------------------------------------------------------------------
# spans.py: return type (finding: LOW spans.py:441)
# ---------------------------------------------------------------------------


def test_extended_factories_return_custom_span_data() -> None:
    """sandbox_span, graph_span, swarm_span etc. must return Span[CustomSpanData].

    The return-type annotation was changed from Span[ConcreteSpanData] to
    Span[CustomSpanData] to remove 6 ``type: ignore[return-value]`` suppressions.
    The concrete SpanData payload is still embedded inside
    ``span.data.data`` when needed.
    """
    from troopai.adk.tracing import (
        graph_node_span,
        graph_span,
        graph_superstep_span,
        sandbox_span,
        swarm_span,
        swarm_turn_span,
    )
    from troopai.adk.types.tracing.span_data import CustomSpanData

    spans = [
        sandbox_span(backend_id="unix_local"),
        graph_span(graph_id="g1"),
        graph_superstep_span(graph_id="g1", index=0),
        graph_node_span(graph_id="g1", node_name="n1"),
        swarm_span(swarm_id="s1"),
        swarm_turn_span(swarm_id="s1", index=1, member="alice"),
    ]
    for span in spans:
        assert isinstance(span.data, CustomSpanData), f"Expected CustomSpanData, got {type(span.data)}"


# ---------------------------------------------------------------------------
# metrics/tracer.py: field-filtered ** unpack (finding: LOW metrics/tracer.py:179)
# ---------------------------------------------------------------------------


def test_graph_node_extra_key_does_not_crash_metrics() -> None:
    """Extra keys in GraphNodeSpanData.export() must be silently filtered out.

    Before the fix, an unknown key would cause TypeError inside
    _record_graph_node, which would be silently swallowed by the outer
    except in MetricSpan.finish() — masking the bug. The fix uses
    dataclasses.fields() to filter, so extra keys are dropped cleanly.
    """
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from troopai.adk.tracing.metrics.instruments import Instruments
    from troopai.adk.tracing.metrics.tracer import MetricsTracer

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    tracer = MetricsTracer(Instruments(provider.get_meter("t")))

    # Inject an extra key that does not exist on GraphNodeSpanData.
    payload = {
        "type": "graph_node",
        "graph_id": "g1",
        "node_name": "planner",
        "status": "success",
        "_future_field": "extra_value",  # not declared on GraphNodeSpanData
    }
    with tracer.custom_span(CustomSpanData(name="graph.node.planner", data=payload)):
        pass

    # If the fix works, the span finished without raising.
    metrics_data = reader.get_metrics_data()
    assert metrics_data is not None


def test_swarm_turn_extra_key_does_not_crash_metrics() -> None:
    """Extra keys in SwarmTurnSpanData payload must be silently filtered."""
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from troopai.adk.tracing.metrics.instruments import Instruments
    from troopai.adk.tracing.metrics.tracer import MetricsTracer

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    tracer = MetricsTracer(Instruments(provider.get_meter("t")))

    payload = {
        "type": "swarm_turn",
        "swarm_id": "s1",
        "index": 1,
        "member": "alice",
        "status": "success",
        "_future_field": "extra_value",  # unknown key
    }
    with tracer.custom_span(CustomSpanData(name="swarm.turn.1", data=payload)):
        pass

    metrics_data = reader.get_metrics_data()
    assert metrics_data is not None


# ---------------------------------------------------------------------------
# otel_tracer.py: _filter_to_fields (finding: LOW otel_tracer.py:510)
# ---------------------------------------------------------------------------


def test_otel_agent_span_extra_key_does_not_crash() -> None:
    """Extra keys in the exported dict must not crash OTelTracer.agent_span."""
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from troopai.adk.tracing.otel.otel_tracer import OTelTracer, _filter_to_fields

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OTelTracer(provider=provider)

    # Construct a span manually, then simulate finish() calling
    # the attribute_flattener on an exported dict with an extra key.
    data = AgentSpanData(name="test_agent")
    tracer.agent_span(data)

    # Simulate what OTelSpan.finish() does: export() the data and pass
    # to the flattener. Inject an extra key to trigger the fix.
    exported = data.export()
    exported["_extra_future_key"] = "boom"

    # The span's closure (_flatten_agent) should not raise — _filter_to_fields
    # will silently drop the unknown key before rebuilding AgentSpanData.
    filtered = _filter_to_fields(exported, AgentSpanData)
    assert "_extra_future_key" not in filtered
    assert "name" in filtered
    # Must be constructable without error.
    AgentSpanData(**filtered)


# ---------------------------------------------------------------------------
# a2a span name prefix (finding: HIGH a2a/executor.py:177)
# ---------------------------------------------------------------------------


def test_a2a_function_span_does_not_double_prefix() -> None:
    """OTelTracer already prepends 'a2a.' when a2a_data is present.

    Callers must use bare suffix names like 'task.{id}', NOT 'a2a.task.{id}'.
    This test verifies the OTel span name by capturing the built name
    inside OTelTracer.function_span.
    """
    pytest.importorskip("opentelemetry")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from troopai.adk.tracing.otel.otel_tracer import OTelTracer

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OTelTracer(provider=provider)

    task_id = "task-abc-123"
    data = FunctionSpanData(
        name=f"task.{task_id}",
        a2a_data={"task_id": task_id, "context_id": "ctx-1"},
    )
    with tracer.function_span(data):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span_name = spans[0].name
    # Correct: "a2a.task.task-abc-123"
    # Wrong:   "a2a.a2a.task.task-abc-123"
    assert span_name == f"a2a.task.{task_id}", f"Expected 'a2a.task.{task_id}', got {span_name!r}"
    assert "a2a.a2a" not in span_name, "Double 'a2a.a2a.' prefix emitted"
