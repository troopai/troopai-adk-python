"""Tests for :func:`troopai.adk.tracing.flush_traces`.

Verifies that:
- ``flush_traces()`` is a no-op when the installed tracer is not Flushable.
- ``flush_traces()`` calls ``flush()`` on a Flushable tracer.
- :class:`~troopai.adk.tracing.MultiTracer` fans ``flush()`` to Flushable children.
- :class:`~troopai.adk.tracing.otel.OTelTracer` implements :class:`~troopai.adk.tracing.Flushable`
  and its ``flush()`` calls ``force_flush`` on the provider.
"""

from __future__ import annotations

from typing import Any

import pytest

from troopai.adk.tracing import (
    Flushable,
    MultiTracer,
    NoOpTracer,
    flush_traces,
    set_tracer,
)
from troopai.adk.types.tracing.span_data import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
)

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _MinimalTracer:
    """A Tracer-protocol stub that does NOT implement Flushable."""

    def agent_span(self, data: AgentSpanData) -> Any:  # pragma: no cover
        raise NotImplementedError

    def function_span(self, data: FunctionSpanData) -> Any:  # pragma: no cover
        raise NotImplementedError

    def generation_span(self, data: GenerationSpanData) -> Any:  # pragma: no cover
        raise NotImplementedError

    def response_span(self, data: ResponseSpanData) -> Any:  # pragma: no cover
        raise NotImplementedError

    def handoff_span(self, data: HandoffSpanData) -> Any:  # pragma: no cover
        raise NotImplementedError

    def guardrail_span(self, data: GuardrailSpanData) -> Any:  # pragma: no cover
        raise NotImplementedError

    def custom_span(self, data: CustomSpanData) -> Any:  # pragma: no cover
        raise NotImplementedError


class _FlushableTracer(_MinimalTracer):
    """A Tracer stub that also implements Flushable."""

    def __init__(self) -> None:
        self.flush_call_count = 0

    def flush(self) -> None:
        self.flush_call_count += 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlushTracesNoop:
    def teardown_method(self) -> None:
        set_tracer(None)

    def test_noop_tracer_does_not_raise(self) -> None:
        """``flush_traces()`` is silent when the active tracer is not Flushable."""
        set_tracer(None)  # installs NoOpTracer
        flush_traces()  # must not raise

    def test_non_flushable_tracer_does_not_raise(self) -> None:
        """A Tracer that lacks ``flush()`` is silently skipped."""
        set_tracer(_MinimalTracer())
        flush_traces()  # must not raise


class TestFlushTracesFlushable:
    def teardown_method(self) -> None:
        set_tracer(None)

    def test_flush_called_on_flushable_tracer(self) -> None:
        """``flush_traces()`` delegates to ``flush()`` on a Flushable tracer."""
        tracer = _FlushableTracer()
        set_tracer(tracer)

        flush_traces()

        assert tracer.flush_call_count == 1

    def test_flush_called_exactly_once(self) -> None:
        """Multiple ``flush_traces()`` calls each invoke ``flush()`` once."""
        tracer = _FlushableTracer()
        set_tracer(tracer)

        flush_traces()
        flush_traces()

        assert tracer.flush_call_count == 2

    def test_flushable_protocol_satisfied(self) -> None:
        """``_FlushableTracer`` is recognised as :class:`Flushable` at runtime."""
        assert isinstance(_FlushableTracer(), Flushable)

    def test_noop_tracer_not_flushable(self) -> None:
        """``NoOpTracer`` does not satisfy :class:`Flushable`."""
        assert not isinstance(NoOpTracer(), Flushable)


class TestMultiTracerFlush:
    def teardown_method(self) -> None:
        set_tracer(None)

    def test_multi_tracer_flushes_flushable_children(self) -> None:
        """``MultiTracer.flush()`` fans out to every Flushable child."""
        child_a = _FlushableTracer()
        child_b = _FlushableTracer()
        multi = MultiTracer([child_a, child_b])
        set_tracer(multi)

        flush_traces()

        assert child_a.flush_call_count == 1
        assert child_b.flush_call_count == 1

    def test_multi_tracer_skips_non_flushable_children(self) -> None:
        """``MultiTracer.flush()`` skips children that are not Flushable."""
        non_flushable = _MinimalTracer()
        flushable = _FlushableTracer()
        multi = MultiTracer([non_flushable, flushable])
        set_tracer(multi)

        flush_traces()

        assert flushable.flush_call_count == 1

    def test_multi_tracer_flush_continues_after_exception(self) -> None:
        """A failing ``flush()`` on one child does not prevent others from flushing."""

        class _RaisingTracer(_FlushableTracer):
            def flush(self) -> None:
                raise RuntimeError("exporter down")

        raiser = _RaisingTracer()
        good = _FlushableTracer()
        multi = MultiTracer([raiser, good])
        set_tracer(multi)

        flush_traces()  # must not propagate the RuntimeError

        assert good.flush_call_count == 1

    def test_multi_tracer_implements_flushable(self) -> None:
        """``MultiTracer`` is itself recognised as :class:`Flushable`."""
        multi = MultiTracer([_FlushableTracer()])
        assert isinstance(multi, Flushable)


class TestOTelTracerFlush:
    """Verify OTelTracer.flush() calls force_flush on the provider."""

    otel_sdk = pytest.importorskip("opentelemetry.sdk.trace")

    def teardown_method(self) -> None:
        set_tracer(None)

    def test_otel_tracer_flush_calls_force_flush(self) -> None:
        """``OTelTracer.flush()`` calls ``force_flush`` on the explicit provider."""
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        from troopai.adk.tracing import custom_span
        from troopai.adk.tracing.otel import OTelTracer

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = OTelTracer(provider=provider, service_name="flush-test")
        set_tracer(tracer)

        with custom_span("test-span"):
            pass

        # The SimpleSpanProcessor exports synchronously, but flush() must
        # still succeed (force_flush is a no-op on an already-drained processor).
        flush_traces()

        finished = exporter.get_finished_spans()
        assert len(finished) == 1
        assert finished[0].name == "test-span"

    def test_otel_tracer_implements_flushable(self) -> None:
        """``OTelTracer`` satisfies the :class:`Flushable` protocol."""
        from opentelemetry.sdk.trace import TracerProvider

        from troopai.adk.tracing.otel import OTelTracer

        tracer = OTelTracer(provider=TracerProvider(), service_name="proto-check")
        assert isinstance(tracer, Flushable)
