"""Tests for :mod:`troopai.adk.tracing.multi_tracer`.

Covers fan-out semantics: every wrapped tracer receives the same
factory call, and the composite span propagates ``start`` / ``finish``
/ ``set_error`` to every child. Also covers the fault-isolation
guarantee — a single sub-tracer raising in a lifecycle hook must not
crash the other children nor the surrounding run.
"""

from __future__ import annotations

from typing import Any, override

import pytest

from troopai.adk.tracing import (
    MultiTracer,
    NoOpSpan,
    Span,
    current_span,
    set_tracer,
)
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


class _TrackedSpan(Span[Any]):
    """Span subclass whose lifecycle hooks flip flags instead of touching
    the framework :class:`~contextvars.ContextVar`.

    Keeps :class:`MultiTracer` tests isolated: real :class:`Span` children
    would stack on :data:`_current_span` and their LIFO tokens would leak
    state across sibling tests. Fan-out semantics are still fully
    observable via ``started`` / ``finished`` / ``errors_recorded``.
    """

    def __init__(self, data: SpanData) -> None:
        super().__init__(data)
        self.started = False
        self.finished = False
        self.errors_recorded: list[tuple[str, dict[str, Any] | None]] = []

    @override
    def start(self) -> None:
        self.started = True

    @override
    def finish(self) -> None:
        self.finished = True
        self._finished = True

    @override
    def set_error(self, message: str, *, data: dict[str, Any] | None = None) -> None:
        super().set_error(message, data=data)
        self.errors_recorded.append((message, data))


class _RecordingTracer:
    """Minimal tracer that logs every factory call and returns
    :class:`_TrackedSpan` instances so lifecycle hooks are observable
    without any shared-state side effects."""

    def __init__(self, label: str = "rec") -> None:
        self.label = label
        self.received: list[tuple[str, SpanData]] = []
        self.spans: list[_TrackedSpan] = []

    def _track(self, kind: str, data: SpanData) -> _TrackedSpan:
        self.received.append((kind, data))
        span = _TrackedSpan(data)
        self.spans.append(span)
        return span

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        return self._track("agent", data)

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        return self._track("function", data)

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        return self._track("generation", data)

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        return self._track("response", data)

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        return self._track("handoff", data)

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        return self._track("guardrail", data)

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        return self._track("custom", data)


class _ExplodingSpan(Span[CustomSpanData]):
    """Span whose lifecycle hooks raise to exercise fault isolation."""

    def __init__(
        self,
        data: CustomSpanData,
        *,
        explode_on: str,
    ) -> None:
        super().__init__(data)
        self.explode_on = explode_on
        self.started = False
        self.finished = False
        self.errors_recorded: list[str] = []

    @override
    def start(self) -> None:
        if self.explode_on == "start":
            raise RuntimeError("sub-tracer start failed")
        self.started = True

    @override
    def finish(self) -> None:
        if self.explode_on == "finish":
            raise RuntimeError("sub-tracer finish failed")
        self.finished = True

    @override
    def set_error(self, message: str, *, data: dict[str, Any] | None = None) -> None:
        del data  # interface contract: accepted but unused on this test span
        if self.explode_on == "set_error":
            raise RuntimeError("sub-tracer set_error failed")
        self.errors_recorded.append(message)


class _RealSpanTracer:
    """Tracer returning real :class:`Span` instances.

    Unlike :class:`_RecordingTracer` (which returns ContextVar-free
    :class:`_TrackedSpan`), these children install themselves on the
    framework :data:`_current_span` ContextVar in ``start`` and restore it
    in ``finish`` — so :class:`CompositeSpan`'s finish ordering is
    observable via :func:`current_span`.
    """

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        return Span(data)

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        return Span(data)

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        return Span(data)

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        return Span(data)

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        return Span(data)

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        return Span(data)

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        return Span(data)


class _ExplodingTracer:
    """Tracer whose ``custom_span`` returns an :class:`_ExplodingSpan`."""

    def __init__(self, *, explode_on: str) -> None:
        self.explode_on = explode_on

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        raise NotImplementedError

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        raise NotImplementedError

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        raise NotImplementedError

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        raise NotImplementedError

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        raise NotImplementedError

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        raise NotImplementedError

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        return _ExplodingSpan(data, explode_on=self.explode_on)


@pytest.fixture(autouse=True)
def _reset_tracer() -> Any:
    yield
    set_tracer(None)


def test_empty_tracer_list_returns_noop_span() -> None:
    """An empty :class:`MultiTracer` must short-circuit to :class:`NoOpSpan`
    so callers still get a valid span without paying to iterate."""
    multi = MultiTracer([])
    span = multi.custom_span(CustomSpanData(name="x", data={}))
    assert isinstance(span, NoOpSpan)


def test_fans_out_to_every_wrapped_tracer() -> None:
    a = _RecordingTracer("a")
    b = _RecordingTracer("b")
    c = _RecordingTracer("c")
    multi = MultiTracer([a, b, c])

    multi.custom_span(CustomSpanData(name="checkout", data={"n": 1}))

    for tracer in (a, b, c):
        assert len(tracer.received) == 1
        kind, data = tracer.received[0]
        assert kind == "custom"
        assert isinstance(data, CustomSpanData)
        assert data.name == "checkout"


def test_composite_start_propagates_to_all_children() -> None:
    a = _RecordingTracer("a")
    b = _RecordingTracer("b")
    multi = MultiTracer([a, b])

    span = multi.agent_span(AgentSpanData(name="planner"))
    assert isinstance(span, CompositeSpan)

    span.start()

    assert len(a.spans) == 1
    assert len(b.spans) == 1
    assert a.spans[0].started is True
    assert b.spans[0].started is True


def test_composite_finish_propagates_to_all_children() -> None:
    a = _RecordingTracer("a")
    b = _RecordingTracer("b")
    multi = MultiTracer([a, b])

    span = multi.custom_span(CustomSpanData(name="x", data={}))
    span.start()
    span.finish()

    assert a.spans[0].finished is True
    assert b.spans[0].finished is True


def test_composite_set_error_propagates_to_all_children() -> None:
    a = _RecordingTracer("a")
    b = _RecordingTracer("b")
    multi = MultiTracer([a, b])

    span = multi.custom_span(CustomSpanData(name="x", data={}))
    span.set_error("boom", data={"code": "E42"})

    for tracer in (a, b):
        child = tracer.spans[0]
        assert len(child.errors_recorded) == 1
        message, error_data = child.errors_recorded[0]
        assert message == "boom"
        assert error_data == {"code": "E42"}


def test_broken_child_start_does_not_break_siblings() -> None:
    """A sub-tracer raising in ``start`` must be caught so every other
    backend still gets its lifecycle hook."""
    exploding = _ExplodingTracer(explode_on="start")
    healthy = _RecordingTracer("healthy")
    multi = MultiTracer([exploding, healthy])

    span = multi.custom_span(CustomSpanData(name="x", data={}))

    span.start()

    assert len(healthy.spans) == 1
    assert healthy.spans[0].started is True


def test_broken_child_finish_does_not_break_siblings() -> None:
    exploding = _ExplodingTracer(explode_on="finish")
    healthy = _RecordingTracer("healthy")
    multi = MultiTracer([exploding, healthy])

    span = multi.custom_span(CustomSpanData(name="x", data={}))
    span.start()
    span.finish()

    assert healthy.spans[0].finished is True


def test_broken_child_set_error_does_not_break_siblings() -> None:
    exploding = _ExplodingTracer(explode_on="set_error")
    healthy = _RecordingTracer("healthy")
    multi = MultiTracer([exploding, healthy])

    span = multi.custom_span(CustomSpanData(name="x", data={}))
    span.set_error("boom")

    assert len(healthy.spans[0].errors_recorded) == 1
    assert healthy.spans[0].errors_recorded[0][0] == "boom"


def test_fans_out_across_every_span_kind() -> None:
    """Every :class:`Tracer` factory method must fan out identically.
    Regression guard against a new span kind landing on the protocol
    without a matching :class:`MultiTracer` method."""
    a = _RecordingTracer("a")
    b = _RecordingTracer("b")
    multi = MultiTracer([a, b])

    multi.agent_span(AgentSpanData(name="A"))
    multi.function_span(FunctionSpanData(name="f"))
    multi.generation_span(GenerationSpanData())
    multi.response_span(ResponseSpanData())
    multi.handoff_span(HandoffSpanData())
    multi.guardrail_span(GuardrailSpanData(name="g"))
    multi.custom_span(CustomSpanData(name="c", data={}))

    expected = [
        "agent",
        "function",
        "generation",
        "response",
        "handoff",
        "guardrail",
        "custom",
    ]
    assert [kind for kind, _ in a.received] == expected
    assert [kind for kind, _ in b.received] == expected


def test_composite_span_works_as_context_manager() -> None:
    """``with MultiTracer(...).custom_span(...)`` must enter and exit
    every child in one block."""
    a = _RecordingTracer("a")
    b = _RecordingTracer("b")
    multi = MultiTracer([a, b])

    set_tracer(multi)
    with multi.custom_span(CustomSpanData(name="block", data={})) as span:
        assert isinstance(span, CompositeSpan)

    for tracer in (a, b):
        child = tracer.spans[0]
        assert child.started is True
        assert child.finished is True


def test_composite_finish_restores_contextvar_lifo() -> None:
    """Children tracking the framework ContextVar must be finished in
    reverse start order.

    ``start()`` installs child[0] then child[1] on the ``_current_span``
    stack, so child[1] is current and its token reverts to child[0].
    Finishing in START order resets child[0]'s (outer) token first —
    restoring the var to ``None`` — then child[1]'s token, which reverts
    the var to child[0], leaving a *finished* span as the current one.
    Finishing in reverse (LIFO) order clears the stack cleanly.
    """
    a = _RealSpanTracer()
    b = _RealSpanTracer()
    multi = MultiTracer([a, b])

    assert current_span() is None
    span = multi.custom_span(CustomSpanData(name="x", data={}))
    span.start()
    span.finish()

    # A balanced start/finish must leave the stack empty — not pointing at
    # one of the now-finished child spans.
    assert current_span() is None


def test_composite_span_records_exception_on_every_child() -> None:
    """When the ``with`` body raises, every child span must receive
    ``set_error`` via :meth:`Span.__exit__`."""
    a = _RecordingTracer("a")
    b = _RecordingTracer("b")
    multi = MultiTracer([a, b])

    with pytest.raises(ValueError), multi.custom_span(CustomSpanData(name="oops", data={})):
        raise ValueError("bad input")

    for tracer in (a, b):
        child = tracer.spans[0]
        assert len(child.errors_recorded) == 1
        message, error_data = child.errors_recorded[0]
        assert message == "bad input"
        assert error_data == {"type": "ValueError"}
