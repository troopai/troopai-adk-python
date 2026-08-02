"""Tests for :mod:`troopai.adk.tracing.spans` contextvars parent tracking.

Verifies that the :data:`troopai_current_span` ContextVar correctly
tracks nested parent-child relationships across ``with`` blocks,
``await`` boundaries, and concurrent :func:`asyncio.gather` tasks.

Also verifies the zero-cost invariant: :class:`NoOpSpan` must never
touch the ContextVar so disabled-tracing paths stay free.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from troopai.adk.tracing import (
    NoOpSpan,
    NoOpTracer,
    Span,
    current_span,
    custom_span,
    get_tracer,
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
    SpanData,
)


class _RecordingTracer:
    """Minimal tracer that returns real :class:`Span` instances so
    parent tracking works via the framework ContextVar."""

    def __init__(self) -> None:
        self.spans: list[SpanData] = []

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        self.spans.append(data)
        return Span(data)

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        self.spans.append(data)
        return Span(data)

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        self.spans.append(data)
        return Span(data)

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        self.spans.append(data)
        return Span(data)

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        self.spans.append(data)
        return Span(data)

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        self.spans.append(data)
        return Span(data)

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        self.spans.append(data)
        return Span(data)


@pytest.fixture(autouse=True)
def _tracer() -> Any:
    set_tracer(_RecordingTracer())
    yield
    set_tracer(None)


def test_current_span_none_by_default() -> None:
    assert current_span() is None


def test_nested_spans_chain_parent() -> None:
    with custom_span("outer", span_id="outer-id") as outer:
        assert current_span() is outer
        with custom_span("inner", span_id="inner-id") as inner:
            assert current_span() is inner
            assert inner.parent_id == "outer-id"
        assert current_span() is outer
    assert current_span() is None


def test_parent_restored_after_exception() -> None:
    with custom_span("outer", span_id="o") as outer:
        try:
            with custom_span("inner"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert current_span() is outer
    assert current_span() is None


@pytest.mark.asyncio
async def test_parent_chain_across_await() -> None:
    with custom_span("outer", span_id="outer-id"):
        await asyncio.sleep(0)
        with custom_span("inner") as inner:
            await asyncio.sleep(0)
            assert inner.parent_id == "outer-id"


@pytest.mark.asyncio
async def test_concurrent_gather_tasks_have_independent_stacks() -> None:
    """Each asyncio task owns its own ContextVar copy: sibling tasks must
    not see each other as parents."""
    observed: list[tuple[str, str | None]] = []

    async def worker(name: str) -> None:
        with custom_span(name, span_id=f"{name}-id") as span:
            await asyncio.sleep(0)
            observed.append((name, span.parent_id))

    async with asyncio.TaskGroup() as tg:
        tg.create_task(worker("a"))
        tg.create_task(worker("b"))
        tg.create_task(worker("c"))

    for _name, parent_id in observed:
        assert parent_id is None


@pytest.mark.asyncio
async def test_gather_children_see_outer_parent() -> None:
    """Workers spawned from inside an outer ``with`` block must still
    chain to it as their parent."""
    observed_parents: list[str | None] = []

    async def worker() -> None:
        with custom_span("child") as span:
            await asyncio.sleep(0)
            observed_parents.append(span.parent_id)

    with custom_span("outer", span_id="outer-id"):
        await asyncio.gather(worker(), worker(), worker())

    assert observed_parents == ["outer-id", "outer-id", "outer-id"]


def test_noop_span_is_zero_cost() -> None:
    """NoOpSpan must not touch the ContextVar — otherwise the disabled
    path is not actually free."""
    set_tracer(NoOpTracer())
    assert isinstance(get_tracer(), NoOpTracer)

    before = current_span()
    with custom_span("ignored") as span:
        assert isinstance(span, NoOpSpan)
        # Entering a NoOpSpan must not shift the ContextVar.
        assert current_span() is before
    assert current_span() is before


def test_disabled_kwarg_forces_noop_even_with_real_tracer() -> None:
    """``disabled=True`` bypasses the tracer and returns NoOpSpan."""
    with custom_span("off", disabled=True) as span:
        assert isinstance(span, NoOpSpan)
        # Still must not affect the parent chain.
        assert current_span() is None
