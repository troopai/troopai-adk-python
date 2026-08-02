"""End-to-end tests for ``tenant_id`` threading through ``Runner.arun`` and
``Runner.arun_graph``.

Three high-value scenarios:

1. **Agent run, span tenant** — ``Runner.arun`` with ``RunConfig(tenant_id=…,
   tracing_enabled=True)`` emits ``AgentSpanData`` and ``GenerationSpanData``
   both carrying ``tenant_id == "acme"``.

2. **Agent run, status tenant + cost** — ``Runner.arun`` with
   ``RunConfig(tenant_id=…)`` and ``StatusTrackingHooks`` records exactly one
   run for ``"acme"`` with the expected cost.

3. **Graph run, tenant threaded (locks the fix)** — ``Runner.arun_graph``
   with ``RunConfig(tenant_id=…)`` and a ``GraphHooks`` observer asserts that
   the ``RunContext`` reaching the graph hooks carries ``tenant_id == "acme"``.
   This test fails against the pre-fix bare ``RunContext(context=context)`` and
   passes once the graph entry-points use ``RunContext.make(..., tenant_id=…)``.
"""

from __future__ import annotations

from typing import Any, override
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.hooks import GraphHooks
from troopai.adk.graphs.result import GraphRunStatus
from troopai.adk.graphs.state import GraphState
from troopai.adk.run.config import RunConfig
from troopai.adk.run.context import RunContext
from troopai.adk.run.runner import Runner
from troopai.adk.status.hooks import StatusTrackingHooks
from troopai.adk.status.store import AgentStatusStore
from troopai.adk.tracing import Span, set_tracer
from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText
from troopai.adk.types.tokens.llm_usage import LLMUsage
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
# Shared recording tracer
# ---------------------------------------------------------------------------


class _RecordingTracer:
    """Minimal in-memory tracer that captures every emitted span."""

    def __init__(self) -> None:
        self.spans: list[SpanData] = []

    def _record(self, data: SpanData) -> Span[Any]:
        self.spans.append(data)
        return Span(data)

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        return self._record(data)

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        return self._record(data)

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        return self._record(data)

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        return self._record(data)

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        return self._record(data)

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        return self._record(data)

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        return self._record(data)


@pytest.fixture(autouse=True)
def _reset_tracer() -> Any:
    yield
    set_tracer(None)


# ---------------------------------------------------------------------------
# Shared fake LLM helpers
# ---------------------------------------------------------------------------


def _text_response(response_id: str = "resp-1") -> LLMResponse:
    return LLMResponse(
        response_id=response_id,
        model="fake",
        response=[LLMResponseText(text="done")],
        usage=LLMUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _make_fake_call_llm(response_id: str = "resp-1") -> AsyncMock:
    async def _fake(*_args: Any, **_kwargs: Any) -> LLMResponse:
        return _text_response(response_id)

    return AsyncMock(side_effect=_fake)


def _patched_arun(agent: Agent, config: RunConfig) -> Any:
    """Context manager that patches the agent loop + guardrails for a clean run."""
    return (
        patch(
            "troopai.adk.run.loop.call_llm",
            new=_make_fake_call_llm(),
        ),
        patch(
            "troopai.adk.run.runner.run_blocking_input_guardrails",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "troopai.adk.run.runner.run_parallel_input_guardrails",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "troopai.adk.run.runner.run_output_guardrails",
            new=AsyncMock(return_value=[]),
        ),
    )


# ---------------------------------------------------------------------------
# Test 1: Agent run — span tenant_id
# ---------------------------------------------------------------------------


async def test_agent_run_span_tenant_id() -> None:
    """``Runner.arun`` with ``tenant_id="acme"`` emits agent + generation spans
    both carrying ``tenant_id == "acme"``."""
    tracer = _RecordingTracer()
    set_tracer(tracer)

    agent = Agent(
        name="tenant-span-agent",
        system_prompt="You are a test agent.",
    )
    config = RunConfig(tenant_id="acme", tracing_enabled=True)

    cm = _patched_arun(agent, config)
    with cm[0], cm[1], cm[2], cm[3]:
        await Runner.arun(agent, "go", max_turns=3, run_config=config)

    agent_spans = [s for s in tracer.spans if isinstance(s, AgentSpanData)]
    generation_spans = [s for s in tracer.spans if isinstance(s, GenerationSpanData)]

    assert len(agent_spans) >= 1, f"expected ≥1 AgentSpanData, got {tracer.spans}"
    assert len(generation_spans) >= 1, f"expected ≥1 GenerationSpanData, got {tracer.spans}"

    for a_span in agent_spans:
        assert a_span.tenant_id == "acme", f"AgentSpanData.tenant_id={a_span.tenant_id!r}, expected 'acme'"
    for g_span in generation_spans:
        assert g_span.tenant_id == "acme", f"GenerationSpanData.tenant_id={g_span.tenant_id!r}, expected 'acme'"


# ---------------------------------------------------------------------------
# Test 2: Agent run — status store tenant + cost
# ---------------------------------------------------------------------------


async def test_agent_run_status_tenant_and_cost() -> None:
    """``Runner.arun`` with ``tenant_id="acme"`` records one run in the status store
    under the correct tenant with the accumulated cost."""
    from collections.abc import AsyncIterator

    from troopai.adk.llms.llm import LLM

    class _FixedCostLLM(LLM):
        """Stub LLM that returns a fixed cost; acomplete is never called here."""

        async def acomplete(  # type: ignore[override]  # patched in loop; never called
            self,
            messages: Any,
            llm_config: Any = None,
            tools: Any = None,
            output_schema: Any = None,
            stream: bool = False,
        ) -> LLMResponse | AsyncIterator[Any]:
            raise NotImplementedError

        @override
        def cost(self, model: str, usage: LLMUsage) -> float | None:
            del model, usage
            return 0.07

    store = AgentStatusStore(path=":memory:")
    hooks: StatusTrackingHooks[object] = StatusTrackingHooks(store=store)
    agent = Agent(
        name="tenant-status-agent",
        system_prompt="You are a test agent.",
    )
    config = RunConfig(tenant_id="acme")
    fake_llm = _FixedCostLLM()

    with (
        patch(
            "troopai.adk.run.loop.call_llm",
            new=_make_fake_call_llm(),
        ),
        patch("troopai.adk.run.loop.resolve_llm", return_value=fake_llm),
        patch(
            "troopai.adk.run.runner.run_blocking_input_guardrails",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "troopai.adk.run.runner.run_parallel_input_guardrails",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "troopai.adk.run.runner.run_output_guardrails",
            new=AsyncMock(return_value=[]),
        ),
    ):
        await Runner.arun(agent, "go", max_turns=3, run_config=config, hooks=hooks)

    status = await store.get_status("tenant-status-agent", tenant_id="acme")
    assert status.total_runs == 1, f"total_runs={status.total_runs}, expected 1"
    assert status.total_cost_usd == pytest.approx(0.07), f"total_cost_usd={status.total_cost_usd}, expected 0.07"

    # A different explicit tenant must see no runs.
    other_status = await store.get_status("tenant-status-agent", tenant_id="other-co")
    assert other_status.total_runs == 0, (
        f"acme run incorrectly visible under tenant 'other-co': total_runs={other_status.total_runs}"
    )
    await store.close()


# ---------------------------------------------------------------------------
# Test 3: Graph run — tenant threaded into RunContext (locks the fix)
# ---------------------------------------------------------------------------


class _TenantCapturingHooks(GraphHooks[object]):
    """Observer that records the tenant_id seen on the RunContext at graph start."""

    def __init__(self) -> None:
        self.seen_tenant_id: str | None | type[_MISSING] = _MISSING

    @override
    async def on_graph_start(
        self,
        context: RunContext[object],
        state: GraphState[object],
    ) -> None:
        del state
        self.seen_tenant_id = context.tenant_id


class _MISSING:
    """Sentinel: hook never fired."""


async def test_arun_graph_threads_tenant_id_into_run_context() -> None:
    """``Runner.arun_graph`` with ``RunConfig(tenant_id="acme")`` must deliver a
    ``RunContext`` carrying ``tenant_id == "acme"`` to graph lifecycle hooks.

    This test is the regression guard for the bare-constructor bug:
    before the fix ``RunContext(context=context)`` silently dropped the
    tenant, causing ``on_graph_start`` to see ``None``.  After the fix
    ``RunContext.make(context, tenant_id=config.tenant_id)`` is used and
    the hook sees ``"acme"``.
    """
    g = Graph.new("tenant-graph").node("only", lambda: "done").entry("only").terminal("only").compile()

    observer = _TenantCapturingHooks()
    config = RunConfig(tenant_id="acme")

    result = await Runner.arun_graph(g, "go", hooks=[observer], run_config=config)

    assert result.status == GraphRunStatus.COMPLETED
    assert observer.seen_tenant_id is not _MISSING, "on_graph_start was never called — hooks not wired"
    assert observer.seen_tenant_id == "acme", (
        f"RunContext.tenant_id={observer.seen_tenant_id!r} reached graph hook, expected 'acme'. "
        "Likely cause: bare RunContext(context=context) drops tenant_id."
    )
