"""Tests for LLMRouter wiring in the agent loop.

Proves:
1. ``RunConfig(router=...)`` falls back to a pricier candidate when the
   cheap one raises a retryable error (end-to-end Runner.arun).
2. A normal ``RunConfig()`` (no router) produces identical behaviour to
   the pre-routing baseline.
3. Per-candidate budget gate raises ``TenantBudgetExceeded`` and does NOT
   escalate to a pricier model when the budget is exhausted.

Tests 1 and 2 go through ``Runner.arun`` with patched ``call_llm`` /
``resolve_llm`` so no real LLM provider is reached. Test 3 covers the
per-candidate gate directly via ``call_llm_with_routing``, because
wiring the full budget plumbing through Runner requires a live
``RunContext`` with a tenant_id that's awkward to thread without the
runner's own context builder — and the gate behaviour is cleanly visible
at the driver level.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, override
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.budgets import TenantBudget
from troopai.adk.exceptions import TenantBudgetExceeded
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.llms.cost import CostEstimate
from troopai.adk.llms.llm import LLM
from troopai.adk.llms.routing import LLMRouter, RoutedModel, RoutingContext
from troopai.adk.run.config import RunConfig
from troopai.adk.run.llm_calls import call_llm_with_routing
from troopai.adk.run.runner import Runner
from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText
from troopai.adk.types.tokens.llm_usage import LLMUsage

# ---------------------------------------------------------------------------
# Shared fake LLM helpers
# ---------------------------------------------------------------------------


def _text_response(content: str = "done") -> LLMResponse:
    return LLMResponse(
        response_id="resp-routing-loop-test",
        model="fake",
        response=[LLMResponseText(text=content)],
        usage=LLMUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
    )


class _FakeLLM(LLM):
    """Stub LLM: ``acomplete`` is never called (patched at loop level)."""

    def __init__(self, estimate_usd: float | None = None, actual_usd: float | None = None) -> None:
        self._estimate_usd = estimate_usd
        self._actual_usd = actual_usd

    async def acomplete(  # type: ignore[override]
        self,
        messages: Any,
        llm_config: Any = None,
        tools: Any = None,
        output_schema: Any = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[Any]:
        raise NotImplementedError("acomplete is patched; should not be called")

    @override
    def cost(self, model: str, usage: LLMUsage) -> float | None:
        del model, usage
        return self._actual_usd

    @override
    def estimate_cost(
        self,
        messages: Any,
        model: str,
        *,
        max_output_tokens: int | None = None,
    ) -> CostEstimate:
        del messages, max_output_tokens
        return CostEstimate(
            model=model,
            input_tokens=10,
            estimated_output_tokens=0,
            estimated_cost_usd=self._estimate_usd,
            output_bounded=False,
        )


class _RaisingFakeLLM(LLM):
    """Raises RuntimeError on acomplete (retryable — not an TroopAIError)."""

    @override
    async def acomplete(  # type: ignore[override]
        self,
        messages: Any,
        llm_config: Any = None,
        tools: Any = None,
        output_schema: Any = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[Any]:
        raise RuntimeError("provider timeout — cheap model failed")

    @override
    def cost(self, model: str, usage: LLMUsage) -> float | None:
        del model, usage
        return None

    @override
    def estimate_cost(
        self,
        messages: Any,
        model: str,
        *,
        max_output_tokens: int | None = None,
    ) -> CostEstimate:
        del messages, max_output_tokens
        return CostEstimate(
            model=model,
            input_tokens=10,
            estimated_output_tokens=0,
            estimated_cost_usd=None,
            output_bounded=False,
        )


class _GoodFakeLLM(LLM):
    """Returns a successful LLMResponse. Tracks whether acomplete was called."""

    def __init__(self, content: str = "pricey output") -> None:
        self._content = content
        self.acomplete_called: bool = False

    @override
    async def acomplete(  # type: ignore[override]
        self,
        messages: Any,
        llm_config: Any = None,
        tools: Any = None,
        output_schema: Any = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[Any]:
        self.acomplete_called = True
        return _text_response(self._content)

    @override
    def cost(self, model: str, usage: LLMUsage) -> float | None:
        del model, usage
        return None

    @override
    def estimate_cost(
        self,
        messages: Any,
        model: str,
        *,
        max_output_tokens: int | None = None,
    ) -> CostEstimate:
        del messages, max_output_tokens
        return CostEstimate(
            model=model,
            input_tokens=10,
            estimated_output_tokens=0,
            estimated_cost_usd=None,
            output_bounded=False,
        )


class _StaticRouter(LLMRouter):
    """Returns a fixed candidate list in order."""

    def __init__(self, candidates_list: list[RoutedModel]) -> None:
        self._candidates = candidates_list

    @override
    def candidates(self, ctx: RoutingContext) -> Sequence[RoutedModel]:
        del ctx
        return self._candidates

    @override
    def should_escalate(self, response: LLMResponse | None) -> bool:
        return False


# ---------------------------------------------------------------------------
# Patch helpers
# ---------------------------------------------------------------------------


def _patch_runner_guardrails() -> tuple[Any, Any, Any]:
    return (
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
# Test 1 — router falls back to pricier candidate on failure
# ---------------------------------------------------------------------------


async def test_router_falls_back_to_pricier_on_failure() -> None:
    """Router [cheap_broken, pricey_ok] → final output from pricey_ok.

    Uses real fake LLMs so ``acomplete`` is actually called through the
    routing driver. ``build_tools`` is patched to None (no real tools).
    ``resolve_llm`` / ``resolve_model_name`` are patched for the
    non-routed resolve step that happens before the router branch.
    """
    cheap_broken = _RaisingFakeLLM()
    pricey_ok = _GoodFakeLLM(content="pricey output")

    router = _StaticRouter(
        [
            RoutedModel(llm=cheap_broken, model="cheap-broken"),
            RoutedModel(llm=pricey_ok, model="pricey-ok"),
        ]
    )
    agent = Agent(name="routing-loop-agent", system_prompt="test agent")
    config = RunConfig(router=router)

    grd1, grd2, grd3 = _patch_runner_guardrails()
    with (
        patch("troopai.adk.run.llm_calls.build_tools", new=AsyncMock(return_value=None)),
        patch("troopai.adk.run.loop.resolve_llm", return_value=pricey_ok),
        patch("troopai.adk.run.loop.resolve_model_name", return_value="pricey-ok"),
        grd1,
        grd2,
        grd3,
    ):
        result = await Runner.arun(agent, "hi", max_turns=3, run_config=config)

    assert result.final_output == "pricey output", f"expected 'pricey output', got {result.final_output!r}"
    assert pricey_ok.acomplete_called is True


# ---------------------------------------------------------------------------
# Test 2 — no router → unchanged behaviour
# ---------------------------------------------------------------------------


async def test_no_router_unchanged() -> None:
    """A ``RunConfig()`` with no router completes exactly like a baseline run."""
    agent = Agent(name="no-router-agent", system_prompt="no router test agent")
    config = RunConfig()  # no router

    fake_llm = _FakeLLM(estimate_usd=None, actual_usd=None)
    grd1, grd2, grd3 = _patch_runner_guardrails()
    with (
        patch(
            "troopai.adk.run.loop.call_llm",
            new=AsyncMock(side_effect=lambda *a, **kw: _text_response("baseline output")),
        ),
        patch("troopai.adk.run.loop.resolve_llm", return_value=fake_llm),
        grd1,
        grd2,
        grd3,
    ):
        result = await Runner.arun(agent, "hi", max_turns=3, run_config=config)

    assert result.final_output == "baseline output", f"expected 'baseline output', got {result.final_output!r}"
    # Confirm no routing code was involved: router is None
    assert config.router is None


# ---------------------------------------------------------------------------
# Test 3 — per-candidate budget gate raises TenantBudgetExceeded; no escalation
# ---------------------------------------------------------------------------


async def test_routed_call_per_candidate_budget_gate() -> None:
    """Per-candidate budget gate raises ``TenantBudgetExceeded`` and does NOT escalate.

    Exercises the gate via ``call_llm_with_routing`` directly.
    ``TenantBudgetExceeded`` is an ``TroopAIError`` subclass → ``_is_routing_retryable``
    returns False → the exception propagates immediately, never reaching the
    second candidate.
    """
    # LLM with a high cost estimate (exceeds per-run cap)
    expensive_llm = _FakeLLM(estimate_usd=0.50, actual_usd=0.50)
    good_llm = _FakeLLM(estimate_usd=None, actual_usd=None)
    good_called: list[bool] = []

    async def _track_good_call(*args: Any, **kwargs: Any) -> LLMResponse:
        good_called.append(True)
        return _text_response("should not reach")

    router = _StaticRouter(
        [
            RoutedModel(llm=expensive_llm, model="expensive"),
            RoutedModel(llm=good_llm, model="good"),
        ]
    )
    agent = Agent(name="budget-gate-agent", system_prompt="budget gate test")
    config = RunConfig(
        tenant_id="t-budget-gate",
        tenant_budget=TenantBudget(dollars_per_run=0.001),
    )
    hooks = RunHooks()

    # Build a real RunContext with the tenant_id threaded in so the budget
    # gate sees it; use the same make() helper the runner uses.
    from troopai.adk.run.context import RunContext

    ctx: RunContext[None] = RunContext.make(None)
    ctx.tenant_id = "t-budget-gate"

    with (
        patch(
            "troopai.adk.run.llm_calls.build_tools",
            new=AsyncMock(return_value=None),
        ),
        patch(
            "troopai.adk.run.llm_calls.call_llm",
            new=AsyncMock(side_effect=_track_good_call),
        ),
        pytest.raises(TenantBudgetExceeded) as exc_info,
    ):
        await call_llm_with_routing(
            router,
            agent,
            [],
            config,
            hooks,
            context=ctx,
        )

    assert exc_info.value.scope == "run"
    assert exc_info.value.tenant_id == "t-budget-gate"
    # The budget gate propagated immediately; the second candidate's call_llm must not have run.
    assert len(good_called) == 0, (
        f"good_llm was invoked {len(good_called)} time(s) — gate should have stopped escalation"
    )
