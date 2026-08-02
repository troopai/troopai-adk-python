"""Tests for per-tenant budget enforcement in the agent loop.

Proves:
(a) per-run budget raises ``TenantBudgetExceeded`` BEFORE the LLM call
    in kill mode.
(b) warn mode (``kill_on_exceed=False``) completes the run AND records
    actual spend to the ledger.
(c) a normal run with no tenant_budget is a complete no-op (zero
    behaviour change when unconfigured).

LLM-mocking pattern: patch ``troopai.adk.run.loop.call_llm`` to return a
fake ``LLMResponse``, patch ``troopai.adk.run.loop.resolve_llm`` to return
a stub ``LLM`` subclass whose ``cost()`` and ``estimate_cost()`` return
controlled values, and patch the runner-level guardrail coroutines (same
pattern as ``test_tenant_e2e.py``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, override
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.budgets import BudgetPeriod, InMemoryCostLedger, TenantBudget, period_key
from troopai.adk.exceptions import TenantBudgetExceeded
from troopai.adk.llms.cost import CostEstimate
from troopai.adk.llms.llm import LLM
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText
from troopai.adk.types.tokens.llm_usage import LLMUsage

# ---------------------------------------------------------------------------
# Shared fake LLM + response helpers
# ---------------------------------------------------------------------------


class _FakeLLM(LLM):
    """Stub LLM with controllable ``estimate_cost`` and ``cost`` returns.

    ``acomplete`` is never called in these tests — ``loop.call_llm`` is
    patched so the real provider is never reached.
    """

    def __init__(
        self,
        estimate_usd: float | None,
        actual_usd: float | None,
    ) -> None:
        self._estimate_usd = estimate_usd
        self._actual_usd = actual_usd

    async def acomplete(  # type: ignore[override]  # patched; never called
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


def _text_response() -> LLMResponse:
    return LLMResponse(
        response_id="resp-budget-test",
        model="fake",
        response=[LLMResponseText(text="done")],
        usage=LLMUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15),
    )


def _patch_runner_guardrails() -> tuple[Any, Any, Any]:
    """Return three patch context managers for the runner-level guardrails."""
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
# Test 1 — per-run budget raises TenantBudgetExceeded BEFORE the LLM call
# ---------------------------------------------------------------------------


async def test_per_run_budget_kills_before_call() -> None:
    """``dollars_per_run=0.001`` with estimate=0.05 → raises before calling LLM.

    Asserts that the ``call_llm`` mock has zero invocations, proving
    ``TenantBudgetExceeded`` was raised in the pre-call gate.
    """
    agent = Agent(name="budget-kill-agent", system_prompt="You are a budget-kill agent.")
    config = RunConfig(
        tenant_id="t1",
        tenant_budget=TenantBudget(dollars_per_run=0.001),
    )
    fake_llm = _FakeLLM(estimate_usd=0.05, actual_usd=0.05)
    call_llm_mock = AsyncMock(side_effect=lambda *a, **kw: _text_response())

    grd1, grd2, grd3 = _patch_runner_guardrails()
    with (  # noqa: SIM117  # pytest.raises cannot be merged into the patch block
        patch("troopai.adk.run.loop.call_llm", new=call_llm_mock),
        patch("troopai.adk.run.loop.resolve_llm", return_value=fake_llm),
        grd1,
        grd2,
        grd3,
    ):
        with pytest.raises(TenantBudgetExceeded) as exc_info:
            await Runner.arun(agent, "go", max_turns=3, run_config=config)

    assert exc_info.value.scope == "run"
    assert exc_info.value.tenant_id == "t1"
    # Gate fires before the call — the LLM must never have been invoked.
    assert call_llm_mock.call_count == 0, (
        f"call_llm was invoked {call_llm_mock.call_count} time(s) — budget gate did not fire pre-call"
    )


# ---------------------------------------------------------------------------
# Test 2 — warn mode completes AND records spend to the ledger
# ---------------------------------------------------------------------------


async def test_warn_mode_continues_and_records() -> None:
    """``kill_on_exceed=False`` → run completes; actual cost recorded in ledger."""
    from datetime import UTC, datetime

    ledger = InMemoryCostLedger()
    agent = Agent(name="budget-warn-agent", system_prompt="You are a budget-warn agent.")
    config = RunConfig(
        tenant_id="t1",
        cost_ledger=ledger,
        tenant_budget=TenantBudget(
            dollars_per_period=0.001,
            kill_on_exceed=False,
        ),
    )
    # estimate=0.05 breaches the 0.001 cap → warn path triggered
    # actual cost returned by cost() is also 0.05 → recorded post-call
    fake_llm = _FakeLLM(estimate_usd=0.05, actual_usd=0.05)

    grd1, grd2, grd3 = _patch_runner_guardrails()
    with (
        patch(
            "troopai.adk.run.loop.call_llm",
            new=AsyncMock(side_effect=lambda *a, **kw: _text_response()),
        ),
        patch("troopai.adk.run.loop.resolve_llm", return_value=fake_llm),
        grd1,
        grd2,
        grd3,
    ):
        result = await Runner.arun(agent, "go", max_turns=3, run_config=config)

    assert result.final_output == "done", f"expected 'done', got {result.final_output!r}"

    # Ledger should have the actual spend recorded for today's period key
    expected_key = period_key(BudgetPeriod.DAY, datetime.now(UTC))
    recorded = await ledger.spend("t1", expected_key)
    assert recorded == pytest.approx(0.05), f"ledger recorded {recorded}, expected ~0.05 for (t1, {expected_key!r})"


# ---------------------------------------------------------------------------
# Test 3 — no budget configured → zero behaviour change
# ---------------------------------------------------------------------------


async def test_no_budget_is_noop() -> None:
    """A ``RunConfig()`` with no ``tenant_budget`` runs exactly like a baseline run."""
    agent = Agent(name="no-budget-agent", system_prompt="You are a no-budget agent.")
    config = RunConfig()  # no tenant_budget, no tenant_id

    grd1, grd2, grd3 = _patch_runner_guardrails()
    with (
        patch(
            "troopai.adk.run.loop.call_llm",
            new=AsyncMock(side_effect=lambda *a, **kw: _text_response()),
        ),
        patch(
            "troopai.adk.run.loop.resolve_llm",
            return_value=_FakeLLM(estimate_usd=None, actual_usd=None),
        ),
        grd1,
        grd2,
        grd3,
    ):
        result = await Runner.arun(agent, "go", max_turns=3, run_config=config)

    assert result.final_output == "done", f"expected 'done', got {result.final_output!r}"
    assert result.context.cost_usd == 0.0


# ---------------------------------------------------------------------------
# Test 4 — ledger outage fails CLOSED by default (cost-conservative)
# ---------------------------------------------------------------------------


class _FailingLedger:
    """A CostLedger whose backing store is unreachable (both ops raise).

    Models a Redis/Postgres outage: ``spend()`` cannot report the period
    total and ``record()`` cannot persist. Used to prove the per-period
    dollar gate fails closed rather than silently permitting unbounded spend.
    """

    async def spend(self, tenant_id: str, period_key: str) -> float:
        del tenant_id, period_key
        raise ConnectionError("cost ledger backend unreachable")

    async def record(self, tenant_id: str, period_key: str, cost_usd: float) -> None:
        del tenant_id, period_key, cost_usd
        raise ConnectionError("cost ledger backend unreachable")


async def test_ledger_outage_fails_closed_by_default() -> None:
    """spend() outage + default config → TenantBudgetExceeded (period), no LLM call.

    The estimate (0.0001) is UNDER the cap (0.001), so the only thing that can
    block the call is the fail-closed treatment of the unreadable ledger.
    """
    agent = Agent(name="ledger-failclosed-agent", system_prompt="You are a test agent.")
    config = RunConfig(
        tenant_id="t1",
        cost_ledger=_FailingLedger(),
        tenant_budget=TenantBudget(dollars_per_period=0.001),  # kill_on_exceed defaults True
    )
    fake_llm = _FakeLLM(estimate_usd=0.0001, actual_usd=0.0001)
    call_llm_mock = AsyncMock(side_effect=lambda *a, **kw: _text_response())

    grd1, grd2, grd3 = _patch_runner_guardrails()
    with (  # noqa: SIM117
        patch("troopai.adk.run.loop.call_llm", new=call_llm_mock),
        patch("troopai.adk.run.loop.resolve_llm", return_value=fake_llm),
        grd1,
        grd2,
        grd3,
    ):
        with pytest.raises(TenantBudgetExceeded) as exc_info:
            await Runner.arun(agent, "go", max_turns=3, run_config=config)

    assert exc_info.value.scope == "period"
    assert exc_info.value.tenant_id == "t1"
    assert call_llm_mock.call_count == 0, (
        f"call_llm invoked {call_llm_mock.call_count}× — fail-closed gate did not block on ledger outage"
    )


async def test_ledger_outage_fail_open_proceeds() -> None:
    """ledger_fail_open=True → the outage is logged but the call proceeds."""
    agent = Agent(name="ledger-failopen-agent", system_prompt="You are a test agent.")
    config = RunConfig(
        tenant_id="t1",
        cost_ledger=_FailingLedger(),
        tenant_budget=TenantBudget(dollars_per_period=0.001),
        ledger_fail_open=True,
    )
    fake_llm = _FakeLLM(estimate_usd=0.0001, actual_usd=0.0001)
    call_llm_mock = AsyncMock(side_effect=lambda *a, **kw: _text_response())

    grd1, grd2, grd3 = _patch_runner_guardrails()
    with (
        patch("troopai.adk.run.loop.call_llm", new=call_llm_mock),
        patch("troopai.adk.run.loop.resolve_llm", return_value=fake_llm),
        grd1,
        grd2,
        grd3,
    ):
        result = await Runner.arun(agent, "go", max_turns=3, run_config=config)

    assert result.final_output == "done"
    assert call_llm_mock.call_count >= 1, "fail-open should have allowed the LLM call"


async def test_ledger_outage_warn_mode_continues() -> None:
    """Fail-closed default + kill_on_exceed=False → warn-and-continue (no raise).

    A soft-budget posture (kill_on_exceed=False) means the developer already
    accepted overspend warnings, so the ledger outage degrades to a warning
    rather than a hard block — even though record() also fails.
    """
    agent = Agent(name="ledger-warn-agent", system_prompt="You are a test agent.")
    config = RunConfig(
        tenant_id="t1",
        cost_ledger=_FailingLedger(),
        tenant_budget=TenantBudget(dollars_per_period=0.001, kill_on_exceed=False),
    )
    fake_llm = _FakeLLM(estimate_usd=0.0001, actual_usd=0.0001)
    call_llm_mock = AsyncMock(side_effect=lambda *a, **kw: _text_response())

    grd1, grd2, grd3 = _patch_runner_guardrails()
    with (
        patch("troopai.adk.run.loop.call_llm", new=call_llm_mock),
        patch("troopai.adk.run.loop.resolve_llm", return_value=fake_llm),
        grd1,
        grd2,
        grd3,
    ):
        result = await Runner.arun(agent, "go", max_turns=3, run_config=config)

    assert result.final_output == "done"
    assert call_llm_mock.call_count >= 1
