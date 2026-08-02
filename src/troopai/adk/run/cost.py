"""Token cost optimization utilities.

Provides tool result post-processing (JSON minification, token-budget
truncation) and usage limit checking — extracted from ``Runner`` to keep
cost-optimization logic in one place.
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING

from troopai.adk.budgets import CostLedger, TenantBudget
from troopai.adk.exceptions import TenantBudgetExceeded, UsageLimitExceeded, UserError

if TYPE_CHECKING:
    from troopai.adk.llms.llm_usage import LLMUsage, LLMUsageLimits
    from troopai.adk.tools.function_tool import FunctionTool

logger = logging.getLogger(__name__)


def check_usage_limits(limits: LLMUsageLimits, usage: LLMUsage) -> None:
    """Check if current usage exceeds configured limits.

    Called after each LLM response. Raises :class:`UsageLimitExceeded`
    on the first limit that is exceeded.

    Args:
        limits: The configured usage limits.
        usage: The current cumulative usage.
    """
    if limits.request_limit is not None and usage.requests > limits.request_limit:
        raise UsageLimitExceeded(f"Request limit exceeded: {usage.requests} > {limits.request_limit}")
    if limits.tool_calls_limit is not None and usage.tool_calls > limits.tool_calls_limit:
        raise UsageLimitExceeded(f"Tool call limit exceeded: {usage.tool_calls} > {limits.tool_calls_limit}")
    if limits.input_tokens_limit is not None and usage.input_tokens > limits.input_tokens_limit:
        raise UsageLimitExceeded(f"Input token limit exceeded: {usage.input_tokens} > {limits.input_tokens_limit}")
    if limits.output_tokens_limit is not None and usage.output_tokens > limits.output_tokens_limit:
        raise UsageLimitExceeded(f"Output token limit exceeded: {usage.output_tokens} > {limits.output_tokens_limit}")
    if limits.total_tokens_limit is not None and usage.total_tokens > limits.total_tokens_limit:
        raise UsageLimitExceeded(f"Total token limit exceeded: {usage.total_tokens} > {limits.total_tokens_limit}")


def check_request_limit_before_call(limits: LLMUsageLimits, usage: LLMUsage) -> None:
    """Raise before a model request would exceed the configured request cap."""
    if limits.request_limit is None:
        return
    next_request = usage.requests + 1
    if next_request > limits.request_limit:
        raise UsageLimitExceeded(f"Request limit exceeded: {next_request} > {limits.request_limit}")


def check_tool_call_limits_before_dispatch(limits: LLMUsageLimits, usage: LLMUsage, pending: int) -> None:
    """Raise before dispatching a tool batch that would exceed the call cap."""
    if limits.tool_calls_limit is None:
        return
    projected = usage.tool_calls + pending
    if projected > limits.tool_calls_limit:
        raise UsageLimitExceeded(f"Tool call limit exceeded: {projected} > {limits.tool_calls_limit}")


def check_tenant_budget(
    budget: TenantBudget,
    tenant_id: str,
    run_cost: float,
    period_spend: float,
    estimate: float | None,
) -> None:
    """Raise ``TenantBudgetExceeded`` if ``estimate`` would breach a cap.

    This is a pure check: it ALWAYS raises on a breach. The caller
    (``enforce_tenant_budget`` in llm_calls.py) catches the exception and
    consults ``budget.kill_on_exceed`` to decide whether to abort the run or
    log a warning and continue.

    Uses strict ``>`` (matching :func:`check_usage_limits`), so a call whose
    projected total exactly meets the cap is allowed. ``estimate is None``
    (provider has no cost table) skips the estimate-based gate — the pre-call
    dollar check is unavailable, so enforcement falls back to post-call
    recording.

    A non-finite ``period_spend`` is the fail-closed sentinel its caller sets
    when the ledger is unreachable: the period cap is then unconditionally
    breached and is raised even when ``estimate is None``, because skipping the
    gate would let the call proceed and the (unreachable) ledger could not
    record it either — defeating the fail-closed guarantee.

    Args:
        budget: The tenant's configured budget policy.
        tenant_id: Identifier of the tenant being checked.
        run_cost: Accumulated cost for the current run so far.
        period_spend: Accumulated cost across runs in the current period.
        estimate: Estimated USD cost of the upcoming call, or ``None`` if
            the provider has no cost table.
    """
    if budget.dollars_per_period is not None and not math.isfinite(period_spend):
        raise TenantBudgetExceeded(tenant_id, "period", period_spend, budget.dollars_per_period, estimate or 0.0)
    if estimate is None:
        return
    if budget.dollars_per_run is not None and run_cost + estimate > budget.dollars_per_run:
        raise TenantBudgetExceeded(tenant_id, "run", run_cost, budget.dollars_per_run, estimate)
    if budget.dollars_per_period is not None and period_spend + estimate > budget.dollars_per_period:
        raise TenantBudgetExceeded(tenant_id, "period", period_spend, budget.dollars_per_period, estimate)


def validate_budget_config(budget: TenantBudget | None, ledger: CostLedger | None) -> None:
    """Fail fast on a budget misconfiguration before the run starts.

    A per-period cap (``dollars_per_period``) needs a ``cost_ledger`` to read
    and record cross-run spend; without one it cannot be enforced. A per-run
    cap needs no ledger.
    """
    if budget is None:
        return
    if budget.dollars_per_period is not None and ledger is None:
        raise UserError("RunConfig.tenant_budget.dollars_per_period requires a cost_ledger")


def minify_json(result: str) -> str:
    """Re-serialize JSON tool results with minimal whitespace.

    If the result is valid JSON, re-encodes it with compact
    separators (no spaces). Non-JSON results pass through unchanged.
    Typically saves 15-20% of tokens for structured API responses.

    Args:
        result: The tool result string.

    Returns:
        Minified JSON string, or the original string if not JSON.
    """
    try:
        parsed = json.loads(result)
        return json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return result


def apply_result_limits(
    result: str,
    tool: FunctionTool,
    model: str,
) -> str:
    """Truncate tool result if it exceeds the tool's token budget.

    Uses :class:`TokenCounter` for accurate estimation, then
    truncates at an approximate character boundary (avg ~4 chars
    per token for English text).

    Args:
        result: The raw tool result string.
        tool: The FunctionTool with optional ``max_result_tokens``.
        model: litellm model identifier for token counting.

    Returns:
        The original result if within budget, or a truncated
        version with a ``[Result truncated: ...]`` suffix.
    """
    from troopai.adk.context.token_counter import TokenCounter

    if tool.max_result_tokens is None:
        return result

    token_count: int = TokenCounter.count_text(result, model)
    if token_count <= tool.max_result_tokens:
        return result

    # Pre-compute the suffix and reserve its token cost so the total
    # never exceeds max_result_tokens.  Using a placeholder token count
    # for the suffix avoids a chicken-and-egg problem (the actual N/M
    # values affect the suffix length) while staying conservative.
    suffix = f"\n[Result truncated: {token_count} → {tool.max_result_tokens} tokens]"
    suffix_tokens: int = TokenCounter.count_text(suffix, model)
    effective_limit = max(tool.max_result_tokens - suffix_tokens, 1)

    # Binary-search for the largest character prefix whose token count
    # does not exceed effective_limit.  This is accurate for all
    # scripts (CJK, code, English) rather than the fixed 4-chars/token
    # English approximation.
    lo, hi = 0, len(result)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if TokenCounter.count_text(result[:mid], model) <= effective_limit:
            lo = mid
        else:
            hi = mid - 1
    truncated: str = result[:lo]

    logger.info(
        "Tool '%s' result truncated: %d tokens → %d max_result_tokens",
        tool.name,
        token_count,
        tool.max_result_tokens,
    )

    return truncated + suffix
