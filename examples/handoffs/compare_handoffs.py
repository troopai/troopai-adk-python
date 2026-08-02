"""
Example: Handoff Cost Comparison — LLM-Orchestrated vs Code-Orchestrated

Runs the SAME queries through both handoff mechanisms so token usage and
costs are directly comparable. Neither triage agent has tools — they
both do pure classification/routing only.

What's the same (fair):
- Same queries, same specialist agents, same system prompt content.
- Both make 2 LLM calls per query: 1 triage + 1 specialist.

What's inherently different (by design):
- **Triage input**: LLM-orch sends 3 tool definitions (``transfer_to_*``).
  Code-orch sends a JSON schema for the Intent Union (``response_format``).
- **Triage output**: LLM-orch generates a tool call (function name + args).
  Code-orch generates a short JSON object (the Intent).
- **Specialist input**: LLM-orch history includes the tool call/result
  pair (verbose). Code-orch history includes just the Intent JSON
  assistant message (compact). This is the fundamental cost difference.

Run:
    python examples/handoffs/compare_handoffs.py
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Literal, Union

from litellm.cost_calculator import cost_per_token
from pydantic import Field

from troopai.adk.agents import Agent
from troopai.adk.handoffs import HandoffRoute
from troopai.adk.llms.llm_usage import LLMUsage
from troopai.adk.run import RunConfig, Runner
from troopai.adk.run.config import DEFAULT_MODEL
from troopai.adk.types.intents import Intent, Respond
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

# Queries both approaches will handle (identical inputs for fair comparison)
QUERIES = [
    "I want a refund for order #12345.",
    "Why was I charged twice this month?",
    "My app keeps crashing after the update.",
]


def est_cost(input_tokens: int, output_tokens: int) -> float:
    """Compute cost using litellm's per-model pricing."""
    prompt_cost, completion_cost = cost_per_token(
        DEFAULT_MODEL,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
    )
    return prompt_cost + completion_cost


def print_usage(label: str, usage: LLMUsage) -> None:
    """Print token usage and estimated cost."""
    total_cost = est_cost(usage.input_tokens, usage.output_tokens)
    logger.info(f"  [{label}]")
    logger.info(f"    Requests:      {usage.requests}")
    logger.info(f"    Input tokens:  {usage.input_tokens:,}")
    logger.info(f"    Output tokens: {usage.output_tokens:,}")
    logger.info(f"    Total tokens:  {usage.total_tokens:,}")
    if usage.input_tokens_details.cached_tokens > 0:
        logger.info(f"    Cached tokens: {usage.input_tokens_details.cached_tokens:,}")
    logger.info(f"    Est. cost:     ${total_cost:.6f}")


# =============================================================================
# Shared specialist agents (both approaches route to these)
# =============================================================================

refunds_agent = Agent(
    name="Refunds Specialist",
    system_prompt="You handle refund requests. Be empathetic and resolve quickly.",
)

billing_agent = Agent(
    name="Billing Specialist",
    system_prompt="You handle billing questions: invoices, charges, payment methods.",
)

technical_agent = Agent(
    name="Technical Support",
    system_prompt="You handle technical issues: bugs, errors, setup problems.",
)


# =============================================================================
# Shared system prompt (identical for both triage agents)
# =============================================================================
# Both agents do the same job: classify the user's request and route to
# a specialist. The ONLY difference is the routing mechanism.

TRIAGE_PROMPT = (
    "You are a customer support triage agent. Classify the user's "
    "request and route to the right specialist:\n"
    "- Refund requests → Refunds Specialist\n"
    "- Billing questions → Billing Specialist\n"
    "- Technical issues → Technical Support\n"
    "Do NOT answer the question yourself. Route immediately."
)


# =============================================================================
# Approach 1: LLM-Orchestrated (transfer_to_* tools)
# =============================================================================
# The LLM routes by calling a transfer_to_* tool.
# Triage cost: tool definitions sent as input, tool call as output.
# Specialist cost: sees tool call + tool result in history.

llm_triage = Agent(
    name="LLM Triage",
    system_prompt=TRIAGE_PROMPT,
    handoffs=[refunds_agent, billing_agent, technical_agent],
)


# =============================================================================
# Approach 2: Code-Orchestrated (structured Intent output)
# =============================================================================
# The LLM routes by outputting a structured Intent via response_format.
# Triage cost: JSON schema sent as input, short JSON as output.
# Specialist cost: sees only the Intent JSON in history (no tool call).


class RefundIntent(Intent):
    """User wants a refund or return."""

    kind: Literal["refund"] = "refund"
    order_id: str | None = Field(None, description="Order ID if mentioned.")


class BillingIntent(Intent):
    """User has a billing question."""

    kind: Literal["billing"] = "billing"
    topic: str | None = Field(None, description="Specific billing topic.")


class TechnicalIssue(Intent):
    """User reports a technical problem."""

    kind: Literal["technical"] = "technical"
    error_message: str | None = Field(None, description="Error message if any.")


TriageOutput = Union[RefundIntent, BillingIntent, TechnicalIssue, Respond]

code_triage = Agent(
    name="Code Triage",
    system_prompt=TRIAGE_PROMPT,
    output_schema=TriageOutput,
    handoffs=(
        HandoffRoute("compare")
        .when(RefundIntent)
        .to(refunds_agent)
        .when(BillingIntent)
        .to(billing_agent)
        .when(TechnicalIssue)
        .to(technical_agent)
        .otherwise(refunds_agent)
    ),
)


# =============================================================================
# Run comparison
# =============================================================================


@dataclass
class UsageBreakdown:
    """Accumulated per-step token counts across all queries."""

    triage_input: int = 0
    triage_output: int = 0
    specialist_input: int = 0
    specialist_output: int = 0
    total: LLMUsage = field(default_factory=LLMUsage)


async def run_approach(
    name: str,
    agent: Agent,
    queries: list[str],
    run_config: RunConfig,
) -> UsageBreakdown:
    """Run all queries through an agent and return per-step usage breakdown."""
    breakdown = UsageBreakdown()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  {name}")
    logger.info(f"{'=' * 60}")

    for i, query in enumerate(queries, 1):
        result = await Runner.arun(agent, query, run_config=run_config)
        per_request = result.context.usage.usage

        # Split per-request usage: first call = triage, second = specialist
        triage_in = per_request[0].input_tokens if len(per_request) >= 1 else 0
        triage_out = per_request[0].output_tokens if len(per_request) >= 1 else 0
        spec_in = per_request[1].input_tokens if len(per_request) >= 2 else 0
        spec_out = per_request[1].output_tokens if len(per_request) >= 2 else 0

        breakdown.triage_input += triage_in
        breakdown.triage_output += triage_out
        breakdown.specialist_input += spec_in
        breakdown.specialist_output += spec_out
        breakdown.total = breakdown.total + result.context.usage

        logger.info(f"\n  Query {i}: {query!r}")
        logger.info(f"  → Routed to: {result.last_agent.name}")
        logger.info(
            f"    Triage:     {triage_in:>5,} in / {triage_out:>4,} out  (${est_cost(triage_in, triage_out):.6f})"
        )
        logger.info(f"    Specialist: {spec_in:>5,} in / {spec_out:>4,} out  (${est_cost(spec_in, spec_out):.6f})")

    logger.info(f"\n{'-' * 60}")
    print_usage(f"{name} TOTAL", breakdown.total)
    return breakdown


def print_comparison_row(label: str, llm_val: int, code_val: int, fmt: str = ",") -> None:
    """Print one row of the comparison table."""
    diff = code_val - llm_val
    logger.info(f"  {label:<24} {llm_val:>10{fmt}} {code_val:>10{fmt}} {diff:>+10{fmt}}")


async def main() -> None:
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())
    llm = await run_approach("LLM-Orchestrated", llm_triage, QUERIES, run_config)
    code = await run_approach("Code-Orchestrated", code_triage, QUERIES, run_config)

    # Per-step comparison (isolates routing cost from specialist variability)
    logger.info(f"\n{'=' * 60}")
    logger.info("  STEP-BY-STEP COMPARISON")
    logger.info(f"{'=' * 60}")
    logger.info(f"  {'Metric':<24} {'LLM-Orch':>10} {'Code-Orch':>10} {'Diff':>10}")
    logger.info(f"  {'-' * 54}")

    logger.info(f"  {'--- TRIAGE STEP ---'}")
    print_comparison_row("Input tokens", llm.triage_input, code.triage_input)
    print_comparison_row("Output tokens", llm.triage_output, code.triage_output)
    llm_triage_cost = est_cost(llm.triage_input, llm.triage_output)
    code_triage_cost = est_cost(code.triage_input, code.triage_output)
    logger.info(
        f"  {'Est. cost':<24} {'$' + f'{llm_triage_cost:.6f}':>10} {'$' + f'{code_triage_cost:.6f}':>10} {'$' + f'{code_triage_cost - llm_triage_cost:+.6f}':>10}"
    )

    logger.info(f"\n  {'--- SPECIALIST STEP ---'}")
    print_comparison_row("Input tokens", llm.specialist_input, code.specialist_input)
    print_comparison_row("Output tokens", llm.specialist_output, code.specialist_output)
    llm_spec_cost = est_cost(llm.specialist_input, llm.specialist_output)
    code_spec_cost = est_cost(code.specialist_input, code.specialist_output)
    logger.info(
        f"  {'Est. cost':<24} {'$' + f'{llm_spec_cost:.6f}':>10} {'$' + f'{code_spec_cost:.6f}':>10} {'$' + f'{code_spec_cost - llm_spec_cost:+.6f}':>10}"
    )

    # Overall totals
    logger.info(f"\n  {'--- OVERALL ---'}")
    print_comparison_row("Requests", llm.total.requests, code.total.requests)
    print_comparison_row("Total tokens", llm.total.total_tokens, code.total.total_tokens)
    llm_cost = est_cost(llm.total.input_tokens, llm.total.output_tokens)
    code_cost = est_cost(code.total.input_tokens, code.total.output_tokens)
    logger.info(
        f"  {'Est. cost':<24} {'$' + f'{llm_cost:.6f}':>10} {'$' + f'{code_cost:.6f}':>10} {'$' + f'{code_cost - llm_cost:+.6f}':>10}"
    )
    logger.info("")

    if code_triage_cost < llm_triage_cost:
        savings_pct = (1 - code_triage_cost / llm_triage_cost) * 100
        logger.info(f"  Triage step: code-orchestrated is {savings_pct:.1f}% cheaper.")
    elif llm_triage_cost < code_triage_cost:
        savings_pct = (1 - llm_triage_cost / code_triage_cost) * 100
        logger.info(f"  Triage step: LLM-orchestrated is {savings_pct:.1f}% cheaper.")

    spec_diff = abs(code_spec_cost - llm_spec_cost)
    logger.info(f"  Specialist step: ${spec_diff:.6f} difference.")
    logger.info("")
    logger.info("  Note: Triage Δ = tool defs vs JSON schema (inherent to mechanism).")
    logger.info("  Note: Specialist Δ = tool call/result pair vs Intent JSON (inherent).")
    logger.info("  Both specialists see the full conversation history (strategy='full').")


if __name__ == "__main__":
    asyncio.run(main())
