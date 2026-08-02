"""
Example: Multi-Turn Handoff Cost Comparison

Shows where code-orchestrated handoffs become cheaper than LLM-orchestrated.
The single-turn compare_handoffs.py shows code-orch as MORE expensive because
the response_format JSON schema (~1,400 tokens) exceeds 3 tool definitions
(~700 tokens). But that's misleading — in production, specialists have tools
and run multiple turns, and the cost dynamics flip.

Why multi-turn matters:
- LLM-orch history includes a tool_call + tool_result pair from triage
  (~200 extra tokens). This overhead is re-sent on EVERY specialist turn.
- Code-orch history includes only a compact Intent JSON (~50 tokens).
- With 3 specialist turns, LLM-orch pays ~600 extra tokens vs ~150 for
  code-orch. The triage schema overhead is paid ONCE; the specialist
  overhead compounds.

What this benchmark adds over compare_handoffs.py:
- Specialists have 2 tools each (so they run 2-3 turns)
- 5 handoff targets (more transfer_to_* tools = more triage input for LLM-orch)
- Queries that trigger tool use

Run:
    python examples/handoffs/compare_handoffs_multiturn.py
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
from troopai.adk.tools import function_tool
from troopai.adk.types.intents import Intent, Respond
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

# Queries that trigger tool use in the specialist
QUERIES = [
    "I want a refund for order #12345. It arrived damaged.",
    "Why was I charged twice this month? My account is ACC-789.",
    "My app crashes after the latest update. Error code E-500.",
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
    total_cost = est_cost(usage.input_tokens, usage.output_tokens)
    logger.info(f"  [{label}]")
    logger.info(f"    Requests:      {usage.requests}")
    logger.info(f"    Input tokens:  {usage.input_tokens:,}")
    logger.info(f"    Output tokens: {usage.output_tokens:,}")
    logger.info(f"    Total tokens:  {usage.total_tokens:,}")
    logger.info(f"    Est. cost:     ${total_cost:.6f}")


# =============================================================================
# Specialist tools (shared by both approaches)
# =============================================================================


@function_tool(name="lookup_order", description="Look up order details by ID.")
def lookup_order(order_id: str) -> str:
    return f'{{"order_id": "{order_id}", "status": "delivered", "total": 49.99, "condition": "damaged"}}'


@function_tool(name="process_refund", description="Process a refund for an order.")
def process_refund(order_id: str, reason: str) -> str:
    return f'{{"refund_id": "RF-{order_id}", "status": "approved", "amount": 49.99}}'


@function_tool(name="get_invoice", description="Get invoice details for an account.")
def get_invoice(account_id: str) -> str:
    return f'{{"account_id": "{account_id}", "invoices": [{{"id": "INV-001", "amount": 29.99}}, {{"id": "INV-002", "amount": 29.99}}]}}'


@function_tool(name="check_payment", description="Check payment status for an invoice.")
def check_payment(invoice_id: str) -> str:
    return f'{{"invoice_id": "{invoice_id}", "status": "paid", "date": "2025-03-15"}}'


@function_tool(name="check_system_status", description="Check system health and known issues.")
def check_system_status(component: str) -> str:
    return f'{{"component": "{component}", "status": "degraded", "known_issue": "current release crash on startup"}}'


@function_tool(name="lookup_error", description="Look up error code details.")
def lookup_error(error_code: str) -> str:
    return f'{{"error_code": "{error_code}", "description": "Null pointer in auth module", "fix": "Apply the latest auth-module patch"}}'


# =============================================================================
# Specialist agents WITH tools (each will run 2-3 turns)
# =============================================================================

refunds_agent = Agent(
    name="Refunds Specialist",
    system_prompt=(
        "You handle refund requests. Look up the order first, then process "
        "the refund. Always use both tools before responding."
    ),
    tools=[lookup_order, process_refund],
)

billing_agent = Agent(
    name="Billing Specialist",
    system_prompt=(
        "You handle billing questions. Get the invoice first, then check "
        "payment status. Always use both tools before responding."
    ),
    tools=[get_invoice, check_payment],
)

technical_agent = Agent(
    name="Technical Support",
    system_prompt=(
        "You handle technical issues. Check system status first, then look up "
        "the error code. Always use both tools before responding."
    ),
    tools=[check_system_status, lookup_error],
)

shipping_agent = Agent(
    name="Shipping Support",
    system_prompt="You handle shipping and delivery questions.",
)

account_agent = Agent(
    name="Account Support",
    system_prompt="You handle account settings, password resets, and profile changes.",
)


# =============================================================================
# Shared triage prompt
# =============================================================================

TRIAGE_PROMPT = (
    "You are a customer support triage agent. Classify the user's "
    "request and route to the right specialist:\n"
    "- Refund/return requests → Refunds Specialist\n"
    "- Billing/invoice/payment questions → Billing Specialist\n"
    "- Technical issues/bugs/errors → Technical Support\n"
    "- Shipping/delivery questions → Shipping Support\n"
    "- Account/password/profile → Account Support\n"
    "Do NOT answer the question yourself. Route immediately."
)


# =============================================================================
# Approach 1: LLM-Orchestrated (5 transfer_to_* tools)
# =============================================================================

llm_triage = Agent(
    name="LLM Triage",
    system_prompt=TRIAGE_PROMPT,
    handoffs=[refunds_agent, billing_agent, technical_agent, shipping_agent, account_agent],
)


# =============================================================================
# Approach 2: Code-Orchestrated (5 Intent types)
# =============================================================================


class RefundIntent(Intent):
    """User wants a refund or return."""

    kind: Literal["refund"] = "refund"
    order_id: str | None = Field(None, description="Order ID if mentioned.")


class BillingIntent(Intent):
    """User has a billing question."""

    kind: Literal["billing"] = "billing"
    account_id: str | None = Field(None, description="Account ID if mentioned.")


class TechnicalIssue(Intent):
    """User reports a technical problem."""

    kind: Literal["technical"] = "technical"
    error_code: str | None = Field(None, description="Error code if any.")


class ShippingIntent(Intent):
    """User has a shipping or delivery question."""

    kind: Literal["shipping"] = "shipping"


class AccountIntent(Intent):
    """User needs account/profile help."""

    kind: Literal["account"] = "account"


TriageOutput = Union[
    RefundIntent,
    BillingIntent,
    TechnicalIssue,
    ShippingIntent,
    AccountIntent,
    Respond,
]

code_triage = Agent(
    name="Code Triage",
    system_prompt=TRIAGE_PROMPT,
    output_schema=TriageOutput,
    handoffs=(
        HandoffRoute("compare_mt")
        .when(RefundIntent)
        .to(refunds_agent)
        .when(BillingIntent)
        .to(billing_agent)
        .when(TechnicalIssue)
        .to(technical_agent)
        .when(ShippingIntent)
        .to(shipping_agent)
        .when(AccountIntent)
        .to(account_agent)
        .otherwise(refunds_agent)
    ),
)


# =============================================================================
# Run comparison
# =============================================================================


@dataclass
class UsageBreakdown:
    triage_input: int = 0
    triage_output: int = 0
    specialist_input: int = 0
    specialist_output: int = 0
    specialist_requests: int = 0
    total: LLMUsage = field(default_factory=LLMUsage)


async def run_approach(
    name: str,
    agent: Agent,
    queries: list[str],
    run_config: RunConfig,
) -> UsageBreakdown:
    breakdown = UsageBreakdown()

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  {name}")
    logger.info(f"{'=' * 60}")

    for i, query in enumerate(queries, 1):
        result = await Runner.arun(agent, query, run_config=run_config)
        per_request = result.context.usage.usage

        # First call = triage, remaining = specialist turns
        triage_in = per_request[0].input_tokens if len(per_request) >= 1 else 0
        triage_out = per_request[0].output_tokens if len(per_request) >= 1 else 0

        spec_in = sum(r.input_tokens for r in per_request[1:])
        spec_out = sum(r.output_tokens for r in per_request[1:])
        spec_calls = len(per_request) - 1

        breakdown.triage_input += triage_in
        breakdown.triage_output += triage_out
        breakdown.specialist_input += spec_in
        breakdown.specialist_output += spec_out
        breakdown.specialist_requests += spec_calls
        breakdown.total = breakdown.total + result.context.usage

        routed_to = result.last_agent.name if result.last_agent is not None else "<none>"

        logger.info(f"\n  Query {i}: {query!r}")
        logger.info(f"  → Routed to: {routed_to} ({spec_calls} specialist turns)")
        logger.info(
            f"    Triage:     {triage_in:>5,} in / {triage_out:>4,} out  (${est_cost(triage_in, triage_out):.6f})"
        )
        logger.info(
            f"    Specialist: {spec_in:>5,} in / {spec_out:>4,} out  (${est_cost(spec_in, spec_out):.6f})  [{spec_calls} turns]"
        )

    logger.info(f"\n{'-' * 60}")
    print_usage(f"{name} TOTAL", breakdown.total)
    return breakdown


def print_row(label: str, llm_val: int, code_val: int, fmt: str = ",") -> None:
    diff = code_val - llm_val
    logger.info(f"  {label:<24} {llm_val:>10{fmt}} {code_val:>10{fmt}} {diff:>+10{fmt}}")


async def main() -> None:
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())
    llm = await run_approach("LLM-Orchestrated", llm_triage, QUERIES, run_config)
    code = await run_approach("Code-Orchestrated", code_triage, QUERIES, run_config)

    logger.info(f"\n{'=' * 60}")
    logger.info("  STEP-BY-STEP COMPARISON")
    logger.info(f"{'=' * 60}")
    logger.info(f"  {'Metric':<24} {'LLM-Orch':>10} {'Code-Orch':>10} {'Diff':>10}")
    logger.info(f"  {'-' * 54}")

    logger.info(f"  {'--- TRIAGE (1 call) ---'}")
    print_row("Input tokens", llm.triage_input, code.triage_input)
    print_row("Output tokens", llm.triage_output, code.triage_output)
    llm_tc = est_cost(llm.triage_input, llm.triage_output)
    code_tc = est_cost(code.triage_input, code.triage_output)
    logger.info(
        f"  {'Est. cost':<24} {'$' + f'{llm_tc:.6f}':>10} {'$' + f'{code_tc:.6f}':>10} {'$' + f'{code_tc - llm_tc:+.6f}':>10}"
    )

    logger.info(f"\n  {'--- SPECIALIST (multi) ---'}")
    print_row("LLM calls", llm.specialist_requests, code.specialist_requests)
    print_row("Input tokens", llm.specialist_input, code.specialist_input)
    print_row("Output tokens", llm.specialist_output, code.specialist_output)
    llm_sc = est_cost(llm.specialist_input, llm.specialist_output)
    code_sc = est_cost(code.specialist_input, code.specialist_output)
    logger.info(
        f"  {'Est. cost':<24} {'$' + f'{llm_sc:.6f}':>10} {'$' + f'{code_sc:.6f}':>10} {'$' + f'{code_sc - llm_sc:+.6f}':>10}"
    )

    logger.info(f"\n  {'--- OVERALL ---'}")
    print_row("Requests", llm.total.requests, code.total.requests)
    print_row("Total tokens", llm.total.total_tokens, code.total.total_tokens)
    llm_cost = est_cost(llm.total.input_tokens, llm.total.output_tokens)
    code_cost = est_cost(code.total.input_tokens, code.total.output_tokens)
    logger.info(
        f"  {'Est. cost':<24} {'$' + f'{llm_cost:.6f}':>10} {'$' + f'{code_cost:.6f}':>10} {'$' + f'{code_cost - llm_cost:+.6f}':>10}"
    )
    logger.info("")

    # Analysis
    triage_delta = code_tc - llm_tc
    spec_delta = code_sc - llm_sc
    if triage_delta > 0:
        logger.info(f"  Triage: code-orch pays ${triage_delta:.6f} MORE (schema > tool defs).")
    else:
        logger.info(f"  Triage: code-orch saves ${-triage_delta:.6f} (schema < tool defs).")

    if spec_delta < 0:
        logger.info(f"  Specialist: code-orch saves ${-spec_delta:.6f} (no tool_call/result in history).")
        logger.info(f"    → Savings compound: {llm.specialist_requests} specialist turns × triage overhead per turn.")
    else:
        logger.info(f"  Specialist: LLM-orch saves ${spec_delta:.6f}.")

    overall_delta = code_cost - llm_cost
    if overall_delta < 0:
        logger.info(f"\n  RESULT: Code-orchestrated is ${-overall_delta:.6f} cheaper overall.")
        logger.info(f"  Specialist savings ({-spec_delta:.6f}) outweigh triage overhead ({triage_delta:.6f}).")
    else:
        logger.info(f"\n  RESULT: LLM-orchestrated is ${overall_delta:.6f} cheaper overall.")
        logger.info("  Add more specialist turns or handoff targets to see code-orch win.")


if __name__ == "__main__":
    asyncio.run(main())
