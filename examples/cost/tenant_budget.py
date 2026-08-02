"""Per-tenant budget enforcement with TenantBudget.

Demonstrates the two key behaviours of the per-tenant run budget:

1. **Generous budget** — a ``dollars_per_run=1.0`` cap easily accommodates a
   single haiku call. The run completes normally and the final output is
   logged.
2. **Tiny budget** — a ``dollars_per_run=0.0000001`` cap is lower than the
   pre-call cost estimate, so :class:`~troopai.adk.exceptions.TenantBudgetExceeded`
   is raised *before* any API call is made. The tenant, scope, and budget are
   logged from the caught exception.

Usage::

    python examples/cost/tenant_budget.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.agents import Agent
from troopai.adk.budgets import TenantBudget
from troopai.adk.exceptions import TenantBudgetExceeded
from troopai.adk.run import Runner
from troopai.adk.run.config import RunConfig
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
TENANT_ID = "demo-tenant"

agent = Agent(
    name="budget-demo",
    system_prompt="Reply in one short sentence.",
    llm=MODEL,
)


async def main() -> None:
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    verbose = VerboseConfig()

    # --- Case 1: generous budget — run completes ---
    logger.info("=== Case 1: generous budget (dollars_per_run=1.0) ===")
    run_config = RunConfig(
        tenant_id=TENANT_ID,
        tenant_budget=TenantBudget(dollars_per_run=1.0),
        verbose=verbose,
    )
    result = await Runner.arun(agent, "What is 2 + 2?", run_config=run_config)
    logger.info("Run completed. final_output: %s", result.final_output)
    logger.info("Accumulated run cost: $%.6f", result.context.cost_usd)

    # --- Case 2: tiny budget — TenantBudgetExceeded raised pre-call ---
    logger.info("=== Case 2: tiny budget (dollars_per_run=0.0000001) ===")
    tight_config = RunConfig(
        tenant_id=TENANT_ID,
        tenant_budget=TenantBudget(dollars_per_run=0.0000001),
        verbose=verbose,
    )
    try:
        await Runner.arun(agent, "What is 3 + 3?", run_config=tight_config)
    except TenantBudgetExceeded as exc:
        logger.info(
            "TenantBudgetExceeded caught — tenant=%r, scope=%r, budget=$%.9f",
            exc.tenant_id,
            exc.scope,
            exc.budget,
        )
        logger.info("No real API call was made (pre-call kill).")
    else:
        logger.warning("Expected TenantBudgetExceeded was NOT raised — budget enforcement may be broken")


if __name__ == "__main__":
    asyncio.run(main())
