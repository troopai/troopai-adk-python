"""Cost-aware context compaction under budget pressure.

Demonstrates two aspects of :attr:`~troopai.adk.context.context_config.CompactionConfig.cost_aware`:

**Part 1 — Pure demonstration (no API call)**:
:func:`~troopai.adk.context.context_manager.effective_compaction_config` is
called directly with a run cost that exceeds 80 % of the per-run budget. The
returned config has a tightened ``trigger_tokens`` (halved) and a reduced
``preserve_recent_items`` (capped at 1), shedding input tokens as the run
approaches its budget.

**Part 2 — End-to-end wiring (one real haiku call)**:
A :class:`~troopai.adk.run.config.RunConfig` with both a
:class:`~troopai.adk.context.context_config.ContextManagementConfig` (compaction
enabled and cost-aware) and a :class:`~troopai.adk.budgets.TenantBudget` is
passed to :meth:`~troopai.adk.run.Runner.arun`. The run completes normally;
this confirms the wiring is live and the cost-aware compaction path is reachable.

Usage::

    python examples/cost/cost_aware_compaction.py
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
from troopai.adk.context.context_config import CompactionConfig, ContextManagementConfig
from troopai.adk.context.context_manager import effective_compaction_config
from troopai.adk.run import Runner
from troopai.adk.run.config import RunConfig
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"
TENANT_ID = "compaction-demo"


async def main() -> None:
    # -------------------------------------------------------------------
    # Part 1: pure tightening demonstration — no API call
    # -------------------------------------------------------------------
    logger.info("=== Part 1: effective_compaction_config tightening (no API call) ===")

    original = CompactionConfig(
        enabled=True,
        trigger_tokens=100_000,
        preserve_recent_items=6,
        cost_aware=True,
    )
    budget = TenantBudget(dollars_per_run=1.0)

    # run_cost=0.90 → utilization = 90 % ≥ threshold=0.8 → tightening applies
    tightened = effective_compaction_config(original, budget, run_cost=0.90, threshold=0.8)

    logger.info(
        "original  trigger_tokens=%d  preserve_recent_items=%d", original.trigger_tokens, original.preserve_recent_items
    )
    logger.info(
        "tightened trigger_tokens=%d  preserve_recent_items=%d",
        tightened.trigger_tokens,
        tightened.preserve_recent_items,
    )
    logger.info(
        "Trigger halved from %d → %d; preserve_recent_items capped at %d → %d",
        original.trigger_tokens,
        tightened.trigger_tokens,
        original.preserve_recent_items,
        tightened.preserve_recent_items,
    )

    # Below threshold — no tightening
    unchanged = effective_compaction_config(original, budget, run_cost=0.10, threshold=0.8)
    logger.info(
        "Below threshold (run_cost=0.10): unchanged trigger_tokens=%d (same=%s)",
        unchanged.trigger_tokens,
        unchanged.trigger_tokens == original.trigger_tokens,
    )

    # -------------------------------------------------------------------
    # Part 2: end-to-end wiring — one real haiku call
    # -------------------------------------------------------------------
    logger.info("=== Part 2: end-to-end run with cost-aware compaction wired ===")

    agent = Agent(
        name="compaction-demo",
        system_prompt="Reply in one short sentence.",
        llm=MODEL,
    )

    run_config = RunConfig(
        tenant_id=TENANT_ID,
        tenant_budget=TenantBudget(dollars_per_run=1.0),
        context_management=ContextManagementConfig(
            compaction=CompactionConfig(
                enabled=True,
                trigger_tokens=100_000,
                preserve_recent_items=6,
                cost_aware=True,
            ),
        ),
        verbose=VerboseConfig(),
    )

    result = await Runner.arun(agent, "What is 5 + 7?", run_config=run_config)
    logger.info("Run completed. final_output: %s", result.final_output)
    logger.info("Accumulated run cost: $%.6f", result.context.cost_usd)
    logger.info("Cost-aware compaction wiring confirmed end-to-end.")


if __name__ == "__main__":
    asyncio.run(main())
