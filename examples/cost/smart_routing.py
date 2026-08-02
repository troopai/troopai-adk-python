"""Cheap-first smart routing with automatic fallback.

Demonstrates ``LLMRouter`` in action: a fixed-order router tries a
deliberately-invalid model first; that attempt fails with a provider error,
the router escalates to a real ``claude-haiku-4-5-20251001`` candidate, and
the run completes normally. One real API call is made — only to haiku.

Key concepts:

- :class:`~troopai.adk.llms.routing.LLMRouter` defines an ordered candidate list.
- The agent loop escalates to the next candidate on any non-framework error
  (provider timeout, bad model name, etc.).
- :class:`~troopai.adk.llms.routing.CheapestFirstRouter` orders candidates by
  estimated USD ascending; this example uses a custom fixed-order subclass
  so the broken candidate is always tried first, making the fallback explicit.
- Pass the router via :attr:`~troopai.adk.run.config.RunConfig.router`.

Usage::

    python examples/cost/smart_routing.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from collections.abc import Sequence
from typing import override

from troopai.adk.agents import Agent
from troopai.adk.llms import LiteLLM
from troopai.adk.llms.routing import LLMRouter, RoutedModel, RoutingContext
from troopai.adk.run import Runner
from troopai.adk.run.config import RunConfig
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

GOOD_MODEL = "claude-haiku-4-5-20251001"
BROKEN_MODEL = "not-a-real-model-xyz-99999"


class FixedOrderRouter(LLMRouter):
    """Returns candidates in the exact order given at construction.

    Used here to guarantee the broken candidate is always tried first,
    making the fallback path explicit regardless of estimated costs.
    """

    def __init__(self, candidates_list: list[RoutedModel]) -> None:
        if len(candidates_list) == 0:
            raise ValueError("FixedOrderRouter requires at least one candidate")
        self._candidates = candidates_list

    @override
    def candidates(self, ctx: RoutingContext) -> Sequence[RoutedModel]:
        del ctx
        return self._candidates


agent = Agent(
    name="routing-demo",
    system_prompt="Reply in one short sentence.",
    # When RunConfig.router is set, agent.llm is bypassed — the router's
    # candidates supply the LLMs. It's set here only as a sensible default.
    llm=GOOD_MODEL,
)


async def main() -> None:
    router = FixedOrderRouter(
        [
            # Candidate 1: intentionally broken — will fail and trigger escalation
            RoutedModel(llm=LiteLLM(model=BROKEN_MODEL), model=BROKEN_MODEL),
            # Candidate 2: the real haiku model — receives the call after fallback
            RoutedModel(llm=LiteLLM(model=GOOD_MODEL), model=GOOD_MODEL),
        ]
    )

    config = RunConfig(router=router, verbose=VerboseConfig())

    logger.info("Starting routed run: broken candidate first, haiku fallback second.")
    result = await Runner.arun(agent, "What is the capital of France?", run_config=config)

    logger.info("Run completed via routing fallback.")
    logger.info("final_output: %s", result.final_output)
    logger.info(
        "Accumulated cost: $%.6f (one real haiku call after broken-model escalation)",
        result.context.cost_usd,
    )


if __name__ == "__main__":
    asyncio.run(main())
