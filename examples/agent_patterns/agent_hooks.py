"""
Example: Per-Agent Lifecycle Hooks (``AgentHooks``)

Demonstrates ``AgentHooks`` — per-agent lifecycle callbacks that fire
alongside run-level ``RunHooks`` but are scoped to a single agent
instance. The load-bearing use case: multi-agent swarms where each
agent wants its own observability (metrics, tracing, side effects)
without the run-level hooks accumulating per-agent conditionals.

This example attaches a different ``MetricsHooks`` instance to each
agent in a two-agent handoff pipeline, so each instance collects only
its own agent's events. It also demonstrates:

- ``on_start`` / ``on_end``             — turn boundaries per agent
- ``on_llm_start`` / ``on_llm_end``     — counted per agent
- ``on_tool_start`` / ``on_tool_end``   — per-tool observation
- ``on_handoff``                        — fires on the INCOMING agent
  with ``source`` as the outgoing agent

Compare to ``RunHooks`` (see docs): ``RunHooks`` fires once for the
whole run, ``AgentHooks`` fires once per agent instance. Both can be
active at the same time.
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.agents import Agent
from troopai.adk.hooks.hooks import AgentHooks
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Metrics hooks — one instance attaches to one agent
# =============================================================================


class MetricsHooks(AgentHooks):
    """Collects lifecycle counts for a single agent.

    Instances are agent-scoped: attach one to each ``Agent`` and each
    instance will only record events for the agent it is attached to.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.starts = 0
        self.ends = 0
        self.llm_calls = 0
        self.tool_calls = 0
        self.handoffs_in = 0

    async def on_start(self, context, agent) -> None:
        del context
        self.starts += 1
        logger.info("[%s] on_start agent=%s starts=%d", self.label, agent.name, self.starts)

    async def on_end(self, context, agent, output) -> None:
        del context, output
        self.ends += 1
        logger.info("[%s] on_end agent=%s ends=%d", self.label, agent.name, self.ends)

    async def on_llm_start(self, context, agent, messages) -> None:
        del context, messages
        self.llm_calls += 1
        logger.info("[%s] on_llm_start agent=%s llm_calls=%d", self.label, agent.name, self.llm_calls)

    async def on_llm_end(self, context, agent, response) -> None:
        del context, agent, response

    async def on_tool_start(self, context, agent, tool_name, tool_input) -> None:
        del context, tool_input
        logger.info("[%s] on_tool_start agent=%s tool=%s", self.label, agent.name, tool_name)

    async def on_tool_end(self, context, agent, tool_name, tool_output) -> None:
        del context, tool_output
        self.tool_calls += 1
        logger.info(
            "[%s] on_tool_end agent=%s tool=%s tool_calls=%d", self.label, agent.name, tool_name, self.tool_calls
        )

    async def on_handoff(self, context, agent, source) -> None:
        del context
        self.handoffs_in += 1
        logger.info(
            "[%s] on_handoff agent=%s ← source=%s handoffs_in=%d", self.label, agent.name, source.name, self.handoffs_in
        )


# =============================================================================
# A tiny tool so we can observe on_tool_start / on_tool_end
# =============================================================================


async def _lookup_order_handler(ctx, raw_args):
    del ctx, raw_args
    return "order #42: status=shipped, eta=tomorrow"


lookup_order_tool = FunctionTool(
    name="lookup_order",
    description="Look up a customer order by id",
    schema={
        "type": "object",
        "properties": {"order_id": {"type": "string"}},
        "required": ["order_id"],
    },
    on_invoke=_lookup_order_handler,
)


# =============================================================================
# Main: router → specialist handoff with per-agent metrics
# =============================================================================


async def main() -> None:
    router_metrics = MetricsHooks(label="router")
    specialist_metrics = MetricsHooks(label="specialist")

    specialist = Agent(
        name="Specialist",
        system_prompt=(
            "You handle customer orders. Use lookup_order if the user asks about an order. Then respond concisely."
        ),
        tools=[lookup_order_tool],
        hooks=specialist_metrics,
    )

    router = Agent(
        name="Router",
        system_prompt=(
            "You are a frontline assistant. If the user asks about a specific "
            "order, hand off to the Specialist. Otherwise answer yourself."
        ),
        handoffs=[specialist],
        hooks=router_metrics,
    )

    result = await Runner.arun(
        router,
        "Can you check the status of order 42?",
        run_config=RunConfig(verbose=VerboseConfig()),
    )

    logger.info("=" * 60)
    logger.info("Final output: %s", result.final_output)
    logger.info("-" * 60)
    logger.info(
        "router  → starts=%d ends=%d llm_calls=%d tool_calls=%d handoffs_in=%d",
        router_metrics.starts,
        router_metrics.ends,
        router_metrics.llm_calls,
        router_metrics.tool_calls,
        router_metrics.handoffs_in,
    )
    logger.info(
        "specialist → starts=%d ends=%d llm_calls=%d tool_calls=%d handoffs_in=%d",
        specialist_metrics.starts,
        specialist_metrics.ends,
        specialist_metrics.llm_calls,
        specialist_metrics.tool_calls,
        specialist_metrics.handoffs_in,
    )


if __name__ == "__main__":
    asyncio.run(main())
