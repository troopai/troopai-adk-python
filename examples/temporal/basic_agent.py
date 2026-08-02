"""Minimal agent running as a Temporal workflow.

Demonstrates: TemporalLLM, TroopAIWorkflow, TroopAITemporalPlugin.

Prerequisites:
    pip install "troopai-adk-python[temporal]"
    temporal server start-dev

Run with:
    python examples/temporal/basic_agent.py
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1 — import guards (temporalio is an optional extra)
# ---------------------------------------------------------------------------

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.worker import Worker
except ImportError as _exc:
    raise SystemExit("temporalio not installed. Run: pip install 'troopai-adk-python[temporal]'") from _exc

# ---------------------------------------------------------------------------
# Step 2 — define the agent (Agent = config, not execution)
# ---------------------------------------------------------------------------

from troopai.adk.agents import Agent
from troopai.adk.llms import LiteLLM

_llm = LiteLLM(model="gpt-4o-mini")
_agent = Agent(
    name="summariser",
    system_prompt="Summarise the user's message in one sentence.",
    llm=_llm,
)

# ---------------------------------------------------------------------------
# Step 3 — wrap the agent LLM so every LLM call runs as a Temporal activity
# ---------------------------------------------------------------------------

from troopai.adk.workflows.temporal import (
    ModelActivityConfig,
    TemporalLLM,
    TroopAITemporalPlugin,
    TroopAIWorkflow,
)

# TemporalLLM.install replaces agent.llm (and every handoff target's LLM)
# with a TemporalLLM shim.  Outside a workflow the shim calls the wrapped
# LLM directly — zero overhead for non-durable usage.
TemporalLLM.install(
    _agent,
    activity_config=ModelActivityConfig(
        start_to_close_timeout=60,  # seconds
        maximum_attempts=3,
    ),
)

# ---------------------------------------------------------------------------
# Step 4 — define the workflow (subclass TroopAIWorkflow, override run())
# ---------------------------------------------------------------------------

from typing import override

from troopai.adk.run.runner import Runner


@workflow.defn
class SummariserWorkflow(TroopAIWorkflow):
    """Temporal workflow: route a single user message through the summariser agent."""

    __troopai_agents__ = (_agent,)

    @override
    @workflow.run
    async def run(self, prompt: str) -> str:
        """Execute the agent loop inside a durable Temporal workflow.

        Args:
            prompt: The user's input message to summarise.

        Returns:
            The agent's final text output.
        """
        workflow.logger.info("SummariserWorkflow started, prompt=%r", prompt)
        result = await Runner.arun(_agent, prompt)
        output: str = result.final_output if isinstance(result.final_output, str) else str(result.final_output)
        workflow.logger.info("SummariserWorkflow finished, output=%r", output)
        return output


# ---------------------------------------------------------------------------
# Step 5 — configure the worker with TroopAITemporalPlugin
# ---------------------------------------------------------------------------

from troopai.adk.workflows.temporal.activity import invoke_model_activity

TASK_QUEUE = "summariser-queue"


async def _run() -> None:
    """Start the worker and execute one workflow run end-to-end."""
    plugin = TroopAITemporalPlugin()
    # Register the underlying LLM under the key TemporalLLM will look up.
    # The key defaults to str(llm) if not overridden in TemporalLLM.install().
    plugin.register_model(str(_llm), _llm)

    client = await Client.connect("localhost:7233")
    logger.info("Connected to Temporal server")

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[SummariserWorkflow],
        activities=[invoke_model_activity],
        **plugin.build_worker_kwargs(),
    ):
        logger.info("Worker started on task queue %r", TASK_QUEUE)

        # Start the workflow and wait for the result.
        result = await client.execute_workflow(
            SummariserWorkflow.run,
            "Artificial intelligence is reshaping how humans interact with software.",
            id="summariser-run-001",
            task_queue=TASK_QUEUE,
        )
        logger.info("Workflow completed. Output: %s", result)


if __name__ == "__main__":
    asyncio.run(_run())
