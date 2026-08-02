"""Graph with Human-In-The-Loop (HITL) interrupt and Temporal signal resume.

Demonstrates:
    - A Graph with two nodes: one normal node, one HITL-interrupting node
    - Wrapping agent LLMs with TemporalLLM for durable LLM calls
    - A Temporal workflow that catches GraphRunStatus.INTERRUPTED, enters
      workflow.wait_condition, and resumes when the client sends a
      send_human_reply signal
    - Sending a HumanReply signal from the client side

Prerequisites:
    pip install "troopai-adk-python[temporal]"
    temporal server start-dev

Run with:
    python examples/temporal/graph_workflow.py
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from temporalio import workflow
    from temporalio.client import Client
    from temporalio.worker import Worker
except ImportError as _exc:
    raise SystemExit("temporalio not installed. Run: pip install 'troopai-adk-python[temporal]'") from _exc

# ---------------------------------------------------------------------------
# Graph definition
# ---------------------------------------------------------------------------

from troopai.adk.graphs.graph import Graph
from troopai.adk.graphs.interrupt import request_human_input
from troopai.adk.graphs.result import GraphRunStatus
from troopai.adk.llms import LiteLLM
from troopai.adk.orchestration.executable import ExecutableInput
from troopai.adk.run.runner import Runner

_llm = LiteLLM(model="gpt-4o-mini")


# Node A: a plain callable node that summarises the input
def _summarise_node(text: str) -> str:
    """Preprocess the input before human review."""
    return f"[PREPROCESSED] {text}"


# Node B: a HITL node that pauses for human approval
def _review_node(inp: ExecutableInput, ctx: Any) -> str:
    """Request human approval before the workflow continues.

    The two-arg (ExecutableInput, context) signature lets the framework inject
    the human reply into ExecutableInput.metadata["__resume_reply__"] on resume.
    """
    human_answer = request_human_input(
        inp,
        question="Approve the preprocessed text for publishing?",
        kind="tool_approval",
        tool="publish",
    )
    return f"approved:{human_answer}"


_graph = (
    Graph.new("hitl-graph")
    .node("preprocess", _summarise_node)
    .node("review", _review_node)
    .edge("preprocess", "review")
    .entry("preprocess")
    .terminal("review")
    .compile()
)

# ---------------------------------------------------------------------------
# Install TemporalLLM on every agent in the graph (none in this example since
# nodes are callables, but the pattern is identical when agents are used).
# ---------------------------------------------------------------------------

from typing import override

from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer
from troopai.adk.workflows.temporal import (
    HumanReply,
    TroopAITemporalPlugin,
    TroopAIWorkflow,
)
from troopai.adk.workflows.temporal.activity import invoke_model_activity

# ---------------------------------------------------------------------------
# Temporal workflow with HITL signal/resume loop
# ---------------------------------------------------------------------------

TASK_QUEUE = "hitl-graph-queue"


@workflow.defn
class HitlGraphWorkflow(TroopAIWorkflow):
    """Temporal workflow: run a graph that may be interrupted for human input.

    HITL lifecycle:
    1. ``Runner.arun_graph()`` returns ``status=INTERRUPTED``.
    2. The workflow records the pending interrupt and enters ``wait_condition``.
    3. An external actor sends the ``send_human_reply`` signal.
    4. The workflow drains the reply queue and resumes via
       ``Runner.arun_graph_from_checkpoint()``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._thread_id = ""

    @override
    @workflow.run
    async def run(self, prompt: str) -> str:
        """Execute the graph, handling any HITL interrupts durably.

        Args:
            prompt: Initial user input passed to the graph entry node.

        Returns:
            The final output from the graph terminal node.
        """
        self._thread_id = workflow.info().workflow_id
        workflow.logger.info("HitlGraphWorkflow started, thread_id=%r", self._thread_id)

        checkpointer = InMemoryCheckpointer()

        # First run — may return INTERRUPTED
        result = await Runner.arun_graph(
            _graph,
            prompt,
            hooks=[checkpointer],
            thread_id=self._thread_id,
        )
        workflow.logger.info("Initial graph run status: %s", result.status)

        # HITL loop: keep waiting and resuming until the graph completes
        from troopai.adk.graphs.interrupt import GraphResume

        while result.status == GraphRunStatus.INTERRUPTED:
            self.update_state({"status": "waiting_for_human", "thread_id": self._thread_id})

            workflow.logger.info(
                "Graph interrupted — waiting for send_human_reply signal. Pending interrupts: %s",
                [iv.node_id for iv in result.interrupts],
            )

            # Wait until at least one human reply arrives via signal
            await workflow.wait_condition(lambda: len(self._pending_replies) > 0)

            replies_batch = self.consume_replies()
            workflow.logger.info("Received %d human reply/replies", len(replies_batch))

            replies: dict[str, str] = {r.node_id: r.value for r in replies_batch}
            resume = GraphResume(replies=replies)

            result = await Runner.arun_graph_from_checkpoint(
                _graph,
                checkpointer=checkpointer,
                thread_id=self._thread_id,
                resume=resume,
            )
            workflow.logger.info("Resumed graph run status: %s", result.status)

        output: str = result.final_output if isinstance(result.final_output, str) else str(result.final_output)
        workflow.logger.info("HitlGraphWorkflow finished, output=%r", output)
        return output


# ---------------------------------------------------------------------------
# Worker and end-to-end run
# ---------------------------------------------------------------------------


async def _run() -> None:
    """Start the worker and run the HITL graph workflow end-to-end."""
    plugin = TroopAITemporalPlugin()
    # No LLM-based agents in this example — register if your graph uses agents:
    # plugin.register_model(str(_llm), _llm)

    client = await Client.connect("localhost:7233")
    logger.info("Connected to Temporal server")

    async with Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[HitlGraphWorkflow],
        activities=[invoke_model_activity],
        **plugin.build_worker_kwargs(),
    ):
        logger.info("Worker started on task queue %r", TASK_QUEUE)

        # Start workflow — it will pause at the review node
        handle = await client.start_workflow(
            HitlGraphWorkflow.run,
            "AI will soon handle all routine knowledge work.",
            id="hitl-graph-run-001",
            task_queue=TASK_QUEUE,
        )
        logger.info("Workflow started, id=%r", handle.id)

        # Simulate an external human approving via a Temporal signal.
        # In production this signal would come from a webhook or approval UI.
        await asyncio.sleep(2)  # give the workflow time to reach the wait_condition
        await handle.signal(
            HitlGraphWorkflow.send_human_reply,
            HumanReply(node_id="review", value="approved-by-operator"),
        )
        logger.info("Sent HumanReply signal for node_id='review'")

        final = await handle.result()
        logger.info("Workflow completed. Output: %s", final)


if __name__ == "__main__":
    asyncio.run(_run())
