"""Graph HITL interrupts: suspend a run for human input, then resume.

Demonstrates:
- ``request_human_input(inp, question, *, kind, **metadata)`` inside a callable
  node — raises ``InterruptException`` on first invocation, returns the reply on
  resume.
- Non-streaming suspend: ``Runner.arun_graph(g, prompt, hooks=[cp], thread_id=t)``
  returns ``status=INTERRUPTED`` with no exception raised; ``result.interrupts``
  carries the pending ``Interrupt`` objects.
- Functional resume: ``Runner.arun_graph_from_checkpoint(g, checkpointer=cp,
  thread_id=t, resume=GraphResume(replies={...}))``
- Streaming suspend: ``Runner.arun_graph_streamed(g, prompt)`` — consumer drains
  via ``async for ev in result.stream_events()`` and receives a
  ``"graph.node_interrupt"`` event before ``"graph.end(status=interrupted)"``.
  No exception raised by the driver.

No LLM or API key required — all nodes are plain Python callables.

Topology::

    ask  ──►  after
    (interrupts on first call; receives human reply on resume)

Run::

    python examples/graphs/hitl.py
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from troopai.adk.graphs import Graph
from troopai.adk.graphs.checkpointers.in_memory import InMemoryCheckpointer
from troopai.adk.graphs.events import GRAPH_END, NODE_INTERRUPT
from troopai.adk.graphs.interrupt import GraphResume, Interrupt, request_human_input
from troopai.adk.graphs.result import GraphRunResultStreaming, GraphRunStatus
from troopai.adk.orchestration.executable import ExecutableInput
from troopai.adk.run import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# -- Node callables -----------------------------------------------------------


def ask_node(inp: ExecutableInput, ctx: Any) -> str:
    """Node that requests human approval before proceeding.

    The two-arg ``(ExecutableInput, context)`` signature tells
    ``CallableExecutable`` to pass the full envelope, giving access to
    ``inp.metadata`` where the loop injects the human reply on resume.

    Args:
        inp: The ``ExecutableInput`` for this invocation.
        ctx: Execution context (unused but required by the 2-arg dispatch).

    Returns:
        A confirmation string incorporating the human's reply.
    """
    reply = request_human_input(inp, "Approve the action?", kind="tool_approval", tool="deploy")
    return f"approved:{reply}"


async def after_node(text: str) -> str:
    """Terminal node — receives the output of ``ask_node`` after resume.

    Args:
        text: Merged output from the upstream node.

    Returns:
        A finalised string.
    """
    return f"done:{text}"


# -- Graph builders -----------------------------------------------------------


def build_checkpoint_graph() -> Graph:
    """Build and compile the two-node ask → after graph.

    Returns:
        A compiled ``ask → after`` graph.
    """
    return (
        Graph.new("hitl-demo", description="ask → after (HITL checkpoint round-trip)")
        .node("ask", ask_node)
        .node("after", after_node)
        .edge("ask", "after")
        .entry("ask")
        .terminal("after")
        .compile()
    )


def build_streaming_graph() -> Graph:
    """Build a single-node graph for the streaming interrupt demonstration.

    Returns:
        A compiled single-node graph that interrupts and can be resumed.
    """
    return (
        Graph.new("hitl-stream-demo", description="ask (streaming HITL demo)")
        .node("ask", ask_node)
        .entry("ask")
        .terminal("ask")
        .compile()
    )


# -- Shared run config --------------------------------------------------------

# Console output comes from the verbose event stream; logger lines
# land in the rotating .log file configured at import time.
_RUN_CONFIG = RunConfig(verbose=VerboseConfig())


# -- Checkpoint round-trip ----------------------------------------------------


async def run_checkpoint_demo(graph: Graph, cp: InMemoryCheckpointer) -> None:
    """Demonstrate non-streaming suspend + functional resume.

    Args:
        graph: Compiled graph with ask → after topology.
        cp: In-memory checkpointer that will capture the interrupted state.
    """
    logger.info("=" * 64)
    logger.info("Non-streaming HITL: suspend then resume via checkpoint")
    logger.info("=" * 64)

    first = await Runner.arun_graph(graph, "go", hooks=[cp], thread_id="hitl-demo", run_config=_RUN_CONFIG)

    logger.info("status after first run: %s", first.status.value)
    assert first.status == GraphRunStatus.INTERRUPTED
    assert first.error is None

    for iv in first.interrupts:
        logger.info(
            "interrupt  node_id=%s  question=%r  kind=%s",
            iv.node_id,
            iv.question,
            iv.kind,
        )

    assert first.state is not None
    assert "ask" in first.state.pending_interrupts

    second = await Runner.arun_graph_from_checkpoint(
        graph,
        checkpointer=cp,
        thread_id="hitl-demo",
        resume=GraphResume(replies={"ask": "the-answer"}),
        run_config=_RUN_CONFIG,
    )

    logger.info("status after resume   : %s", second.status.value)
    logger.info("final_output          : %s", second.final_output)
    assert second.status == GraphRunStatus.COMPLETED
    assert second.state is not None
    assert "ask" not in second.state.pending_interrupts

    logger.info("-" * 64)


# -- Streaming HITL -----------------------------------------------------------


async def run_streaming_demo(graph: Graph) -> None:
    """Demonstrate streamed suspend: drain events without raising.

    Args:
        graph: Compiled graph whose single node will interrupt.
    """
    logger.info("=" * 64)
    logger.info("Streaming HITL: drain events; node_interrupt arrives before graph.end")
    logger.info("=" * 64)

    result: GraphRunResultStreaming = await Runner.arun_graph_streamed(graph, "go", run_config=_RUN_CONFIG)

    interrupt_events: list[Interrupt] = []

    async for ev in result.stream_events():
        ev_type: str = ev["type"]
        if ev_type == NODE_INTERRUPT:
            iv: Interrupt = ev["interrupt"]
            interrupt_events.append(iv)
            logger.info(
                "[graph.node_interrupt]  node_id=%s  graph_path=%s  question=%r",
                ev["node_id"],
                ev["graph_path"],
                iv.question,
            )
        elif ev_type == GRAPH_END:
            logger.info(
                "[graph.end]             status=%s  supersteps=%d",
                ev["status"].value,
                ev["total_supersteps"],
            )
        else:
            logger.info("[%s]", ev_type)

    status_val = result.status.value if result.status is not None else "none"
    logger.info("result.status     : %s", status_val)
    logger.info("result.interrupts : %d pending", len(result.interrupts))

    assert result.status == GraphRunStatus.INTERRUPTED
    assert len(interrupt_events) == 1
    assert len(result.interrupts) == 1

    logger.info("-" * 64)


# -- Driver -------------------------------------------------------------------


async def main() -> None:
    """Build the graphs and run both demonstrations."""
    cp = InMemoryCheckpointer()
    checkpoint_graph = build_checkpoint_graph()
    await run_checkpoint_demo(checkpoint_graph, cp)

    streaming_graph = build_streaming_graph()
    await run_streaming_demo(streaming_graph)


if __name__ == "__main__":
    asyncio.run(main())
