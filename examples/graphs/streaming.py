"""Graph streaming: consume real-time structural events from a graph run.

Demonstrates:
- ``Runner.arun_graph_streamed(graph, prompt)`` → ``GraphRunResultStreaming``
- ``async for ev in result.stream_events()`` — iterate events in real time
- Event ``["type"]`` discriminators: ``"graph.start"``, ``"graph.superstep_start"``,
  ``"graph.node_start"``, ``"graph.node_end"``, ``"graph.superstep_end"``,
  ``"graph.end"``; and the interior envelope ``"graph.node_stream"`` (carries
  ``["graph_path"]``, ``["node_id"]``, ``["inner"]``)
- ``result.final_output`` and ``result.status.value`` (a ``GraphRunStatus`` string,
  e.g. ``"completed"``) after the stream drains
- ``result.cancel("after_superstep")`` to show cooperative cancellation

No LLM or API key required — all nodes are plain Python callables.

Topology::

    prepare
    ├── enrich_a  ─┐
    └── enrich_b  ─┤─→ summarise
                   ┘

``prepare`` runs in superstep 1.  ``enrich_a`` and ``enrich_b`` run in
parallel in superstep 2.  ``summarise`` joins them in superstep 3.

Run::

    python examples/graphs/streaming.py
"""

from __future__ import annotations

import asyncio
import logging

from troopai.adk.graphs import Graph, Merge
from troopai.adk.graphs.events import (
    GRAPH_END,
    GRAPH_START,
    NODE_END,
    NODE_ERROR,
    NODE_START,
    NODE_STREAM,
    SUPERSTEP_END,
    SUPERSTEP_START,
)
from troopai.adk.graphs.result import GraphRunResultStreaming
from troopai.adk.run import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

# Console output comes from the verbose event stream; logger lines
# land in the rotating .log file configured at import time.
_RUN_CONFIG = RunConfig(verbose=VerboseConfig())


# -- Node callables -----------------------------------------------------------


async def prepare(text: str) -> str:
    """Entry node — normalises the prompt text."""
    logger.debug("prepare: got %r", text)
    return f"prepared:{text}"


async def enrich_a(text: str) -> str:
    """Fan-out branch A — adds a label."""
    return f"A({text})"


async def enrich_b(text: str) -> str:
    """Fan-out branch B — adds a label."""
    return f"B({text})"


async def summarise(text: str) -> str:
    """AND-join terminal — receives both branches merged."""
    return f"summary:[{text}]"


# -- Graph --------------------------------------------------------------------


def build_graph() -> Graph:
    """Build and compile the fan-out/join graph.

    Returns:
        A compiled ``prepare → (enrich_a ∥ enrich_b) → summarise`` graph.
    """
    return (
        Graph.new("streaming-demo", description="prepare → (enrich_a ∥ enrich_b) → summarise")
        .node("prepare", prepare)
        .node("enrich_a", enrich_a)
        .node("enrich_b", enrich_b)
        .node("summarise", summarise, merge=Merge.concat_text)
        .edge("prepare", "enrich_a")
        .edge("prepare", "enrich_b")
        .edge("enrich_a", "summarise")
        .edge("enrich_b", "summarise")
        .entry("prepare")
        .terminal("summarise")
        .compile()
    )


# -- Event logging ------------------------------------------------------------


def log_structural_event(ev: dict) -> None:  # type: ignore[type-arg]
    """Log a structural graph event.

    Args:
        ev: A ``GraphStreamEvent`` dict (structural type).
    """
    ev_type: str = ev["type"]
    if ev_type == GRAPH_START:
        logger.info(
            "[graph.start]       graph_id=%s  entry=%s  terminals=%s",
            ev["graph_id"],
            ev["entry_node"],
            ev["terminal_nodes"],
        )
    elif ev_type == SUPERSTEP_START:
        logger.info(
            "[graph.superstep_start] superstep=%d  ready=%s",
            ev["superstep"],
            ev["ready_nodes"],
        )
    elif ev_type == NODE_START:
        logger.info(
            "[graph.node_start]  node=%s  superstep=%d",
            ev["node_id"],
            ev["superstep"],
        )
    elif ev_type == NODE_END:
        logger.info(
            "[graph.node_end]    node=%s  superstep=%d",
            ev["node_id"],
            ev["superstep"],
        )
    elif ev_type == NODE_ERROR:
        logger.warning(
            "[graph.node_error]  node=%s  error_type=%s  message=%s",
            ev["node_id"],
            ev["error_type"],
            ev["error_message"],
        )
    elif ev_type == SUPERSTEP_END:
        logger.info(
            "[graph.superstep_end]   superstep=%d  fired=%s  errored=%s",
            ev["superstep"],
            ev["fired_nodes"],
            ev["errored_nodes"],
        )
    elif ev_type == GRAPH_END:
        logger.info(
            "[graph.end]         graph_id=%s  status=%s  supersteps=%d",
            ev["graph_id"],
            ev["status"].value,
            ev["total_supersteps"],
        )


def log_node_stream_event(ev: dict) -> None:  # type: ignore[type-arg]
    """Log an interior ``graph.node_stream`` envelope event.

    The ``graph_path`` is always a single-element tuple for the emitting
    graph — nested graphs run non-streaming and do not contribute combined
    paths here.

    Args:
        ev: A ``NodeStreamEvent`` dict.
    """
    logger.debug(
        "[graph.node_stream] node=%s  graph_path=%s  inner_type=%s",
        ev["node_id"],
        ev["graph_path"],
        ev["inner"].get("type") if isinstance(ev["inner"], dict) else type(ev["inner"]).__name__,
    )


# -- Streaming run ------------------------------------------------------------


async def run_full(graph: Graph) -> None:
    """Drive a complete streamed run and log every event.

    Args:
        graph: The compiled graph to execute.
    """
    logger.info("=" * 64)
    logger.info("Full streaming run: prepare → (enrich_a ∥ enrich_b) → summarise")
    logger.info("=" * 64)

    result: GraphRunResultStreaming = await Runner.arun_graph_streamed(graph, "hello-stream", run_config=_RUN_CONFIG)

    async for ev in result.stream_events():
        if ev["type"] == NODE_STREAM:
            log_node_stream_event(ev)
        else:
            log_structural_event(ev)

    logger.info("-" * 64)
    logger.info("final_output : %s", result.final_output)
    logger.info("status       : %s", result.status.value if result.status is not None else "none")
    logger.info("-" * 64)


# -- Cancellation demo --------------------------------------------------------


async def run_cancel_demo(graph: Graph) -> None:
    """Drive a streamed run cancelled at the next superstep boundary.

    ``result.cancel("after_superstep")`` sets a cooperative flag; the
    driver stops scheduling new supersteps after the current one drains.

    Args:
        graph: The compiled graph to execute.
    """
    logger.info("=" * 64)
    logger.info("Cancellation demo (after_superstep)")
    logger.info("=" * 64)

    result: GraphRunResultStreaming = await Runner.arun_graph_streamed(graph, "cancel-me", run_config=_RUN_CONFIG)
    supersteps_seen: list[int] = []

    async for ev in result.stream_events():
        if ev["type"] == SUPERSTEP_START:
            superstep: int = ev["superstep"]
            supersteps_seen.append(superstep)
            logger.info("[graph.superstep_start] superstep=%d  ready=%s", superstep, ev["ready_nodes"])
            # Cancel after the first superstep completes.
            if superstep == 1:
                result.cancel("after_superstep")
                logger.info("cancel(after_superstep) requested after superstep 1")
        elif ev["type"] == NODE_STREAM:
            log_node_stream_event(ev)
        else:
            log_structural_event(ev)

    logger.info("-" * 64)
    logger.info("supersteps observed : %s", supersteps_seen)
    logger.info("final_output        : %s", result.final_output)
    logger.info("status              : %s", result.status.value if result.status is not None else "none")
    logger.info("-" * 64)


# -- Driver -------------------------------------------------------------------


async def main() -> None:
    """Build the graph and run both demonstrations."""
    graph: Graph = build_graph()
    await run_full(graph)
    await run_cancel_demo(graph)


if __name__ == "__main__":
    asyncio.run(main())
