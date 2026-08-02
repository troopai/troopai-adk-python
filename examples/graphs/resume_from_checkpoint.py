"""Checkpoint and resume: a linear graph interrupted mid-flight, then continued.

Demonstrates:
- :class:`~troopai.adk.graphs.checkpointers.sqlite.SQLiteCheckpointer` for
  durable persistence across process boundaries.
- :meth:`~troopai.adk.run.runner.Runner.arun_graph` with
  ``GraphConfig(max_supersteps=2)`` — the run halts after nodes ``a`` and
  ``b`` with status ``max_supersteps``.
- A fresh :class:`SQLiteCheckpointer` on the same database file (simulating
  a separate process) loading the persisted state.
- :meth:`~troopai.adk.run.runner.Runner.arun_graph_from_checkpoint` picking
  up where the first run stopped — only node ``c`` fires; ``a`` and ``b``
  are not re-executed (selective re-fire / idempotency).

Topology::

    a ──► b ──► c

No LLM or API key required — all nodes are plain Python callables.

Run::

    python examples/graphs/resume_from_checkpoint.py
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from troopai.adk.graphs import Graph, GraphConfig
from troopai.adk.graphs.checkpointers.sqlite import SQLiteCheckpointer
from troopai.adk.run import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

# Console output comes from the verbose event stream; logger lines
# land in the rotating .log file configured at import time.
_RUN_CONFIG = RunConfig(verbose=VerboseConfig())

# -- Call counters (prove nodes do not double-execute) ----------------------

_call_counts: dict[str, int] = {"a": 0, "b": 0, "c": 0}


# -- Node callables ---------------------------------------------------------


async def node_a(text: str) -> str:
    """Entry node — echoes input with a label."""
    _call_counts["a"] += 1
    logger.info("node_a executing (call #%d)", _call_counts["a"])
    return f"a-done:{text}"


async def node_b(text: str) -> str:
    """Middle node — transforms the output from ``a``."""
    _call_counts["b"] += 1
    logger.info("node_b executing (call #%d)", _call_counts["b"])
    return f"b-done:{text}"


async def node_c(text: str) -> str:
    """Terminal node — produces the final result."""
    _call_counts["c"] += 1
    logger.info("node_c executing (call #%d)", _call_counts["c"])
    return f"c-done:{text}"


# -- Graph builder ----------------------------------------------------------


def build_graph(*, max_supersteps: int | None) -> Graph:
    """Return a compiled ``a → b → c`` graph with the given superstep cap.

    The graph id is always ``"resume-demo"`` so the checkpoint written by
    the capped first run is recognised by the uncapped resumed run.

    Args:
        max_supersteps: Cap to inject via ``GraphConfig``. ``None`` uses
            the framework default (50), which is well above three nodes.
    """
    builder = (
        Graph.new("resume-demo", description="a → b → c with SQLite checkpointing")
        .node("a", node_a)
        .node("b", node_b)
        .node("c", node_c)
        .pipe("a", "b", "c")
        .entry("a")
        .terminal("c")
    )
    if max_supersteps is not None:
        builder = builder.with_config(GraphConfig(max_supersteps=max_supersteps))
    return builder.compile()


# -- First run (deliberately capped) ---------------------------------------


async def first_run(db_path: str, thread_id: str) -> None:
    """Run the graph capped at 2 supersteps so it halts before node ``c``.

    Args:
        db_path: Path to the SQLite database to write checkpoints to.
        thread_id: Logical run key stored in the checkpoint.
    """
    graph = build_graph(max_supersteps=2)
    checkpointer = SQLiteCheckpointer(db_path)
    logger.info("=" * 60)
    logger.info("First run (max_supersteps=2) — expecting halt after a, b")
    logger.info("=" * 60)

    result = await Runner.arun_graph(
        graph,
        "hello",
        hooks=[checkpointer],
        thread_id=thread_id,
        run_config=_RUN_CONFIG,
    )

    completed = sorted(result.node_results.keys())
    logger.info("Status   : %s", result.status.value)
    logger.info("Supersteps: %d", result.total_supersteps)
    logger.info("Completed nodes: %s", completed)

    await checkpointer.close()


# -- Resume run (fresh checkpointer, uncapped graph) -----------------------


async def resume_run(db_path: str, thread_id: str) -> None:
    """Resume from the persisted checkpoint using a fresh checkpointer.

    A new :class:`SQLiteCheckpointer` is constructed on the same database
    file — simulating a separate process that re-opens durable storage.
    Only node ``c`` should fire; ``a`` and ``b`` are not re-executed.

    Args:
        db_path: Path to the SQLite database holding the checkpoint.
        thread_id: The run key to resume.
    """
    graph = build_graph(max_supersteps=None)
    checkpointer = SQLiteCheckpointer(db_path)
    logger.info("=" * 60)
    logger.info("Resuming thread_id=%s with a fresh checkpointer", thread_id)
    logger.info("=" * 60)

    result = await Runner.arun_graph_from_checkpoint(
        graph,
        checkpointer=checkpointer,
        thread_id=thread_id,
        run_config=_RUN_CONFIG,
    )

    logger.info("Status   : %s", result.status.value)
    logger.info("Supersteps: %d", result.total_supersteps)
    logger.info("Final output: %s", result.final_output)

    await checkpointer.close()


# -- Driver -----------------------------------------------------------------


async def main() -> None:
    """Orchestrate the two-phase checkpoint/resume demonstration."""
    thread_id = "resume-demo-001"

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    try:
        await first_run(db_path=db_path, thread_id=thread_id)
        await resume_run(db_path=db_path, thread_id=thread_id)

        logger.info("=" * 60)
        logger.info("Call counts — a:%d  b:%d  c:%d", _call_counts["a"], _call_counts["b"], _call_counts["c"])
        logger.info("Expected    — a:1   b:1   c:1  (no double-execution)")
        logger.info("=" * 60)

        assert _call_counts["a"] == 1, f"node_a fired {_call_counts['a']} times; expected 1"
        assert _call_counts["b"] == 1, f"node_b fired {_call_counts['b']} times; expected 1"
        assert _call_counts["c"] == 1, f"node_c fired {_call_counts['c']} times; expected 1"

        logger.info("Idempotency verified: each node executed exactly once across both runs.")
    finally:
        Path(db_path).unlink(missing_ok=True)
        logger.info("Temporary database removed.")


if __name__ == "__main__":
    asyncio.run(main())
