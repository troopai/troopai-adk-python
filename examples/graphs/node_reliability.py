"""Per-node timeout and retry: three self-contained scenarios.

Demonstrates:
- **Retry-to-success**: a flaky node that raises twice then succeeds under
  ``GraphConfig(default_retry=NodeRetryPolicy(max_attempts=3,
  initial_backoff=0.01))``.  The graph completes; attempt count and final
  status are logged.
- **Timeout → GraphNodeTimeoutError**: a node that sleeps beyond
  ``GraphConfig(per_node_timeout=...)``.  The run ends with
  ``status == "failed"`` and ``result.error`` names ``GraphNodeTimeoutError``.
- **Per-node override beats graph default**: the graph default allows only 1
  attempt; one node carries a per-node ``retry=NodeRetryPolicy(max_attempts=4,
  initial_backoff=0.01)`` attached via ``dataclasses.replace`` (both ``Graph``
  and ``GraphNode`` are frozen dataclasses).  That node succeeds on its third
  call despite the stricter graph default.

No LLM or API key required — all nodes are plain Python callables.

Topology::

    Scenario 1  flaky-node ──► sink
    Scenario 2  slow-node
    Scenario 3  stubborn-node ──► sink

Run::

    python examples/graphs/node_reliability.py
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging

from troopai.adk.graphs import Graph
from troopai.adk.graphs.config import GraphConfig, NodeRetryPolicy
from troopai.adk.run import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

# Console output comes from the verbose event stream; logger lines
# land in the rotating .log file configured at import time.
_RUN_CONFIG = RunConfig(verbose=VerboseConfig())


# ---------------------------------------------------------------------------
# Scenario 1 — Retry-to-success
# ---------------------------------------------------------------------------


async def demo_retry_to_success() -> None:
    """Graph-level retry: flaky node succeeds after two transient failures."""
    logger.info("=" * 60)
    logger.info("Scenario 1 — Retry-to-success")
    logger.info("=" * 60)

    attempts = [0]

    def _flaky_node(text: str) -> str:
        """Raises twice, succeeds on the third call."""
        attempts[0] += 1
        logger.info("[S1] flaky-node attempt %d", attempts[0])
        if attempts[0] < 3:
            raise ValueError(f"transient error on attempt {attempts[0]}")
        return f"recovered:{text}"

    graph = (
        Graph.new("retry-success", description="flaky-node retries to success")
        .node("flaky-node", _flaky_node)
        .node("sink", lambda t: f"sink:{t}")
        .edge("flaky-node", "sink")
        .entry("flaky-node")
        .terminal("sink")
        .with_config(GraphConfig(default_retry=NodeRetryPolicy(max_attempts=3, initial_backoff=0.01)))
        .compile()
    )

    result = await Runner.arun_graph(graph, "hello", run_config=_RUN_CONFIG)

    logger.info("[S1] total attempts on flaky-node: %d", attempts[0])
    logger.info("[S1] status: %s", result.status.value)
    logger.info("[S1] final_output: %s", result.final_output)
    assert result.status.value == "completed", f"expected completed, got {result.status}"
    assert attempts[0] == 3, f"expected 3 attempts, got {attempts[0]}"
    logger.info("[S1] PASS — graph completed after %d attempts", attempts[0])


# ---------------------------------------------------------------------------
# Scenario 2 — Timeout → GraphNodeTimeoutError
# ---------------------------------------------------------------------------


async def _slow_node(text: str) -> str:
    """Sleeps for 10 s — far beyond the configured per-node timeout."""
    await asyncio.sleep(10.0)
    return "never"


async def demo_timeout_error() -> None:
    """A node that exceeds its per-attempt timeout surfaces GraphNodeTimeoutError."""
    logger.info("=" * 60)
    logger.info("Scenario 2 — Timeout → GraphNodeTimeoutError")
    logger.info("=" * 60)

    graph = (
        Graph.new("timeout-demo", description="slow-node exceeds timeout")
        .node("slow-node", _slow_node)
        .entry("slow-node")
        .terminal("slow-node")
        .with_config(GraphConfig(per_node_timeout=0.05))
        .compile()
    )

    result = await Runner.arun_graph(graph, "go", run_config=_RUN_CONFIG)

    logger.info("[S2] status: %s", result.status.value)
    logger.info("[S2] error field: %s", result.error)
    assert result.status.value == "failed", f"expected failed, got {result.status}"
    error_text = result.error or ""
    assert "GraphNodeTimeoutError" in error_text, f"expected GraphNodeTimeoutError in error, got: {error_text!r}"
    logger.info("[S2] PASS — GraphNodeTimeoutError surfaced as expected")


# ---------------------------------------------------------------------------
# Scenario 3 — Per-node override beats graph default
# ---------------------------------------------------------------------------


async def demo_per_node_override() -> None:
    """Per-node retry=4 overrides graph default=1; node succeeds on attempt 3."""
    logger.info("=" * 60)
    logger.info("Scenario 3 — Per-node override beats graph default")
    logger.info("=" * 60)

    attempts = [0]

    def _stubborn_node(text: str) -> str:
        """Needs 3 calls to succeed.  Graph default allows only 1 attempt."""
        attempts[0] += 1
        logger.info("[S3] stubborn-node attempt %d", attempts[0])
        if attempts[0] < 3:
            raise ValueError(f"not yet on attempt {attempts[0]}")
        return f"stubborn-ok:{text}"

    # Graph default: 1 attempt only — would fail without a per-node override.
    g = (
        Graph.new("per-node-override", description="stubborn-node with override")
        .node("stubborn-node", _stubborn_node)
        .node("sink", lambda t: f"sink:{t}")
        .edge("stubborn-node", "sink")
        .entry("stubborn-node")
        .terminal("sink")
        .with_config(GraphConfig(default_retry=NodeRetryPolicy(max_attempts=1)))
        .compile()
    )

    # Both Graph and GraphNode are frozen dataclasses.  Attach the per-node
    # policy via dataclasses.replace — the only correct mutation path.
    node = g.get_node("stubborn-node")
    node_override = dataclasses.replace(
        node,
        retry=NodeRetryPolicy(max_attempts=4, initial_backoff=0.01),
    )
    g = dataclasses.replace(
        g,
        nodes=tuple(node_override if n.id == "stubborn-node" else n for n in g.nodes),
    )

    result = await Runner.arun_graph(g, "push", run_config=_RUN_CONFIG)

    logger.info("[S3] total attempts on stubborn-node: %d", attempts[0])
    logger.info("[S3] status: %s", result.status.value)
    logger.info("[S3] final_output: %s", result.final_output)
    assert result.status.value == "completed", f"expected completed, got {result.status}"
    assert attempts[0] == 3, f"expected 3 attempts, got {attempts[0]}"
    logger.info("[S3] PASS — per-node override allowed %d attempts", attempts[0])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    await demo_retry_to_success()
    await demo_timeout_error()
    await demo_per_node_override()
    logger.info("=" * 60)
    logger.info("All three reliability scenarios passed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
