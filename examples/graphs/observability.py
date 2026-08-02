"""Graph observability: GraphHooks callbacks + OTel spans, side-by-side.

Demonstrates the two parallel observability surfaces a graph run exposes:

- ``GraphHooks`` — async callbacks fired at lifecycle boundaries. Attach
  any number of ``GraphHooks`` instances via ``Runner.arun_graph(...,
  hooks=[...])`` to react in-process to graph / superstep / node events.
- **OpenTelemetry spans** — when an ``OTelTracer`` is installed, the BSP
  loop opens a span tree shaped ``graph.<id>`` → ``graph.superstep.<n>``
  → ``graph.node.<name>``. Attributes live under the ``troopai.graph.*``
  namespace (id, entry, status, supersteps_total, node.name, node.status,
  node.attempts, ...).

Both surfaces fire at the same lifecycle points and are decoupled — use
one, the other, or both. Default cost is zero when neither is opted in
(``NoOpTracer`` returns ``NoOpSpan`` for every factory call; no hooks
attached means no callbacks fired).

Topology::

    root ──► fetch_a ──┐
       └──► fetch_b ──┴──► join

The two fetches run in parallel inside one BSP superstep, so the OTel
trace shows them as siblings under the same ``graph.superstep.1`` span.

Run::

    python examples/graphs/observability.py
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, override

from troopai.adk.graphs import Graph
from troopai.adk.graphs.hooks import GraphHooks
from troopai.adk.graphs.interrupt import Interrupt
from troopai.adk.graphs.result import GraphRunStatus
from troopai.adk.graphs.state import GraphState
from troopai.adk.orchestration.executable import ExecutableInput, NodeResult
from troopai.adk.run import RunConfig
from troopai.adk.run.context import RunContext
from troopai.adk.run.runner import Runner
from troopai.adk.tracing import set_tracer
from troopai.adk.types.items.items import RunItem
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# OTel is an optional dependency; the example degrades gracefully without it.
try:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

    from troopai.adk.tracing.otel import OTelTracer

    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False


class LoggingHooks(GraphHooks[Any]):
    """Log every GraphHooks lifecycle callback at INFO level."""

    @override
    async def on_graph_start(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
    ) -> None:
        del context
        logger.info("[hook] graph_start  thread_id=%s", state.thread_id)

    @override
    async def on_superstep_start(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        ready_nodes: tuple[str, ...],
    ) -> None:
        del context
        logger.info("[hook] superstep_start  #%d  ready=%s", state.superstep, list(ready_nodes))

    @override
    async def on_node_start(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        node_id: str,
        input: ExecutableInput,
    ) -> None:
        del context, state, input
        logger.info("[hook] node_start  %s", node_id)

    @override
    async def on_node_end(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        node_id: str,
        result: NodeResult,
    ) -> None:
        del context, state
        logger.info("[hook] node_end    %s  output=%r", node_id, result.output)

    @override
    async def on_node_interrupt(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        node_id: str,
        interrupt: Interrupt,
    ) -> None:
        del context, state
        logger.info(
            "[hook] node_interrupt  %s  kind=%s  question=%r",
            node_id,
            interrupt.kind,
            interrupt.question,
        )

    @override
    async def on_node_error(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        node_id: str,
        error: BaseException,
    ) -> None:
        del context, state
        logger.info("[hook] node_error  %s  %s: %s", node_id, type(error).__name__, error)

    @override
    async def on_superstep_end(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        fired_nodes: tuple[str, ...],
        items: list[RunItem],
    ) -> None:
        del context, state, items
        logger.info("[hook] superstep_end  fired=%s", list(fired_nodes))

    @override
    async def on_graph_end(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        status: GraphRunStatus,
        final_output: Any,
    ) -> None:
        del context, state
        logger.info("[hook] graph_end    status=%s  final=%r", status.value, final_output)


def _install_otel_console_exporter() -> None:
    """Install an OTel tracer that prints each finished span to stderr."""
    if not _OTEL_AVAILABLE:
        logger.info("(opentelemetry-sdk not installed; skipping OTel span output)")
        return
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    set_tracer(OTelTracer(provider=provider, service_name="observability-example"))
    logger.info("OTel tracer installed (spans → stderr)")


def _build_graph() -> Graph:
    """root fans out to two parallel fetches; both feed into a join terminal."""

    def _root() -> str:
        return "go"

    def _fetch_a(text: str) -> str:
        return f"a-{text}"

    def _fetch_b(text: str) -> str:
        return f"b-{text}"

    def _join(text: str) -> str:
        return f"joined({text})"

    return (
        Graph.new("observability-demo")
        .node("root", _root)
        .node("fetch_a", _fetch_a)
        .node("fetch_b", _fetch_b)
        .node("join", _join)
        .edge("root", "fetch_a")
        .edge("root", "fetch_b")
        .edge("fetch_a", "join")
        .edge("fetch_b", "join")
        .entry("root")
        .terminal("join")
        .compile()
    )


async def _run() -> None:
    _install_otel_console_exporter()
    graph = _build_graph()
    logger.info("=" * 64)
    logger.info("Running graph %s with LoggingHooks attached", graph.id)
    logger.info("=" * 64)
    result = await Runner.arun_graph(graph, "go", hooks=[LoggingHooks()], run_config=RunConfig(verbose=VerboseConfig()))
    logger.info("=" * 64)
    logger.info("final_output: %r", result.final_output)
    logger.info("total_supersteps: %d", result.total_supersteps)
    set_tracer(None)


if __name__ == "__main__":
    asyncio.run(_run())
