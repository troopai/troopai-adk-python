"""Streamed flow example — ``Runner.arun_flow_streamed(flow)``.

Demonstrates event streaming as a flow runs: ``flow.start``,
``flow.step_start`` / ``flow.step_end`` per step, ``flow.route_evaluated``
when a ``@flow_router`` returns a label, and ``flow.end`` once the run
completes. Use for live dashboards, progress UIs, or long-running
flows where you want events emitted as they happen.

Pattern: use ``isinstance`` to narrow the discriminated event union —
pyright / mypy validate field access per-arm.

For the async one-shot variant, see ``examples/flows/async_flow.py``.
For the sync wrapper, see ``examples/flows/sync_flow.py``.

Run::

    python examples/flows/streamed_flow.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from pydantic import BaseModel

from troopai.adk import Flow, FlowConfig, Runner, flow_listen, flow_router, flow_start
from troopai.adk.flows.events import (
    FlowEndEvent,
    FlowRouteEvaluatedEvent,
    FlowStepEndEvent,
    FlowStepStartEvent,
)

logger = logging.getLogger(__name__)


class CounterState(BaseModel):
    """Trivial typed state for the example."""

    visits: list[str] = []
    """Records the name of every step that ran, in completion order."""

    final_label: str = ""
    """Set by the chosen branch after the router fires."""


class CounterFlow(Flow[CounterState]):
    """Tiny synthetic flow: ``intake → router → branch_alpha OR branch_beta``."""

    @flow_start
    async def intake(self) -> None:
        self.state.visits.append("intake")
        logger.info("intake")

    @flow_router(intake)
    async def decide(self) -> str:
        self.state.visits.append("decide")
        return "alpha" if len(self.state.visits) % 2 == 1 else "beta"

    @flow_listen("alpha")
    async def branch_alpha(self) -> None:
        self.state.visits.append("alpha")
        self.state.final_label = "ALPHA"
        logger.info("alpha")

    @flow_listen("beta")
    async def branch_beta(self) -> None:
        self.state.visits.append("beta")
        self.state.final_label = "BETA"
        logger.info("beta")


async def main() -> None:
    """Drive the flow with ``Runner.arun_flow_streamed(...)`` and observe events."""
    flow = CounterFlow(CounterState)
    streaming = Runner.arun_flow_streamed(flow, config=FlowConfig(max_steps=20))
    async for event in streaming.stream_events():
        if isinstance(event, FlowStepStartEvent):
            logger.info("→ start: %s (count=%d)", event.step_name, event.step_count)
        elif isinstance(event, FlowStepEndEvent):
            logger.info("← end:   %s", event.step_name)
        elif isinstance(event, FlowRouteEvaluatedEvent):
            logger.info("↳ route: %s → %r → %s", event.router_step, event.route_label, event.triggered_steps)
        elif isinstance(event, FlowEndEvent):
            logger.info("▣ run:   status=%s completed=%s", event.status, event.completed_steps)
    logger.info("FINAL status=%s visits=%s final_label=%s", streaming.status, flow.state.visits, flow.state.final_label)


if __name__ == "__main__":
    asyncio.run(main())
