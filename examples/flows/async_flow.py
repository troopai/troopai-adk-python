"""Async flow example — ``await Runner.arun_flow(flow)``.

Demonstrates non-blocking flow execution from inside an ``async def``
context. Use this mode from web handlers, async tasks, or
pytest-asyncio tests. Returns a :class:`FlowRunResult` once the flow
completes.

For the blocking wrapper, see ``examples/flows/sync_flow.py``.
For event streaming, see ``examples/flows/streamed_flow.py``.

Run::

    python examples/flows/async_flow.py
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
    """Drive the flow with ``await Runner.arun_flow(...)``."""
    flow = CounterFlow(CounterState)
    result = await Runner.arun_flow(flow, config=FlowConfig(max_steps=20))
    logger.info("status=%s", result.status)
    logger.info("visits=%s", flow.state.visits)
    logger.info("final_label=%s", flow.state.final_label)


if __name__ == "__main__":
    asyncio.run(main())
