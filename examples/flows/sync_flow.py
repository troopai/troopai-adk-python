"""Sync flow example — ``Runner.run_flow(flow)``.

Demonstrates the blocking wrapper for flow execution. Same event-loop
strategy as :meth:`Runner.run` / :meth:`Runner.run_task_pipeline`:
when invoked inside a running loop, the call offloads to a worker
thread so the parent loop is not blocked. When invoked from a plain
script (no loop), :func:`asyncio.run` drives the coroutine.

Use from synchronous entry points (scripts, REPLs, framework hooks
that don't expose async surfaces).

For the async variant, see ``examples/flows/async_flow.py``.
For event streaming, see ``examples/flows/streamed_flow.py``.

Run::

    python examples/flows/sync_flow.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

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


def main() -> None:
    """Drive the flow with ``Runner.run_flow(...)`` — the sync wrapper."""
    flow = CounterFlow(CounterState)
    result = Runner.run_flow(flow, config=FlowConfig(max_steps=20))
    logger.info("status=%s", result.status)
    logger.info("visits=%s", flow.state.visits)
    logger.info("final_label=%s", flow.state.final_label)


if __name__ == "__main__":
    main()
