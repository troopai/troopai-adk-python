"""AND-gate flow example — fan-in via ``method_a & method_b``.

Demonstrates how an AND gate built fluently with the ``&`` operator on
:class:`FlowStep` instances fires its listener exactly once after every
required trigger has arrived in the run.

Run: ``python examples/flows/and_join_flow.py``
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from troopai.adk import Flow, Runner, flow_listen, flow_start

logger = logging.getLogger(__name__)


class ReportState(BaseModel):
    """Typed state collected by parallel research branches."""

    numbers_section: str = ""
    """Output of the numerical-research branch."""

    quotes_section: str = ""
    """Output of the quote-research branch."""

    merged_report: str = ""
    """Final synthesis after both branches complete."""


class ReportFlow(Flow[ReportState]):
    """Run two research branches in parallel, then synthesize."""

    @flow_start
    async def gather_numbers(self) -> None:
        """One of two parallel research branches."""
        await asyncio.sleep(0)  # Yield to demonstrate concurrency
        self.state.numbers_section = "42% growth Y/Y"
        logger.info("gather_numbers: done")

    @flow_start
    async def gather_quotes(self) -> None:
        """Second parallel branch — fires alongside :meth:`gather_numbers`."""
        await asyncio.sleep(0)
        self.state.quotes_section = "'The future is decentralized.'"
        logger.info("gather_quotes: done")

    @flow_listen(gather_numbers & gather_quotes)
    async def synthesize(self) -> None:
        """AND-gate listener — fires ONCE after BOTH branches complete.

        Built fluently via the ``&`` operator. The framework tracks
        arrivals by name and fires this method on the arrival that
        completes the gate.
        """
        self.state.merged_report = f"REPORT: {self.state.numbers_section} | {self.state.quotes_section}"
        logger.info("synthesize: %s", self.state.merged_report)


async def main() -> None:
    flow = ReportFlow(ReportState)
    result = await Runner.arun_flow(flow)
    logger.info("status=%s", result.status)
    logger.info("merged_report=%s", result.final_state.merged_report)
    logger.info("completed_steps=%s", result.completed_steps)


if __name__ == "__main__":
    asyncio.run(main())
