"""OR-gate flow example — fires once on first arrival.

Demonstrates the ``method_a | method_b`` OR gate semantics: the gated
listener fires exactly ONCE when either trigger arrives; the gate is
consumed so the other trigger completing later does not re-fire it.

Run: ``python examples/flows/or_join_flow.py``
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from troopai.adk import Flow, Runner, flow_listen, flow_start

logger = logging.getLogger(__name__)


class RacedState(BaseModel):
    """State collected by a race between two backends."""

    primary_done: bool = False
    """Set when the primary backend finishes."""

    fallback_done: bool = False
    """Set when the fallback backend finishes."""

    chosen_response: str = ""
    """First response that arrived — populated exactly once."""

    listener_fire_count: int = 0
    """Counter to verify single-fire semantic."""


class RaceFlow(Flow[RacedState]):
    """Run two backends in parallel; consume whichever responds first."""

    @flow_start
    async def hit_primary(self) -> None:
        """Synthetic primary backend with a small latency."""
        await asyncio.sleep(0.01)
        self.state.primary_done = True
        logger.info("primary done")

    @flow_start
    async def hit_fallback(self) -> None:
        """Synthetic fallback backend with a larger latency."""
        await asyncio.sleep(0.02)
        self.state.fallback_done = True
        logger.info("fallback done")

    @flow_listen(hit_primary | hit_fallback)
    async def consume_first(self) -> None:
        """OR-gate listener — fires ONCE on the first arrival.

        Built fluently via the ``|`` operator. The OR gate is consumed
        on first fire; if the second trigger completes later, this
        listener does NOT fire a second time.
        """
        self.state.listener_fire_count += 1
        if self.state.primary_done:
            self.state.chosen_response = "primary"
        else:
            self.state.chosen_response = "fallback"
        logger.info("consume_first: chose %s", self.state.chosen_response)


async def main() -> None:
    flow = RaceFlow(RacedState)
    result = await Runner.arun_flow(flow)
    logger.info("status=%s", result.status)
    logger.info("chosen_response=%s", result.final_state.chosen_response)
    logger.info("listener_fire_count=%d (expect 1)", result.final_state.listener_fire_count)
    assert result.final_state.listener_fire_count == 1


if __name__ == "__main__":
    asyncio.run(main())
