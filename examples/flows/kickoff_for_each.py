"""Batch-kickoff example — fan out a Flow over a list of inputs.

Demonstrates :meth:`Runner.arun_flow_for_each` driving the same
``SentimentFlow`` over a batch of headlines.

Three call shapes are exercised:

1. Sequential (``concurrency=1``, the default) — strict ordering, no
   parallelism overhead. The cost-conservative path.
2. Bounded-parallel (``concurrency=3``) — capped fan-out via
   ``asyncio.Semaphore``.
3. Error isolation — one bad input produces ``status="failed"`` for
   that index only; the rest of the batch completes.

Synthetic only (no agent / LLM call) so the example runs without any
provider API key — its purpose is to exercise the batch helper's
shape, not to demonstrate a sentiment classifier.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from troopai.adk.flows import Flow, flow_start
from troopai.adk.run.runner import Runner

logger = logging.getLogger(__name__)


class HeadlineState(BaseModel):
    headline: str = ""
    label: str = "unknown"


class SentimentFlow(Flow[HeadlineState]):
    @flow_start
    async def classify(self) -> None:
        text = self.state.headline.lower()
        if "boom" in text:
            raise RuntimeError("explicit fault for error-isolation demo")
        if "down" in text or "loss" in text:
            self.state.label = "negative"
        elif "up" in text or "win" in text or "growth" in text:
            self.state.label = "positive"
        else:
            self.state.label = "neutral"


def _factory(state: HeadlineState) -> SentimentFlow:
    return SentimentFlow(state)


async def _sequential() -> None:
    headlines = [
        "Markets up across the board",
        "Earnings down sharply",
        "Steady growth quarter",
        "No significant change in policy",
    ]
    results = await Runner.arun_flow_for_each(
        _factory,
        [HeadlineState(headline=h) for h in headlines],
    )
    logger.info("SEQUENTIAL — %d results", len(results))
    for r in results:
        logger.info("  %r → %s", r.final_state.headline, r.final_state.label)


async def _bounded_parallel() -> None:
    headlines = [f"Item {i}: market up" for i in range(8)]
    results = await Runner.arun_flow_for_each(
        _factory,
        [HeadlineState(headline=h) for h in headlines],
        concurrency=3,
    )
    logger.info("PARALLEL (concurrency=3) — %d results", len(results))
    for r in results:
        logger.info("  %r → %s", r.final_state.headline, r.final_state.label)


async def _error_isolation() -> None:
    # Use a step body exception (routed through ``error_policy`` by the
    # executor, captured into ``status="failed"``). Programmer errors
    # raised by the factory would NOT be captured — they propagate to
    # surface real bugs.
    headlines = [
        "Markets up",
        "BOOM crash",  # step body raises here
        "Earnings down",
    ]
    results = await Runner.arun_flow_for_each(
        _factory,
        [HeadlineState(headline=h) for h in headlines],
    )
    logger.info("ERROR ISOLATION — per-item outcomes:")
    for i, r in enumerate(results):
        if r.status == "completed":
            logger.info("  [%d] %s → %s", i, r.final_state.headline, r.final_state.label)
        else:
            logger.info("  [%d] %s → status=%s, error=%s", i, r.final_state.headline, r.status, r.error)


async def main() -> None:
    await _sequential()
    await _bounded_parallel()
    await _error_isolation()


if __name__ == "__main__":
    asyncio.run(main())
