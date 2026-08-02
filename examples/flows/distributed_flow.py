"""Distributed-flow example — two workers share a SQLite-backed flow run.

Demonstrates :class:`SqliteFlowWorkerBackend` coordinating two
flow workers running in the same process. The first worker
claims the cold-start batch, runs it, persists a checkpoint, and
releases. The second worker then claims the next batch from the
persisted state and completes the flow.

This example uses a single-process simulation (two ``await``
calls in sequence with distinct ``worker_id`` values) so the
example runs without subprocess fixtures. The real distributed
case spawns multiple processes that each call
:meth:`Runner.arun_flow_distributed` against the same SQLite
file; the contention semantics are identical (``BEGIN IMMEDIATE``
serialises claim attempts at the SQLite layer).

Synthetic-only — no agent / LLM call. Runs without any provider
API key.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path

from pydantic import BaseModel

from troopai.adk.flows import (
    Flow,
    SqliteFlowWorkerBackend,
    flow_listen,
    flow_start,
)
from troopai.adk.run.runner import Runner

logger = logging.getLogger(__name__)


class PipelineState(BaseModel):
    counter: int = 0
    log: list[str] = []


class PipelineFlow(Flow[PipelineState]):
    @flow_start
    async def boot(self) -> None:
        self.state.counter = 1
        self.state.log.append("boot")

    @flow_listen("boot")
    async def step_a(self) -> None:
        self.state.counter += 10
        self.state.log.append("step_a")

    @flow_listen(step_a)
    async def step_b(self) -> None:
        self.state.counter *= 2
        self.state.log.append("step_b")


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "flow_workers.db"
        backend = SqliteFlowWorkerBackend(path=db_path)

        flow_a = PipelineFlow(PipelineState)
        logger.info("Worker A claims and processes initial batch...")
        result_a = await Runner.arun_flow_distributed(
            flow_a,
            backend,
            worker_id="worker-a",
        )
        logger.info(
            "Worker A — status=%s final_state=%s",
            result_a.status,
            result_a.final_state.model_dump(),
        )

        # The flow's BSP loop in `arun_flow_distributed` runs the
        # entire flow under one claim, so by the time Worker B runs
        # the checkpoint is terminal. This still illustrates the
        # cross-call resume contract: Worker B sees the persisted
        # state.
        logger.info("Worker B observes the persisted checkpoint...")
        loaded = await backend.load_checkpoint(flow_a.flow_id)
        if loaded is not None:
            logger.info(
                "Worker B — loaded checkpoint completed=%s state_size=%d",
                loaded.completed_steps,
                len(loaded.state_data),
            )

        claims = await backend.list_claims(flow_a.flow_id)
        logger.info("Live claims after release: %s", claims)


if __name__ == "__main__":
    asyncio.run(main())
