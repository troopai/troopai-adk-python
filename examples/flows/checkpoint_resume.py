"""FlowCheckpoint persistence + resume example.

Demonstrates the explicit checkpoint workflow:

1. Run a flow partially, capture state at a step boundary.
2. Serialize :class:`FlowCheckpoint` to JSON (portable across processes).
3. Reload the checkpoint, rehydrate the state, and resume execution.

Persistence is **explicit** — there is no ``@persist`` decorator. The
developer drives capture and resume themselves.

Run: ``python examples/flows/checkpoint_resume.py``
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from troopai.adk import Flow, FlowCheckpoint, Runner, flow_listen, flow_start

logger = logging.getLogger(__name__)


class StagedState(BaseModel):
    """State carried across multiple staged steps."""

    stage1_done: bool = False
    stage2_done: bool = False
    stage3_done: bool = False
    log: list[str] = []


class StagedFlow(Flow[StagedState]):
    """Three-stage flow used for the resume demo."""

    @flow_start
    async def stage1(self) -> None:
        self.state.stage1_done = True
        self.state.log.append("stage1")
        logger.info("stage1: done")

    @flow_listen(stage1)
    async def stage2(self) -> None:
        self.state.stage2_done = True
        self.state.log.append("stage2")
        logger.info("stage2: done")

    @flow_listen(stage2)
    async def stage3(self) -> None:
        self.state.stage3_done = True
        self.state.log.append("stage3")
        logger.info("stage3: done")


async def demo_capture_and_resume() -> None:
    """Run, capture between stages, serialize, rehydrate, resume.

    The canonical capture point is after a full run completes —
    the executor's terminal state IS the resumable state. Mid-flow
    capture via hooks is not yet implemented.
    """
    # Stage 1: run partially and capture (simulated — we run the
    # full flow and snapshot the final state as the resumable artifact).
    flow1 = StagedFlow(StagedState)
    result1 = await Runner.arun_flow(flow1)
    logger.info("Initial run: status=%s log=%s", result1.status, result1.final_state.log)

    # Capture: build a checkpoint from the final state.
    checkpoint = FlowCheckpoint(
        flow_id=result1.flow_id,
        completed_steps=result1.completed_steps,
        pending_steps=(),
        and_gate_arrivals={},
        consumed_gates=(),
        state_data=result1.final_state.model_dump_json(),
        state_type_name=f"{StagedState.__module__}.{StagedState.__qualname__}",
    )
    serialized = checkpoint.to_json()
    logger.info("Serialized checkpoint (%d chars)", len(serialized))

    # Hand off to "another process" — for the demo we just deserialize
    # and rehydrate state in-memory.
    rehydrated_cp = FlowCheckpoint.from_json(serialized)
    rehydrated_state = StagedState.model_validate_json(rehydrated_cp.state_data)
    logger.info("Rehydrated state.log=%s", rehydrated_state.log)

    # Resume: reconstruct the same Flow class with the rehydrated state.
    flow2 = StagedFlow(initial_state=rehydrated_state)
    result2 = await Runner.arun_flow_from_checkpoint(flow2, rehydrated_cp)
    logger.info("Resumed run: status=%s log=%s", result2.status, result2.final_state.log)


async def main() -> None:
    await demo_capture_and_resume()


if __name__ == "__main__":
    asyncio.run(main())
