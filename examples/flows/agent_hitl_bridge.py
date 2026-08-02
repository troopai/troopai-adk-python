"""Agent-internal HITL bridge example — agent deferral propagates up to Flow.

Demonstrates :func:`arun_flow_agent` carrying an inner agent run's
``requires_approval`` tool gate up through the Flow layer:

1. A step body calls :func:`arun_flow_agent` to run an agent.
2. The agent's deferred-tool short-circuit normally surfaces on
   ``RunResult.requires_action``; here it propagates as
   :class:`FlowAgentDeferred`, which the executor captures into
   :attr:`FlowDeferredStep.agent_run_state`.
3. The flow halts with ``status="deferred"`` and a
   :class:`FlowCheckpoint`.
4. The developer rehydrates the agent's :class:`RunState`, records
   decisions via :meth:`RunState.approve` / :meth:`RunState.reject`,
   serialises it back, and resumes via
   :meth:`Runner.arun_flow_from_checkpoint(..., agent_resolutions=...)`.

The example uses an ``AsyncMock`` over :meth:`Runner.arun` so the
agent path doesn't need a provider API key. Synthetic-only, no
``load_dotenv`` required.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel

from troopai.adk.flows import Flow, arun_flow_agent, flow_start
from troopai.adk.run.runner import Runner
from troopai.adk.run.state import RunState

logger = logging.getLogger(__name__)


class CartState(BaseModel):
    approved_amount: int = 0


class CartFlow(Flow[CartState]):
    @flow_start
    async def approve_purchase(self) -> None:
        result = await arun_flow_agent(self, _STUB_AGENT, "approve $5000 purchase", defer_key="purchase")
        self.state.approved_amount = 5000 if not result.requires_action else 0


_STUB_AGENT: Any = object()


def _deferred_run_result() -> Any:
    """Stub that mimics an agent run halted on a tool's ``requires_approval``."""

    class _Deferred:
        requires_action = True

    out = _Deferred()
    # ``state`` is declared as a ClassVar=None for static typing; the
    # runtime contract from the Agent layer is that the attribute
    # carries a populated RunState whenever requires_action is True.
    out.state = RunState(  # type: ignore[attr-defined]
        original_user_prompt="approve $5000 purchase",
        current_agent_name="purchase_agent",
    )
    return out


def _completed_run_result() -> Any:
    class _Completed:
        requires_action = False
        state: Any = None
        final_output = "approved"

    return _Completed()


async def main() -> None:
    # First run: agent defers → flow halts.
    with patch("troopai.adk.run.runner.Runner.arun", new=AsyncMock(return_value=_deferred_run_result())):
        flow = CartFlow(CartState)
        initial = await Runner.arun_flow(flow)
    if initial.checkpoint is None:
        raise RuntimeError("Expected a checkpoint after agent deferral")
    ds = initial.deferred_steps[0]
    logger.info(
        "DEFERRED: step=%r defer_key=%r agent_state_size=%d",
        ds.step_name,
        ds.defer_key,
        len(ds.agent_run_state or ""),
    )

    # Out-of-band: rehydrate the agent's RunState and approve (here we
    # just round-trip — real code would call state.approve(...) on the
    # deferred tool calls).
    if ds.agent_run_state is None:
        raise RuntimeError("Deferred step is missing the agent_run_state payload")
    resumed_agent_state = RunState.from_dict(json.loads(ds.agent_run_state))
    resolved_json = json.dumps(resumed_agent_state.to_dict())
    logger.info("Approved out-of-band; resuming.")

    # Resume: bridge consumes the resolved state, agent completes.
    with patch("troopai.adk.run.runner.Runner.arun", new=AsyncMock(return_value=_completed_run_result())):
        resumed_flow = CartFlow(CartState)
        if ds.defer_key is None:
            raise RuntimeError("Deferred step is missing its defer_key")
        resumed = await Runner.arun_flow_from_checkpoint(
            resumed_flow,
            initial.checkpoint,
            agent_resolutions={ds.defer_key: resolved_json},
        )
    logger.info(
        "RESUMED: status=%s final_state=%s",
        resumed.status,
        resumed.final_state.model_dump(),
    )


if __name__ == "__main__":
    asyncio.run(main())
