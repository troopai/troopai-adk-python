"""Tests that flow step-guardrail evaluation records audit entries on the result."""

from __future__ import annotations

from pydantic import BaseModel

from troopai.adk.flows import (
    Flow,
    FlowStepContext,
    FlowStepGuardrails,
    FlowStepGuardrailVerdict,
    flow_start,
)
from troopai.adk.run.runner import Runner
from troopai.adk.types.guardrails import GuardrailAction


class _State(BaseModel):
    count: int = 0


def _allow(_ctx: FlowStepContext[_State]) -> FlowStepGuardrailVerdict:
    return FlowStepGuardrailVerdict.allow()


def _reject(_ctx: FlowStepContext[_State]) -> FlowStepGuardrailVerdict:
    return FlowStepGuardrailVerdict.reject_content("nope")


class TestFlowGuardrailAudit:
    async def test_pre_allow_records_flow_pre_pass(self) -> None:
        class _Flow(Flow[_State]):
            @flow_start(guardrails=FlowStepGuardrails(pre=(_allow,)))
            async def step(self) -> None:
                self.state.count += 1

        result = await Runner.arun_flow(_Flow(_State))
        records = [r for r in result.guardrail_audit if r.level == "flow_pre"]
        assert len(records) == 1
        assert records[0].action is GuardrailAction.PASS
        assert records[0].agent_name is None
        assert records[0].transformed_hash is None

    async def test_post_allow_records_flow_post_pass(self) -> None:
        class _Flow(Flow[_State]):
            @flow_start(guardrails=FlowStepGuardrails(post=(_allow,)))
            async def step(self) -> None:
                self.state.count += 1

        result = await Runner.arun_flow(_Flow(_State))
        assert any(r.level == "flow_post" and r.action is GuardrailAction.PASS for r in result.guardrail_audit)

    async def test_pre_reject_records_flow_pre_raise(self) -> None:
        class _Flow(Flow[_State]):
            @flow_start(guardrails=FlowStepGuardrails(pre=(_reject,)))
            async def step(self) -> None:
                self.state.count += 1

        result = await Runner.arun_flow(_Flow(_State))
        records = [r for r in result.guardrail_audit if r.level == "flow_pre"]
        assert len(records) == 1
        assert records[0].action is GuardrailAction.RAISE
