"""Unit tests for the agent-bridge HITL surface."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from troopai.adk.flows import (
    Flow,
    FlowAgentDeferred,
    FlowApprovalPolicy,
    FlowCheckpoint,
    arun_flow_agent,
    flow_start,
)
from troopai.adk.run.runner import Runner
from troopai.adk.run.state import RunState


class _State(BaseModel):
    completed: bool = False


class _Bridge(Flow[_State]):
    """Tiny flow whose start step calls into an agent via the bridge."""

    @flow_start
    async def go(self) -> None:
        # Pass a stub object as the "agent" — the patched Runner.arun
        # never inspects it.
        result = await arun_flow_agent(self, _STUB_AGENT, "ping", defer_key="agent_a")
        self.state.completed = result is not None


_STUB_AGENT: Any = object()


def _completed_run_result(text: str = "ok") -> Any:
    """Build a minimal ``RunResult``-like stub the bridge can inspect.

    ``requires_action`` is a property on the real type but the test
    only needs the attribute access; SimpleNamespace would also work
    but the explicit class makes intent clearer.
    """

    class _CompletedResult:
        requires_action = False
        state: Any = None
        final_output = text

    return _CompletedResult()


def _deferred_run_result(state: RunState) -> Any:
    """Build a stub mimicking an agent run that deferred via a tool gate."""

    class _DeferredResult:
        requires_action = True

    out = _DeferredResult()
    # The stub class declares ``state`` only at the class level for
    # static typing; pyright reads the body as immutable. At runtime
    # we attach the real RunState — the executor's only contract is
    # the attribute exists when requires_action is True.
    out.state = state  # type: ignore[attr-defined]
    return out


class TestAgentBridgeCompletes:
    async def test_agent_run_returning_normally_yields_result(self) -> None:
        completed = _completed_run_result()
        with patch("troopai.adk.run.runner.Runner.arun", new=AsyncMock(return_value=completed)):
            flow = _Bridge(_State)
            result = await Runner.arun_flow(flow)
        assert result.status == "completed"
        assert result.final_state.completed is True


class TestAgentBridgeDefers:
    async def test_agent_deferral_propagates_to_flow_checkpoint(self) -> None:
        run_state = RunState(original_user_prompt="ping", current_agent_name="agent_a")
        deferred = _deferred_run_result(run_state)

        with patch("troopai.adk.run.runner.Runner.arun", new=AsyncMock(return_value=deferred)):
            flow = _Bridge(_State)
            result = await Runner.arun_flow(flow)

        assert result.status == "deferred"
        assert result.requires_action is True
        assert result.checkpoint is not None
        assert len(result.deferred_steps) == 1
        ds = result.deferred_steps[0]
        assert ds.step_name == "go"
        assert ds.defer_key == "agent_a"
        assert ds.agent_run_state is not None  # serialised JSON of the agent's RunState

    async def test_agent_deferral_carries_step_approval_policy(self) -> None:
        """The step's ``approval_policy=`` must survive the agent-deferral capture path too."""
        policy = FlowApprovalPolicy(quorum=2, allowed_roles=("ops_lead",))

        class _PolicyBridge(Flow[_State]):
            @flow_start(approval_policy=policy)
            async def go(self) -> None:
                await arun_flow_agent(self, _STUB_AGENT, "ping", defer_key="agent_a")

        run_state = RunState(original_user_prompt="ping", current_agent_name="agent_a")
        deferred = _deferred_run_result(run_state)

        with patch("troopai.adk.run.runner.Runner.arun", new=AsyncMock(return_value=deferred)):
            result = await Runner.arun_flow(_PolicyBridge(_State))

        assert result.status == "deferred"
        assert result.deferred_steps[0].policy is policy

    async def test_resume_threads_agent_resolution_back_to_runner(self) -> None:
        run_state = RunState(original_user_prompt="ping", current_agent_name="agent_a")
        deferred = _deferred_run_result(run_state)

        with patch("troopai.adk.run.runner.Runner.arun", new=AsyncMock(return_value=deferred)):
            flow = _Bridge(_State)
            initial = await Runner.arun_flow(flow)
        assert initial.checkpoint is not None
        ds = initial.deferred_steps[0]
        assert ds.agent_run_state is not None
        assert ds.defer_key is not None

        # Approve the inner agent's deferred tool via the standard
        # RunState surface — same shape as the function-tool path.
        resumed_state = RunState.from_dict(json.loads(ds.agent_run_state))

        # On resume, patch Runner.arun to return a completed result
        # and assert it was called with the RunState argument.
        completed = _completed_run_result()
        resume_mock = AsyncMock(return_value=completed)
        with patch("troopai.adk.run.runner.Runner.arun", new=resume_mock):
            resumed_flow = _Bridge(_State)
            resumed = await Runner.arun_flow_from_checkpoint(
                resumed_flow,
                initial.checkpoint,
                agent_resolutions={ds.defer_key: json.dumps(resumed_state.to_dict())},
            )
        assert resumed.status == "completed"
        assert resumed.final_state.completed is True
        # On resume, the bridge called Runner.arun with the resumed
        # state as the second positional argument (vs the raw prompt
        # on the cold-start path).
        call_args = resume_mock.call_args
        assert call_args is not None
        # The second arg passed to Runner.arun is either the prompt
        # (cold start) or a RunState (resume) — assert the latter.
        assert isinstance(call_args.args[1], RunState)


class TestFlowAgentDeferredExceptionShape:
    def test_carries_step_defer_key_and_run_state_data(self) -> None:
        exc = FlowAgentDeferred(step_name="go", defer_key="k", run_state_data='{"x":1}')
        assert exc.step_name == "go"
        assert exc.defer_key == "k"
        assert exc.run_state_data == '{"x":1}'


class TestRoundTripAgentRunStateOnCheckpoint:
    def test_checkpoint_json_round_trips_agent_run_state(self) -> None:
        from troopai.adk.flows import FlowDeferredStep

        deferred = FlowDeferredStep(
            step_name="go",
            defer_key="agent_a",
            agent_run_state='{"current_agent_name": "agent_a"}',
        )
        cp = FlowCheckpoint(
            flow_id="f",
            completed_steps=(),
            pending_steps=("go",),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data="{}",
            deferred_steps=(deferred,),
        )
        rehydrated = FlowCheckpoint.from_json(cp.to_json())
        out = rehydrated.deferred_steps[0]
        assert out.defer_key == "agent_a"
        assert out.agent_run_state == '{"current_agent_name": "agent_a"}'


@pytest.mark.parametrize(
    "agent_resolutions",
    [None, {}],
)
async def test_arun_flow_from_checkpoint_accepts_empty_agent_resolutions(
    agent_resolutions: Any,
) -> None:
    """Plain step-level deferrals resume without an ``agent_resolutions`` mapping."""
    flow = _Bridge(_State)
    completed = _completed_run_result()
    with patch("troopai.adk.run.runner.Runner.arun", new=AsyncMock(return_value=completed)):
        # Build a trivial checkpoint with no deferred steps; resume should be a no-op completion.
        empty_cp = FlowCheckpoint(
            flow_id=flow.flow_id,
            completed_steps=("go",),
            pending_steps=(),
            and_gate_arrivals={},
            consumed_gates=(),
            state_data=flow.state.model_dump_json(),
        )
        resumed = await Runner.arun_flow_from_checkpoint(flow, empty_cp, agent_resolutions=agent_resolutions)
    assert resumed.status == "completed"
