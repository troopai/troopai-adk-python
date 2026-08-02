"""Non-streaming ``AgentExecutable.invoke`` must suspend on a TOP-LEVEL HITL
deferral.

The real top-level ``Runner.arun`` does NOT raise
:class:`~troopai.adk.exceptions.AgentToolDeferral` when a tool is gated on
human approval — it RETURNS a ``RunResult`` with ``requires_action`` True and
``deferred_requests`` + ``state`` populated (see the agent loop's
``NextStepInterruption`` arm). The sibling
``test_agent_executable_deferral_invoke.py`` only exercises the nested
sub-agent path, where ``Runner.arun`` is monkeypatched to RAISE — which does
not reflect the primary top-level behaviour.

These tests patch ``Runner.arun`` to RETURN (not raise) so the bridge's
fall-through path is exercised: a returned ``requires_action`` result must be
lifted to ``InterruptException(NestedAgentInterrupt)`` and the deferral's
``RunState`` deposited into the side-channel dict the BSP loop owns. Without
the fix the node silently records ``final_output=None`` and the graph loop
treats it as a completed node, defeating the HITL contract.
"""

from __future__ import annotations

from typing import Any

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs.adapters import AgentExecutable
from troopai.adk.graphs.interrupt import InterruptException, NestedAgentInterrupt
from troopai.adk.orchestration.executable import ExecutableInput
from troopai.adk.run.context import RunContext
from troopai.adk.run.state import RunState
from troopai.adk.tools.deferred_tool import DeferredToolCall, DeferredToolRequests
from troopai.adk.types.run.run_result import RunResult
from troopai.adk.types.tokens.llm_usage import LLMUsage


def _deferral_run_result(
    *,
    with_state: bool = True,
    approvals: int = 1,
) -> RunResult[Any]:
    """A ``RunResult`` shaped exactly as the top-level runner returns it on a
    HITL deferral: ``final_output=None``, ``requires_action`` True via
    populated ``deferred_requests``, and (optionally) a resumable ``state``."""
    deferred = DeferredToolRequests(
        approvals=[
            DeferredToolCall(
                tool_call_id=f"c{i}",
                tool_name="t",
                tool_arguments={},
                raw_arguments="{}",
            )
            for i in range(approvals)
        ],
    )
    state = (
        RunState(
            current_agent_name="planner",
            turn_count=4,
            deferred_tool_requests=deferred,
        )
        if with_state
        else None
    )
    return RunResult(
        final_output=None,
        user_prompt="hi",
        context=RunContext(context=None, usage=LLMUsage()),
        last_agent=None,
        deferred_requests=deferred,
        state=state,
    )


async def test_invoke_lifts_returned_requires_action_to_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-level path: ``Runner.arun`` RETURNS ``requires_action`` (does not
    raise). The bridge must still suspend with a ``NestedAgentInterrupt`` and
    deposit the snapshot — not record a ``final_output=None`` node."""
    result = _deferral_run_result(approvals=1)

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        return result

    from troopai.adk.run import runner as runner_mod

    monkeypatch.setattr(runner_mod.Runner, "arun", classmethod(fake_arun))

    snapshots: dict[str, RunState] = {}
    executable: AgentExecutable[Any] = AgentExecutable(agent=Agent(name="planner", system_prompt="x"))
    input_ = ExecutableInput(
        content=[],
        metadata={
            "__interrupt_node_id__": "n",
            "__nested_agent_snapshots__": snapshots,
        },
    )
    context: RunContext[Any] = RunContext(context=None, usage=LLMUsage())

    with pytest.raises(InterruptException) as exc_info:
        await executable.invoke(input_, context, config=...)  # type: ignore[arg-type]

    interrupt = exc_info.value.interrupt
    assert isinstance(interrupt, NestedAgentInterrupt)
    assert interrupt.node_id == "n"
    assert interrupt.tool_call_ids == ("c0",)
    # Snapshot deposited so the resume path can re-enter the inner agent.
    assert snapshots["n"] is result.state


async def test_invoke_raises_runtime_error_when_returned_deferral_missing_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A returned ``requires_action`` result with no ``state`` cannot be
    resumed — the bridge must refuse loudly rather than silently complete."""
    result = _deferral_run_result(with_state=True)
    # Force the inconsistent shape: deferred but no resumable state.
    result.state = None

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        return result

    from troopai.adk.run import runner as runner_mod

    monkeypatch.setattr(runner_mod.Runner, "arun", classmethod(fake_arun))

    executable: AgentExecutable[Any] = AgentExecutable(agent=Agent(name="planner", system_prompt="x"))
    input_ = ExecutableInput(
        content=[],
        metadata={
            "__interrupt_node_id__": "n",
            "__nested_agent_snapshots__": {},
        },
    )
    context: RunContext[Any] = RunContext(context=None, usage=LLMUsage())

    with pytest.raises(RuntimeError, match="not\\s+state"):
        await executable.invoke(input_, context, config=...)  # type: ignore[arg-type]


async def test_invoke_completes_normally_when_no_deferral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path is unaffected: a normal ``RunResult`` (no deferral) yields a
    populated ``NodeResult`` with the agent's final output."""
    result: RunResult[Any] = RunResult(
        final_output="done",
        user_prompt="hi",
        context=RunContext(context=None, usage=LLMUsage()),
        last_agent=None,
    )

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        return result

    from troopai.adk.run import runner as runner_mod

    monkeypatch.setattr(runner_mod.Runner, "arun", classmethod(fake_arun))

    executable: AgentExecutable[Any] = AgentExecutable(agent=Agent(name="planner", system_prompt="x"))
    input_ = ExecutableInput(content=[], metadata={})
    context: RunContext[Any] = RunContext(context=None, usage=LLMUsage())

    node_result = await executable.invoke(input_, context, config=...)  # type: ignore[arg-type]

    assert node_result.output == "done"
    assert node_result.final_text == "done"
