"""``AgentExecutable.invoke`` translates ``AgentToolDeferral`` into
``InterruptException(NestedAgentInterrupt)``, and stores the deferral's
``RunState`` in the surrounding ``GraphState`` via a side-channel write.

The reserved metadata keys ``__interrupt_node_id__`` and
``__nested_agent_snapshots__`` are the BSP loop's side-channel; the
producer side lives in the BSP loop, the consumer side is exercised
here on the adapter.
"""

from __future__ import annotations

from typing import Any

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.exceptions import AgentToolDeferral
from troopai.adk.graphs.adapters import AgentExecutable
from troopai.adk.graphs.interrupt import InterruptException, NestedAgentInterrupt
from troopai.adk.orchestration.executable import ExecutableInput
from troopai.adk.run.context import RunContext
from troopai.adk.run.state import RunState
from troopai.adk.tools.deferred_tool import DeferredToolCall, DeferredToolRequests
from troopai.adk.types.tokens.llm_usage import LLMUsage


async def test_invoke_translates_deferral_to_nested_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    defer = AgentToolDeferral(
        agent_name="planner",
        deferred_requests=DeferredToolRequests(
            approvals=[
                DeferredToolCall(tool_call_id="c1", tool_name="t", tool_arguments={}, raw_arguments="{}"),
            ],
        ),
        state=RunState(current_agent_name="planner", turn_count=3),
    )

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        raise defer

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
        # config is unused on the failure path; fake_arun raises before
        # the executable reads config.

    interrupt = exc_info.value.interrupt
    assert isinstance(interrupt, NestedAgentInterrupt)
    assert interrupt.node_id == "n"
    assert interrupt.agent_name == "planner"
    assert interrupt.tool_call_ids == ("c1",)
    # Snapshot deposited into the side-channel dict the BSP loop owns.
    assert "n" in snapshots
    assert snapshots["n"].turn_count == 3


async def test_invoke_raises_when_node_id_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing ``__interrupt_node_id__`` is a producer-side programmer
    error — the bridge MUST refuse rather than silently lifting to an
    interrupt whose node_id is empty (which would later trip the
    GraphState cross-reference check with no breadcrumb to the bridge)."""
    defer = AgentToolDeferral(
        agent_name="planner",
        deferred_requests=DeferredToolRequests(
            approvals=[
                DeferredToolCall(tool_call_id="c1", tool_name="t", tool_arguments={}, raw_arguments="{}"),
            ],
        ),
        state=RunState(current_agent_name="planner", turn_count=1),
    )

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        raise defer

    from troopai.adk.run import runner as runner_mod

    monkeypatch.setattr(runner_mod.Runner, "arun", classmethod(fake_arun))

    executable: AgentExecutable[Any] = AgentExecutable(agent=Agent(name="planner", system_prompt="x"))
    # No __interrupt_node_id__ in metadata — bridge must raise RuntimeError.
    input_ = ExecutableInput(content=[], metadata={"__nested_agent_snapshots__": {}})
    context: RunContext[Any] = RunContext(context=None, usage=LLMUsage())

    with pytest.raises(RuntimeError, match="__interrupt_node_id__"):
        await executable.invoke(input_, context, config=...)  # type: ignore[arg-type]
        # config is unused on the failure path; fake_arun raises before
        # the executable reads config.


async def test_invoke_raises_when_snapshots_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing ``__nested_agent_snapshots__`` is the producer-side bug
    that would leave the bridge silently dropping snapshots — refusing
    here surfaces the bug at the source rather than at the downstream
    cross-reference check."""
    defer = AgentToolDeferral(
        agent_name="planner",
        deferred_requests=DeferredToolRequests(
            approvals=[
                DeferredToolCall(tool_call_id="c1", tool_name="t", tool_arguments={}, raw_arguments="{}"),
            ],
        ),
        state=RunState(current_agent_name="planner", turn_count=1),
    )

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        raise defer

    from troopai.adk.run import runner as runner_mod

    monkeypatch.setattr(runner_mod.Runner, "arun", classmethod(fake_arun))

    executable: AgentExecutable[Any] = AgentExecutable(agent=Agent(name="planner", system_prompt="x"))
    input_ = ExecutableInput(content=[], metadata={"__interrupt_node_id__": "n"})
    context: RunContext[Any] = RunContext(context=None, usage=LLMUsage())

    with pytest.raises(RuntimeError, match="__nested_agent_snapshots__"):
        await executable.invoke(input_, context, config=...)  # type: ignore[arg-type]
        # config is unused on the failure path; fake_arun raises before
        # the executable reads config.


async def test_invoke_deposits_snapshot_with_full_run_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bridge round-trip: the deposited RunState preserves every field
    that matters for resume (turn_count, current_agent_name,
    deferred_tool_requests). This is the integration test that catches
    bugs where the bridge truncates or shadows the snapshot."""
    deferred_requests = DeferredToolRequests(
        approvals=[
            DeferredToolCall(tool_call_id="c1", tool_name="t", tool_arguments={"x": 1}, raw_arguments='{"x":1}'),
            DeferredToolCall(tool_call_id="c2", tool_name="u", tool_arguments={}, raw_arguments="{}"),
        ],
    )
    pre_defer_state = RunState(
        current_agent_name="planner",
        turn_count=7,
        deferred_tool_requests=deferred_requests,
    )
    defer = AgentToolDeferral(
        agent_name="planner",
        deferred_requests=deferred_requests,
        state=pre_defer_state,
    )

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        raise defer

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
        # config is unused on the failure path; fake_arun raises before
        # the executable reads config.

    # The deposit happened — identity, not equality.
    assert snapshots["n"] is pre_defer_state
    # The deposited state preserves every field the resume path will need.
    deposited = snapshots["n"]
    assert deposited.turn_count == 7
    assert deposited.current_agent_name == "planner"
    assert len(deposited.deferred_tool_requests.approvals) == 2
    # The interrupt payload references the same two tool calls.
    assert isinstance(exc_info.value.interrupt, NestedAgentInterrupt)
    assert exc_info.value.interrupt.tool_call_ids == ("c1", "c2")
