"""AgentExecutable.resume_from_snapshot applies a NestedAgentReply
to the snapshot via RunState.approve/reject, re-enters Runner.arun
with the mutated state, and returns the resulting NodeResult.
Reply-snapshot mismatches raise NestedAgentResumeError. When the
resumed agent defers again, the bridge re-lifts to a fresh
NestedAgentInterrupt and re-deposits the new snapshot."""

from __future__ import annotations

from typing import Any

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs.adapters import AgentExecutable
from troopai.adk.graphs.interrupt import (
    InterruptException,
    NestedAgentApproval,
    NestedAgentInterrupt,
    NestedAgentRejection,
    NestedAgentReply,
    NestedAgentResumeError,
)
from troopai.adk.run.context import RunContext
from troopai.adk.run.state import RunState
from troopai.adk.tools.deferred_tool import DeferredToolCall, DeferredToolRequests
from troopai.adk.types.tokens.llm_usage import LLMUsage


def _deferred_call(call_id: str) -> DeferredToolCall:
    return DeferredToolCall(
        tool_call_id=call_id,
        tool_name="t",
        tool_arguments={},
        raw_arguments="{}",
    )


def _snapshot_with_calls(*ids: str) -> RunState:
    return RunState(
        deferred_tool_requests=DeferredToolRequests(approvals=[_deferred_call(i) for i in ids]),
        current_agent_name="planner",
    )


class _FakeRunResult:
    """Stand-in for the RunResult returned by Runner.arun(agent, state)."""

    def __init__(self, final_output: Any = "ok") -> None:
        self.context = type("_FakeRC", (), {"usage": LLMUsage()})()
        self.final_output = final_output
        self.new_items: list[Any] = []
        self.last_agent = type("_FakeAgent", (), {"name": "planner"})()
        # Completed run — no re-deferral.
        self.requires_action = False
        self.deferred_requests = None
        self.state = None


async def test_resume_applies_approval_then_re_enters_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    """A NestedAgentApproval lands as state.approve before the runner re-entry."""
    seen_state: dict[str, RunState] = {}

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        # config is unused on this fast path; the fake just records what state arrived.
        state_arg = args[1] if len(args) >= 2 else kwargs.get("user_prompt")
        assert isinstance(state_arg, RunState)
        seen_state["resumed"] = state_arg
        return _FakeRunResult()

    from troopai.adk.run import runner as runner_mod

    monkeypatch.setattr(runner_mod.Runner, "arun", classmethod(fake_arun))

    snapshot = _snapshot_with_calls("c1", "c2")
    reply = NestedAgentReply(
        decisions=(
            NestedAgentApproval(tool_call_id="c1", approver_id="alice", reason="ok"),
            NestedAgentRejection(tool_call_id="c2", message="blocked", approver_id="bob"),
        )
    )
    executable = AgentExecutable(agent=Agent(name="planner", system_prompt="x"))
    context: RunContext[Any] = RunContext(context=None, usage=LLMUsage())

    result = await executable.resume_from_snapshot(
        snapshot=snapshot,
        reply=reply,
        context=context,
        config=...,  # type: ignore[arg-type]
        # config is unused on this success path; fake_arun doesn't read it.
        node_id="n",
        nested_agent_snapshots={},
    )

    mutated = seen_state["resumed"]
    # The mutated state arrived at Runner.arun
    assert mutated is snapshot
    assert len(mutated.approved_tools) == 1
    assert mutated.approved_tools[0].tool_call_id == "c1"
    assert len(mutated.rejected_tools) == 1
    rejected_call, message = mutated.rejected_tools[0]
    assert rejected_call.tool_call_id == "c2"
    assert message == "blocked"
    # NodeResult shaped by _run_agent_node_result returned cleanly
    assert result is not None


async def test_resume_rejects_unknown_tool_call_id() -> None:
    """A reply targeting a tool_call_id not in the snapshot's deferred
    approvals raises NestedAgentResumeError BEFORE any state mutation
    or runner re-entry — fail-fast preserves the snapshot for retry."""
    snapshot = _snapshot_with_calls("c1")
    reply = NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="cX"),))
    executable = AgentExecutable(agent=Agent(name="planner", system_prompt="x"))
    context: RunContext[Any] = RunContext(context=None, usage=LLMUsage())

    with pytest.raises(NestedAgentResumeError) as exc_info:
        await executable.resume_from_snapshot(
            snapshot=snapshot,
            reply=reply,
            context=context,
            config=...,  # type: ignore[arg-type]
            # config never reached — NestedAgentResumeError raised at validation.
            node_id="n",
            nested_agent_snapshots={},
        )
    assert exc_info.value.node_id == "n"
    assert "cX" in exc_info.value.detail
    # Snapshot is untouched — caller can retry with a fixed reply.
    assert len(snapshot.approved_tools) == 0
    assert len(snapshot.rejected_tools) == 0


async def test_resume_rejects_duplicate_decision_for_same_call() -> None:
    """A reply with two decisions for the same tool_call_id raises
    NestedAgentResumeError — ambiguous intent."""
    snapshot = _snapshot_with_calls("c1")
    reply = NestedAgentReply(
        decisions=(
            NestedAgentApproval(tool_call_id="c1"),
            NestedAgentRejection(tool_call_id="c1", message="contradiction"),
        )
    )
    executable = AgentExecutable(agent=Agent(name="planner", system_prompt="x"))
    context: RunContext[Any] = RunContext(context=None, usage=LLMUsage())

    with pytest.raises(NestedAgentResumeError) as exc_info:
        await executable.resume_from_snapshot(
            snapshot=snapshot,
            reply=reply,
            context=context,
            config=...,  # type: ignore[arg-type]
            # config never reached — NestedAgentResumeError raised at validation.
            node_id="n",
            nested_agent_snapshots={},
        )
    assert exc_info.value.node_id == "n"
    assert "c1" in exc_info.value.detail
    assert len(snapshot.approved_tools) == 0
    assert len(snapshot.rejected_tools) == 0


async def test_resume_with_empty_decisions_re_enters_runner_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty decisions tuple is a valid no-op — the caller chose to
    leave every pending call to re-defer. The snapshot is unmodified
    and the agent re-enters as-is. (Useful for partial-fan-out resume
    where one node had no decisions ready.)"""
    seen_state: dict[str, RunState] = {}

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        state_arg = args[1] if len(args) >= 2 else kwargs.get("user_prompt")
        assert isinstance(state_arg, RunState)
        seen_state["resumed"] = state_arg
        return _FakeRunResult()

    from troopai.adk.run import runner as runner_mod

    monkeypatch.setattr(runner_mod.Runner, "arun", classmethod(fake_arun))

    snapshot = _snapshot_with_calls("c1")
    reply = NestedAgentReply(decisions=())
    executable = AgentExecutable(agent=Agent(name="planner", system_prompt="x"))
    context: RunContext[Any] = RunContext(context=None, usage=LLMUsage())

    result = await executable.resume_from_snapshot(
        snapshot=snapshot,
        reply=reply,
        context=context,
        config=...,  # type: ignore[arg-type]
        # config is unused on this fast path.
        node_id="n",
        nested_agent_snapshots={},
    )
    # Snapshot unchanged
    assert len(snapshot.approved_tools) == 0
    assert len(snapshot.rejected_tools) == 0
    # Runner re-entered with the unchanged snapshot
    assert seen_state["resumed"] is snapshot
    assert result is not None


async def test_resume_lifts_re_deferral_to_fresh_nested_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the resumed agent defers AGAIN, the bridge re-lifts to a fresh
    NestedAgentInterrupt and re-deposits the new snapshot — the second
    human-approval gate is never silently lost."""
    # Build the re-deferral RunResult shape: requires_action=True with
    # populated deferred_requests + state.
    re_defer_state = RunState(
        deferred_tool_requests=DeferredToolRequests(approvals=[_deferred_call("c-second")]),
        current_agent_name="planner",
        turn_count=9,
    )

    class _ReDeferringRunResult:
        def __init__(self) -> None:
            self.context = type("_FakeRC", (), {"usage": LLMUsage()})()
            self.final_output = None
            self.new_items: list[Any] = []
            self.last_agent = type("_FakeAgent", (), {"name": "planner"})()
            self.requires_action = True
            self.deferred_requests = re_defer_state.deferred_tool_requests
            self.state = re_defer_state

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        return _ReDeferringRunResult()

    from troopai.adk.run import runner as runner_mod

    monkeypatch.setattr(runner_mod.Runner, "arun", classmethod(fake_arun))

    snapshot = _snapshot_with_calls("c1")
    reply = NestedAgentReply(decisions=(NestedAgentApproval(tool_call_id="c1"),))
    snapshots: dict[str, RunState] = {}
    executable = AgentExecutable(agent=Agent(name="planner", system_prompt="x"))
    context: RunContext[Any] = RunContext(context=None, usage=LLMUsage())

    with pytest.raises(InterruptException) as exc_info:
        await executable.resume_from_snapshot(
            snapshot=snapshot,
            reply=reply,
            context=context,
            config=...,  # type: ignore[arg-type]
            # config unused on this fast path; fake_arun ignores it.
            node_id="n",
            nested_agent_snapshots=snapshots,
        )

    # First decision applied to the original snapshot.
    assert len(snapshot.approved_tools) == 1
    assert snapshot.approved_tools[0].tool_call_id == "c1"
    # Re-deferral lifted to a fresh interrupt with the NEW pending call.
    interrupt = exc_info.value.interrupt
    assert isinstance(interrupt, NestedAgentInterrupt)
    assert interrupt.node_id == "n"
    assert interrupt.tool_call_ids == ("c-second",)
    # The new snapshot was deposited for the BSP loop to checkpoint.
    assert "n" in snapshots
    assert snapshots["n"] is re_defer_state
    assert snapshots["n"].turn_count == 9


async def test_resume_raises_when_node_id_empty() -> None:
    """node_id is required and must be non-empty — empty string is a
    footgun because it lets the empty string flow through error messages
    and the caller's GraphResume.replies lookup."""
    snapshot = _snapshot_with_calls("c1")
    reply = NestedAgentReply(decisions=())
    executable = AgentExecutable(agent=Agent(name="planner", system_prompt="x"))
    context: RunContext[Any] = RunContext(context=None, usage=LLMUsage())
    with pytest.raises(ValueError, match="node_id must be a non-empty"):
        await executable.resume_from_snapshot(
            snapshot=snapshot,
            reply=reply,
            context=context,
            config=...,  # type: ignore[arg-type]
            # config unused — ValueError raised by node_id guard.
            node_id="",
            nested_agent_snapshots={},
        )
