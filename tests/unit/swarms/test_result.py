"""Tests for ``SwarmRunResult`` — non-streaming result ergonomics."""

from __future__ import annotations

from troopai.adk.agents.agent import Agent
from troopai.adk.swarms.result import SwarmRunResult
from troopai.adk.swarms.state import SwarmState
from troopai.adk.swarms.stop_reason import StopReason
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.types.tokens.llm_usage import LLMUsage


def _result(
    *,
    swarm_name: str | None = "code-review",
    final_output: object = "Refactored the module.",
    total_turns: int = 8,
    handoff_count: int = 3,
    total_tokens: int = 12_400,
) -> SwarmRunResult:
    a = Agent(name="author", system_prompt="noop")
    b = Agent(name="reviewer", system_prompt="noop")
    swarm = Swarm(members=(a, b), entry="author", name=swarm_name)
    state = SwarmState(
        swarm=swarm,
        current_agent=a,
        current_agent_name="author",
        total_turns=total_turns,
        handoff_count=handoff_count,
        cumulative_usage=LLMUsage(total_tokens=total_tokens),
    )
    return SwarmRunResult(
        final_output=final_output,
        stop_reason=StopReason(kind="explicit_done", detail="done"),
        user_prompt="Refactor this module.",
        state=state,
        last_agent=a,
        total_turns=total_turns,
        handoff_count=handoff_count,
    )


class TestSwarmRunResultRepr:
    def test_repr_summarizes_the_run(self) -> None:
        result = _result()
        assert repr(result) == (
            "SwarmRunResult(swarm='code-review', stop='explicit_done', "
            "turns=8, handoffs=3, tokens=12400, "
            "final_output='Refactored the module.')"
        )

    def test_repr_truncates_long_output(self) -> None:
        result = _result(final_output="x" * 500)
        r = repr(result)
        assert "x" * 500 not in r
        assert r.endswith("…')")

    def test_repr_without_swarm_name(self) -> None:
        result = _result(swarm_name=None)
        r = repr(result)
        assert r.startswith("SwarmRunResult(stop='explicit_done'")

    def test_repr_without_state(self) -> None:
        result = SwarmRunResult(
            final_output=None,
            stop_reason=StopReason(kind="max_turns", detail=""),
            user_prompt="go",
        )
        assert repr(result) == ("SwarmRunResult(stop='max_turns', turns=0, handoffs=0, final_output=None)")

    def test_repr_survives_release_agents(self) -> None:
        result = _result()
        result.release_agents()
        r = repr(result)
        assert r.startswith("SwarmRunResult(stop='explicit_done'")
