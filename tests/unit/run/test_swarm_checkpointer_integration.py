"""Runner-level integration tests for swarm checkpointer auto-wiring.

Verifies that ``Runner.arun_swarm``, ``Runner.arun_swarm_from_checkpoint``,
and ``SwarmRunner.checkpointer`` correctly thread a
:class:`SwarmCheckpointer` through the runner and into the swarm loop,
where it auto-saves after each turn via the swarm hook registry.

The patched-member harness from ``test_swarm_loop_checkpointer.py`` is
reused: ``run_agent_loop`` is replaced with an ``AsyncMock`` that returns a
minimal :class:`RunResult` so the driver can complete one iteration via
:class:`MaxTurnsTermination` without touching an LLM.
"""

from __future__ import annotations

from typing import Any, override
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs.interrupt import Interrupt, InterruptException
from troopai.adk.run.context import RunContext
from troopai.adk.run.runner import Runner
from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer
from troopai.adk.swarms.hooks import SwarmHooks
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.state import SwarmState
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination
from troopai.adk.types.run.run_result import RunResult


def _make_swarm(*, hooks: SwarmHooks[Any] | None = None, max_turns: int = 1) -> Swarm[Any]:
    """Single-member swarm capped at ``max_turns``."""
    member = Agent(name="worker", system_prompt="x")
    return Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(max_turns),
        hooks=hooks,
    )


def _stub_result(ctx: RunContext[Any], member: Agent[Any]) -> RunResult[Any]:
    """Minimal RunResult that lets the loop's step-8 advance cleanly."""
    return RunResult(
        final_output=None,
        user_prompt="",
        new_items=[],
        context=ctx,
        last_agent=member,
        swarm_yield=None,
    )


class TestArunSwarmAutoSave:
    async def test_arun_swarm_checkpointer_saves_after_turn(self) -> None:
        """Runner.arun_swarm with a checkpointer auto-saves after each turn."""
        sw = _make_swarm()
        cp = InMemorySwarmCheckpointer(thread_id="t1")

        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(side_effect=lambda **kwargs: _stub_result(kwargs["context"], kwargs["agent"])),
        ):
            await Runner.arun_swarm(sw, "hello", checkpointer=cp)

        saved = await cp.list_checkpoints()
        assert saved == ["t1"], f"expected ['t1'], got {saved}"
        loaded = await cp.load("t1", sw)
        assert loaded is not None
        assert loaded.turn >= 1


class TestArunSwarmFromCheckpointContinuesSaving:
    async def test_resumed_run_continues_saving(self) -> None:
        """arun_swarm_from_checkpoint passes the checkpointer into the loop
        so the resumed run continues auto-saving (checkpointer is no longer
        dropped after load).

        Seed run uses MaxTurnsTermination(1), producing a checkpoint at turn 1.
        Resume uses the same member roster but MaxTurnsTermination(2) so the
        loaded state (total_turns=1) allows one more turn before terminating.
        """
        member = Agent(name="worker", system_prompt="x")
        sw_seed = Swarm(
            members=(member,),
            entry=member,
            policy=RoundRobinPolicy(),
            termination=MaxTurnsTermination(1),
        )
        sw_resume = Swarm(
            members=(member,),
            entry=member,
            policy=RoundRobinPolicy(),
            termination=MaxTurnsTermination(2),
        )
        cp = InMemorySwarmCheckpointer(thread_id="t2")

        # Seed: run 1 turn, checkpoint saved at turn=1.
        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(side_effect=lambda **kwargs: _stub_result(kwargs["context"], kwargs["agent"])),
        ):
            await Runner.arun_swarm(sw_seed, "seed", checkpointer=cp)

        assert await cp.list_checkpoints() == ["t2"]
        seed = await cp.load("t2", sw_seed)
        assert seed is not None
        assert seed.turn == 1

        # Resume: loaded state has total_turns=1; with MaxTurnsTermination(2) one
        # more turn runs, saving a checkpoint at turn=2.
        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(side_effect=lambda **kwargs: _stub_result(kwargs["context"], kwargs["agent"])),
        ):
            await Runner.arun_swarm_from_checkpoint(
                sw_resume,
                checkpointer=cp,
                thread_id="t2",
            )

        final = await cp.load("t2", sw_resume)
        assert final is not None
        assert final.turn > seed.turn


class TestArunSwarmSaveFailurePropagates:
    async def test_save_failure_raises_from_arun_swarm(self) -> None:
        """A checkpointer whose save() raises propagates the error out of
        arun_swarm (propagate_errors=True on SwarmCheckpointerHooks ensures
        it is not swallowed)."""

        class _FailingCheckpointer(InMemorySwarmCheckpointer):
            async def save(self, checkpoint: Any) -> None:  # type: ignore[override]  # test fixture widens checkpoint type intentionally
                raise RuntimeError("backend unavailable")

        sw = _make_swarm()
        cp = _FailingCheckpointer(thread_id="fail")

        with (
            patch(
                "troopai.adk.run.swarm_loop.run_agent_loop",
                new=AsyncMock(side_effect=lambda **kwargs: _stub_result(kwargs["context"], kwargs["agent"])),
            ),
            pytest.raises(RuntimeError, match="backend unavailable"),
        ):
            await Runner.arun_swarm(sw, "hello", checkpointer=cp)


class TestArunSwarmInterruptPathSaveFailurePropagates:
    async def test_interrupt_path_save_failure_raises(self) -> None:
        """A checkpointer whose save() raises on the interrupt path propagates
        the error out of arun_swarm.

        The interrupt hook (on_swarm_turn_interrupt) calls save() when a
        member turn raises InterruptException.  With propagate_errors=True on
        SwarmCheckpointerHooks the error must not be swallowed.
        """

        class _FailingCheckpointer(InMemorySwarmCheckpointer):
            async def save(self, checkpoint: Any) -> None:  # type: ignore[override]  # test fixture widens checkpoint type intentionally
                raise RuntimeError("interrupt-path backend unavailable")

        sw = _make_swarm()
        cp = _FailingCheckpointer(thread_id="fail-interrupt")

        interrupt = Interrupt(
            node_id="worker",
            question="approve?",
            kind="tool_approval",
        )

        async def _raise_interrupt(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise InterruptException(interrupt)

        with (
            patch(
                "troopai.adk.run.swarm_loop.run_agent_loop",
                new=AsyncMock(side_effect=_raise_interrupt),
            ),
            pytest.raises(RuntimeError, match="interrupt-path backend unavailable"),
        ):
            await Runner.arun_swarm(sw, "hello", checkpointer=cp)


class TestObserverStillBestEffort:
    async def test_observer_error_does_not_abort_run(self) -> None:
        """A swarm.hooks observer that raises does NOT abort the run.
        propagate_errors defaults to False on SwarmHooks, so the error
        is logged and the run continues to completion."""
        completed: list[bool] = []

        class _RaisingObserver(SwarmHooks[None]):
            @override
            async def on_swarm_turn_end(
                self,
                context: RunContext[None],
                state: SwarmState[None],
                items: list[Any],
            ) -> None:
                del context, state, items
                raise ValueError("observer noise")

        sw = _make_swarm(hooks=_RaisingObserver())

        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(side_effect=lambda **kwargs: _stub_result(kwargs["context"], kwargs["agent"])),
        ):
            result = await Runner.arun_swarm(sw, "hello")

        completed.append(True)
        assert len(completed) == 1, "run did not complete despite observer error"
        assert result is not None


class TestSwarmRunnerWithCheckpointer:
    async def test_swarm_runner_with_checkpointer_auto_saves(self) -> None:
        """SwarmRunner.checkpointer(cp).arun(prompt) auto-saves."""
        sw = _make_swarm()
        cp = InMemorySwarmCheckpointer(thread_id="b1")

        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(side_effect=lambda **kwargs: _stub_result(kwargs["context"], kwargs["agent"])),
        ):
            await Runner.configure().swarm(sw).checkpointer(cp).arun("hello")

        saved = await cp.list_checkpoints()
        assert saved == ["b1"], f"expected ['b1'], got {saved}"
        loaded = await cp.load("b1", sw)
        assert loaded is not None
        assert loaded.turn >= 1


__all__: list[str] = []
