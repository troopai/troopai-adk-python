"""Unit tests for ``run_swarm_loop`` + ``run_swarm_loop_streamed``
HookRegistry wiring.

Verifies two properties:

1. **Auto-save**: a supplied ``InMemorySwarmCheckpointer`` saves after
   each completed turn.
2. **Observer preserved**: a ``swarm.hooks`` observer's
   ``on_swarm_turn_end`` still fires when no checkpointer is provided
   (registry with a single element behaves identically to the old direct
   call).

Both tests use the lightest real harness: a patched ``run_agent_loop``
(non-streamed) or ``_stream_member_turn`` (streamed) that returns a
minimal :class:`RunResult` so the driver can complete one iteration via
:class:`MaxTurnsTermination`.  The registry-construction path is
exercised by the loop itself, so these tests cover the new routing
end-to-end without needing a real LLM.
"""

from __future__ import annotations

import asyncio
from typing import Any, override
from unittest.mock import AsyncMock, patch

from troopai.adk.agents.agent import Agent
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.config import DEFAULT_RUN_CONFIG
from troopai.adk.run.context import RunContext
from troopai.adk.run.swarm_loop import run_swarm_loop
from troopai.adk.run.swarm_loop_streamed import run_swarm_loop_streamed
from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer
from troopai.adk.swarms.hooks import SwarmHooks
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.result import SwarmRunResultStreaming
from troopai.adk.swarms.state import SwarmState
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination
from troopai.adk.types.run.run_result import RunResult


def _make_swarm(*, hooks: SwarmHooks[Any] | None = None, max_turns: int = 1) -> Swarm[Any]:
    """Single-member swarm capped at ``max_turns`` for loop tests."""
    member = Agent(name="worker", system_prompt="x")
    return Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(max_turns),
        hooks=hooks,
    )


def _stub_result(member: Agent[Any], ctx: RunContext[Any]) -> RunResult[Any]:
    """Minimal RunResult that lets the loop's step-8 advance cleanly."""
    return RunResult(
        final_output=None,
        user_prompt="",
        new_items=[],
        context=ctx,
        last_agent=member,
        swarm_yield=None,
    )


class TestSwarmLoopCheckpointerAutoSave:
    async def test_checkpointer_saves_after_turn(self) -> None:
        """Supplying a checkpointer causes it to save after each turn."""
        sw = _make_swarm()
        ctx: RunContext[None] = RunContext.make(None)
        cp = InMemorySwarmCheckpointer(thread_id="t1")

        mock_result = _stub_result(sw.entry, ctx)
        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(return_value=mock_result),
        ):
            await run_swarm_loop(
                swarm=sw,
                user_prompt="hello",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                checkpointer=cp,
            )

        saved = await cp.list_checkpoints()
        assert saved == ["t1"], f"expected ['t1'], got {saved}"
        loaded = await cp.load("t1", sw)
        assert loaded is not None
        assert loaded.turn >= 1


class TestSwarmLoopObserverPreserved:
    async def test_swarm_hooks_observer_fires_without_checkpointer(self) -> None:
        """A swarm.hooks observer's on_swarm_turn_end fires with no checkpointer."""
        fired: list[str] = []

        class _Recorder(SwarmHooks[None]):
            @override
            async def on_swarm_turn_end(
                self,
                context: RunContext[None],
                state: SwarmState[None],
                items: list[Any],
            ) -> None:
                del context, state, items
                fired.append("on_swarm_turn_end")

        sw = _make_swarm(hooks=_Recorder())
        ctx: RunContext[None] = RunContext.make(None)
        mock_result = _stub_result(sw.entry, ctx)

        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(return_value=mock_result),
        ):
            await run_swarm_loop(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
            )

        assert fired == ["on_swarm_turn_end"], f"expected one fire, got {fired}"


class TestSwarmLoopRegistryBothHookTypes:
    async def test_observer_and_checkpointer_both_fire(self) -> None:
        """With both swarm.hooks and a checkpointer, both receive on_swarm_turn_end."""
        fired: list[str] = []

        class _Recorder(SwarmHooks[None]):
            @override
            async def on_swarm_turn_end(
                self,
                context: RunContext[None],
                state: SwarmState[None],
                items: list[Any],
            ) -> None:
                del context, state, items
                fired.append("observer")

        sw = _make_swarm(hooks=_Recorder())
        ctx: RunContext[None] = RunContext.make(None)
        cp = InMemorySwarmCheckpointer(thread_id="t2")
        mock_result = _stub_result(sw.entry, ctx)

        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(return_value=mock_result),
        ):
            await run_swarm_loop(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                checkpointer=cp,
            )

        assert fired == ["observer"], f"observer did not fire: {fired}"
        assert await cp.list_checkpoints() == ["t2"]


class TestSwarmLoopStreamedCheckpointerAutoSave:
    async def test_streamed_checkpointer_saves_after_turn(self) -> None:
        """Supplying a checkpointer to run_swarm_loop_streamed causes it to save."""
        sw = _make_swarm()
        ctx: RunContext[None] = RunContext.make(None)
        cp = InMemorySwarmCheckpointer(thread_id="s1")
        sr: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(user_prompt="go")

        async def _fake_stream(**kwargs: Any) -> RunResult[Any]:
            del kwargs
            return _stub_result(sw.entry, ctx)

        with patch(
            "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
            new=AsyncMock(side_effect=_fake_stream),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=sr,
                checkpointer=cp,
            )

        sr.set_run_task(asyncio.get_running_loop().create_task(asyncio.sleep(0)))
        async for _ in sr.stream_events():
            pass

        saved = await cp.list_checkpoints()
        assert saved == ["s1"], f"expected ['s1'], got {saved}"


class TestSwarmLoopStreamedObserverPreserved:
    async def test_streamed_observer_fires_without_checkpointer(self) -> None:
        """A swarm.hooks observer fires on_swarm_turn_end in the streamed path."""
        fired: list[str] = []

        class _Recorder(SwarmHooks[None]):
            @override
            async def on_swarm_turn_end(
                self,
                context: RunContext[None],
                state: SwarmState[None],
                items: list[Any],
            ) -> None:
                del context, state, items
                fired.append("on_swarm_turn_end")

        sw = _make_swarm(hooks=_Recorder())
        ctx: RunContext[None] = RunContext.make(None)
        sr: SwarmRunResultStreaming[None] = SwarmRunResultStreaming(user_prompt="go")

        async def _fake_stream(**kwargs: Any) -> RunResult[Any]:
            del kwargs
            return _stub_result(sw.entry, ctx)

        with patch(
            "troopai.adk.run.swarm_loop_streamed._stream_member_turn",
            new=AsyncMock(side_effect=_fake_stream),
        ):
            await run_swarm_loop_streamed(
                swarm=sw,
                user_prompt="go",
                ctx_wrapper=ctx,
                hooks=RunHooks(),
                config=DEFAULT_RUN_CONFIG,
                result=sr,
            )

        sr.set_run_task(asyncio.get_running_loop().create_task(asyncio.sleep(0)))
        async for _ in sr.stream_events():
            pass

        assert fired == ["on_swarm_turn_end"], f"observer did not fire: {fired}"


__all__: list[str] = []
