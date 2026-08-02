"""Verify Runner.arun_swarm / arun_swarm_from_checkpoint open a swarm_span.

The runner opens a typed swarm_span keyed by a UUID; the same UUID is
reused on resume so troopai.swarm.id correlates suspend and resume sides
of one logical run. A checkpoint missing the swarm_id triggers a
warning log and regeneration.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.runner import Runner
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination


def _swarm() -> Swarm:
    m = Agent(name="m", system_prompt="x")
    return Swarm(
        members=(m,),
        entry=m,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(1),
    )


def _make_mock_span() -> MagicMock:
    span = MagicMock()
    span.data.data = {}
    return span


class TestArunSwarmOpensSwarmSpan:
    async def test_swarm_id_is_a_uuid_and_stamped_on_state(self) -> None:
        sw = _swarm()
        captured_span_kwargs: list[dict[str, Any]] = []

        def _track_swarm_span(**kwargs: Any) -> MagicMock:
            captured_span_kwargs.append(kwargs)
            return _make_mock_span()

        async def _fake_run_swarm_loop(**kwargs: Any) -> Any:
            from troopai.adk.swarms.result import SwarmRunResult
            from troopai.adk.swarms.state import SwarmState
            from troopai.adk.swarms.stop_reason import StopReason

            state = SwarmState(
                swarm=sw,
                current_agent=sw.entry,
                current_agent_name=sw.entry.name,
            )
            state.swarm_id = kwargs.get("swarm_id")
            state.total_turns = 1
            return SwarmRunResult(
                final_output=None,
                stop_reason=StopReason(kind="max_turns", detail=""),
                user_prompt="go",
                state=state,
                last_agent=sw.entry,
            )

        with (
            patch("troopai.adk.run.runner.swarm_span", side_effect=_track_swarm_span),
            patch(
                "troopai.adk.run.swarm_loop.run_swarm_loop",
                new=AsyncMock(side_effect=_fake_run_swarm_loop),
            ),
        ):
            result = await Runner.arun_swarm(sw, "go")

        # swarm_span was called once with a UUID-shaped swarm_id
        assert len(captured_span_kwargs) == 1
        swarm_id = captured_span_kwargs[0]["swarm_id"]
        assert isinstance(swarm_id, str)
        assert len(swarm_id) == 36  # UUID4 format with dashes
        # And the same id was threaded into run_swarm_loop, persisted on state
        assert result.state is not None
        assert result.state.swarm_id == swarm_id


class TestArunSwarmFromCheckpointReusesSwarmId:
    async def test_loaded_swarm_id_is_reused(self) -> None:
        from troopai.adk.swarms.checkpointer import SwarmCheckpoint
        from troopai.adk.swarms.checkpointers.in_memory import (
            InMemorySwarmCheckpointer,
        )
        from troopai.adk.swarms.state import SwarmState

        sw = _swarm()
        cp = InMemorySwarmCheckpointer(thread_id="thr-1")
        state = SwarmState(
            swarm=sw,
            current_agent=sw.entry,
            current_agent_name=sw.entry.name,
        )
        state.swarm_id = "previously-generated-id"
        await cp.save(
            SwarmCheckpoint(
                thread_id="thr-1",
                state=dict(state.to_dict()),
                turn=0,
            )
        )

        captured_span_kwargs: list[dict[str, Any]] = []

        def _track_swarm_span(**kwargs: Any) -> MagicMock:
            captured_span_kwargs.append(kwargs)
            return _make_mock_span()

        async def _fake_run_swarm_loop(**kwargs: Any) -> Any:
            from troopai.adk.swarms.result import SwarmRunResult
            from troopai.adk.swarms.stop_reason import StopReason

            loaded = kwargs["initial_state"]
            return SwarmRunResult(
                final_output=None,
                stop_reason=StopReason(kind="max_turns", detail=""),
                user_prompt="go",
                state=loaded,
                last_agent=sw.entry,
            )

        with (
            patch("troopai.adk.run.runner.swarm_span", side_effect=_track_swarm_span),
            patch(
                "troopai.adk.run.swarm_loop.run_swarm_loop",
                new=AsyncMock(side_effect=_fake_run_swarm_loop),
            ),
        ):
            await Runner.arun_swarm_from_checkpoint(sw, checkpointer=cp, thread_id="thr-1")

        assert captured_span_kwargs[0]["swarm_id"] == "previously-generated-id"

    async def test_missing_swarm_id_regenerates_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """A checkpoint payload missing swarm_id triggers warning + fresh UUID."""
        import logging

        from troopai.adk.swarms.checkpointer import SwarmCheckpoint
        from troopai.adk.swarms.checkpointers.in_memory import (
            InMemorySwarmCheckpointer,
        )
        from troopai.adk.swarms.state import SwarmState

        sw = _swarm()
        cp = InMemorySwarmCheckpointer(thread_id="thr-2")
        state = SwarmState(
            swarm=sw,
            current_agent=sw.entry,
            current_agent_name=sw.entry.name,
        )
        # Save WITHOUT a swarm_id to drive the regenerate branch.
        payload = dict(state.to_dict())
        payload.pop("swarm_id", None)
        await cp.save(SwarmCheckpoint(thread_id="thr-2", state=payload, turn=0))

        captured_span_kwargs: list[dict[str, Any]] = []

        def _track_swarm_span(**kwargs: Any) -> MagicMock:
            captured_span_kwargs.append(kwargs)
            return _make_mock_span()

        async def _fake_run_swarm_loop(**kwargs: Any) -> Any:
            from troopai.adk.swarms.result import SwarmRunResult
            from troopai.adk.swarms.stop_reason import StopReason

            loaded = kwargs["initial_state"]
            return SwarmRunResult(
                final_output=None,
                stop_reason=StopReason(kind="max_turns", detail=""),
                user_prompt="go",
                state=loaded,
                last_agent=sw.entry,
            )

        with (
            patch("troopai.adk.run.runner.swarm_span", side_effect=_track_swarm_span),
            patch(
                "troopai.adk.run.swarm_loop.run_swarm_loop",
                new=AsyncMock(side_effect=_fake_run_swarm_loop),
            ),
            caplog.at_level(logging.WARNING, logger="troopai.adk.run.runner"),
        ):
            result = await Runner.arun_swarm_from_checkpoint(sw, checkpointer=cp, thread_id="thr-2")

        # A fresh UUID was generated (different from any prior value).
        regenerated_swarm_id = captured_span_kwargs[0]["swarm_id"]
        assert isinstance(regenerated_swarm_id, str)
        assert len(regenerated_swarm_id) == 36
        # And a warning was logged.
        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("swarm_id" in m for m in warning_messages)
        # And the regenerated id is stamped back onto returned state, so
        # the next persisted checkpoint will carry it forward.
        assert result.state is not None
        assert result.state.swarm_id == regenerated_swarm_id
