"""Verify Runner.arun_swarm_streamed entry-point shape.

The runner must return a SwarmRunResultStreaming immediately, open
a swarm_span keyed by a generated UUID (or reused on resume), and
schedule the background driver task.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.runner import Runner
from troopai.adk.swarms.events import SwarmDoneEvent, SwarmStartEvent
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.result import SwarmRunResultStreaming
from troopai.adk.swarms.stop_reason import StopReason
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination


def _swarm() -> Swarm[Any]:
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


class TestArunSwarmStreamedEntryPoint:
    async def test_returns_streaming_result_with_uuid_swarm_id(self) -> None:
        sw = _swarm()
        captured_span_kwargs: list[dict[str, Any]] = []

        def _track_swarm_span(**kwargs: Any) -> MagicMock:
            captured_span_kwargs.append(kwargs)
            return _make_mock_span()

        async def _fake_driver(**kwargs: Any) -> None:
            res: SwarmRunResultStreaming[Any] = kwargs["result"]
            res.state = None
            res.stop_reason = StopReason(kind="max_turns", detail="")
            await res.put_event(SwarmStartEvent(entry_agent="m", member_names=("m",)))
            await res.put_event(
                SwarmDoneEvent(
                    reason=StopReason(kind="max_turns", detail=""),
                    final_output=None,
                )
            )

        with (
            patch("troopai.adk.run.runner.swarm_span", side_effect=_track_swarm_span),
            patch(
                "troopai.adk.run.swarm_loop_streamed.run_swarm_loop_streamed",
                new=AsyncMock(side_effect=_fake_driver),
            ),
        ):
            result = await Runner.arun_swarm_streamed(sw, "go")
            assert isinstance(result, SwarmRunResultStreaming)

            events: list[Any] = []
            async for ev in result.stream_events():
                events.append(ev)

        # swarm_span called once with a UUID-shaped id.
        assert len(captured_span_kwargs) == 1
        swarm_id = captured_span_kwargs[0]["swarm_id"]
        assert isinstance(swarm_id, str)
        assert len(swarm_id) == 36  # UUID4 dash-separated

        assert len(events) == 2
        assert isinstance(events[0], SwarmStartEvent)
        assert isinstance(events[1], SwarmDoneEvent)


class TestArunSwarmStreamedResumeReusesSwarmId:
    async def test_loaded_swarm_id_is_reused(self) -> None:
        from troopai.adk.swarms.state import SwarmState

        sw = _swarm()
        loaded_state: SwarmState[Any] = SwarmState(
            swarm=sw,
            current_agent=sw.entry,
            current_agent_name=sw.entry.name,
        )
        loaded_state.swarm_id = "previously-generated-id"

        captured_span_kwargs: list[dict[str, Any]] = []

        def _track_swarm_span(**kwargs: Any) -> MagicMock:
            captured_span_kwargs.append(kwargs)
            return _make_mock_span()

        async def _fake_driver(**kwargs: Any) -> None:
            res: SwarmRunResultStreaming[Any] = kwargs["result"]
            res.state = loaded_state
            res.stop_reason = StopReason(kind="max_turns", detail="")

        with (
            patch("troopai.adk.run.runner.swarm_span", side_effect=_track_swarm_span),
            patch(
                "troopai.adk.run.swarm_loop_streamed.run_swarm_loop_streamed",
                new=AsyncMock(side_effect=_fake_driver),
            ),
        ):
            result = await Runner.arun_swarm_streamed(sw, "", initial_state=loaded_state)
            async for _ in result.stream_events():
                pass

        assert len(captured_span_kwargs) == 1
        assert captured_span_kwargs[0]["swarm_id"] == "previously-generated-id"


class TestArunSwarmStreamedMissingSwarmIdRegenerates:
    async def test_missing_swarm_id_regenerates_with_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        import logging

        from troopai.adk.swarms.state import SwarmState

        sw = _swarm()
        loaded_state: SwarmState[Any] = SwarmState(
            swarm=sw,
            current_agent=sw.entry,
            current_agent_name=sw.entry.name,
        )
        # swarm_id intentionally left as None (older payload simulation).
        assert loaded_state.swarm_id is None

        captured_span_kwargs: list[dict[str, Any]] = []

        def _track_swarm_span(**kwargs: Any) -> MagicMock:
            captured_span_kwargs.append(kwargs)
            return _make_mock_span()

        async def _fake_driver(**kwargs: Any) -> None:
            res: SwarmRunResultStreaming[Any] = kwargs["result"]
            res.state = loaded_state
            res.stop_reason = StopReason(kind="max_turns", detail="")

        with (
            patch("troopai.adk.run.runner.swarm_span", side_effect=_track_swarm_span),
            patch(
                "troopai.adk.run.swarm_loop_streamed.run_swarm_loop_streamed",
                new=AsyncMock(side_effect=_fake_driver),
            ),
            caplog.at_level(logging.WARNING, logger="troopai.adk.run.runner"),
        ):
            result = await Runner.arun_swarm_streamed(
                sw,
                "",
                initial_state=loaded_state,
            )
            async for _ in result.stream_events():
                pass

        # A fresh UUID was generated.
        swarm_id = captured_span_kwargs[0]["swarm_id"]
        assert isinstance(swarm_id, str)
        assert len(swarm_id) == 36
        # The regenerated UUID was stamped onto the loaded state.
        assert loaded_state.swarm_id == swarm_id
        # A warning was logged.
        warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
        assert any("swarm_id" in m for m in warning_messages)
