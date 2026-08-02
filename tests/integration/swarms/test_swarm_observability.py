"""End-to-end OTel span tree for swarm runs.

Exercises Runner.arun_swarm and arun_swarm_from_checkpoint with a real
in-memory OTel exporter to assert:

- One root swarm.<id> span per invocation.
- One swarm.turn.<n> span per iteration that runs a member turn.
- Stable troopai.swarm.id across suspend/resume.
- resume_attempt set only on the resumed turn span.
- Zero spans when config.tracing_enabled is False.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs.interrupt import Interrupt, InterruptException
from troopai.adk.run.config import RunConfig
from troopai.adk.run.context import RunContext
from troopai.adk.run.runner import Runner
from troopai.adk.swarms.checkpointer import SwarmCheckpoint
from troopai.adk.swarms.checkpointers.in_memory import InMemorySwarmCheckpointer
from troopai.adk.swarms.interrupt import SwarmResume
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination
from troopai.adk.tracing.otel.otel_tracer import OTelTracer
from troopai.adk.tracing.tracer import set_tracer
from troopai.adk.types.run.run_result import RunResult


@pytest.fixture
def otel_exporter() -> Iterator[InMemorySpanExporter]:
    """Wire an OTel exporter + matching OTelTracer for the test.

    Mirrors the fixture from ``tests/unit/tracing/otel/test_swarm_attributes.py``.
    The ``set_tracer(None)`` teardown restores the NoOpTracer default so
    subsequent tests do not leak the exporter binding.
    """
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    set_tracer(OTelTracer(provider=provider, service_name="troopai-adk-python-test"))
    yield exporter
    exporter.clear()
    set_tracer(None)


def _make_swarm(*, max_turns: int = 1) -> Swarm[Any]:
    member = Agent(name="approver", system_prompt="x")
    return Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(max_turns),
    )


class TestSwarmObservabilityHappyPath:
    async def test_one_turn_run_produces_root_plus_one_turn_span(
        self,
        otel_exporter: InMemorySpanExporter,
    ) -> None:
        sw = _make_swarm()

        async def _fake_run_agent_loop(**kwargs: Any) -> RunResult[Any]:
            ctx: RunContext[Any] = kwargs["ctx_wrapper"]
            return RunResult(
                final_output="ok",
                user_prompt="",
                new_items=[],
                context=ctx,
                last_agent=sw.entry,
            )

        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(side_effect=_fake_run_agent_loop),
        ):
            result = await Runner.arun_swarm(sw, "go", run_config=RunConfig(tracing_enabled=True))

        spans = otel_exporter.get_finished_spans()
        roots = [s for s in spans if s.name.startswith("swarm.") and not s.name.startswith("swarm.turn.")]
        turns = [s for s in spans if s.name.startswith("swarm.turn.")]
        assert len(roots) == 1
        assert len(turns) == 1

        assert result.state is not None
        swarm_id = result.state.swarm_id
        assert swarm_id is not None
        assert roots[0].name == f"swarm.{swarm_id}"
        assert turns[0].name == "swarm.turn.1"

        root_attrs = roots[0].attributes or {}
        turn_attrs = turns[0].attributes or {}
        assert root_attrs.get("troopai.swarm.id") == swarm_id
        assert root_attrs.get("troopai.swarm.entry") == "approver"
        assert root_attrs.get("troopai.swarm.status") == "max_turns"
        assert root_attrs.get("troopai.swarm.turns_total") == 1
        assert turn_attrs.get("troopai.swarm.id") == swarm_id
        assert turn_attrs.get("troopai.swarm.turn.index") == 1
        assert turn_attrs.get("troopai.swarm.turn.member") == "approver"
        assert turn_attrs.get("troopai.swarm.turn.status") == "success"


class TestSwarmObservabilitySuspendResume:
    async def test_suspend_then_resume_share_swarm_id(
        self,
        otel_exporter: InMemorySpanExporter,
    ) -> None:
        # max_turns=2: leaves room for one suspended turn + one resumed turn
        # before MaxTurnsTermination fires on the resumed invocation.
        sw = _make_swarm(max_turns=2)
        interrupt = Interrupt(node_id="approver", question="ok?", kind="generic")

        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(side_effect=InterruptException(interrupt)),
        ):
            first = await Runner.arun_swarm(sw, "go", run_config=RunConfig(tracing_enabled=True))

        assert first.state is not None
        original_swarm_id = first.state.swarm_id
        assert original_swarm_id is not None

        cp = InMemorySwarmCheckpointer(thread_id="thr-1")
        await cp.save(
            SwarmCheckpoint(
                thread_id="thr-1",
                state=dict(first.state.to_dict()),
                turn=first.state.total_turns,
            )
        )

        async def _fake_resumed_run_agent_loop(**kwargs: Any) -> RunResult[Any]:
            ctx: RunContext[Any] = kwargs["ctx_wrapper"]
            return RunResult(
                final_output="ok",
                user_prompt="",
                new_items=[],
                context=ctx,
                last_agent=sw.entry,
            )

        # Patch both modules: swarm_loop.run_agent_loop is the fresh-turn
        # path, swarm_resume.run_agent_loop is the HITL-pure helper
        # which imports the symbol at its own module level.
        with (
            patch(
                "troopai.adk.run.swarm_loop.run_agent_loop",
                new=AsyncMock(side_effect=_fake_resumed_run_agent_loop),
            ),
            patch(
                "troopai.adk.run.swarm_resume.run_agent_loop",
                new=AsyncMock(side_effect=_fake_resumed_run_agent_loop),
            ),
        ):
            await Runner.arun_swarm_from_checkpoint(
                sw,
                checkpointer=cp,
                thread_id="thr-1",
                resume=SwarmResume(replies={"approver": "yes"}),
                run_config=RunConfig(tracing_enabled=True),
            )

        spans = otel_exporter.get_finished_spans()
        roots = [s for s in spans if s.name.startswith("swarm.") and not s.name.startswith("swarm.turn.")]
        turns = [s for s in spans if s.name.startswith("swarm.turn.")]
        # Two root spans (one per invocation), both keyed by the same swarm_id;
        # the suspend side opens one turn span that closes with
        # status="interrupted", the resume side opens one more turn span
        # that closes with status="success".
        assert len(roots) == 2
        assert len(turns) == 2
        ids = {(s.attributes or {}).get("troopai.swarm.id") for s in roots}
        assert ids == {original_swarm_id}

        # The resumed turn span carries resume_attempt=1.
        resumed_turns = [s for s in turns if (s.attributes or {}).get("troopai.swarm.turn.resume_attempt") == 1]
        assert len(resumed_turns) == 1


class TestSwarmObservabilityDisabled:
    async def test_tracing_disabled_produces_zero_spans(
        self,
        otel_exporter: InMemorySpanExporter,
    ) -> None:
        sw = _make_swarm()

        async def _fake_run_agent_loop(**kwargs: Any) -> RunResult[Any]:
            ctx: RunContext[Any] = kwargs["ctx_wrapper"]
            return RunResult(
                final_output="ok",
                user_prompt="",
                new_items=[],
                context=ctx,
                last_agent=sw.entry,
            )

        with patch(
            "troopai.adk.run.swarm_loop.run_agent_loop",
            new=AsyncMock(side_effect=_fake_run_agent_loop),
        ):
            await Runner.arun_swarm(sw, "go", run_config=RunConfig(tracing_enabled=False))

        assert otel_exporter.get_finished_spans() == ()
