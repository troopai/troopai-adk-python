"""Unit tests for :attr:`Task.output_schema` per-call override.

The runner MUST build a transient agent via
``dataclasses.replace(agent, output_schema=task.output_schema)`` for
each call — the original ``Agent`` definition is untouched.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel

from troopai.adk.agents.agent import Agent
from troopai.adk.run.context import RunContext
from troopai.adk.tasks import Task
from troopai.adk.types.run.run_result import RunResult


class Summary(BaseModel):
    """Test schema."""

    headline: str


def _agent_with_schema(schema: Any = None) -> Agent:
    return Agent(name="t", system_prompt="x", output_schema=schema)


async def test_task_schema_overrides_agent_schema() -> None:
    """When Task.output_schema is set, the inner arun receives an
    Agent whose output_schema is the task's, not the original's."""
    original_agent = _agent_with_schema(None)  # agent has NO schema
    task = Task(description="x", agent=original_agent, output_schema=Summary)

    captured: dict[str, Any] = {}

    async def fake_arun(agent: Agent, *_a: Any, **_kw: Any) -> RunResult:
        captured["agent"] = agent
        return RunResult(
            final_output="ok",
            user_prompt="ignored",
            new_items=[],
            context=RunContext.make(None),
        )

    from troopai.adk.run.runner import Runner

    with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake_arun)):
        await Runner.arun_task(task)

    effective_agent: Agent = captured["agent"]
    assert effective_agent.output_schema is Summary
    # Original is unchanged
    assert original_agent.output_schema is None


async def test_no_override_passes_original_agent_unchanged() -> None:
    """When Task.output_schema is None, the inner arun receives the
    original Agent reference — no replace."""
    original_agent = _agent_with_schema(Summary)
    task = Task(description="x", agent=original_agent)  # no schema override

    captured: dict[str, Any] = {}

    async def fake_arun(agent: Agent, *_a: Any, **_kw: Any) -> RunResult:
        captured["agent"] = agent
        return RunResult(
            final_output="ok",
            user_prompt="ignored",
            new_items=[],
            context=RunContext.make(None),
        )

    from troopai.adk.run.runner import Runner

    with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake_arun)):
        await Runner.arun_task(task)

    # Exact identity — no replace made a new instance.
    assert captured["agent"] is original_agent
