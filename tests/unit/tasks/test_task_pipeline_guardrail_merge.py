"""Unit tests for guardrail merging in :meth:`Runner.arun_task`.

Task input/output guardrails MUST be APPENDED after the caller's
``RunConfig.guardrails`` lists — run-scope first, task-scope second.
Duplicates are NOT de-duplicated.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

from troopai.adk.agents.agent import Agent
from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentGuardrails,
    AgentInputGuardrail,
    AgentInputGuardrailData,
    AgentOutputGuardrail,
    AgentOutputGuardrailData,
)
from troopai.adk.run.config import RunConfig
from troopai.adk.run.context import RunContext
from troopai.adk.tasks import Task
from troopai.adk.types.run.run_result import RunResult


def _agent() -> Agent:
    return Agent(name="t", system_prompt="x")


def _passing_input_guardrail(name: str) -> AgentInputGuardrail[Any]:
    async def _fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
        return AgentGuardrailFunctionOutput(tripwire_triggered=False)

    return AgentInputGuardrail(guardrail_function=_fn, name=name)


def _passing_output_guardrail(name: str) -> AgentOutputGuardrail[Any]:
    async def _fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
        return AgentGuardrailFunctionOutput(tripwire_triggered=False)

    return AgentOutputGuardrail(guardrail_function=_fn, name=name)


async def test_task_guardrails_appended_after_run_config_guardrails() -> None:
    """The transient RunConfig passed to inner arun MUST have
    run-config guardrails FIRST, task-scope guardrails SECOND."""
    run_g_in = _passing_input_guardrail("run_in")
    run_g_out = _passing_output_guardrail("run_out")
    task_g_in = _passing_input_guardrail("task_in")
    task_g_out = _passing_output_guardrail("task_out")

    base_config = RunConfig(
        guardrails=AgentGuardrails(input=[run_g_in], output=[run_g_out]),
    )

    task = Task(
        description="x",
        agent=_agent(),
        guardrails=AgentGuardrails(input=[task_g_in], output=[task_g_out]),
    )

    captured: dict[str, Any] = {}

    async def fake_arun(*_a: Any, **kw: Any) -> RunResult:
        captured["run_config"] = kw["run_config"]
        return RunResult(
            final_output="ok",
            user_prompt="ignored",
            new_items=[],
            context=RunContext.make(None),
        )

    from troopai.adk.run.runner import Runner

    with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake_arun)):
        await Runner.arun_task(task, run_config=base_config)

    transient_cfg: RunConfig = captured["run_config"]
    assert [g.get_name() for g in transient_cfg.guardrails.input] == [
        "run_in",
        "task_in",
    ]
    assert [g.get_name() for g in transient_cfg.guardrails.output] == [
        "run_out",
        "task_out",
    ]


async def test_no_replace_when_no_task_overrides() -> None:
    """Empty task overrides do not alter the caller's RunConfig values."""
    base_config = RunConfig(verbose=None)

    task = Task(description="x", agent=_agent())
    captured: dict[str, Any] = {}

    async def fake_arun(*_a: Any, **kw: Any) -> RunResult:
        captured["run_config"] = kw["run_config"]
        return RunResult(
            final_output="ok",
            user_prompt="ignored",
            new_items=[],
            context=RunContext.make(None),
        )

    from troopai.adk.run.runner import Runner

    with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake_arun)):
        await Runner.arun_task(task, run_config=base_config)

    transient_cfg: RunConfig = captured["run_config"]
    assert transient_cfg is not base_config
    assert transient_cfg.guardrails.input == base_config.guardrails.input
    assert transient_cfg.guardrails.output == base_config.guardrails.output
    transient_cfg.guardrails.input.append(_passing_input_guardrail("leaked"))
    assert base_config.guardrails.input == []


async def test_duplicate_guardrails_not_deduplicated() -> None:
    """A guardrail appearing in both run-config and task-scope MUST
    appear TWICE in the merged list — no de-duplication."""
    shared = _passing_input_guardrail("shared")
    base_config = RunConfig(guardrails=AgentGuardrails(input=[shared]))
    task = Task(description="x", agent=_agent(), guardrails=AgentGuardrails(input=[shared]))

    captured: dict[str, Any] = {}

    async def fake_arun(*_a: Any, **kw: Any) -> RunResult:
        captured["run_config"] = kw["run_config"]
        return RunResult(
            final_output="ok",
            user_prompt="ignored",
            new_items=[],
            context=RunContext.make(None),
        )

    from troopai.adk.run.runner import Runner

    with patch.object(Runner, "arun", new=AsyncMock(side_effect=fake_arun)):
        await Runner.arun_task(task, run_config=base_config)

    assert len(captured["run_config"].guardrails.input) == 2
