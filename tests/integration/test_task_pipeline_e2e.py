"""End-to-end integration test for the Task abstraction.

Drives :meth:`Runner.arun_task` and :meth:`Runner.arun_task_pipeline`
against the real agent loop with a stubbed LLM. Verifies:

- Two-call explicit chaining: the developer feeds the upstream task's
  output into the downstream task's description at construction time.
  The framework does NOT rewrite prompts at runtime —
  ``Task.description`` is what the agent sees verbatim.
- ``TaskPipelineResult.final_output`` is the last non-skipped task's
  final output.
- Pipeline halts on task exception, returning a partial result with
  ``TaskOutput.error`` set on the failing task — the exception is
  NOT re-raised by ``arun_task_pipeline``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.tasks import Task, TaskPipeline
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseText,
)


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        response_id="resp-x",
        model="fake",
        response=[LLMResponseText(text=text)],
    )


def _last_user_text(messages: Any) -> str:
    """Extract the last user message text from a Layer-1 message list."""
    if not isinstance(messages, list):
        return ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            content = msg.get("content")
            return content if isinstance(content, str) else str(content)
    return ""


@pytest.mark.integration
async def test_explicit_chaining_via_two_arun_task_calls() -> None:
    """The developer constructs the downstream task with a
    description that embeds the upstream task's output. Each task's
    ``description`` is fed verbatim — the framework never rewrites
    prompts at runtime."""
    agent = Agent(name="t", system_prompt="You are helpful.")

    seen_prompts: list[str] = []

    async def fake_call_llm(*_args: Any, **kwargs: Any) -> LLMResponse:
        messages = kwargs.get("messages") or _args[1]
        text = _last_user_text(messages)
        seen_prompts.append(text)
        return _text_response(f"answer-for: {text}")

    with (
        patch("troopai.adk.run.loop.call_llm", new=AsyncMock(side_effect=fake_call_llm)),
        patch("troopai.adk.run.runner.run_blocking_input_guardrails", new=AsyncMock(return_value=[])),
        patch("troopai.adk.run.runner.run_parallel_input_guardrails", new=AsyncMock(return_value=[])),
        patch("troopai.adk.run.runner.run_output_guardrails", new=AsyncMock(return_value=[])),
    ):
        from troopai.adk.run.runner import Runner

        upstream = Task(description="A", agent=agent)
        upstream_out = await Runner.arun_task(upstream)

        downstream = Task(
            description=f"chained-from: {upstream_out.final_output}",
            agent=agent,
        )
        downstream_out = await Runner.arun_task(downstream)

    # First task sees its own description verbatim.
    assert seen_prompts[0] == "A"
    # Second task sees the developer-built description verbatim — the
    # framework didn't transform anything.
    assert seen_prompts[1] == "chained-from: answer-for: A"
    assert upstream_out.final_output == "answer-for: A"
    assert downstream_out.final_output == "answer-for: chained-from: answer-for: A"


@pytest.mark.integration
async def test_pipeline_halts_on_task_exception() -> None:
    agent = Agent(name="t", system_prompt="x")

    call_count = {"n": 0}

    async def fake_call_llm(*_args: Any, **_kwargs: Any) -> LLMResponse:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated agent failure")
        return _text_response("ok")

    with (
        patch("troopai.adk.run.loop.call_llm", new=AsyncMock(side_effect=fake_call_llm)),
        patch("troopai.adk.run.runner.run_blocking_input_guardrails", new=AsyncMock(return_value=[])),
        patch("troopai.adk.run.runner.run_parallel_input_guardrails", new=AsyncMock(return_value=[])),
        patch("troopai.adk.run.runner.run_output_guardrails", new=AsyncMock(return_value=[])),
    ):
        from troopai.adk.run.runner import Runner

        t1 = Task(description="A", agent=agent)
        t2 = Task(description="B", agent=agent)
        t3 = Task(description="C", agent=agent)

        result = await Runner.arun_task_pipeline(TaskPipeline(tasks=(t1, t2, t3)))

    # Pipeline halted at t2 → t3 never ran.
    assert len(result.task_outputs) == 2
    assert result.task_outputs[0].error is None
    assert result.task_outputs[1].error is not None
    assert "RuntimeError" in result.task_outputs[1].error
    # final_output reflects the last successful task only.
    assert result.final_output == "ok"
