"""Integration tests for the 📋 Task boundary panel.

The task panel brackets every outer ``Runner.arun()`` /
``arun_swarm()`` invocation — it opens before the agent loop and
closes once the loop returns (or an exception propagates). These
tests use the line backend so we can assert on captured text without
requiring a real TTY. The Rich-panel visual is exercised in
``test_panel_renderer.py``.
"""

from __future__ import annotations

import io
from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseText,
)
from troopai.adk.verbose import VerboseConfig


def _fake_text(text: str = "ok") -> LLMResponse:
    return LLMResponse(
        response_id="t",
        model="fake",
        response=[LLMResponseText(text=text)],
    )


def _patches(fake_call_llm: Any) -> ExitStack:
    stack = ExitStack()
    stack.enter_context(
        patch(
            "troopai.adk.run.loop.call_llm",
            new=AsyncMock(side_effect=fake_call_llm),
        )
    )
    stack.enter_context(
        patch(
            "troopai.adk.run.runner.run_blocking_input_guardrails",
            new=AsyncMock(return_value=[]),
        )
    )
    stack.enter_context(
        patch(
            "troopai.adk.run.runner.run_parallel_input_guardrails",
            new=AsyncMock(return_value=[]),
        )
    )
    stack.enter_context(
        patch(
            "troopai.adk.run.runner.run_output_guardrails",
            new=AsyncMock(return_value=[]),
        )
    )
    return stack


def _line_config() -> tuple[VerboseConfig, io.StringIO]:
    sink = io.StringIO()
    cfg = VerboseConfig(mode="line", output=sink, use_color=False)
    return cfg, sink


class TestTaskPanelLifecycle:
    @pytest.mark.asyncio
    async def test_task_started_fires_at_arun_entry(self) -> None:
        cfg, sink = _line_config()
        agent = Agent(name="alpha", system_prompt="hi")

        async def fake_call_llm(_agent, _msgs, _cfg, **_kw):
            del _agent, _msgs, _cfg, _kw
            return _fake_text("done")

        with _patches(fake_call_llm):
            await Runner.arun(agent, "Summarise the docs", run_config=RunConfig(verbose=cfg))

        output = sink.getvalue()
        assert "task started" in output.lower()
        assert "Summarise the docs" in output

    @pytest.mark.asyncio
    async def test_task_completed_fires_on_clean_exit(self) -> None:
        cfg, sink = _line_config()
        agent = Agent(name="alpha", system_prompt="hi")

        async def fake_call_llm(_agent, _msgs, _cfg, **_kw):
            del _agent, _msgs, _cfg, _kw
            return _fake_text("done")

        with _patches(fake_call_llm):
            await Runner.arun(agent, "small task", run_config=RunConfig(verbose=cfg))

        output = sink.getvalue()
        assert "task completed" in output.lower()

    @pytest.mark.asyncio
    async def test_task_failed_fires_on_exception(self) -> None:
        cfg, sink = _line_config()
        agent = Agent(name="alpha", system_prompt="hi")

        async def fake_call_llm(_agent, _msgs, _cfg, **_kw):
            del _agent, _msgs, _cfg, _kw
            raise RuntimeError("boom")

        with _patches(fake_call_llm), pytest.raises(RuntimeError, match="boom"):
            await Runner.arun(agent, "small task", run_config=RunConfig(verbose=cfg))

        output = sink.getvalue()
        assert "task failed" in output.lower()
        assert "RuntimeError" in output

    @pytest.mark.asyncio
    async def test_long_prompt_truncated_in_task_name(self) -> None:
        cfg, sink = _line_config()
        agent = Agent(name="alpha", system_prompt="hi")

        async def fake_call_llm(_agent, _msgs, _cfg, **_kw):
            del _agent, _msgs, _cfg, _kw
            return _fake_text("done")

        long_prompt = "x" * 200
        with _patches(fake_call_llm):
            await Runner.arun(agent, long_prompt, run_config=RunConfig(verbose=cfg))

        output = sink.getvalue()
        # 80-char cap + "..." suffix from _derive_task_name.
        assert "..." in output
        # The original 200-char prompt should NOT appear verbatim.
        assert long_prompt not in output

    @pytest.mark.asyncio
    async def test_no_task_panel_when_verbose_disabled(self) -> None:
        agent = Agent(name="alpha", system_prompt="hi")

        async def fake_call_llm(_agent, _msgs, _cfg, **_kw):
            del _agent, _msgs, _cfg, _kw
            return _fake_text("done")

        # No verbose config attached to the run.
        with _patches(fake_call_llm):
            result = await Runner.arun(agent, "small task")

        assert result.final_output == "done"
