"""Tests that ``Runner.arun`` disposes ``Toolset`` instances on
``agent.tools`` after the run completes.

The disposal hook is the load-bearing seam of the auto-managed
lifecycle: without it, ``MCPToolset`` would leak subprocesses on
every run. These tests use a minimal ``Toolset`` subclass to prove
``adispose`` is invoked across both success and exception paths.

LLM mocking pattern adapted from ``tests/unit/run/test_runner_tracing.py``
— patch ``troopai.adk.run.loop.call_llm`` plus the three guardrail
runners so the agent loop completes with a deterministic final-text
response and no real provider call.
"""

from __future__ import annotations

import logging
from typing import Any, override
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.runner import Runner
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.tools.toolsets.abstract import Toolset
from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText


class _RecordingToolset(Toolset):
    """A toolset that records calls to ``get_tools`` and ``adispose``."""

    def __init__(self) -> None:
        self.get_tools_calls = 0
        self.adispose_calls = 0

    @override
    async def get_tools(self, ctx: Any = None) -> dict[str, FunctionTool]:
        self.get_tools_calls += 1
        return {}

    @override
    async def adispose(self) -> None:
        self.adispose_calls += 1


def _final_text_response() -> LLMResponse:
    return LLMResponse(
        response_id="resp-1",
        model="fake",
        response=[LLMResponseText(text="done")],
    )


def _patches_for_quiet_run(call_llm_mock: AsyncMock) -> Any:
    """Return the three patches needed to drive ``Runner.arun`` without
    a real LLM provider or any guardrails.
    """
    return (
        patch("troopai.adk.run.loop.call_llm", new=call_llm_mock),
        patch(
            "troopai.adk.run.runner.run_blocking_input_guardrails",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "troopai.adk.run.runner.run_parallel_input_guardrails",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "troopai.adk.run.runner.run_output_guardrails",
            new=AsyncMock(return_value=[]),
        ),
    )


async def test_arun_disposes_toolset_on_success() -> None:
    ts = _RecordingToolset()
    agent = Agent(name="t", system_prompt="p", tools=[ts])

    call_llm = AsyncMock(return_value=_final_text_response())
    patches = _patches_for_quiet_run(call_llm)
    with patches[0], patches[1], patches[2], patches[3]:
        await Runner.arun(agent, "hi")

    assert ts.adispose_calls == 1


async def test_arun_disposes_toolset_on_exception() -> None:
    """Even when the run blows up, ``adispose`` MUST run."""
    ts = _RecordingToolset()
    agent = Agent(name="t", system_prompt="p", tools=[ts])

    call_llm = AsyncMock(side_effect=RuntimeError("boom"))
    patches = _patches_for_quiet_run(call_llm)
    with patches[0], patches[1], patches[2], patches[3], pytest.raises(RuntimeError):
        await Runner.arun(agent, "hi")

    assert ts.adispose_calls == 1


async def test_arun_continues_disposal_after_one_toolset_raises(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A failing ``adispose`` must not block other toolsets' cleanup."""

    class _BadToolset(Toolset):
        @override
        async def get_tools(self, ctx: Any = None) -> dict[str, FunctionTool]:
            return {}

        @override
        async def adispose(self) -> None:
            raise RuntimeError("first one explodes")

    bad = _BadToolset()
    good = _RecordingToolset()
    agent = Agent(name="t", system_prompt="p", tools=[bad, good])

    call_llm = AsyncMock(return_value=_final_text_response())
    patches = _patches_for_quiet_run(call_llm)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        caplog.at_level(logging.WARNING, logger="troopai.adk.run.runner"),
    ):
        await Runner.arun(agent, "hi")

    assert good.adispose_calls == 1
    assert any("adispose() raised" in rec.message for rec in caplog.records)


async def test_default_toolset_adispose_is_a_noop() -> None:
    """Pre-existing toolsets that do not override ``adispose`` MUST work."""

    class _Plain(Toolset):
        @override
        async def get_tools(self, ctx: Any = None) -> dict[str, FunctionTool]:
            return {}

    ts = _Plain()
    await ts.adispose()  # The base ABC's no-op default
    await ts.adispose()  # Idempotent


async def test_arun_disposes_toolsets_in_reverse_registration_order() -> None:
    """Disposal order MUST be LIFO so anyio cancel scopes opened later
    by ``MCPToolset.get_tools`` close before earlier ones.

    Regression guard: ``_dispose_agent_toolsets`` originally iterated
    ``agent.tools`` in FIFO order, which raised
    ``RuntimeError: Attempted to exit a cancel scope ...`` whenever
    two ``MCPServerStdio`` instances were composed in a single agent.
    """
    order: list[str] = []

    class _OrderedToolset(Toolset):
        def __init__(self, label: str) -> None:
            self.label = label

        @override
        async def get_tools(self, ctx: Any = None) -> dict[str, FunctionTool]:
            return {}

        @override
        async def adispose(self) -> None:
            order.append(self.label)

    a = _OrderedToolset("a")
    b = _OrderedToolset("b")
    c = _OrderedToolset("c")
    agent = Agent(name="t", system_prompt="p", tools=[a, b, c])

    call_llm = AsyncMock(return_value=_final_text_response())
    patches = _patches_for_quiet_run(call_llm)
    with patches[0], patches[1], patches[2], patches[3]:
        await Runner.arun(agent, "hi")

    assert order == ["c", "b", "a"], (
        f"Disposal must run in REVERSE registration order to honour the anyio LIFO cancel-scope invariant; got {order}"
    )
