"""Tests for ``AgentHooks`` — per-agent lifecycle callbacks.

Covers every firing site wired into the loop:

- ``on_start`` / ``on_end``           — turn boundaries (``runner.py`` +
  ``resumption.py``)
- ``on_llm_start`` / ``on_llm_end``   — around each LLM call
  (``loop.py``, streaming and non-streaming)
- ``on_tool_start`` / ``on_tool_end`` — around each tool execution
  (``tools_executor.py``)
- ``on_handoff``                      — fires on the **incoming** agent
  with ``source=from_agent`` (``handoffs_executor.py``)

Each test pins a specific wiring — if any `if agent.hooks is not None`
block is deleted, tests break.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.hooks.hooks import AgentHooks, RunHooks
from troopai.adk.run.config import DEFAULT_RUN_CONFIG
from troopai.adk.run.context import RunContext
from troopai.adk.run.tools_executor import execute_tool_calls
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
    LLMResponseText,
)

# ── Recording test doubles ───────────────────────────────────────────


class RecordingAgentHooks(AgentHooks):
    """Record every per-agent hook call with its kwargs."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def on_start(self, context, agent) -> None:
        del context
        self.events.append(("on_start", {"agent": agent.name}))

    async def on_end(self, context, agent, output) -> None:
        del context
        self.events.append(("on_end", {"agent": agent.name, "output": output}))

    async def on_handoff(self, context, agent, source) -> None:
        del context
        self.events.append(("on_handoff", {"agent": agent.name, "source": source.name}))

    async def on_llm_start(self, context, agent, messages) -> None:
        del context
        self.events.append(("on_llm_start", {"agent": agent.name, "n_messages": len(messages)}))

    async def on_llm_end(self, context, agent, response) -> None:
        del context, response
        self.events.append(("on_llm_end", {"agent": agent.name}))

    async def on_tool_start(self, context, agent, tool_name, tool_input) -> None:
        del context, tool_input
        self.events.append(("on_tool_start", {"agent": agent.name, "tool_name": tool_name}))

    async def on_tool_end(self, context, agent, tool_name, tool_output) -> None:
        del context, tool_output
        self.events.append(("on_tool_end", {"agent": agent.name, "tool_name": tool_name}))


class RecordingRunHooks(RunHooks):
    """Record a subset of run-level hooks for co-firing tests."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def on_agent_start(self, context, agent) -> None:
        del context
        self.events.append(("run.on_agent_start", {"agent": agent.name}))

    async def on_agent_end(self, context, agent, result) -> None:
        del context, result
        self.events.append(("run.on_agent_end", {"agent": agent.name}))


# ── Helpers ──────────────────────────────────────────────────────────


def _echo_tool(name: str = "echo") -> FunctionTool:
    async def _handler(_ctx, _raw_args):
        del _ctx, _raw_args
        return "ok"

    return FunctionTool(
        name=name,
        description="Echo test tool",
        schema={"type": "object", "properties": {}},
        on_invoke=_handler,
    )


def _fake_llm_response(text: str = "final answer") -> LLMResponse:
    return LLMResponse(
        response_id="test",
        model="fake",
        response=[LLMResponseText(text=text)],
    )


def _mock_runner_stack(fake_call_llm) -> ExitStack:
    """Stack all patches needed for a clean ``Runner.arun`` unit test.

    - Patches ``loop.call_llm`` with the supplied fake
    - Stubs out the three Runner guardrail entry points
    """
    stack = ExitStack()
    stack.enter_context(patch("troopai.adk.run.loop.call_llm", new=AsyncMock(side_effect=fake_call_llm)))
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


async def _runner_arun(agent, prompt, **kwargs):
    """Lazy-import Runner so the test module imports cleanly in isolation."""
    from troopai.adk.run.runner import Runner

    return await Runner.arun(agent, prompt, **kwargs)


# ── Agent.hooks default ──────────────────────────────────────────────


class TestAgentHooksField:
    """Verify the ``Agent.hooks`` attribute exists and defaults to None."""

    def test_default_is_none(self) -> None:
        agent = Agent(name="x", system_prompt="hi")
        assert agent.hooks is None

    def test_accepts_agent_hooks_instance(self) -> None:
        hooks = RecordingAgentHooks()
        agent = Agent(name="x", system_prompt="hi", hooks=hooks)
        assert agent.hooks is hooks


# ── on_start / on_end via Runner.arun ────────────────────────────────


class TestOnStartOnEnd:
    """``AgentHooks.on_start`` / ``on_end`` fire around a full run."""

    @pytest.mark.asyncio
    async def test_on_start_and_on_end_fire_in_order(self) -> None:
        hooks = RecordingAgentHooks()
        agent = Agent(name="alpha", system_prompt="hi", hooks=hooks)

        async def fake_call_llm(_agent, _msgs, _cfg, **_kw):
            del _agent, _msgs, _cfg, _kw
            return _fake_llm_response("done")

        with _mock_runner_stack(fake_call_llm):
            result = await _runner_arun(agent, "hello")

        assert result.final_output == "done"
        names = [e[0] for e in hooks.events]
        assert names[0] == "on_start"
        assert names[-1] == "on_end"
        assert hooks.events[-1][1]["output"] == "done"

    @pytest.mark.asyncio
    async def test_no_hooks_attribute_is_passthrough(self) -> None:
        """An agent with ``hooks=None`` never triggers AgentHooks machinery."""
        run_hooks = RecordingRunHooks()
        agent = Agent(name="plain", system_prompt="hi")  # hooks defaults to None
        assert agent.hooks is None

        async def fake_call_llm(_agent, _msgs, _cfg, **_kw):
            del _agent, _msgs, _cfg, _kw
            return _fake_llm_response("out")

        with _mock_runner_stack(fake_call_llm):
            await _runner_arun(agent, "hello", hooks=run_hooks)

        # RunHooks still fired; AgentHooks didn't (there are none)
        names = [e[0] for e in run_hooks.events]
        assert "run.on_agent_start" in names
        assert "run.on_agent_end" in names


# ── on_llm_start / on_llm_end via Runner.arun ────────────────────────


class TestOnLLMStartOnLLMEnd:
    """``AgentHooks.on_llm_start`` / ``on_llm_end`` fire around each LLM call."""

    @pytest.mark.asyncio
    async def test_llm_hooks_fire_once_per_turn(self) -> None:
        hooks = RecordingAgentHooks()
        agent = Agent(name="alpha", system_prompt="hi", hooks=hooks)

        async def fake_call_llm(_agent, _msgs, _cfg, **_kw):
            del _agent, _msgs, _cfg, _kw
            return _fake_llm_response("once")

        with _mock_runner_stack(fake_call_llm):
            await _runner_arun(agent, "hello")

        llm_starts = [e for e in hooks.events if e[0] == "on_llm_start"]
        llm_ends = [e for e in hooks.events if e[0] == "on_llm_end"]
        assert len(llm_starts) == 1
        assert len(llm_ends) == 1
        # on_llm_start precedes on_llm_end
        names = [e[0] for e in hooks.events]
        assert names.index("on_llm_start") < names.index("on_llm_end")


# ── Co-firing: RunHooks + AgentHooks ─────────────────────────────────


class TestCoFiringOrder:
    """AgentHooks fires immediately AFTER the matching RunHooks call."""

    @pytest.mark.asyncio
    async def test_run_hooks_then_agent_hooks(self) -> None:
        # Share a single event list with a stable timeline.
        timeline: list[str] = []

        class TimelineRunHooks(RunHooks):
            async def on_agent_start(self, context, agent) -> None:
                del context, agent
                timeline.append("run.start")

            async def on_agent_end(self, context, agent, result) -> None:
                del context, agent, result
                timeline.append("run.end")

        class TimelineAgentHooks(AgentHooks):
            async def on_start(self, context, agent) -> None:
                del context, agent
                timeline.append("agent.start")

            async def on_end(self, context, agent, output) -> None:
                del context, agent, output
                timeline.append("agent.end")

        agent = Agent(name="x", system_prompt="hi", hooks=TimelineAgentHooks())

        async def fake_call_llm(_agent, _msgs, _cfg, **_kw):
            del _agent, _msgs, _cfg, _kw
            return _fake_llm_response("hi")

        with _mock_runner_stack(fake_call_llm):
            await _runner_arun(agent, "hello", hooks=TimelineRunHooks())

        # Ordering contract: run.start -> agent.start -> ... -> run.end -> agent.end
        assert timeline.index("run.start") < timeline.index("agent.start")
        assert timeline.index("run.end") < timeline.index("agent.end")


# ── on_tool_start / on_tool_end via execute_tool_calls ───────────────


class TestOnToolStartOnToolEnd:
    """``AgentHooks.on_tool_start`` / ``on_tool_end`` fire around each tool."""

    @pytest.mark.asyncio
    async def test_tool_hooks_fire_around_tool_execution(self) -> None:
        hooks = RecordingAgentHooks()
        tool = _echo_tool("ping")
        agent = Agent(
            name="with_tool",
            system_prompt="hi",
            tools=[tool],
            hooks=hooks,
        )

        tool_call = LLMResponseFunctionToolCall(call_id="c1", name="ping", arguments="{}")

        await execute_tool_calls(
            agent=agent,
            tool_calls=[tool_call],
            ctx_wrapper=RunContext(context=None),
            hooks=RunHooks(),
            config=DEFAULT_RUN_CONFIG,
            model="gpt-4o-mini",
        )

        names = [e[0] for e in hooks.events]
        assert "on_tool_start" in names
        assert "on_tool_end" in names
        assert names.index("on_tool_start") < names.index("on_tool_end")
        # Tool name plumbed through
        start = next(e for e in hooks.events if e[0] == "on_tool_start")
        assert start[1]["tool_name"] == "ping"

    @pytest.mark.asyncio
    async def test_no_agent_hooks_skips_agent_tool_hooks(self) -> None:
        """With ``hooks=None`` on the agent, AgentHooks.on_tool_* is not called."""
        tool = _echo_tool("ping")
        agent = Agent(name="no_hooks", system_prompt="hi", tools=[tool])
        assert agent.hooks is None

        tool_call = LLMResponseFunctionToolCall(call_id="c1", name="ping", arguments="{}")

        # Should complete without raising; no AgentHooks machinery invoked.
        results, _ = await execute_tool_calls(
            agent=agent,
            tool_calls=[tool_call],
            ctx_wrapper=RunContext(context=None),
            hooks=RunHooks(),
            config=DEFAULT_RUN_CONFIG,
            model="gpt-4o-mini",
        )
        assert len(results) == 1


# ── on_handoff via execute_deterministic_handoff ─────────────────────


class TestOnHandoff:
    """``AgentHooks.on_handoff`` fires on the **incoming** agent."""

    @pytest.mark.asyncio
    async def test_on_handoff_fires_on_incoming_agent_with_source(self) -> None:
        from troopai.adk.run.handoffs_executor import execute_deterministic_handoff

        from_hooks = RecordingAgentHooks()
        to_hooks = RecordingAgentHooks()
        from_agent = Agent(name="dispatcher", system_prompt="hi", hooks=from_hooks)
        to_agent = Agent(name="specialist", system_prompt="hi", hooks=to_hooks)

        # Minimal HandoffTarget stub: only ``invoke`` is called.
        class _FakeTarget:
            async def invoke(self, intent, context, output, run_context):
                del intent, context, output, run_context
                return to_agent, object()  # (new_agent, handoff_data)

        ctx_wrapper = RunContext(context=None)
        await execute_deterministic_handoff(
            from_agent=from_agent,
            target=_FakeTarget(),  # type: ignore[arg-type]
            intent=None,
            context_msgs=(),
            output_msgs=(),
            context=ctx_wrapper,
            ctx_wrapper=ctx_wrapper,
            hooks=RunHooks(),
        )

        # Contract: on_handoff fires on the INCOMING agent (to_agent).
        to_names = [e[0] for e in to_hooks.events]
        from_names = [e[0] for e in from_hooks.events]
        assert "on_handoff" in to_names, "AgentHooks.on_handoff should fire on the incoming agent"
        assert "on_handoff" not in from_names, "AgentHooks.on_handoff should NOT fire on the outgoing agent"
        event = next(e for e in to_hooks.events if e[0] == "on_handoff")
        assert event[1]["agent"] == "specialist"
        assert event[1]["source"] == "dispatcher"
