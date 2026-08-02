"""V7 smoke test — end-to-end verbose wiring through the Runner.

Exercises the paths V6 instrumented:

* ``emit_turn_start`` / ``emit_turn_end`` at the loop boundaries
* ``emit_usage_recorded`` after each LLM call
* ``emit_cache_miss`` when a tool executes
* ``emit_tool_error`` when a tool raises

Each test installs a :class:`VerboseConfig` on the :class:`RunConfig`,
captures the renderer output on an in-memory stream, and asserts
that the expected event lines appear. The fake LLM pattern is lifted
from ``tests/unit/hooks/test_agent_hooks.py``; see that file for the
original helper rationale.
"""

from __future__ import annotations

import io
from contextlib import ExitStack
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.config import RunConfig
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
    LLMResponseText,
)
from troopai.adk.verbose import VerboseConfig

# ---------------------------------------------------------------------------
# Fake LLM + stack patches (mirrors tests/unit/hooks/test_agent_hooks.py)
# ---------------------------------------------------------------------------


def _fake_text(text: str = "final answer") -> LLMResponse:
    return LLMResponse(
        response_id="test",
        model="fake",
        response=[LLMResponseText(text=text)],
    )


def _fake_text_with_usage(text: str = "final answer") -> LLMResponse:
    """Fake response carrying a non-None usage so ``emit_usage_recorded`` fires."""
    from troopai.adk.types.tokens.llm_usage import LLMUsage

    return LLMResponse(
        response_id="test",
        model="fake",
        response=[LLMResponseText(text=text)],
        usage=LLMUsage(input_tokens=3, output_tokens=2, total_tokens=5),
    )


def _fake_tool_call(tool_name: str, args: str = "{}") -> LLMResponse:
    return LLMResponse(
        response_id="test-tc",
        model="fake",
        response=[
            LLMResponseFunctionToolCall(
                call_id="call-1",
                name=tool_name,
                arguments=args,
            ),
        ],
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


async def _arun(agent: Agent, prompt: str, **kwargs: Any):
    from troopai.adk.run.runner import Runner

    return await Runner.arun(agent, prompt, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _line_config() -> tuple[VerboseConfig, io.StringIO]:
    """Build a ``VerboseConfig`` pinned to line mode + an in-memory sink."""
    sink = io.StringIO()
    cfg = VerboseConfig(mode="line", output=sink, use_color=False)
    return cfg, sink


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunnerVerboseIntegration:
    """End-to-end smoke tests for the V6 runner emitter wiring."""

    @pytest.mark.asyncio
    async def test_turn_and_usage_events_fire(self) -> None:
        """A single-turn text run emits turn.start, turn.end, usage.recorded."""
        cfg, sink = _line_config()
        agent = Agent(name="alpha", system_prompt="hi")

        async def fake_call_llm(_agent, _msgs, _cfg, **_kw):
            del _agent, _msgs, _cfg, _kw
            return _fake_text_with_usage("ok")

        with _patches(fake_call_llm):
            result = await _arun(agent, "hello", run_config=RunConfig(verbose=cfg))

        assert result.final_output == "ok"
        output = sink.getvalue()
        # Runner emits turn boundaries each turn (V6 wiring in loop.py)
        assert "turn" in output
        # Usage recorded fires after every LLM call that returns usage
        assert "usage" in output

    @pytest.mark.asyncio
    async def test_tool_cache_miss_event_fires(self) -> None:
        """A tool call with no cache configured emits cache.miss."""
        cfg, sink = _line_config()

        call_count = {"n": 0}

        async def _echo_handler(_ctx, _raw_args):
            del _ctx, _raw_args
            return "42"

        echo = FunctionTool(
            name="echo",
            description="echo tool",
            schema={"type": "object", "properties": {}},
            on_invoke=_echo_handler,
            cache=True,  # enables cache miss emission
        )

        agent = Agent(name="alpha", system_prompt="hi", tools=[echo])

        async def fake_call_llm(_agent, _msgs, _cfg, **_kw):
            del _agent, _msgs, _cfg, _kw
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _fake_tool_call("echo")
            return _fake_text("done")

        with _patches(fake_call_llm):
            result = await _arun(agent, "call echo", run_config=RunConfig(verbose=cfg))

        assert result.final_output == "done"
        output = sink.getvalue()
        # Tool start/end line
        assert "tool" in output
        # Cache miss logged (first invocation of a cache-enabled tool)
        assert "cache" in output

    @pytest.mark.asyncio
    async def test_tool_error_event_fires(self) -> None:
        """A tool that raises surfaces a tool.error event before propagating.

        The runner is configured with ``fail_on_tool_error=True`` so the
        exception propagates after the verbose layer captures it. We
        assert the V6 ``emit_tool_error`` call wrote something to the
        stream before the exception bubbles up.
        """
        cfg, sink = _line_config()

        async def _boom_handler(_ctx, _raw_args):
            del _ctx, _raw_args
            raise ValueError("deliberate")

        boom = FunctionTool(
            name="boom",
            description="always fails",
            schema={"type": "object", "properties": {}},
            on_invoke=_boom_handler,
            max_retries=0,
        )

        agent = Agent(name="alpha", system_prompt="hi", tools=[boom])

        call_count = {"n": 0}

        async def fake_call_llm(_agent, _msgs, _cfg, **_kw):
            del _agent, _msgs, _cfg, _kw
            call_count["n"] += 1
            if call_count["n"] == 1:
                return _fake_tool_call("boom")
            return _fake_text("recovered")

        with _patches(fake_call_llm), pytest.raises(ValueError, match="deliberate"):
            await _arun(
                agent,
                "call boom",
                run_config=RunConfig(verbose=cfg, fail_on_tool_error=True),
            )

        output = sink.getvalue()
        # V6 emit_tool_error fired before the raise — the panel must
        # surface both the failing tool name and the exception type so
        # operators can distinguish error paths from successful tool
        # calls. ``tool`` alone would match ``on_tool_start`` and would
        # not prove the error close-block ran.
        assert "boom" in output
        assert "failed" in output.lower()
        assert "ValueError" in output

    @pytest.mark.asyncio
    async def test_auto_mode_is_safe_in_non_tty(self) -> None:
        """``mode='auto'`` downgrades to line renderer when stdout is an ``io.StringIO``.

        Pins the promise that auto mode never crashes in CI, piped, or
        file-redirected runs because no TTY is attached to the explicit
        ``output`` stream passed to :class:`VerboseConfig`.
        """
        sink = io.StringIO()
        cfg = VerboseConfig(mode="auto", output=sink, use_color=False)
        agent = Agent(name="alpha", system_prompt="hi")

        async def fake_call_llm(_agent, _msgs, _cfg, **_kw):
            del _agent, _msgs, _cfg, _kw
            return _fake_text("ok")

        with _patches(fake_call_llm):
            result = await _arun(agent, "hello", run_config=RunConfig(verbose=cfg))

        assert result.final_output == "ok"
        output = sink.getvalue()
        # Output MUST exist (the line renderer kicked in), not be Rich panels
        assert len(output) > 0
        # Rich's border drawing characters MUST NOT appear in line mode
        assert "╭" not in output
        assert "╰" not in output

    @pytest.mark.asyncio
    async def test_verbose_disabled_emits_nothing(self) -> None:
        """``enabled=False`` + any mode is a total no-op."""
        sink = io.StringIO()
        cfg = VerboseConfig(enabled=False, output=sink, use_color=False)
        agent = Agent(name="alpha", system_prompt="hi")

        async def fake_call_llm(_agent, _msgs, _cfg, **_kw):
            del _agent, _msgs, _cfg, _kw
            return _fake_text("ok")

        with _patches(fake_call_llm):
            result = await _arun(agent, "hello", run_config=RunConfig(verbose=cfg))

        assert result.final_output == "ok"
        assert len(sink.getvalue()) == 0
