"""Regression tests for runner-level HITL / streaming defects.

Covers four confirmed defects in ``runner.py``:

1. ``_inject_memories`` str()-coerced structured (list-of-parts) system
   content, flattening it to a repr; it must append a text part instead.
2. The streamed producer stored the ``CancelledError`` from an
   ``cancel(mode="immediate")`` unconditionally, so the consumer saw a
   spurious error instead of a clean stop; the twin path already guarded
   on ``cancel_mode``.
3. The streamed exception path recovered via ``error_handlers`` while a
   parallel input-guardrail tripwire was ignored; the tripwire must take
   precedence.
4. The sandbox bracket + task panel were opened before the try/finally, so
   a setup-step failure leaked the sandbox session.
"""

from __future__ import annotations

import asyncio
import contextlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.exceptions.exceptions import AgentInputGuardrailTripwireTriggered
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner, _inject_memories
from troopai.adk.run.stream import CancelMode

# ── Helpers ──────────────────────────────────────────────────────────


def _make_agent() -> Agent:
    return Agent(name="test-agent", system_prompt="You are a test agent.")


def _make_input_tripwire(name: str = "gr") -> AgentInputGuardrailTripwireTriggered:
    gr_result = SimpleNamespace(guardrail=SimpleNamespace(get_name=lambda: name))
    return AgentInputGuardrailTripwireTriggered(gr_result)  # type: ignore[arg-type]


class _FakeMemory:
    def __init__(self, contents: list[str]) -> None:
        self._contents = contents

    async def search(self, query: str, *, namespace: Any = None, limit: Any = None) -> list[Any]:
        return [SimpleNamespace(entry=SimpleNamespace(content=c)) for c in self._contents]


def _make_memory_config(position: Any) -> Any:
    return SimpleNamespace(
        memory=_FakeMemory(["user prefers tea"]),
        namespace="ns",
        inject_limit=5,
        inject_position=position,
    )


# ── Finding: _inject_memories preserves structured system content ──────


class TestInjectMemoriesStructuredContent:
    async def test_list_content_gets_text_part_not_str_coercion(self) -> None:
        """A system message whose content is a list of parts must keep its
        list shape with an appended text part, not be str()-flattened."""
        from troopai.adk.memory.memory_config import MemoryInjectionPosition

        effective_input = [
            {"role": "system", "content": [{"type": "input_text", "text": "You are helpful."}]},
            {"role": "user", "content": "hello"},
        ]
        cfg = _make_memory_config(MemoryInjectionPosition.SYSTEM_SUFFIX)

        result = await _inject_memories(effective_input, cfg)

        sys_msg = result[0]
        assert isinstance(sys_msg["content"], list)
        # Original part survives + a new text part carries the memory text.
        assert sys_msg["content"][0] == {"type": "input_text", "text": "You are helpful."}
        assert sys_msg["content"][-1]["type"] == "input_text"
        assert "user prefers tea" in sys_msg["content"][-1]["text"]

    async def test_string_content_still_concatenates(self) -> None:
        """Scalar string content keeps the existing concatenation behavior."""
        from troopai.adk.memory.memory_config import MemoryInjectionPosition

        effective_input = [
            {"role": "system", "content": "Base prompt."},
            {"role": "user", "content": "hello"},
        ]
        cfg = _make_memory_config(MemoryInjectionPosition.SYSTEM_SUFFIX)

        result = await _inject_memories(effective_input, cfg)

        assert isinstance(result[0]["content"], str)
        assert result[0]["content"].startswith("Base prompt.")
        assert "user prefers tea" in result[0]["content"]


# ── Finding: immediate cancel must not store a spurious exception ──────


class TestStreamedImmediateCancelGuard:
    async def test_immediate_cancel_does_not_store_cancellederror(self) -> None:
        """When the producer is cancelled under ``cancel_mode == IMMEDIATE``,
        the CancelledError must NOT be stored — the consumer sees a clean
        completion (mirrors the task-based streamed twin)."""

        async def _cancel_and_raise(**kwargs: Any) -> None:
            result = kwargs["result"]
            result._cancel_mode = CancelMode.IMMEDIATE
            raise asyncio.CancelledError()

        agent = _make_agent()
        config = RunConfig()

        with (
            patch("troopai.adk.run.runner.run_blocking_input_guardrails", new=AsyncMock(return_value=[])),
            patch("troopai.adk.run.runner.run_parallel_input_guardrails", new=AsyncMock(return_value=[])),
            patch("troopai.adk.run.runner.run_agent_loop_streamed", new=_cancel_and_raise),
        ):
            streaming = await Runner.arun(agent, "hi", run_config=config, stream=True)
            # Must NOT re-raise the immediate-cancel CancelledError.
            async for _ in streaming.stream_events():
                pass

        assert streaming._stored_exception is None


# ── Finding: input-guardrail tripwire precedence over error-handler ────


class TestStreamedTripwirePrecedence:
    async def test_parallel_tripwire_wins_over_error_handler_recovery(self) -> None:
        """A parallel input-guardrail tripwire that already fired must be
        surfaced even though a loop error would otherwise be recovered."""
        guardrail_done = asyncio.Event()
        tripwire = _make_input_tripwire()

        async def _tripping_parallel(*_args: Any, **_kwargs: Any) -> list[Any]:
            guardrail_done.set()
            raise tripwire

        async def _failing_loop(**_kwargs: Any) -> None:
            await guardrail_done.wait()
            raise RuntimeError("loop boom")

        agent = _make_agent()
        # An error handler that would happily "recover" the RuntimeError.
        config = RunConfig(error_handlers={Exception: lambda _e: "recovered!"})

        with (
            patch("troopai.adk.run.runner.run_blocking_input_guardrails", new=AsyncMock(return_value=[])),
            patch("troopai.adk.run.runner.run_parallel_input_guardrails", new=_tripping_parallel),
            patch("troopai.adk.run.runner.run_agent_loop_streamed", new=_failing_loop),
            patch("troopai.adk.run.runner.run_output_guardrails", new=AsyncMock(return_value=[])),
        ):
            streaming = await Runner.arun(agent, "hi", run_config=config, stream=True)
            with pytest.raises(AgentInputGuardrailTripwireTriggered):
                async for _ in streaming.stream_events():
                    pass

        # The run was blocked, not silently recovered.
        assert streaming.recovered is False


# ── Finding: sandbox bracket must close when a setup step raises ───────


class _BoomStartHooks(RunHooks[Any]):
    async def on_agent_start(self, context: Any, agent: Any) -> None:
        raise RuntimeError("start boom")


class TestSandboxBracketClosedOnSetupFailure:
    async def test_sandbox_closed_when_setup_step_raises(self) -> None:
        """A failure in a setup step opened after the sandbox bracket must
        still run the finally that closes the sandbox session."""
        closed = {"v": False}

        async def _fake_open(*, stack: Any, agent: Any, config: Any, run_context: Any, hooks: Any) -> None:
            async def _cleanup() -> None:
                closed["v"] = True

            stack.push_async_callback(_cleanup)

        def _boom_reset(_tools: Any) -> None:
            raise RuntimeError("reset boom")

        agent = _make_agent()

        with (
            patch("troopai.adk.run.runner._maybe_open_sandbox_bracket", new=_fake_open),
            patch("troopai.adk.tools.tool_search.reset_revealed_sets", new=_boom_reset),
            pytest.raises(RuntimeError, match="reset boom"),
        ):
            await Runner.arun(agent, "hi", run_config=RunConfig())

        assert closed["v"] is True

    async def test_sandbox_closed_when_start_hook_raises(self) -> None:
        """A start-hook failure (opened after the sandbox bracket) must not
        leak the sandbox session."""
        closed = {"v": False}

        async def _fake_open(*, stack: Any, agent: Any, config: Any, run_context: Any, hooks: Any) -> None:
            async def _cleanup() -> None:
                closed["v"] = True

            stack.push_async_callback(_cleanup)

        agent = _make_agent()

        with (
            patch("troopai.adk.run.runner._maybe_open_sandbox_bracket", new=_fake_open),
            contextlib.suppress(RuntimeError),
        ):
            await Runner.arun(agent, "hi", hooks=_BoomStartHooks(), run_config=RunConfig())

        assert closed["v"] is True
