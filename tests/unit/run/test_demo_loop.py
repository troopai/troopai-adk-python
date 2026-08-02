"""Tests for ``run_demo_loop`` REPL helper.

Covers control flow: exit tokens, EOF/Ctrl-C, empty-input skip,
``to_input_list()`` accumulation across turns, and agent tracking
after handoffs. The LLM layer is mocked out at ``Runner.arun``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.demo import run_demo_loop


class _InputFeeder:
    """Replaces builtin ``input()`` with a scripted sequence.

    Each call pops the next line. When the sequence is exhausted it
    raises ``EOFError`` so the loop exits cleanly — the same signal
    Ctrl-D would deliver.
    """

    def __init__(self, lines: list[str]) -> None:
        self._iter: Iterator[str] = iter(lines)

    def __call__(self, _prompt: str = "") -> str:
        try:
            return next(self._iter)
        except StopIteration:
            raise EOFError


def _fake_run_result(
    *,
    final_output: str = "ok",
    last_agent: Agent,
    input_list_return: list[Any],
) -> Any:
    """Build a minimal RunResult-shaped stub with the attributes the loop reads."""
    from unittest.mock import MagicMock

    result = MagicMock()
    result.final_output = final_output
    result.last_agent = last_agent
    result.to_input_list.return_value = input_list_return
    return result


def _fake_streaming_result(
    *,
    current_agent: Agent,
    user_prompt: Any,
    assistant_text: str,
) -> Any:
    """Build a streaming-result stub that runs the REAL ``to_input_list``.

    ``stream_events()`` yields nothing (the demo loop only logs); the
    crucial behaviour under test is that ``to_input_list()`` reconstructs
    the full turn from ``user_prompt`` + ``new_items``, exactly as the
    real ``RunResultStreaming`` does.
    """
    from unittest.mock import MagicMock

    from troopai.adk.run.stream import RunResultStreaming
    from troopai.adk.types.items.items import MessageOutputItem
    from troopai.adk.types.responses.llm_response import LLMResponseText

    assistant_item = MessageOutputItem(
        raw=[LLMResponseText(text=assistant_text)],
        agent_name=current_agent.name,
    )

    result = MagicMock()
    result.current_agent = current_agent
    result.user_prompt = user_prompt
    result.new_items = [assistant_item]

    async def _empty_stream() -> Any:
        return
        yield  # pragma: no cover - makes this an async generator

    result.stream_events.side_effect = _empty_stream
    # Bind the real implementation so the test exercises the actual fix.
    result.to_input_list.side_effect = lambda: RunResultStreaming.to_input_list(result)
    return result


# ── Control flow ─────────────────────────────────────────────────────


class TestRunDemoLoopControl:
    @pytest.mark.asyncio
    async def test_exit_token_breaks_loop(self) -> None:
        """``exit`` ends the loop without invoking Runner.arun."""
        agent = Agent(name="demo", system_prompt="hi")
        feeder = _InputFeeder(["exit"])

        arun_mock = AsyncMock()
        with patch("builtins.input", feeder), patch("troopai.adk.run.demo.Runner.arun", arun_mock):
            await run_demo_loop(agent, stream=False)

        assert arun_mock.call_count == 0

    @pytest.mark.asyncio
    async def test_quit_token_breaks_loop(self) -> None:
        """``quit`` (case-insensitive) also ends the loop."""
        agent = Agent(name="demo", system_prompt="hi")
        feeder = _InputFeeder(["QUIT"])

        arun_mock = AsyncMock()
        with patch("builtins.input", feeder), patch("troopai.adk.run.demo.Runner.arun", arun_mock):
            await run_demo_loop(agent, stream=False)

        assert arun_mock.call_count == 0

    @pytest.mark.asyncio
    async def test_eof_breaks_loop(self) -> None:
        """EOFError (Ctrl-D) ends the loop."""
        agent = Agent(name="demo", system_prompt="hi")

        def eof_input(_prompt: str = "") -> str:
            raise EOFError

        arun_mock = AsyncMock()
        with patch("builtins.input", eof_input), patch("troopai.adk.run.demo.Runner.arun", arun_mock):
            await run_demo_loop(agent, stream=False)

        assert arun_mock.call_count == 0

    @pytest.mark.asyncio
    async def test_keyboard_interrupt_breaks_loop(self) -> None:
        """KeyboardInterrupt (Ctrl-C) ends the loop."""
        agent = Agent(name="demo", system_prompt="hi")

        def interrupt_input(_prompt: str = "") -> str:
            raise KeyboardInterrupt

        arun_mock = AsyncMock()
        with patch("builtins.input", interrupt_input), patch("troopai.adk.run.demo.Runner.arun", arun_mock):
            await run_demo_loop(agent, stream=False)

        assert arun_mock.call_count == 0

    @pytest.mark.asyncio
    async def test_empty_input_is_skipped(self) -> None:
        """Blank and whitespace-only lines don't call Runner.arun."""
        agent = Agent(name="demo", system_prompt="hi")
        feeder = _InputFeeder(["", "   ", "\t", "exit"])

        arun_mock = AsyncMock()
        with patch("builtins.input", feeder), patch("troopai.adk.run.demo.Runner.arun", arun_mock):
            await run_demo_loop(agent, stream=False)

        assert arun_mock.call_count == 0


# ── Conversation state accumulation ──────────────────────────────────


class TestRunDemoLoopStatePropagation:
    @pytest.mark.asyncio
    async def test_single_turn_non_streaming(self) -> None:
        """One user input invokes arun once and logs the final output."""
        agent = Agent(name="demo", system_prompt="hi")
        feeder = _InputFeeder(["hello"])

        result = _fake_run_result(
            final_output="world",
            last_agent=agent,
            input_list_return=[
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ],
        )

        with (
            patch("builtins.input", feeder),
            patch(
                "troopai.adk.run.demo.Runner.arun",
                new=AsyncMock(return_value=result),
            ) as arun_mock,
        ):
            await run_demo_loop(agent, stream=False)

        assert arun_mock.call_count == 1
        call_kwargs = arun_mock.call_args
        # First positional is the agent
        assert call_kwargs.args[0] is agent
        # Second positional is the input list; first item is the user message
        passed_input = call_kwargs.args[1]
        assert len(passed_input) == 1
        assert passed_input[0]["role"] == "user"
        assert passed_input[0]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_multi_turn_accumulates_history(self) -> None:
        """Second turn is invoked with the first turn's ``to_input_list()``."""
        agent = Agent(name="demo", system_prompt="hi")
        feeder = _InputFeeder(["first", "second"])

        turn1_history: list[Any] = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "r1"},
        ]
        result1 = _fake_run_result(final_output="r1", last_agent=agent, input_list_return=turn1_history)
        result2 = _fake_run_result(
            final_output="r2",
            last_agent=agent,
            input_list_return=[
                *turn1_history,
                {"role": "user", "content": "second"},
                {"role": "assistant", "content": "r2"},
            ],
        )

        arun_mock = AsyncMock(side_effect=[result1, result2])
        with patch("builtins.input", feeder), patch("troopai.adk.run.demo.Runner.arun", arun_mock):
            await run_demo_loop(agent, stream=False)

        assert arun_mock.call_count == 2
        # Second call must see the first turn's history PLUS the new user msg
        second_call_input = arun_mock.call_args_list[1].args[1]
        # First-turn history (2 items) + new user msg = 3
        assert len(second_call_input) == 3
        assert second_call_input[-1]["role"] == "user"
        assert second_call_input[-1]["content"] == "second"

    @pytest.mark.asyncio
    async def test_handoff_updates_current_agent(self) -> None:
        """After a handoff, the next turn is driven by the new agent."""
        agent_a = Agent(name="alpha", system_prompt="a")
        agent_b = Agent(name="beta", system_prompt="b")

        feeder = _InputFeeder(["turn1", "turn2"])

        result1 = _fake_run_result(
            final_output="r1",
            last_agent=agent_b,  # handoff occurred
            input_list_return=[{"role": "user", "content": "turn1"}],
        )
        result2 = _fake_run_result(
            final_output="r2",
            last_agent=agent_b,
            input_list_return=[],
        )

        arun_mock = AsyncMock(side_effect=[result1, result2])
        with patch("builtins.input", feeder), patch("troopai.adk.run.demo.Runner.arun", arun_mock):
            await run_demo_loop(agent_a, stream=False)

        # First call uses agent_a, second uses agent_b (post-handoff).
        assert arun_mock.call_args_list[0].args[0] is agent_a
        assert arun_mock.call_args_list[1].args[0] is agent_b


# ── Streaming-path history preservation (regression) ─────────────────


class TestRunResultStreamingToInputList:
    """``RunResultStreaming.to_input_list()`` must carry the user turn.

    ``new_items`` holds only LLM-generated items — never the user
    message. The streaming variant previously returned ``new_items``
    alone, dropping every user turn and producing a malformed history
    (an assistant message with no preceding user turn). It must mirror
    the non-streaming ``RunResult.to_input_list()`` by prepending the
    original ``user_prompt``.
    """

    def test_prepends_string_user_prompt(self) -> None:
        from troopai.adk.run.stream import RunResultStreaming
        from troopai.adk.types.items.items import MessageOutputItem
        from troopai.adk.types.responses.llm_response import LLMResponseText

        agent = Agent(name="demo", system_prompt="hi")
        result = RunResultStreaming(
            current_agent=agent,
            user_prompt="hello",
            new_items=[MessageOutputItem(raw=[LLMResponseText(text="world")], agent_name="demo")],
        )

        items = result.to_input_list()

        # User turn first, assistant output second — full turn preserved.
        assert len(items) == 2
        assert items[0] == {"role": "user", "content": "hello"}
        assert items[1]["role"] == "assistant"

    def test_prepends_list_user_prompt(self) -> None:
        from troopai.adk.run.stream import RunResultStreaming
        from troopai.adk.types.items.items import MessageOutputItem
        from troopai.adk.types.responses.llm_response import LLMResponseText

        agent = Agent(name="demo", system_prompt="hi")
        prior_history: list[Any] = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "r1"},
            {"role": "user", "content": "second"},
        ]
        result = RunResultStreaming(
            current_agent=agent,
            user_prompt=prior_history,
            new_items=[MessageOutputItem(raw=[LLMResponseText(text="r2")], agent_name="demo")],
        )

        items = result.to_input_list()

        # The full prior conversation survives, followed by the new reply.
        assert len(items) == 4
        assert items[:3] == prior_history
        assert items[3]["role"] == "assistant"


class TestRunDemoLoopStreamingState:
    @pytest.mark.asyncio
    async def test_streaming_multi_turn_preserves_user_history(self) -> None:
        """Streaming demo loop must keep prior user turns across turns.

        Before the fix the loop overwrote ``input_items`` with only the
        assistant-side items, so the second turn's request lost the first
        user message entirely. The second ``arun`` call must see both user
        turns plus the first assistant reply.
        """
        agent = Agent(name="demo", system_prompt="hi")
        feeder = _InputFeeder(["first", "second"])

        # Turn 1: input was [u1]; result echoes that as user_prompt.
        result1 = _fake_streaming_result(
            current_agent=agent,
            user_prompt=[{"role": "user", "content": "first"}],
            assistant_text="r1",
        )

        captured_second_input: list[Any] = []

        async def arun_side_effect(_agent: Any, turn_input: Any, **_kwargs: Any) -> Any:
            if len(captured_second_input) == 0 and turn_input[-1]["content"] == "second":
                captured_second_input.extend(turn_input)
            # Turn 2 user_prompt is whatever was fed in this turn.
            return result1 if turn_input[-1]["content"] == "first" else result2

        result2 = _fake_streaming_result(
            current_agent=agent,
            user_prompt=[],  # unused after turn 2 (loop ends)
            assistant_text="r2",
        )

        arun_mock = AsyncMock(side_effect=arun_side_effect)
        with patch("builtins.input", feeder), patch("troopai.adk.run.demo.Runner.arun", arun_mock):
            await run_demo_loop(agent, stream=True)

        assert arun_mock.call_count == 2
        # Turn 2 input = [u1, assistant1, u2] — the first user turn survived.
        assert len(captured_second_input) == 3
        assert captured_second_input[0] == {"role": "user", "content": "first"}
        assert captured_second_input[1]["role"] == "assistant"
        assert captured_second_input[2] == {"role": "user", "content": "second"}
