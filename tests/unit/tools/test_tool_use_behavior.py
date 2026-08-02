"""Unit tests for check_tool_use_behavior in tools_executor."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from troopai.adk.run.tools_executor import check_tool_use_behavior
from troopai.adk.types.output import FunctionToolCallResult
from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall
from troopai.adk.types.tools.tool_use_behavior import (
    StopAtTools,
    ToolsToFinalOutputResult,
)


def _make_tool_call(
    call_id: str = "tc_1",
    name: str = "my_tool",
    arguments: str = "{}",
) -> LLMResponseFunctionToolCall:
    """Build an LLMResponseFunctionToolCall instance."""
    return LLMResponseFunctionToolCall(call_id=call_id, name=name, arguments=arguments)


class TestRunLlmAgain:
    @pytest.mark.asyncio
    async def test_returns_none(self) -> None:
        agent = MagicMock()
        agent.tool_use_behavior = "run_llm_again"
        ctx_wrapper = MagicMock()

        tool_call = _make_tool_call()
        tool_result = FunctionToolCallResult(call_id="tc_1", output="output")

        result = await check_tool_use_behavior(
            agent=agent,
            tool_results=[tool_result],
            tool_calls=[tool_call],
            ctx_wrapper=ctx_wrapper,
        )

        assert result is None


class TestStopOnFirstTool:
    @pytest.mark.asyncio
    async def test_returns_first_result(self) -> None:
        agent = MagicMock()
        agent.tool_use_behavior = "stop_on_first_tool"
        ctx_wrapper = MagicMock()

        tool_calls = [
            _make_tool_call(call_id="tc_1", name="tool_a"),
            _make_tool_call(call_id="tc_2", name="tool_b"),
        ]
        tool_results = [
            FunctionToolCallResult(call_id="tc_1", output="first_output"),
            FunctionToolCallResult(call_id="tc_2", output="second_output"),
        ]

        result = await check_tool_use_behavior(
            agent=agent,
            tool_results=tool_results,
            tool_calls=tool_calls,
            ctx_wrapper=ctx_wrapper,
        )

        assert result == "first_output"

    @pytest.mark.asyncio
    async def test_empty_results_returns_none(self) -> None:
        agent = MagicMock()
        agent.tool_use_behavior = "stop_on_first_tool"
        ctx_wrapper = MagicMock()

        result = await check_tool_use_behavior(
            agent=agent,
            tool_results=[],
            tool_calls=[],
            ctx_wrapper=ctx_wrapper,
        )

        assert result is None


class TestStopAtTools:
    @pytest.mark.asyncio
    async def test_matching_tool_returns_result(self) -> None:
        agent = MagicMock()
        agent.tool_use_behavior = StopAtTools(stop_at_tool_names=["done"])
        ctx_wrapper = MagicMock()

        tool_calls = [
            _make_tool_call(call_id="tc_1", name="search"),
            _make_tool_call(call_id="tc_2", name="done"),
        ]
        tool_results = [
            FunctionToolCallResult(call_id="tc_1", output="search_output"),
            FunctionToolCallResult(call_id="tc_2", output="final_answer"),
        ]

        result = await check_tool_use_behavior(
            agent=agent,
            tool_results=tool_results,
            tool_calls=tool_calls,
            ctx_wrapper=ctx_wrapper,
        )

        assert result == "final_answer"

    @pytest.mark.asyncio
    async def test_no_matching_tool_returns_none(self) -> None:
        agent = MagicMock()
        agent.tool_use_behavior = StopAtTools(stop_at_tool_names=["done"])
        ctx_wrapper = MagicMock()

        tool_calls = [
            _make_tool_call(call_id="tc_1", name="search"),
            _make_tool_call(call_id="tc_2", name="fetch"),
        ]
        tool_results = [
            FunctionToolCallResult(call_id="tc_1", output="search_output"),
            FunctionToolCallResult(call_id="tc_2", output="fetch_output"),
        ]

        result = await check_tool_use_behavior(
            agent=agent,
            tool_results=tool_results,
            tool_calls=tool_calls,
            ctx_wrapper=ctx_wrapper,
        )

        assert result is None


class TestCallableBehavior:
    @pytest.mark.asyncio
    async def test_final_output(self) -> None:
        def custom_behavior(_ctx, _results):
            return ToolsToFinalOutputResult(
                is_final_output=True,
                final_output="done",
            )

        agent = MagicMock()
        agent.tool_use_behavior = custom_behavior
        ctx_wrapper = MagicMock()

        tool_call = _make_tool_call()
        tool_result = FunctionToolCallResult(call_id="tc_1", output="output")

        result = await check_tool_use_behavior(
            agent=agent,
            tool_results=[tool_result],
            tool_calls=[tool_call],
            ctx_wrapper=ctx_wrapper,
        )

        assert result == "done"

    @pytest.mark.asyncio
    async def test_not_final(self) -> None:
        def custom_behavior(_ctx, _results):
            return ToolsToFinalOutputResult(is_final_output=False)

        agent = MagicMock()
        agent.tool_use_behavior = custom_behavior
        ctx_wrapper = MagicMock()

        tool_call = _make_tool_call()
        tool_result = FunctionToolCallResult(call_id="tc_1", output="output")

        result = await check_tool_use_behavior(
            agent=agent,
            tool_results=[tool_result],
            tool_calls=[tool_call],
            ctx_wrapper=ctx_wrapper,
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_correct_call_id(self) -> None:
        """Verify call_id in FunctionToolResult is the string ID, not the full tool call object."""
        captured_results = []

        def custom_behavior(_ctx, results):
            captured_results.extend(results)
            return ToolsToFinalOutputResult(is_final_output=True, final_output="ok")

        agent = MagicMock()
        agent.tool_use_behavior = custom_behavior
        ctx_wrapper = MagicMock()

        tool_call = _make_tool_call(call_id="tc_abc_123", name="greet")
        tool_result = FunctionToolCallResult(call_id="tc_abc_123", output="hello")

        await check_tool_use_behavior(
            agent=agent,
            tool_results=[tool_result],
            tool_calls=[tool_call],
            ctx_wrapper=ctx_wrapper,
        )

        assert len(captured_results) == 1
        assert captured_results[0].call_id == "tc_abc_123"
        assert isinstance(captured_results[0].call_id, str)
        assert captured_results[0].name == "greet"
        assert captured_results[0].output == "hello"
