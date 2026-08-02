"""Tests for ``RunResult`` convenience helpers.

- ``last_response_id`` property
- ``release_agents()`` method
- ``to_input_list()``
"""

from __future__ import annotations

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.context import RunContext
from troopai.adk.types.items.items import (
    MessageOutputItem,
    ToolCallItem,
)
from troopai.adk.types.responses.llm_response import (
    LLMResponseFunctionToolCall,
    LLMResponseText,
)
from troopai.adk.types.run import RunResult

# ── Helpers ──────────────────────────────────────────────────────────


def _make_message_item(resp_id: str, text: str = "ok") -> MessageOutputItem:
    return MessageOutputItem(
        raw=[LLMResponseText(text=text)],
        id=resp_id,
        status="completed",
        agent_name="test-agent",
    )


def _make_tool_call_item(call_id: str = "c1", name: str = "tool") -> ToolCallItem:
    return ToolCallItem(
        raw=LLMResponseFunctionToolCall(call_id=call_id, name=name, arguments="{}"),
        agent_name="test-agent",
    )


def _make_result(**kwargs) -> RunResult:
    defaults = {
        "final_output": "ok",
        "user_prompt": "hello",
        "new_items": [],
        "context": RunContext(context=None),
        "last_agent": Agent(name="test", system_prompt="hi"),
    }
    defaults.update(kwargs)
    return RunResult(**defaults)  # type: ignore[arg-type]


# ── last_response_id ─────────────────────────────────────────────────


class TestLastResponseId:
    def test_returns_none_when_no_message_items(self) -> None:
        """No MessageOutputItem → last_response_id is None."""
        result = _make_result(new_items=[_make_tool_call_item()])
        assert result.last_response_id is None

    def test_returns_none_on_empty_items(self) -> None:
        """Empty ``new_items`` → last_response_id is None."""
        result = _make_result(new_items=[])
        assert result.last_response_id is None

    def test_returns_id_of_single_message_item(self) -> None:
        """One MessageOutputItem → its id is returned."""
        result = _make_result(
            new_items=[_make_message_item("resp-abc")],
        )
        assert result.last_response_id == "resp-abc"

    def test_returns_id_of_latest_message_item(self) -> None:
        """When multiple message items exist, the LAST one wins."""
        result = _make_result(
            new_items=[
                _make_message_item("resp-1"),
                _make_tool_call_item(),
                _make_message_item("resp-2"),
                _make_tool_call_item(),  # trailing tool calls don't reset
            ]
        )
        assert result.last_response_id == "resp-2"

    def test_skips_message_items_with_none_id(self) -> None:
        """Items with id=None are skipped; walks back to find the first real id."""
        item_none_id = MessageOutputItem(
            raw=[LLMResponseText(text="ok")],
            id=None,
            status="completed",
            agent_name="test-agent",
        )
        result = _make_result(
            new_items=[
                _make_message_item("resp-earliest"),
                item_none_id,
            ]
        )
        assert result.last_response_id == "resp-earliest"


# ── release_agents ───────────────────────────────────────────────────


class TestReleaseAgents:
    def test_nulls_last_agent(self) -> None:
        """``last_agent`` reference is dropped."""
        agent = Agent(name="test", system_prompt="hi")
        result = _make_result(last_agent=agent)

        result.release_agents()
        assert result.last_agent is None

    def test_default_clears_new_items(self) -> None:
        """By default, ``new_items`` is also cleared."""
        items = [_make_message_item("r1")]
        result = _make_result(new_items=items)

        result.release_agents()
        assert len(result.new_items) == 0

    def test_keep_new_items_flag(self) -> None:
        """``release_new_items=False`` keeps conversation history intact."""
        items = [_make_message_item("r1")]
        result = _make_result(new_items=items)

        result.release_agents(release_new_items=False)
        assert result.last_agent is None
        assert len(result.new_items) == 1
        assert result.last_response_id == "r1"  # still accessible

    def test_preserves_cheap_metadata(self) -> None:
        """``final_output``, ``user_prompt``, ``context`` survive release."""
        result = _make_result(
            final_output="answer",
            user_prompt="question",
        )

        result.release_agents()
        assert result.final_output == "answer"
        assert result.user_prompt == "question"
        assert result.context is not None


# ── to_input_list ─────────────────────────────────────────


class TestToInputListMode:
    def test_returns_user_prompt_first(self) -> None:
        """First item in the list is always the user prompt."""
        result = _make_result(
            user_prompt="hello",
            new_items=[_make_message_item("r1", text="world")],
        )
        out = result.to_input_list()
        # First item is the user prompt
        assert len(out) >= 1
        assert out[0]["role"] == "user"
        assert out[0]["content"] == "hello"

    def test_user_prompt_list_preserved(self) -> None:
        """If ``user_prompt`` is already a list, it's passed through."""
        user_list = [
            {"role": "user", "content": "line1"},
            {"role": "user", "content": "line2"},
        ]
        result = _make_result(user_prompt=user_list, new_items=[])  # type: ignore[arg-type]
        out = result.to_input_list()
        assert out[:2] == user_list

    def test_includes_run_item_params(self) -> None:
        """Run items are appended after the user prompt via to_param()."""
        result = _make_result(
            user_prompt="hi",
            new_items=[_make_message_item("r1", text="there")],
        )
        out = result.to_input_list()
        # At minimum: 1 user message + 1 assistant message
        assert len(out) >= 2


# ── Integration: helpers work together ──────────────────────────────


class TestHelpersComposition:
    def test_release_then_to_input_list(self) -> None:
        """After ``release_agents``, ``to_input_list`` returns just the user prompt."""
        result = _make_result(
            user_prompt="hello",
            new_items=[_make_message_item("r1")],
        )
        result.release_agents()
        out = result.to_input_list()
        # Only the user prompt remains
        assert len(out) == 1
        assert out[0]["content"] == "hello"

    def test_release_then_last_response_id_is_none(self) -> None:
        """After release, ``last_response_id`` returns None (items are gone)."""
        result = _make_result(new_items=[_make_message_item("r1")])
        assert result.last_response_id == "r1"
        result.release_agents()
        assert result.last_response_id is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
