"""Tests for RunItem typed item classes and factory functions."""

from __future__ import annotations

import pytest

from troopai.adk.types.items import (
    CompactionItem,
    HandoffCallItem,
    HandoffOutputItem,
    ItemHelpers,
    MCPApprovalRequestItem,
    MCPApprovalResponseItem,
    MCPListToolsItem,
    MessageOutputItem,
    ReasoningItem,
    SystemItem,
    ToolApprovalItem,
    ToolCallItem,
    ToolCallOutputItem,
    ToolSearchCallItem,
    ToolSearchOutputItem,
    UserItem,
)

message_to_items = ItemHelpers.message_to_run_items
messages_to_items = ItemHelpers.messages_to_run_items
items_to_params = ItemHelpers.run_items_to_params


# ==================================================================
# Helpers
# ==================================================================


def _easy_msg(role: str = "system", content: str = "") -> dict:
    return {"role": role, "content": content}


def _make_message(text: str = "Hi there!", **kwargs: object) -> MessageOutputItem:
    from troopai.adk.types.responses.llm_response import LLMResponseRefusal, LLMResponseText

    parts: list = []
    if text is not None:
        parts.append(LLMResponseText(text=text))
    refusal = kwargs.get("refusal")
    if refusal is not None:
        parts.append(LLMResponseRefusal(refusal=str(refusal)))
    if len(parts) == 0:
        parts.append(LLMResponseText(text=""))
    return MessageOutputItem(
        raw=parts,
        id=kwargs.get("id"),  # type: ignore[arg-type]
        status=kwargs.get("status", "completed"),  # type: ignore[arg-type]
    )


def _make_tool_call(**kwargs: object) -> ToolCallItem:
    from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

    return ToolCallItem(
        raw=LLMResponseFunctionToolCall(
            call_id=str(kwargs.get("call_id", "call_1")),
            name=str(kwargs.get("name", "search")),
            arguments=str(kwargs.get("arguments", "{}")),
            id=kwargs.get("id"),  # type: ignore[arg-type]
            status=kwargs.get("status", "completed"),  # type: ignore[arg-type]
        )
    )


def _make_tool_result(call_id: str = "call_1", output: str = "42") -> ToolCallOutputItem:
    from troopai.adk.types.output import FunctionToolCallResult

    return ToolCallOutputItem(raw=FunctionToolCallResult(call_id=call_id, output=output))


def _make_reasoning(**kwargs: object) -> ReasoningItem:
    from troopai.adk.types.responses.llm_response import LLMResponseReasoning

    summary_texts = kwargs.get("summary_texts") or [""]
    content_texts = kwargs.get("content_texts") or []
    return ReasoningItem(
        raw=LLMResponseReasoning(
            id=kwargs.get("id"),  # type: ignore[arg-type]
            thinking=" ".join(str(c) for c in content_texts) if len(content_texts) > 0 else "",
            summary=" ".join(str(s) for s in summary_texts),
            encrypted_content=kwargs.get("encrypted_content"),  # type: ignore[arg-type]
            status=kwargs.get("status", "completed"),  # type: ignore[arg-type]
        )
    )


# ==================================================================
# Item class tests — all data accessed via raw
# ==================================================================


class TestSystemItem:
    def test_raw_access(self) -> None:
        item = SystemItem(raw=_easy_msg("system", "You are helpful."))
        assert item.raw["content"] == "You are helpful."
        assert item.raw["role"] == "system"

    def test_type_discriminator(self) -> None:
        assert SystemItem(raw=_easy_msg()).type == "system"

    def test_agent_name(self) -> None:
        assert SystemItem(raw=_easy_msg(), agent_name="triage").agent_name == "triage"

    def test_to_param_returns_raw(self) -> None:
        raw = _easy_msg("system", "test")
        assert SystemItem(raw=raw).to_param() is raw

    def test_raw_is_required(self) -> None:
        with pytest.raises(TypeError):
            SystemItem()  # type: ignore[call-arg]


class TestUserItem:
    def test_raw_access(self) -> None:
        assert UserItem(raw=_easy_msg("user", "Hello!")).raw["content"] == "Hello!"

    def test_type_discriminator(self) -> None:
        assert UserItem(raw=_easy_msg("user")).type == "user"


class TestCompactionItem:
    def test_raw_access(self) -> None:
        assert CompactionItem(raw=_easy_msg("system", "Summary")).raw["content"] == "Summary"

    def test_type_discriminator(self) -> None:
        assert CompactionItem(raw=_easy_msg()).type == "compaction"


class TestMessageOutputItem:
    def test_raw_access(self) -> None:
        item = _make_message("Hi", status="completed")
        assert item.status == "completed"
        assert item.raw[0].text == "Hi"  # type: ignore[union-attr]

    def test_type_discriminator(self) -> None:
        assert _make_message().type == "message_output"

    def test_no_properties(self) -> None:
        """MessageOutputItem has no convenience properties — use ItemHelpers."""
        item = _make_message("test")
        assert not hasattr(type(item), "content")
        assert not hasattr(type(item), "refusal")

    def test_raw_is_required(self) -> None:
        with pytest.raises(TypeError):
            MessageOutputItem()  # type: ignore[call-arg]


class TestToolCallItem:
    def test_raw_access(self) -> None:
        item = _make_tool_call(call_id="c1", name="search")
        assert item.raw.call_id == "c1"
        assert item.raw.name == "search"

    def test_type_discriminator(self) -> None:
        assert _make_tool_call().type == "tool_call"


class TestToolCallOutputItem:
    def test_raw_access(self) -> None:
        item = _make_tool_result("c1", "42")
        assert item.raw.call_id == "c1"
        assert item.raw.output == "42"

    def test_no_output_property(self) -> None:
        """Use ItemHelpers.tool_call_output_str() instead."""
        item = _make_tool_result()
        assert not hasattr(type(item), "output")

    def test_type_discriminator(self) -> None:
        assert _make_tool_result().type == "tool_call_output"


class TestReasoningItem:
    def test_raw_access(self) -> None:
        item = _make_reasoning(id="r1", content_texts=["Think..."], encrypted_content="sig")
        assert item.raw.id == "r1"
        assert item.raw.encrypted_content == "sig"
        assert len(item.raw.thinking) > 0
        assert item.raw.thinking == "Think..."

    def test_type_discriminator(self) -> None:
        assert _make_reasoning().type == "reasoning"


class TestHandoffCallItem:
    def test_raw_access(self) -> None:
        from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

        item = HandoffCallItem(
            raw=LLMResponseFunctionToolCall(
                call_id="c1", name="transfer_to_billing", arguments="{}", status="completed"
            ),
            target_agent="billing",
        )
        assert item.raw.call_id == "c1"
        assert item.raw.name == "transfer_to_billing"
        assert item.target_agent == "billing"

    def test_type_discriminator(self) -> None:
        from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

        assert (
            HandoffCallItem(
                raw=LLMResponseFunctionToolCall(call_id="c", name="f", arguments="{}"),
            ).type
            == "handoff_call"
        )


class TestHandoffOutputItem:
    def test_source_and_target(self) -> None:
        from troopai.adk.types.output.function_tool_call_result_param import FunctionToolCallResultParam

        item = HandoffOutputItem(
            raw=FunctionToolCallResultParam(type="function_call_output", call_id="c1", output="Transferred."),
            source="triage",
            target="billing",
        )
        assert item.source == "triage"
        assert item.target == "billing"
        assert item.raw["call_id"] == "c1"

    def test_type_discriminator(self) -> None:
        from troopai.adk.types.output.function_tool_call_result_param import FunctionToolCallResultParam

        assert (
            HandoffOutputItem(
                raw=FunctionToolCallResultParam(type="function_call_output", call_id="c", output=""),
            ).type
            == "handoff_output"
        )


class TestMCPListToolsItem:
    def test_raw_access(self) -> None:
        from troopai.adk.types.tools.builtin_tool_types import MCPListTools, MCPListToolsTool

        item = MCPListToolsItem(
            raw=MCPListTools(server="srv", tools=[MCPListToolsTool(name="a"), MCPListToolsTool(name="b")])
        )
        assert item.raw.server == "srv"
        assert len(item.raw.tools) == 2
        assert item.raw.tools[0].name == "a"
        assert item.raw.tools[1].name == "b"

    def test_to_param_omits_none_tool_fields(self) -> None:
        """to_param must not emit explicit nulls for optional tool fields.

        Regression: previously description/input_schema/annotations were
        unconditionally included even when None, which providers may reject.
        """
        from typing import Any, cast

        from troopai.adk.types.tools.builtin_tool_types import MCPListTools, MCPListToolsTool

        # Tool with no optional fields set
        item_no_opts = MCPListToolsItem(raw=MCPListTools(server="srv", tools=[MCPListToolsTool(name="only_name")]))
        param_no_opts = cast(dict[str, Any], item_no_opts.to_param())
        tool_dict = param_no_opts["raw"]["tools"][0]
        assert "name" in tool_dict
        assert "description" not in tool_dict
        assert "input_schema" not in tool_dict
        assert "annotations" not in tool_dict

        # Tool with all optional fields set
        item_full = MCPListToolsItem(
            raw=MCPListTools(
                server="srv",
                tools=[
                    MCPListToolsTool(
                        name="full_tool",
                        input_schema={"type": "object"},
                        description="A full tool",
                        annotations={"readOnly": True},
                    )
                ],
            )
        )
        param_full = cast(dict[str, Any], item_full.to_param())
        full_tool_dict = param_full["raw"]["tools"][0]
        assert full_tool_dict["name"] == "full_tool"
        assert full_tool_dict["description"] == "A full tool"
        assert full_tool_dict["input_schema"] == {"type": "object"}
        assert full_tool_dict["annotations"] == {"readOnly": True}


class TestMCPApprovalRequestItem:
    def test_raw_access(self) -> None:
        from troopai.adk.types.tools.builtin_tool_types import MCPApprovalRequest

        item = MCPApprovalRequestItem(raw=MCPApprovalRequest(id="c1", server="srv", name="tool1"))
        assert item.raw.server == "srv"
        assert item.raw.name == "tool1"
        assert item.raw.id == "c1"


class TestMCPApprovalResponseItem:
    def test_raw_access(self) -> None:
        from troopai.adk.types.tools.builtin_tool_types import MCPApprovalResponse

        item = MCPApprovalResponseItem(raw=MCPApprovalResponse(approval_request_id="c1", approved=False, reason="No"))
        assert item.raw.approved is False
        assert item.raw.reason == "No"

    def test_to_param(self) -> None:
        from typing import Any, cast

        from troopai.adk.types.tools.builtin_tool_types import MCPApprovalResponse

        item = MCPApprovalResponseItem(raw=MCPApprovalResponse(approval_request_id="c1", approved=True))
        # Approval responses now round-trip via the provider_item
        # channel carrying the OpenAI Responses wire shape — `approve`
        # (bool), `approval_request_id`. Previously the param was a
        # function_call_output with a free-form "approved/rejected"
        # string, which the API does not actually accept.
        param = cast(dict[str, Any], item.to_param())
        assert param["type"] == "provider_item"
        assert param["item_type"] == "mcp_approval_response"
        assert param["raw"]["approve"] is True
        assert param["raw"]["approval_request_id"] == "c1"


class TestToolApprovalItem:
    def _make_deferred(self, **kwargs: object) -> object:
        from troopai.adk.tools.deferred_tool import DeferredToolCall

        return DeferredToolCall(
            tool_call_id=str(kwargs.get("call_id", "c1")),
            tool_name=str(kwargs.get("name", "delete_file")),
            tool_arguments=kwargs.get("tool_arguments") or {},  # type: ignore[arg-type]
            raw_arguments=str(kwargs.get("arguments", "{}")),
        )

    def test_raw_access(self) -> None:
        item = ToolApprovalItem(raw=self._make_deferred(call_id="c1", name="rm"))  # type: ignore[arg-type]
        assert item.raw.tool_call_id == "c1"
        assert item.raw.tool_name == "rm"

    def test_pending(self) -> None:
        item = ToolApprovalItem(raw=self._make_deferred())  # type: ignore[arg-type]
        assert item.approved is None
        assert "awaiting" in str(item.to_param()["output"]).lower()


# ==================================================================
# Factory function tests
# ==================================================================


class TestMessageToItems:
    def test_system_message(self) -> None:
        items = message_to_items({"role": "system", "content": "You are helpful."})
        assert isinstance(items[0], SystemItem)
        assert items[0].raw["content"] == "You are helpful."

    def test_user_message(self) -> None:
        items = message_to_items({"role": "user", "content": "Hello!"})
        assert isinstance(items[0], UserItem)
        assert items[0].raw["content"] == "Hello!"

    def test_assistant_message(self) -> None:
        items = message_to_items({"role": "assistant", "content": "Hi there!"})
        assert isinstance(items[0], MessageOutputItem)
        assert ItemHelpers.text_message_output(items[0]) == "Hi there!"

    def test_assistant_with_tool_calls(self) -> None:
        msg = {
            "role": "assistant",
            "content": "Searching.",
            "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search", "arguments": "{}"}}],
        }
        items = message_to_items(msg)
        assert isinstance(items[1], ToolCallItem)
        assert items[1].raw.name == "search"

    def test_assistant_with_thinking_blocks(self) -> None:
        msg = {
            "role": "assistant",
            "content": "42.",
            "thinking_blocks": [{"type": "thinking", "thinking": "Let me think...", "signature": "sig1"}],
        }
        items = message_to_items(msg)
        assert isinstance(items[0], ReasoningItem)
        assert items[0].raw.encrypted_content == "sig1"

    def test_tool_result(self) -> None:
        items = message_to_items({"role": "tool", "tool_call_id": "c1", "content": "42"})
        assert isinstance(items[0], ToolCallOutputItem)
        assert items[0].raw.call_id == "c1"

    def test_handoff_result_detection(self) -> None:
        items = message_to_items({"role": "tool", "tool_call_id": "c1", "content": "Transferred to billing."})
        assert isinstance(items[0], HandoffOutputItem)
        assert items[0].target == "billing"


class TestMessagesToItems:
    def test_full_conversation(self) -> None:
        items = messages_to_items(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        )
        assert len(items) == 3
        assert isinstance(items[0], SystemItem)
        assert isinstance(items[1], UserItem)
        assert isinstance(items[2], MessageOutputItem)


class TestItemsToParams:
    def test_system_and_user(self) -> None:
        params = items_to_params(
            [
                SystemItem(raw=_easy_msg("system", "You are helpful.")),
                UserItem(raw=_easy_msg("user", "Hello!")),
            ]
        )
        assert params[0]["role"] == "system"
        assert params[0]["content"] == "You are helpful."


# ==================================================================
# Raw field tests
# ==================================================================


class TestRawField:
    def test_all_items_require_raw(self) -> None:
        with pytest.raises(TypeError):
            SystemItem()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            MessageOutputItem()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            ToolCallItem()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            ToolCallOutputItem()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            ReasoningItem()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            HandoffCallItem()  # type: ignore[call-arg]
        with pytest.raises(TypeError):
            HandoffOutputItem()  # type: ignore[call-arg]


# ==================================================================
# Roundtrip
# ==================================================================


class TestRoundtrip:
    def test_simple_conversation(self) -> None:
        original = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        restored = items_to_params(messages_to_items(original))
        assert restored[0]["role"] == "system"
        assert restored[0]["content"] == "You are helpful."


# ==================================================================
# ItemHelpers
# ==================================================================


class TestItemHelpers:
    def test_text_message_output(self) -> None:
        assert ItemHelpers.text_message_output(_make_message("Hello!")) == "Hello!"

    def test_text_message_output_multi_part(self) -> None:
        from troopai.adk.types.responses.llm_response import LLMResponseText

        item = MessageOutputItem(
            raw=[LLMResponseText(text="Hello "), LLMResponseText(text="world!")],
        )
        assert ItemHelpers.text_message_output(item) == "Hello world!"

    def test_text_message_output_empty(self) -> None:
        from troopai.adk.types.responses.llm_response import LLMResponseRefusal

        item = MessageOutputItem(
            raw=[LLMResponseRefusal(refusal="No")],
        )
        assert ItemHelpers.text_message_output(item) == ""

    def test_text_message_outputs(self) -> None:
        items = [
            UserItem(raw=_easy_msg("user", "ignored")),
            _make_message("First"),
            _make_message("Second"),
        ]
        assert ItemHelpers.text_message_outputs(items) == "First\nSecond"

    def test_extract_last_text(self) -> None:
        items = [_make_message("First"), UserItem(raw=_easy_msg("user", "mid")), _make_message("Last")]
        assert ItemHelpers.extract_last_text(items) == "Last"

    def test_extract_last_text_none(self) -> None:
        assert ItemHelpers.extract_last_text([UserItem(raw=_easy_msg("user"))]) is None

    def test_refusal_message_output(self) -> None:
        from troopai.adk.types.responses.llm_response import LLMResponseRefusal

        item = MessageOutputItem(
            raw=[LLMResponseRefusal(refusal="I can't.")],
        )
        assert ItemHelpers.refusal_message_output(item) == "I can't."

    def test_refusal_message_output_none(self) -> None:
        assert ItemHelpers.refusal_message_output(_make_message("Hi")) is None

    def test_tool_call_output_str(self) -> None:
        assert ItemHelpers.tool_call_output_str(_make_tool_result("c1", "42")) == "42"

    def test_reasoning_summary_text(self) -> None:
        item = _make_reasoning(summary_texts=["Part one.", "Part two."])
        assert ItemHelpers.reasoning_summary_text(item) == "Part one. Part two."

    def test_reasoning_content_text(self) -> None:
        item = _make_reasoning(content_texts=["First.", "Second."])
        assert ItemHelpers.reasoning_content_text(item) == "First. Second."

    def test_reasoning_content_text_none(self) -> None:
        assert ItemHelpers.reasoning_content_text(_make_reasoning()) is None

    def test_tool_call_output_item(self) -> None:
        tc = ToolCallItem(raw=_make_tool_call(call_id="c1").raw, agent_name="triage")
        result = ItemHelpers.tool_call_output_item(tc, "Found it")
        assert isinstance(result, ToolCallOutputItem)
        assert result.raw.call_id == "c1"
        assert result.agent_name == "triage"


# ==================================================================
# Type discriminator completeness
# ==================================================================


class TestTypeDiscriminators:
    def test_all_items_have_unique_type(self) -> None:
        from troopai.adk.tools.deferred_tool import DeferredToolCall
        from troopai.adk.types.output import FunctionToolCallResult
        from troopai.adk.types.output.function_tool_call_result_param import FunctionToolCallResultParam
        from troopai.adk.types.responses.llm_response import (
            LLMResponseFunctionToolCall,
            LLMResponseReasoning,
            LLMResponseText,
        )
        from troopai.adk.types.tools.builtin_tool_types import (
            MCPApprovalRequest,
            MCPApprovalResponse,
            MCPListTools,
            MCPListToolsTool,
            ToolSearchToolCall,
            ToolSearchToolCallResult,
        )

        items = [
            SystemItem(raw=_easy_msg()),
            UserItem(raw=_easy_msg("user")),
            MessageOutputItem(raw=[LLMResponseText(text="")]),
            ToolCallItem(raw=LLMResponseFunctionToolCall(call_id="c", name="f", arguments="{}")),
            ToolCallOutputItem(raw=FunctionToolCallResult(call_id="c", output="")),
            ReasoningItem(raw=LLMResponseReasoning(thinking="", summary="")),
            HandoffCallItem(raw=LLMResponseFunctionToolCall(call_id="c", name="f", arguments="{}")),
            HandoffOutputItem(raw=FunctionToolCallResultParam(type="function_call_output", call_id="c", output="")),
            CompactionItem(raw=_easy_msg()),
            MCPListToolsItem(raw=MCPListTools(server="s", tools=[MCPListToolsTool(name="t")])),
            MCPApprovalRequestItem(raw=MCPApprovalRequest(id="c", server="s", name="t")),
            MCPApprovalResponseItem(raw=MCPApprovalResponse(approval_request_id="c", approved=True)),
            ToolApprovalItem(
                raw=DeferredToolCall(tool_call_id="c", tool_name="t", tool_arguments={}, raw_arguments="{}")
            ),
            ToolSearchCallItem(raw=ToolSearchToolCall(id="c", query="q")),
            ToolSearchOutputItem(raw=ToolSearchToolCallResult(call_id="c")),
        ]
        types = [item.type for item in items]
        assert len(types) == len(set(types))
        assert "run_item" not in types


# ==================================================================
# Round-trip tests: to_param() → message_to_run_items()
# ==================================================================


class TestRoundTripToolCallItem:
    """ToolCallItem survives to_param → message_to_run_items round-trip."""

    def test_basic_round_trip(self) -> None:
        original = _make_tool_call(call_id="c1", name="search", arguments='{"q":"test"}')
        param = original.to_param()
        assert param["type"] == "function_call"
        assert "role" not in param

        restored = message_to_items(param)
        assert len(restored) == 1
        item = restored[0]
        assert isinstance(item, ToolCallItem)
        assert item.raw.call_id == "c1"
        assert item.raw.name == "search"
        assert item.raw.arguments == '{"q":"test"}'

    def test_round_trip_with_agent_name(self) -> None:
        original = _make_tool_call(call_id="c2", name="delete")
        param = original.to_param()
        restored = message_to_items(param, agent_name="worker")
        assert len(restored) == 1
        assert restored[0].agent_name == "worker"

    def test_round_trip_preserves_status(self) -> None:
        original = _make_tool_call(status="completed")
        param = original.to_param()
        restored = message_to_items(param)
        assert restored[0].raw.status == "completed"


class TestRoundTripReasoningItem:
    """ReasoningItem survives to_param → message_to_run_items round-trip."""

    def test_basic_round_trip(self) -> None:
        original = _make_reasoning(
            id="r1",
            summary_texts=["thinking..."],
            content_texts=["step 1", "step 2"],
        )
        param = original.to_param()
        assert param["type"] == "reasoning"
        assert "role" not in param

        restored = message_to_items(param)
        assert len(restored) == 1
        item = restored[0]
        assert isinstance(item, ReasoningItem)
        assert len(item.raw.summary) > 0
        assert len(item.raw.thinking) > 0

    def test_round_trip_with_encrypted_content(self) -> None:
        original = _make_reasoning(
            summary_texts=[""],
            encrypted_content="sig123",
        )
        param = original.to_param()
        restored = message_to_items(param)
        assert restored[0].raw.encrypted_content == "sig123"

    def test_round_trip_no_content(self) -> None:
        original = _make_reasoning(summary_texts=["summary only"])
        param = original.to_param()
        restored = message_to_items(param)
        # With no content_texts, thinking is empty
        assert len(restored[0].raw.thinking) == 0


class TestRoundTripMessageOutputItem:
    """MessageOutputItem survives to_param → message_to_run_items round-trip."""

    def test_basic_round_trip(self) -> None:
        original = _make_message("Hello!")
        param = original.to_param()
        assert param["type"] == "message"
        assert param["role"] == "assistant"
        # Content is a list of typed dicts after to_param()
        assert isinstance(param["content"], list)

        restored = message_to_items(param)
        assert len(restored) == 1
        item = restored[0]
        assert isinstance(item, MessageOutputItem)
        assert item.raw[0].text == "Hello!"  # type: ignore[union-attr]

    def test_round_trip_with_refusal(self) -> None:
        original = _make_message(text="", refusal="I can't do that")
        param = original.to_param()
        restored = message_to_items(param)
        assert len(restored) == 1
        item = restored[0]
        assert isinstance(item, MessageOutputItem)
        assert any(hasattr(p, "refusal") and p.refusal == "I can't do that" for p in item.raw)


class TestRoundTripMixed:
    """Mixed conversation history round-trips correctly (simulates nested HITL)."""

    def test_full_conversation_round_trip(self) -> None:
        """Simulate a sub-agent conversation: user → assistant → tool_call → tool_result."""
        original_items = [
            UserItem(raw=_easy_msg("user", "Delete user bob")),
            _make_message("I'll look up the user first."),
            _make_tool_call(call_id="c1", name="list_users", arguments="{}"),
            _make_tool_result(call_id="c1", output="Users: alice, bob, charlie"),
            _make_message("Found bob. Deleting now."),
            _make_tool_call(call_id="c2", name="delete_user", arguments='{"user_id":"bob"}'),
        ]

        # Simulate to_dict → from_dict round-trip
        params = items_to_params(original_items)
        restored = messages_to_items(params)

        assert len(restored) == len(original_items)
        assert isinstance(restored[0], UserItem)
        assert isinstance(restored[1], MessageOutputItem)
        assert isinstance(restored[2], ToolCallItem)
        assert restored[2].raw.name == "list_users"
        assert isinstance(restored[3], ToolCallOutputItem)
        assert isinstance(restored[4], MessageOutputItem)
        assert isinstance(restored[5], ToolCallItem)
        assert restored[5].raw.name == "delete_user"
        assert restored[5].raw.arguments == '{"user_id":"bob"}'

    def test_state_round_trip(self) -> None:
        """Simulate RunState.to_dict() → RunState.from_dict() round-trip."""
        from troopai.adk.run.state import RunState

        original_items = [
            UserItem(raw=_easy_msg("user", "Delete user bob")),
            _make_message("Let me check..."),
            _make_tool_call(call_id="c1", name="list_users"),
            _make_tool_result(call_id="c1", output="bob found"),
        ]

        state = RunState(
            conversation_history=original_items,
            original_user_prompt="Delete user bob",
            current_agent_name="worker",
            turn_count=1,
        )

        state_dict = state.to_dict()
        restored = RunState.from_dict(state_dict)

        assert len(restored.conversation_history) == 4
        assert isinstance(restored.conversation_history[0], UserItem)
        assert isinstance(restored.conversation_history[1], MessageOutputItem)
        assert isinstance(restored.conversation_history[2], ToolCallItem)
        assert restored.conversation_history[2].raw.name == "list_users"
        assert isinstance(restored.conversation_history[3], ToolCallOutputItem)
        assert restored.original_user_prompt == "Delete user bob"
        assert restored.current_agent_name == "worker"
        assert restored.turn_count == 1


class TestRoundTripPreservesData:
    """``run_items_to_params`` → ``messages_to_run_items`` must not lose data.

    This reverse path runs on every session reload, history-processor pass,
    and handoff-context rebuild. Multimodal tool output, provider-hosted
    items, and text annotations must survive it intact.
    """

    def test_multimodal_tool_output_survives_roundtrip(self) -> None:
        from troopai.adk.types.output import FunctionToolCallResult

        multimodal_output = [
            {"type": "input_text", "text": "Here is the screenshot:"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]
        item = ToolCallOutputItem(raw=FunctionToolCallResult(call_id="c1", output=multimodal_output))

        back = messages_to_items(items_to_params([item]))

        assert len(back) == 1
        assert isinstance(back[0], ToolCallOutputItem)
        # The list must come back as a list — NOT stringified to its repr.
        assert back[0].raw.output == multimodal_output
        assert back[0].raw.call_id == "c1"

    def test_generic_provider_item_survives_roundtrip(self) -> None:
        from troopai.adk.types.items import ProviderItem
        from troopai.adk.types.responses.llm_response import LLMResponseProviderItem

        item = ProviderItem(
            raw=LLMResponseProviderItem(
                item_type="web_search_call",
                raw={"type": "web_search_call", "id": "ws_1", "status": "completed"},
            )
        )

        back = messages_to_items(items_to_params([item]))

        assert len(back) == 1
        # Must round-trip as a ProviderItem — NOT a silent empty UserItem.
        assert isinstance(back[0], ProviderItem)
        assert back[0].raw.item_type == "web_search_call"
        assert back[0].raw.raw["id"] == "ws_1"

    def test_mcp_list_tools_survives_roundtrip(self) -> None:
        from troopai.adk.types.tools.builtin_tool_types import MCPListTools, MCPListToolsTool

        item = MCPListToolsItem(
            raw=MCPListTools(
                id="mcp_1",
                server="prod-api",
                tools=[MCPListToolsTool(name="search", input_schema={"type": "object"}, description="d")],
            )
        )

        back = messages_to_items(items_to_params([item]))

        assert len(back) == 1
        # Typed MCP item must reconstruct via the provider_item channel,
        # NOT degrade to an empty UserItem.
        assert isinstance(back[0], MCPListToolsItem)
        assert back[0].raw.server == "prod-api"

    def test_text_annotations_survive_roundtrip(self) -> None:
        from troopai.adk.types.responses.llm_response import LLMResponseAnnotation, LLMResponseText

        annotation = LLMResponseAnnotation(url="https://example.com", title="Example", start_index=0, end_index=4)
        item = MessageOutputItem(raw=[LLMResponseText(text="docs", annotations=[annotation])], status="completed")

        back = messages_to_items(items_to_params([item]))

        assert len(back) == 1
        assert isinstance(back[0], MessageOutputItem)
        text_parts = [p for p in back[0].raw if isinstance(p, LLMResponseText)]
        assert len(text_parts) == 1
        assert text_parts[0].annotations is not None
        assert text_parts[0].annotations[0].url == "https://example.com"
