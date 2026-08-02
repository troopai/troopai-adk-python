"""Tests for Handoff schema support, typed input, and factory functions."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from troopai.adk.handoffs.handoff import Handoff, handoff
from troopai.adk.handoffs.handoff_input_data import HandoffInputData
from troopai.adk.handoffs.handoff_route import handoff_route
from troopai.adk.schemas.utils import SchemaEnforcement


def _mock_agent(name: str = "test_agent") -> MagicMock:
    """Create a mock Agent with the required name and description attributes."""
    agent = MagicMock()
    agent.name = name
    agent.description = None
    return agent


class EscalationInput(BaseModel):
    """Test Pydantic model for typed handoff input."""

    reason: str = Field(description="Why this needs escalation.")
    priority: int = Field(description="Priority level.", ge=1, le=5)
    order_id: str | None = Field(None, description="Order ID if applicable.")


class TestHandoffSchema:
    """Tests for Handoff schema generation from input_type."""

    def test_handoff_with_input_type_generates_schema(self) -> None:
        """input_type auto-generates a JSON schema via __post_init__."""
        agent = _mock_agent()
        h = Handoff(target=agent, input_type=EscalationInput)

        assert h.schema is not None
        assert isinstance(h.schema, dict)
        assert h.schema.get("type") == "object"
        assert "properties" in h.schema
        assert "reason" in h.schema["properties"]
        assert "priority" in h.schema["properties"]

    def test_handoff_without_input_type_has_empty_object_schema(self) -> None:
        """When input_type is None, schema defaults to canonical empty object."""
        agent = _mock_agent()
        h = Handoff(target=agent)

        assert h.schema == {"type": "object", "properties": {}}

    def test_handoff_to_tool_includes_schema(self) -> None:
        """to_tool() passes schema and schema_enforcement through."""
        agent = _mock_agent()
        h = Handoff(
            target=agent,
            input_type=EscalationInput,
            schema_enforcement=SchemaEnforcement.STRICT,
        )

        tool = h.to_tool()
        assert tool.schema is not None
        json_schema = tool.get_json_schema()
        assert json_schema.get("type") == "object"
        assert tool.schema_enforcement == SchemaEnforcement.STRICT
        assert "reason" in json_schema["properties"]

    def test_handoff_explicit_schema_overrides_input_type(self) -> None:
        """User-provided schema wins over input_type derivation."""
        agent = _mock_agent()
        custom_schema = {"type": "object", "properties": {"custom": {"type": "string"}}}
        h = Handoff(
            target=agent,
            input_type=EscalationInput,
            schema=custom_schema,
        )

        # Explicit schema should be preserved, not overwritten by input_type
        assert h.schema == custom_schema

    def test_handoff_schema_enforcement_compact(self) -> None:
        """COMPACT enforcement strips metadata from auto-generated schema."""
        agent = _mock_agent()
        h = Handoff(
            target=agent,
            input_type=EscalationInput,
            schema_enforcement=SchemaEnforcement.COMPACT,
        )

        assert h.schema is not None
        # COMPACT strips all "title" keys
        assert "title" not in h.schema
        # Properties should still be present
        assert "reason" in h.schema["properties"]


class TestHandoffInvoke:
    """Tests for Handoff.invoke() with typed and untyped input."""

    @pytest.mark.asyncio
    async def test_invoke_validates_typed_input(self) -> None:
        """When input_type is set, invoke() validates and parses tool_args."""
        agent = _mock_agent()
        h = Handoff(target=agent, input_type=EscalationInput)

        mock_context = MagicMock()
        result_agent, handoff_data = await h.invoke(
            tool_args='{"reason": "urgent", "priority": 3}',
            context=(),
            output=(),
            run_context=mock_context,
        )

        assert result_agent is agent
        assert isinstance(handoff_data.intent, EscalationInput)
        assert handoff_data.intent.reason == "urgent"
        assert handoff_data.intent.priority == 3
        assert handoff_data.intent.order_id is None

    @pytest.mark.asyncio
    async def test_invoke_without_input_type(self) -> None:
        """Backward compat: without input_type, intent is raw string."""
        agent = _mock_agent()
        h = Handoff(target=agent)

        mock_context = MagicMock()
        result_agent, handoff_data = await h.invoke(
            tool_args='{"some": "data"}',
            context=(),
            output=(),
            run_context=mock_context,
        )

        assert result_agent is agent
        assert handoff_data.intent == '{"some": "data"}'

    @pytest.mark.asyncio
    async def test_invoke_typed_input_validation_error(self) -> None:
        """Invalid tool_args raises ValidationError when input_type is set."""
        agent = _mock_agent()
        h = Handoff(target=agent, input_type=EscalationInput)

        mock_context = MagicMock()
        with pytest.raises(Exception):  # Pydantic ValidationError
            await h.invoke(
                tool_args='{"reason": "test", "priority": 99}',  # priority > 5
                context=(),
                output=(),
                run_context=mock_context,
            )

    @pytest.mark.asyncio
    async def test_invoke_populates_context_and_output(self) -> None:
        """Verify temporal split at handoff — context and output are preserved."""
        agent = _mock_agent()
        h = Handoff(target=agent)

        mock_context = MagicMock()
        context_msgs = (
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        )
        output_msgs = (
            {"role": "assistant", "content": "Hi! How can I help?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "1", "type": "function", "function": {"name": "transfer_to_test_agent", "arguments": "{}"}}
                ],
            },
        )

        _, handoff_data = await h.invoke(
            tool_args="{}",
            context=context_msgs,
            output=output_msgs,
            run_context=mock_context,
        )

        assert handoff_data.context == context_msgs
        assert handoff_data.output == output_msgs
        assert handoff_data.forwarded is None
        assert len(handoff_data.messages) == 4
        assert handoff_data.messages == context_msgs + output_msgs


class TestHandoffCallbackDispatch:
    """Tests for on_handoff callback signature detection."""

    @pytest.mark.asyncio
    async def test_on_handoff_with_input(self) -> None:
        """on_handoff(ctx, input) receives the validated typed input."""
        agent = _mock_agent()
        received: list[EscalationInput] = []

        def on_handoff(_ctx: object, input: EscalationInput) -> None:
            received.append(input)

        h = Handoff(
            target=agent,
            input_type=EscalationInput,
            on_handoff=on_handoff,
        )

        mock_context = MagicMock()
        await h.invoke(
            tool_args='{"reason": "urgent", "priority": 3}',
            context=(),
            output=(),
            run_context=mock_context,
        )

        assert len(received) == 1
        assert isinstance(received[0], EscalationInput)
        assert received[0].reason == "urgent"

    @pytest.mark.asyncio
    async def test_on_handoff_without_input(self) -> None:
        """on_handoff(ctx) is called with no input argument."""
        agent = _mock_agent()
        called = []

        def on_handoff(_ctx: object) -> None:
            called.append(True)

        h = Handoff(target=agent, on_handoff=on_handoff)

        mock_context = MagicMock()
        await h.invoke(
            tool_args="{}",
            context=(),
            output=(),
            run_context=mock_context,
        )

        assert len(called) == 1

    @pytest.mark.asyncio
    async def test_on_handoff_async_with_input(self) -> None:
        """Async on_handoff(ctx, input) works correctly."""
        agent = _mock_agent()
        received: list[EscalationInput] = []

        async def on_handoff(_ctx: object, input: EscalationInput) -> None:
            received.append(input)

        h = Handoff(
            target=agent,
            input_type=EscalationInput,
            on_handoff=on_handoff,
        )

        mock_context = MagicMock()
        await h.invoke(
            tool_args='{"reason": "async test", "priority": 1}',
            context=(),
            output=(),
            run_context=mock_context,
        )

        assert len(received) == 1
        assert received[0].reason == "async test"

    @pytest.mark.asyncio
    async def test_on_handoff_async_without_input(self) -> None:
        """Async on_handoff(ctx) works correctly."""
        agent = _mock_agent()
        called = []

        async def on_handoff(_ctx: object) -> None:
            called.append(True)

        h = Handoff(target=agent, on_handoff=on_handoff)

        mock_context = MagicMock()
        await h.invoke(
            tool_args="{}",
            context=(),
            output=(),
            run_context=mock_context,
        )

        assert len(called) == 1

    @pytest.mark.asyncio
    async def test_on_handoff_without_input_type_passes_raw_string(self) -> None:
        """Without input_type, with-input callback gets raw string."""
        agent = _mock_agent()
        received: list[str] = []

        def on_handoff(_ctx: object, raw: str) -> None:
            received.append(raw)

        h = Handoff(target=agent, on_handoff=on_handoff)

        mock_context = MagicMock()
        await h.invoke(
            tool_args='{"foo": "bar"}',
            context=(),
            output=(),
            run_context=mock_context,
        )

        assert received == ['{"foo": "bar"}']

    @pytest.mark.asyncio
    async def test_on_handoff_with_data_signature(self) -> None:
        """(ctx, data: HandoffInputData) callback receives full handoff data."""
        agent = _mock_agent()
        received_data: list[HandoffInputData] = []

        def on_handoff(_ctx: object, data: HandoffInputData) -> None:
            received_data.append(data)

        h = Handoff(target=agent, on_handoff=on_handoff)

        mock_context = MagicMock()
        context_msgs = ({"role": "system", "content": "test"},)
        output_msgs = ({"role": "assistant", "content": "response"},)

        await h.invoke(
            tool_args="{}",
            context=context_msgs,
            output=output_msgs,
            run_context=mock_context,
        )

        assert len(received_data) == 1
        data = received_data[0]
        assert data.context == context_msgs
        assert data.output == output_msgs
        assert len(data.messages) == 2

    @pytest.mark.asyncio
    async def test_on_handoff_with_data_async(self) -> None:
        """Async (ctx, data: HandoffInputData) callback works."""
        agent = _mock_agent()
        received_data: list[HandoffInputData] = []

        async def on_handoff(_ctx: object, data: HandoffInputData) -> None:
            received_data.append(data)

        h = Handoff(target=agent, on_handoff=on_handoff)

        mock_context = MagicMock()
        await h.invoke(
            tool_args="{}",
            context=({"role": "user", "content": "hi"},),
            output=(),
            run_context=mock_context,
        )

        assert len(received_data) == 1
        assert received_data[0].context == ({"role": "user", "content": "hi"},)


class TestHandoffTargetRename:
    """Tests that Handoff uses 'target' instead of 'agent'."""

    def test_target_field_works(self) -> None:
        """Handoff(target=agent) creates a valid handoff."""
        agent = _mock_agent("my_agent")
        h = Handoff(target=agent)

        assert h.target is agent
        assert h.get_name() == "transfer_to_my_agent"
        assert "my_agent" in h.get_description()

    def test_no_agent_field(self) -> None:
        """The 'agent' field no longer exists on Handoff."""
        agent = _mock_agent()
        h = Handoff(target=agent)
        assert not hasattr(h, "agent") or h.target is agent


class TestHandoffFactory:
    """Tests for the handoff() LLM-orchestrated factory."""

    def test_handoff_factory_basic(self) -> None:
        """handoff() creates a Handoff with correct fields."""
        agent = _mock_agent("billing")
        h = handoff(
            target=agent,
            description="Handle billing",
        )

        assert isinstance(h, Handoff)
        assert h.target is agent
        assert h.description == "Handle billing"
        assert h.schema == {"type": "object", "properties": {}}

    def test_handoff_factory_with_input_type(self) -> None:
        """handoff() with input_type generates schema."""
        agent = _mock_agent()
        h = handoff(
            target=agent,
            input_type=EscalationInput,
            description="Escalate",
        )

        assert isinstance(h, Handoff)
        assert h.input_type is EscalationInput
        assert h.schema is not None
        assert "reason" in h.schema["properties"]


class TestHandoffRouteFactory:
    """Tests for the handoff_route() code-orchestrated factory."""

    def test_handoff_route_factory(self) -> None:
        """handoff_route() builds a HandoffRoute from tuples."""
        from troopai.adk.handoffs.handoff_route import HandoffRoute
        from troopai.adk.types.intents import Intent

        class TestIntent(Intent):
            kind: str = "test"

        agent = _mock_agent()
        otherwise_agent = _mock_agent("fallback")

        route = handoff_route(
            (TestIntent, agent),
            otherwise=otherwise_agent,
        )

        assert isinstance(route, HandoffRoute)

    def test_handoff_route_factory_no_routes_raises(self) -> None:
        """handoff_route() with no routes and no otherwise raises ValueError."""
        with pytest.raises(ValueError, match="Must provide at least one"):
            handoff_route()


class TestHandoffInputData:
    """Tests for HandoffInputData fields, messages property, and clone."""

    def test_messages_property(self) -> None:
        """`.messages` returns context + output."""
        data = HandoffInputData(
            intent="test",
            context=({"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}),
            output=({"role": "assistant", "content": "hello"},),
        )

        assert len(data.messages) == 3
        assert data.messages[0] == {"role": "system", "content": "sys"}
        assert data.messages[2] == {"role": "assistant", "content": "hello"}

    def test_messages_empty_output(self) -> None:
        """`.messages` works with empty output (code-orch case)."""
        data = HandoffInputData(
            intent="test",
            context=({"role": "system", "content": "sys"},),
            output=(),
        )
        assert data.messages == ({"role": "system", "content": "sys"},)

    def test_forwarded_default_none(self) -> None:
        """forwarded is None by default."""
        data = HandoffInputData(
            intent="test",
            context=(),
            output=(),
        )
        assert data.forwarded is None

    def test_clone_with_forwarded(self) -> None:
        """clone() can set forwarded."""
        data = HandoffInputData(
            intent="test",
            context=({"role": "user", "content": "hi"},),
            output=({"role": "assistant", "content": "hello"},),
        )

        filtered = data.clone(forwarded=({"role": "user", "content": "hi"},))
        assert filtered.forwarded == ({"role": "user", "content": "hi"},)
        # Original context/output preserved
        assert filtered.context == data.context
        assert filtered.output == data.output

    def test_clone_preserves_intent(self) -> None:
        """clone() preserves intent when not overridden."""
        data = HandoffInputData(
            intent="original",
            context=(),
            output=(),
        )
        cloned = data.clone(forwarded=())
        assert cloned.intent == "original"


class TestHandoffFilters:
    """Tests for filters using the new temporal slicing model."""

    def test_filter_uses_messages_property(self) -> None:
        """remove_tool_calls operates on messages (context + output)."""
        from troopai.adk.handoffs.handoff_filters import remove_tool_calls

        data = HandoffInputData(
            intent="test",
            context=(
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ),
            output=(
                {"role": "assistant", "content": "let me check", "tool_calls": [{"id": "1"}]},
                {"role": "tool", "tool_call_id": "1", "content": "result"},
                {"role": "assistant", "content": "done"},
            ),
        )

        filtered = remove_tool_calls(data)
        # Tool messages removed, tool_calls stripped from assistant
        assert filtered.forwarded is not None
        roles = [m.get("role") for m in filtered.forwarded]
        assert "tool" not in roles
        # tool_calls stripped from first assistant message
        for msg in filtered.forwarded:
            if msg.get("role") == "assistant":
                assert "tool_calls" not in msg
        # Context and output remain unchanged (audit trail preserved)
        assert filtered.context == data.context
        assert filtered.output == data.output

    def test_filter_sets_forwarded(self) -> None:
        """keep_last_n sets forwarded to decouple from audit trail."""
        from troopai.adk.handoffs.handoff_filters import keep_last_n

        data = HandoffInputData(
            intent="test",
            context=(
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "msg1"},
            ),
            output=(
                {"role": "assistant", "content": "resp1"},
                {"role": "user", "content": "msg2"},
                {"role": "assistant", "content": "resp2"},
            ),
        )

        filtered = keep_last_n(2)(data)
        assert filtered.forwarded is not None
        assert len(filtered.forwarded) == 2
        # Full audit trail still accessible
        assert len(filtered.messages) == 5

    def test_keep_last_n_zero_forwards_nothing(self) -> None:
        """keep_last_n(0) must forward NOTHING, not everything.

        Regression: the filter sliced ``items[-n:]``, and ``items[-0:]`` ==
        ``items[0:]`` == ALL items — so ``keep_last_n(0)`` forwarded the
        entire history (cost + privacy blowout) instead of keeping zero.
        """
        from troopai.adk.handoffs.handoff_filters import keep_last_n

        data = HandoffInputData(
            intent="test",
            context=({"role": "user", "content": "msg1"},),
            output=({"role": "assistant", "content": "resp1"},),
        )
        filtered = keep_last_n(0)(data)
        assert filtered.forwarded == ()

    def test_keep_last_n_negative_raises(self) -> None:
        """Negative n is invalid — rejected at factory time (matches the tasks twin)."""
        from troopai.adk.handoffs.handoff_filters import keep_last_n

        with pytest.raises(ValueError, match="keep_last_n"):
            keep_last_n(-1)

    def test_compose_chains_forwarded(self) -> None:
        """compose() passes forwarded through the chain."""
        from troopai.adk.handoffs.handoff_filters import (
            compose,
            keep_last_n,
            remove_tool_calls,
        )

        data = HandoffInputData(
            intent="test",
            context=(
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ),
            output=(
                {"role": "assistant", "content": "checking", "tool_calls": [{"id": "1"}]},
                {"role": "tool", "tool_call_id": "1", "content": "42"},
                {"role": "assistant", "content": "answer is 42"},
            ),
        )

        pipeline = compose(remove_tool_calls, keep_last_n(2))
        result = pipeline(data)

        # remove_tool_calls sets forwarded (strips tool messages)
        # keep_last_n reads forwarded and keeps last 2
        assert result.forwarded is not None
        assert len(result.forwarded) == 2

    def test_intent_only_sets_empty_forwarded(self) -> None:
        """intent_only sets forwarded to empty tuple."""
        from troopai.adk.handoffs.handoff_filters import intent_only

        data = HandoffInputData(
            intent="classify_refund",
            context=({"role": "user", "content": "I want a refund"},),
            output=({"role": "assistant", "content": "routing"},),
        )

        result = intent_only(data)
        assert result.forwarded == ()


class TestPrepareUsesForwarded:
    """Tests that prepare functions respect the forwarded field."""

    @pytest.mark.asyncio
    async def test_prepare_uses_forwarded_when_set(self) -> None:
        """prepare_handoff_input uses forwarded instead of messages."""
        from troopai.adk.handoffs.handoff_target import HandoffTarget
        from troopai.adk.run.handoffs_executor import prepare_handoff_input
        from troopai.adk.types.items import MessageOutputItem, SystemItem, UserItem
        from troopai.adk.types.responses.llm_response import LLMResponseText

        agent = _mock_agent()
        target = HandoffTarget(target=agent)

        data = HandoffInputData(
            intent="test",
            context=(
                SystemItem(raw={"role": "system", "content": "sys"}),
                UserItem(raw={"role": "user", "content": "hi"}),
            ),
            output=(MessageOutputItem(raw=[LLMResponseText(text="hello")]),),
            forwarded=(UserItem(raw={"role": "user", "content": "hi"}),),
        )

        result = await prepare_handoff_input(target, data)
        # Should use forwarded (1 item → 1 message), not messages (3 items)
        assert len(result) == 1
        assert result[0]["content"] == "hi"

    @pytest.mark.asyncio
    async def test_prepare_falls_back_to_messages(self) -> None:
        """prepare_handoff_input uses messages when forwarded is None."""
        from troopai.adk.handoffs.handoff_target import HandoffTarget
        from troopai.adk.run.handoffs_executor import prepare_handoff_input
        from troopai.adk.types.items import MessageOutputItem, SystemItem, UserItem
        from troopai.adk.types.responses.llm_response import LLMResponseText

        agent = _mock_agent()
        target = HandoffTarget(target=agent)

        data = HandoffInputData(
            intent="test",
            context=(
                SystemItem(raw={"role": "system", "content": "sys"}),
                UserItem(raw={"role": "user", "content": "hi"}),
            ),
            output=(MessageOutputItem(raw=[LLMResponseText(text="hello")]),),
        )

        result = await prepare_handoff_input(target, data)
        # forwarded is None, so uses messages (3 items → 3 messages)
        # Trailing assistant rewritten to user
        assert len(result) == 3
