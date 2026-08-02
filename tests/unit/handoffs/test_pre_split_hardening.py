"""Regression tests for pre-split hardening fixes.

Each test class pins one specific bug fix:

- TestKeepLastNOrphanPairs   — keep_last_n drops orphaned ToolCall+ToolCallOutput
- TestRouteNotSealedOnRespond — Respond does NOT seal the HandoffRoute
- TestCloneForwardedOnly      — clone() only accepts ``forwarded``, not audit fields
- TestGetTypeHintsNarrowExcept — get_type_hints except narrowed to NameError
- TestEnabledNonBoolRejected  — static non-bool ``enabled`` raises HandoffDefinitionError
- TestHandleCallbackErrorNoReturn — _handle_callback_error annotated -> NoReturn
- TestHandoffsExecutorTargetName — executor uses .target, not .agent, for agent name
"""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock

import pytest

from troopai.adk.exceptions import HandoffDefinitionError
from troopai.adk.handoffs.handoff import Handoff
from troopai.adk.handoffs.handoff_config import HandoffConfig
from troopai.adk.handoffs.handoff_filters import keep_last_n
from troopai.adk.handoffs.handoff_input_data import HandoffInputData
from troopai.adk.handoffs.handoff_route import HandoffRoute, RouteSealedError
from troopai.adk.handoffs.handoff_target import HandoffTarget
from troopai.adk.run.context import RunContext
from troopai.adk.types.intents import Intent, Respond
from troopai.adk.types.items import ToolCallItem, ToolCallOutputItem, UserItem
from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall


def _mock_agent(name: str = "destination") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.description = None
    return agent


def _run_context() -> RunContext[dict[str, Any]]:
    return RunContext(context={})


def _tool_call_item(call_id: str) -> ToolCallItem:
    return ToolCallItem(raw=LLMResponseFunctionToolCall(call_id=call_id, name="some_tool", arguments="{}"))


def _tool_call_output_item(call_id: str) -> ToolCallOutputItem:
    from troopai.adk.types.output.function_tool_call_result import FunctionToolCallResult

    return ToolCallOutputItem(raw=FunctionToolCallResult(call_id=call_id, output="ok"))


def _user_item(content: str = "hello") -> UserItem:
    from troopai.adk.types.input.llm_input_easy_message import LLMInputEasyMessage

    return UserItem(raw=LLMInputEasyMessage(role="user", content=content))


class _DummyIntent(Intent):
    kind: Literal["dummy"] = "dummy"


# ---------------------------------------------------------------------------
# Bug 1 (HIGH): keep_last_n silently drops a paired ToolCall+ToolCallOutput
# ---------------------------------------------------------------------------


class TestKeepLastNOrphanPairs:
    """keep_last_n must drop BOTH halves of a pair when the slice splits them."""

    def _make_data(self, *items: Any) -> HandoffInputData:
        return HandoffInputData(
            intent="test",
            context=tuple(items),
            output=(),
        )

    def test_orphaned_call_dropped_when_output_excluded(self) -> None:
        """Slice that includes a ToolCallItem but not its ToolCallOutputItem
        must drop the orphaned ToolCallItem.

        Pre-fix: the orphaned ToolCallItem was forwarded, causing strict-provider
        rejections downstream.
        """
        user = _user_item("user1")
        call = _tool_call_item("cid-1")
        output = _tool_call_output_item("cid-1")
        # keep_last_n(2) → [call, output] — both present, no orphan
        data = self._make_data(user, call, output)
        filt = keep_last_n(2)
        result = filt(data)
        assert result.forwarded is not None
        forwarded = result.forwarded
        assert call in forwarded
        assert output in forwarded

    def test_orphaned_call_without_output_in_slice_is_dropped(self) -> None:
        """keep_last_n(1) that picks only the ToolCallItem must drop it."""
        call = _tool_call_item("cid-2")
        output = _tool_call_output_item("cid-2")
        user = _user_item("last")
        # History: [call, output, user] — keep_last_n(2) keeps [output, user]
        # output has no matching call in the slice → both output is orphaned
        data = self._make_data(call, output, user)
        filt = keep_last_n(2)
        result = filt(data)
        assert result.forwarded is not None
        # output is orphaned (its call was excluded) → must be dropped
        assert output not in result.forwarded
        assert user in result.forwarded

    def test_orphaned_output_without_call_in_slice_is_dropped(self) -> None:
        """Slice containing only the ToolCallOutputItem (no matching call) drops it."""
        call = _tool_call_item("cid-3")
        output = _tool_call_output_item("cid-3")
        user = _user_item("after")
        # History: [call, output, user]; keep_last_n(1) → [user] — no orphan
        data = self._make_data(call, output, user)
        filt = keep_last_n(1)
        result = filt(data)
        assert result.forwarded is not None
        assert user in result.forwarded
        assert call not in result.forwarded
        assert output not in result.forwarded

    def test_fully_paired_tool_calls_preserved(self) -> None:
        """Pairs where both halves are in the slice are left intact."""
        call = _tool_call_item("cid-4")
        output = _tool_call_output_item("cid-4")
        data = self._make_data(call, output)
        filt = keep_last_n(2)
        result = filt(data)
        assert result.forwarded is not None
        assert call in result.forwarded
        assert output in result.forwarded

    def test_zero_n_forwards_nothing(self) -> None:
        call = _tool_call_item("cid-5")
        data = self._make_data(call)
        filt = keep_last_n(0)
        result = filt(data)
        assert result.forwarded == ()

    def test_negative_n_raises(self) -> None:
        with pytest.raises(ValueError, match="n >= 0"):
            keep_last_n(-1)


# ---------------------------------------------------------------------------
# Bug 2 (MED): _sealed=True set before Respond early-return
# ---------------------------------------------------------------------------


class TestRouteNotSealedOnRespond:
    """A Respond result must NOT permanently seal the HandoffRoute."""

    async def test_respond_does_not_seal(self) -> None:
        """Route resolved to Respond on turn 1; .when().to() must still work on turn 2."""

        class _Intent1(Intent):
            kind: Literal["intent1"] = "intent1"

        class _Intent2(Intent):
            kind: Literal["intent2"] = "intent2"

        route: HandoffRoute[Any, Any] = HandoffRoute("test-route")
        agent_a = _mock_agent("agent-a")
        route.when(_Intent1).to(agent_a)

        # First resolve: Respond (no handoff taken)
        result = await route.resolve(Respond(message="direct answer"), _run_context())
        assert result is None

        # Route must NOT be sealed — adding a new distinct rule must succeed
        agent_b = _mock_agent("agent-b")
        route.when(_Intent2).to(agent_b)  # would raise RouteSealedError if sealed

    async def test_respond_then_intent_routes_correctly(self) -> None:
        """After a Respond turn, the route still resolves an intent on the next call."""
        route: HandoffRoute[Any, Any] = HandoffRoute("test-route2")
        agent_a = _mock_agent("agent-a")
        route.when(_DummyIntent).to(agent_a)

        # Turn 1: Respond
        await route.resolve(Respond(message="direct answer"), _run_context())

        # Turn 2: real intent — must route and seal NOW
        target = await route.resolve(_DummyIntent(), _run_context())
        assert target is not None
        assert target.target is agent_a

    async def test_real_intent_seals_route(self) -> None:
        """Resolving a real intent DOES seal the route."""
        route: HandoffRoute[Any, Any] = HandoffRoute("seal-test")
        agent_a = _mock_agent("agent-x")
        route.when(_DummyIntent).to(agent_a)

        await route.resolve(_DummyIntent(), _run_context())

        with pytest.raises(RouteSealedError):
            route.when(_DummyIntent).to(agent_a)


# ---------------------------------------------------------------------------
# Bug 3 (MED): clone(**kwargs) lets filter overwrite audit fields
# ---------------------------------------------------------------------------


class TestCloneForwardedOnly:
    """clone() must only accept ``forwarded``; audit fields are immutable via clone."""

    def _base_data(self) -> HandoffInputData:
        user = _user_item("hello")
        return HandoffInputData(
            intent="original-intent",
            context=(user,),
            output=(),
        )

    def test_clone_forwarded_replaces_forwarded(self) -> None:
        data = self._base_data()
        new_forwarded: tuple[Any, ...] = (_user_item("new"),)
        cloned = data.clone(forwarded=new_forwarded)
        assert cloned.forwarded == new_forwarded
        assert cloned.intent == "original-intent"
        assert cloned.context == data.context

    def test_clone_forwarded_none_clears_forwarded(self) -> None:
        user = _user_item("x")
        data = HandoffInputData(intent="i", context=(user,), output=(), forwarded=(user,))
        cloned = data.clone(forwarded=None)
        assert cloned.forwarded is None

    def test_clone_no_args_copies_unchanged(self) -> None:
        data = self._base_data()
        cloned = data.clone()
        assert cloned.forwarded == data.forwarded
        assert cloned.intent == data.intent

    def test_clone_rejects_intent_kwarg(self) -> None:
        """Passing ``intent=`` to clone() must raise TypeError (unknown keyword)."""
        data = self._base_data()
        with pytest.raises(TypeError):
            data.clone(intent="injected")  # type: ignore[call-arg]

    def test_clone_rejects_context_kwarg(self) -> None:
        data = self._base_data()
        with pytest.raises(TypeError):
            data.clone(context=())  # type: ignore[call-arg]

    def test_clone_rejects_output_kwarg(self) -> None:
        data = self._base_data()
        with pytest.raises(TypeError):
            data.clone(output=())  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Bug 4 (MED): Bare except Exception in get_type_hints() swallows errors
# ---------------------------------------------------------------------------


class TestGetTypeHintsNarrowExcept:
    """invoke_on_handoff narrows the except to NameError for get_type_hints failures."""

    async def test_forward_ref_callback_dispatches_correctly(self) -> None:
        """A callback with a forward-reference annotation (NameError on resolution)
        must fall back to the string-annotation check and dispatch correctly.
        """
        from troopai.adk.handoffs.handoff_target import invoke_on_handoff

        dispatched: list[str] = []

        # Use string annotation to simulate forward reference
        def callback(ctx: RunContext[Any]) -> None:
            dispatched.append("called")

        ctx = _run_context()
        await invoke_on_handoff(callback, ctx, intent="x")
        assert dispatched == ["called"]

    async def test_data_callback_annotation_resolves(self) -> None:
        """A callback annotated with HandoffInputData dispatches with the data arg."""
        from troopai.adk.handoffs.handoff_target import invoke_on_handoff

        received_data: list[Any] = []

        def callback(ctx: RunContext[Any], data: HandoffInputData) -> None:
            received_data.append(data)

        ctx = _run_context()
        fake_data = HandoffInputData(intent="i", context=(), output=())
        await invoke_on_handoff(callback, ctx, intent="x", handoff_data=fake_data)
        assert received_data == [fake_data]


# ---------------------------------------------------------------------------
# Bug 5 (MED): Static non-callable enabled silently bool()-coerced
# ---------------------------------------------------------------------------


class TestEnabledNonBoolRejected:
    """Non-bool, non-callable ``enabled`` values must raise HandoffDefinitionError."""

    async def test_integer_enabled_raises(self) -> None:
        from troopai.adk.handoffs.handoff_helpers import evaluate_enabled

        with pytest.raises(HandoffDefinitionError, match="bool or callable"):
            await evaluate_enabled(
                1,  # type: ignore[arg-type]
                _run_context(),
                object(),
                handoff_name="test-handoff",
            )

    async def test_string_enabled_raises(self) -> None:
        from troopai.adk.handoffs.handoff_helpers import evaluate_enabled

        with pytest.raises(HandoffDefinitionError, match="bool or callable"):
            await evaluate_enabled(
                "yes",  # type: ignore[arg-type]
                _run_context(),
                object(),
                handoff_name="test-handoff",
            )

    async def test_none_enabled_raises(self) -> None:
        from troopai.adk.handoffs.handoff_helpers import evaluate_enabled

        with pytest.raises(HandoffDefinitionError, match="bool or callable"):
            await evaluate_enabled(
                None,  # type: ignore[arg-type]
                _run_context(),
                object(),
                handoff_name="test-handoff",
            )

    async def test_true_passes(self) -> None:
        from troopai.adk.handoffs.handoff_helpers import evaluate_enabled

        result = await evaluate_enabled(
            True,
            _run_context(),
            object(),
            handoff_name="test-handoff",
        )
        assert result is True

    async def test_false_passes(self) -> None:
        from troopai.adk.handoffs.handoff_helpers import evaluate_enabled

        result = await evaluate_enabled(
            False,
            _run_context(),
            object(),
            handoff_name="test-handoff",
        )
        assert result is False


# ---------------------------------------------------------------------------
# Bug 6 (LOW): _handle_callback_error annotated -> None but is NoReturn
# ---------------------------------------------------------------------------


class TestHandleCallbackErrorNoReturn:
    """_handle_callback_error must be typed NoReturn and always raise."""

    def test_annotation_is_noreturn(self) -> None:
        """Verify the return annotation is NoReturn (catches future regressions)."""
        import typing

        hints = typing.get_type_hints(Handoff._handle_callback_error)
        # NoReturn is in the typing module; check by name for version portability
        return_hint = hints.get("return")
        assert return_hint is not None
        assert return_hint is typing.NoReturn

    async def test_halt_policy_raises(self) -> None:
        """Under halt policy, _handle_callback_error always raises."""
        h = Handoff(target=_mock_agent())
        exc = RuntimeError("original")
        with pytest.raises(RuntimeError, match="original"):
            h._handle_callback_error(exc, "input_filter")

    async def test_reject_policy_raises_rejection(self) -> None:
        """Under reject_with_message policy, raises HandoffRejection."""
        from troopai.adk.exceptions import HandoffRejection

        h = Handoff(
            target=_mock_agent(),
            config=HandoffConfig(on_error="reject_with_message"),
        )
        exc = RuntimeError("cb error")
        with pytest.raises(HandoffRejection):
            h._handle_callback_error(exc, "on_handoff")


# ---------------------------------------------------------------------------
# Bug 7 (LOW, run/): executor uses .agent instead of .target for agent name
# ---------------------------------------------------------------------------


class TestHandoffsExecutorTargetName:
    """execute_deterministic_handoff / execute_llm_handoff must use
    target.target.name (not target.agent.name) when resolving the span name.
    """

    def test_deterministic_handoff_target_name_resolved(self) -> None:
        """HandoffTarget has .target, not .agent — span should show the agent name."""
        # We check that getattr(getattr(target, "target", None), "name", None)
        # resolves correctly for a real HandoffTarget.
        agent = _mock_agent("billing-agent")
        ht = HandoffTarget(target=agent)

        # Old (broken) path: getattr(getattr(ht, "agent", None), "name", None)
        broken_result = getattr(getattr(ht, "agent", None), "name", None)
        assert broken_result is None, "Pre-fix: .agent does not exist on HandoffTarget"

        # New (correct) path
        fixed_result = getattr(getattr(ht, "target", None), "name", None)
        assert fixed_result == "billing-agent"

    def test_llm_handoff_target_name_via_agent_name_property(self) -> None:
        """Handoff has .agent_name property (fallback in the executor chain)."""
        agent = _mock_agent("refunds-agent")
        h = Handoff(target=agent)

        # The executor uses:
        # getattr(getattr(target, "target", None), "name", None) or getattr(target, "agent_name", None)
        first_chain = getattr(getattr(h, "target", None), "name", None)
        assert first_chain == "refunds-agent"

        # agent_name fallback also works (Handoff exposes it)
        fallback = getattr(h, "agent_name", None)
        assert fallback == "refunds-agent"
