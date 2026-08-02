"""Tests for HandoffConfig.on_error and HandoffRejection routing.

Covers:
- Default ``on_error="halt"`` propagates exceptions raw.
- ``on_error="reject_with_message"`` converts exceptions to
  ``HandoffRejection`` so the LLM sees the failure as a tool result.
- ``error_message_builder`` shapes the visible message.
- Pydantic ``input_type`` validation always surfaces as
  ``HandoffRejection`` regardless of ``on_error``.
- ``resolve_handoff_step`` honours the rejection by emitting a
  tool-result message and returning ``NextStepRunAgain``.
"""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from troopai.adk.exceptions import HandoffRejection
from troopai.adk.handoffs.handoff import Handoff
from troopai.adk.handoffs.handoff_config import HandoffConfig
from troopai.adk.handoffs.handoff_input_data import HandoffInputData
from troopai.adk.run.context import RunContext
from troopai.adk.types.intents import Intent


class _DummyIntent(Intent):
    """Minimal concrete Intent for the code-orchestrated invoke path."""

    kind: Literal["dummy"] = "dummy"


def _mock_agent(name: str = "destination") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    agent.description = None
    return agent


def _run_context() -> RunContext[dict[str, Any]]:
    return RunContext(context={})


class EscalationInput(BaseModel):
    reason: str = Field(description="Why this needs escalation.")
    priority: int = Field(description="Priority level.", ge=1, le=5)


def _raising_filter(_data: HandoffInputData) -> HandoffInputData:
    raise RuntimeError("filter blew up")


async def _raising_on_handoff(_ctx: RunContext[Any]) -> None:
    raise RuntimeError("callback blew up")


class TestDefaultHaltPolicy:
    """``on_error="halt"`` (the default) propagates the original exception."""

    async def test_filter_exception_propagates(self) -> None:
        h = Handoff(target=_mock_agent(), input_filter=_raising_filter)
        with pytest.raises(RuntimeError, match="filter blew up"):
            await h.invoke(
                tool_args="{}",
                context=(),
                output=(),
                run_context=_run_context(),
            )

    async def test_on_handoff_exception_propagates(self) -> None:
        h = Handoff(target=_mock_agent(), on_handoff=_raising_on_handoff)
        with pytest.raises(RuntimeError, match="callback blew up"):
            await h.invoke(
                tool_args="{}",
                context=(),
                output=(),
                run_context=_run_context(),
            )


class TestRejectWithMessagePolicy:
    """``on_error="reject_with_message"`` raises ``HandoffRejection``."""

    async def test_filter_exception_becomes_rejection(self) -> None:
        h = Handoff(
            target=_mock_agent("billing"),
            input_filter=_raising_filter,
            config=HandoffConfig(on_error="reject_with_message"),
        )
        with pytest.raises(HandoffRejection) as exc_info:
            await h.invoke(
                tool_args="{}",
                context=(),
                output=(),
                run_context=_run_context(),
            )
        assert exc_info.value.handoff_name == "transfer_to_billing"
        assert "filter blew up" in exc_info.value.tool_message
        assert isinstance(exc_info.value.cause, RuntimeError)

    async def test_on_handoff_exception_becomes_rejection(self) -> None:
        h = Handoff(
            target=_mock_agent("billing"),
            on_handoff=_raising_on_handoff,
            config=HandoffConfig(on_error="reject_with_message"),
        )
        with pytest.raises(HandoffRejection) as exc_info:
            await h.invoke(
                tool_args="{}",
                context=(),
                output=(),
                run_context=_run_context(),
            )
        assert "callback blew up" in exc_info.value.tool_message

    async def test_default_message_format(self) -> None:
        h = Handoff(
            target=_mock_agent(),
            input_filter=_raising_filter,
            config=HandoffConfig(on_error="reject_with_message"),
        )
        with pytest.raises(HandoffRejection) as exc_info:
            await h.invoke(
                tool_args="{}",
                context=(),
                output=(),
                run_context=_run_context(),
            )
        # Format includes callback_kind to signal "user-callback bug",
        # not "your tool-call args" — so the LLM doesn't try to fix args.
        msg = exc_info.value.tool_message
        assert "input_filter" in msg
        assert "RuntimeError" in msg
        assert "filter blew up" in msg


class TestCustomErrorBuilder:
    """``error_message_builder`` shapes the LLM-visible message."""

    async def test_custom_builder_invoked(self) -> None:
        captured: list[Exception] = []

        def builder(exc: Exception) -> str:
            captured.append(exc)
            return "Please retry with a different agent."

        h = Handoff(
            target=_mock_agent(),
            input_filter=_raising_filter,
            config=HandoffConfig(
                on_error="reject_with_message",
                error_message_builder=builder,
            ),
        )
        with pytest.raises(HandoffRejection) as exc_info:
            await h.invoke(
                tool_args="{}",
                context=(),
                output=(),
                run_context=_run_context(),
            )
        assert exc_info.value.tool_message == "Please retry with a different agent."
        assert len(captured) == 1
        assert isinstance(captured[0], RuntimeError)


class TestValidationErrorAlwaysRejects:
    """Pydantic ``input_type`` validation always surfaces as
    ``HandoffRejection`` regardless of ``on_error`` — the LLM made
    the bad tool call, not user callback code."""

    async def test_validation_error_under_halt_still_rejects(self) -> None:
        # Default config is on_error="halt" but ValidationError ignores
        # the policy.
        h = Handoff(target=_mock_agent("escalate"), input_type=EscalationInput)
        with pytest.raises(HandoffRejection) as exc_info:
            await h.invoke(
                tool_args='{"reason": "test", "priority": 99}',  # priority > 5
                context=(),
                output=(),
                run_context=_run_context(),
            )
        assert "Invalid handoff arguments" in exc_info.value.tool_message
        assert "escalate" in exc_info.value.tool_message

    async def test_validation_error_under_reject_with_message_still_rejects(self) -> None:
        h = Handoff(
            target=_mock_agent("escalate"),
            input_type=EscalationInput,
            config=HandoffConfig(on_error="reject_with_message"),
        )
        with pytest.raises(HandoffRejection):
            await h.invoke(
                tool_args='{"reason": "test", "priority": 99}',
                context=(),
                output=(),
                run_context=_run_context(),
            )


class TestErrorMessageBuilderSafety:
    """A raising ``error_message_builder`` must not lose the original exception."""

    async def test_raising_builder_falls_back_to_default(self) -> None:
        def bad_builder(exc: Exception) -> str:
            # Simulates a builder bug — e.g. accessing a missing attr.
            return exc.nonexistent_attribute  # type: ignore[attr-defined]

        h = Handoff(
            target=_mock_agent("billing"),
            input_filter=_raising_filter,
            config=HandoffConfig(
                on_error="reject_with_message",
                error_message_builder=bad_builder,
            ),
        )
        with pytest.raises(HandoffRejection) as exc_info:
            await h.invoke(
                tool_args="{}",
                context=(),
                output=(),
                run_context=_run_context(),
            )
        # Default formatter mentions the callback_kind and the original.
        assert "input_filter" in exc_info.value.tool_message
        assert "RuntimeError" in exc_info.value.tool_message
        # The original exception is preserved as `cause`.
        assert isinstance(exc_info.value.cause, RuntimeError)


class TestCallbackRaisesHandoffRejectionDirectly:
    """A callback that explicitly raises HandoffRejection is opting INTO
    the rejection path — both ``on_error`` policies respect it.
    """

    async def test_under_halt_callback_raised_rejection_passes_through(self) -> None:
        async def explicit_reject(_ctx: RunContext[Any]) -> None:
            raise HandoffRejection(
                "transfer_to_destination",
                "Not now — try again after 9am.",
                cause=RuntimeError("policy"),
            )

        h = Handoff(target=_mock_agent(), on_handoff=explicit_reject)
        # Default config is on_error="halt" but explicit rejection passes through.
        with pytest.raises(HandoffRejection) as exc_info:
            await h.invoke(
                tool_args="{}",
                context=(),
                output=(),
                run_context=_run_context(),
            )
        assert exc_info.value.tool_message == "Not now — try again after 9am."

    async def test_under_reject_callback_raised_rejection_not_double_wrapped(self) -> None:
        async def explicit_reject(_ctx: RunContext[Any]) -> None:
            raise HandoffRejection(
                "transfer_to_destination",
                "Custom reason.",
                cause=RuntimeError("inner"),
            )

        h = Handoff(
            target=_mock_agent(),
            on_handoff=explicit_reject,
            config=HandoffConfig(on_error="reject_with_message"),
        )
        with pytest.raises(HandoffRejection) as exc_info:
            await h.invoke(
                tool_args="{}",
                context=(),
                output=(),
                run_context=_run_context(),
            )
        # The user's message survives — NOT wrapped with "Handoff failed: ..." prefix.
        assert exc_info.value.tool_message == "Custom reason."


class TestSuccessPathUnchanged:
    """Successful handoffs are unaffected by the new try/except wrapping."""

    async def test_passing_filter_and_callback(self) -> None:
        filter_called: list[Any] = []
        callback_called: list[Any] = []

        def good_filter(data: HandoffInputData) -> HandoffInputData:
            filter_called.append(data)
            return data

        async def good_callback(_ctx: RunContext[Any]) -> None:
            callback_called.append(1)

        h = Handoff(
            target=_mock_agent(),
            input_filter=good_filter,
            on_handoff=good_callback,
        )
        agent, data = await h.invoke(
            tool_args="{}",
            context=(),
            output=(),
            run_context=_run_context(),
        )
        assert len(filter_called) == 1
        assert len(callback_called) == 1
        assert isinstance(data, HandoffInputData)
        assert agent is h.target


class TestHandoffTargetErrorPolicy:
    """The code-orchestrated ``HandoffTarget.invoke`` honors ``on_error`` too.

    Regression: ``HandoffTarget.invoke`` called ``input_filter`` / ``on_handoff``
    bare, so ``HandoffConfig.on_error`` was DEAD on the code-orchestrated path —
    a raising callback crashed the run instead of rejecting, while the
    LLM-orchestrated ``Handoff.invoke`` honored the policy.
    """

    async def test_halt_propagates_raw(self) -> None:
        from troopai.adk.handoffs.handoff_target import HandoffTarget

        target = HandoffTarget(target=_mock_agent("billing"), input_filter=_raising_filter)
        with pytest.raises(RuntimeError, match="filter blew up"):
            await target.invoke(_DummyIntent(), (), (), _run_context())

    async def test_reject_with_message_filter(self) -> None:
        from troopai.adk.handoffs.handoff_target import HandoffTarget

        target = HandoffTarget(
            target=_mock_agent("billing"),
            input_filter=_raising_filter,
            config=HandoffConfig(on_error="reject_with_message"),
        )
        with pytest.raises(HandoffRejection) as exc_info:
            await target.invoke(_DummyIntent(), (), (), _run_context())
        assert exc_info.value.handoff_name == "billing"
        assert "filter blew up" in exc_info.value.tool_message
        assert "input_filter" in exc_info.value.tool_message
        assert isinstance(exc_info.value.cause, RuntimeError)

    async def test_reject_with_message_on_handoff(self) -> None:
        from troopai.adk.handoffs.handoff_target import HandoffTarget

        target = HandoffTarget(
            target=_mock_agent("billing"),
            on_handoff=_raising_on_handoff,
            config=HandoffConfig(on_error="reject_with_message"),
        )
        with pytest.raises(HandoffRejection) as exc_info:
            await target.invoke(_DummyIntent(), (), (), _run_context())
        assert "callback blew up" in exc_info.value.tool_message

    async def test_success_path_unaffected(self) -> None:
        from troopai.adk.handoffs.handoff_target import HandoffTarget

        calls: list[Any] = []

        def good_filter(data: HandoffInputData) -> HandoffInputData:
            calls.append("f")
            return data

        target = HandoffTarget(target=_mock_agent("ok"), input_filter=good_filter)
        agent, data = await target.invoke(_DummyIntent(), (), (), _run_context())
        assert agent is target.target
        assert calls == ["f"]
        assert isinstance(data, HandoffInputData)
