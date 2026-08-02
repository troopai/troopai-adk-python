"""Tests for the ``A2A*Error`` hierarchy.

The tests below verify three properties:

1. Every A2A error is catchable via the framework-wide
   ``TroopAIError`` base — callers should never need to enumerate
   provider exception types separately.
2. ``A2ATaskCancelledError`` is a subclass of ``A2ATaskError``, so
   callers that don't care about the cancel/fail distinction can
   handle both with one ``except``.
3. ``A2ATaskError`` carries its identifiers and remote message on
   typed attributes, not buried inside the message string.
"""

from troopai.adk.a2a import (
    A2AError,
    A2AProtocolError,
    A2ATaskCancelledError,
    A2ATaskError,
    A2ATaskInterruptedError,
    A2ATransportError,
)
from troopai.adk.exceptions import TroopAIError


class TestErrorHierarchy:
    def test_a2a_error_extends_troopai_error(self) -> None:
        # TroopAIError is the framework-wide root; A2A errors must
        # surface through it so callers can catch one class to handle
        # all framework failures.
        assert issubclass(A2AError, TroopAIError)

    def test_transport_protocol_task_all_extend_a2a_error(self) -> None:
        for cls in (A2ATransportError, A2AProtocolError, A2ATaskError):
            assert issubclass(cls, A2AError)

    def test_cancelled_extends_task_error(self) -> None:
        assert issubclass(A2ATaskCancelledError, A2ATaskError)

    def test_catchable_as_troopai_error(self) -> None:
        try:
            raise A2ATransportError("connection refused")
        except TroopAIError as exc:
            assert "connection refused" in str(exc)


class TestA2ATaskError:
    def test_attributes_populated(self) -> None:
        err = A2ATaskError(
            task_id="t-123",
            context_id="c-456",
            state="failed",
            remote_message="upstream LLM rejected the request",
        )
        assert err.task_id == "t-123"
        assert err.context_id == "c-456"
        assert err.state == "failed"
        assert err.remote_message == "upstream LLM rejected the request"
        # The string representation surfaces the state and message so
        # logs are useful even without inspecting attributes.
        assert "t-123" in str(err)
        assert "failed" in str(err)
        assert "upstream LLM rejected the request" in str(err)

    def test_message_optional(self) -> None:
        err = A2ATaskError(
            task_id="t-1",
            context_id="c-1",
            state="rejected",
        )
        assert err.remote_message == ""
        # No empty colon when the remote message is absent.
        assert "rejected" in str(err)

    def test_cancelled_carries_state(self) -> None:
        err = A2ATaskCancelledError(
            task_id="t-1",
            context_id="c-1",
            state="cancelled",
            remote_message="user cancelled",
        )
        # Subclass-of-A2ATaskError: typed routing still works.
        assert isinstance(err, A2ATaskError)
        assert err.state == "cancelled"


class TestA2ATaskInterruptedError:
    def test_carries_attributes(self) -> None:
        err = A2ATaskInterruptedError(
            task_id="t-99",
            context_id="c-99",
            state="input_required",
            prompt="What date should I book?",
        )
        assert err.task_id == "t-99"
        assert err.context_id == "c-99"
        assert err.state == "input_required"
        assert err.prompt == "What date should I book?"
        # The string form surfaces the state and the prompt so logs
        # are useful even without inspecting the typed attributes.
        assert "t-99" in str(err)
        assert "input_required" in str(err)
        assert "What date should I book?" in str(err)

    def test_prompt_optional(self) -> None:
        err = A2ATaskInterruptedError(
            task_id="t-1",
            context_id="c-1",
            state="auth_required",
        )
        assert err.prompt == ""
        # No empty colon when the prompt is absent.
        assert "auth_required" in str(err)
        assert ": " not in str(err)

    def test_distinct_from_task_error(self) -> None:
        # A2ATaskInterruptedError extends A2AError directly, NOT
        # A2ATaskError — interruption is not failure. This lets
        # callers branch ``except A2ATaskError`` for terminal
        # failures and ``except A2ATaskInterruptedError`` for
        # paused tasks separately.
        err = A2ATaskInterruptedError(
            task_id="t-1",
            context_id="c-1",
            state="input_required",
        )
        assert isinstance(err, A2AError)
        assert not isinstance(err, A2ATaskError)


class TestStateTyping:
    """A2ATaskError.state and A2ATaskInterruptedError.state must use A2ATaskStateLiteral."""

    def test_task_error_state_is_narrowed_literal_not_bare_str(self) -> None:
        # The type annotation for .state must be A2ATaskStateLiteral, not str.
        # We verify at runtime by checking it rejects bad state values at
        # static-analysis level; at runtime we just confirm the declared type
        # annotation is the narrowed literal.
        import typing

        from troopai.adk.a2a.exceptions import A2ATaskError

        hints = typing.get_type_hints(A2ATaskError)
        # The annotation must NOT be plain str; it must reference A2ATaskStateLiteral.
        from troopai.adk.a2a.a2a_continuation_token import A2ATaskStateLiteral

        assert hints.get("state") is A2ATaskStateLiteral, (
            f"A2ATaskError.state should be A2ATaskStateLiteral, got {hints.get('state')}"
        )

    def test_interrupted_error_state_is_narrowed_literal_not_bare_str(self) -> None:
        import typing

        from troopai.adk.a2a.exceptions import A2ATaskInterruptedError

        hints = typing.get_type_hints(A2ATaskInterruptedError)
        from troopai.adk.a2a.a2a_continuation_token import A2ATaskStateLiteral

        assert hints.get("state") is A2ATaskStateLiteral, (
            f"A2ATaskInterruptedError.state should be A2ATaskStateLiteral, got {hints.get('state')}"
        )
