"""Tests for the LLM-orchestrated handoff ``enabled`` callback and the
shared ``evaluate_enabled`` dispatch helper.

The callable contract:
- Bool values pass through unchanged.
- 0-arg callables are invoked with no arguments.
- 1-arg callables receive ``(context)``.
- 2-arg / ``*args`` callables receive ``(context, second_arg)`` — where
  ``second_arg`` is the target ``Agent`` for LLM-orch handoffs or the
  matched ``Intent`` for code-orch routing.
- Both sync and async forms are supported. The return value MUST be a
  bool; non-bool returns, async generators, and unintrospectable
  signatures raise ``HandoffDefinitionError``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from troopai.adk.exceptions import HandoffDefinitionError
from troopai.adk.handoffs.handoff import Handoff
from troopai.adk.handoffs.handoff_helpers import evaluate_enabled, is_handoff_enabled
from troopai.adk.handoffs.handoff_route import HandoffRoute
from troopai.adk.handoffs.handoff_target import HandoffTarget
from troopai.adk.run.context import RunContext


def _mock_agent(name: str = "destination") -> MagicMock:
    """Build a mock Agent with the attributes Handoff touches."""
    agent = MagicMock()
    agent.name = name
    agent.description = None
    return agent


def _run_context(payload: dict[str, Any] | None = None) -> RunContext[dict[str, Any]]:
    """Build a real RunContext carrying an optional payload."""
    return RunContext(context=payload or {})


class TestEnabledBool:
    """Plain bool ``enabled`` resolves without invoking any callable."""

    async def test_true_is_enabled(self) -> None:
        h = Handoff(target=_mock_agent(), enabled=True)
        assert await is_handoff_enabled(h, _run_context()) is True

    async def test_false_is_disabled(self) -> None:
        h = Handoff(target=_mock_agent(), enabled=False)
        assert await is_handoff_enabled(h, _run_context()) is False

    async def test_bool_does_not_require_context(self) -> None:
        # Bool form must not raise when context is None — only the
        # callable form needs a context.
        h = Handoff(target=_mock_agent(), enabled=True)
        assert await is_handoff_enabled(h, None) is True


class TestEnabledCallableTwoArg:
    """2-arg ``(context, target_agent) -> bool``."""

    async def test_sync_callable_receives_context_and_target(self) -> None:
        target = _mock_agent("refunds")
        captured: dict[str, Any] = {}

        def gate(ctx: RunContext[Any], agent: Any) -> bool:
            captured["ctx"] = ctx
            captured["agent"] = agent
            return True

        h = Handoff(target=target, enabled=gate)
        ctx = _run_context({"tier": "premium"})
        assert await is_handoff_enabled(h, ctx) is True
        assert captured["ctx"] is ctx
        assert captured["agent"] is target

    async def test_sync_callable_returns_false_disables(self) -> None:
        def gate(ctx: RunContext[Any], agent: Any) -> bool:
            return False

        h = Handoff(target=_mock_agent(), enabled=gate)
        assert await is_handoff_enabled(h, _run_context()) is False

    async def test_async_callable_is_awaited(self) -> None:
        async def gate(ctx: RunContext[Any], agent: Any) -> bool:
            return bool(ctx.context.get("premium"))

        h = Handoff(target=_mock_agent(), enabled=gate)
        assert await is_handoff_enabled(h, _run_context({"premium": True})) is True
        assert await is_handoff_enabled(h, _run_context({"premium": False})) is False


class TestEnabledCallableOneArg:
    """1-arg ``(context) -> bool`` — context-only gate."""

    async def test_one_arg_callable_receives_context(self) -> None:
        captured: dict[str, Any] = {}

        def gate(ctx: RunContext[Any]) -> bool:
            captured["ctx"] = ctx
            return True

        h = Handoff(target=_mock_agent(), enabled=gate)
        ctx = _run_context()
        assert await is_handoff_enabled(h, ctx) is True
        assert captured["ctx"] is ctx

    async def test_one_arg_async_callable(self) -> None:
        async def gate(ctx: RunContext[Any]) -> bool:
            return False

        h = Handoff(target=_mock_agent(), enabled=gate)
        assert await is_handoff_enabled(h, _run_context()) is False


class TestEnabledCallableZeroArg:
    """0-arg ``() -> bool`` — for global feature flags."""

    async def test_zero_arg_sync(self) -> None:
        invocations = []

        def gate() -> bool:
            invocations.append(1)
            return True

        h = Handoff(target=_mock_agent(), enabled=gate)
        assert await is_handoff_enabled(h, _run_context()) is True
        assert len(invocations) == 1

    async def test_zero_arg_async(self) -> None:
        invocations = []

        async def gate() -> bool:
            invocations.append(1)
            return False

        h = Handoff(target=_mock_agent(), enabled=gate)
        assert await is_handoff_enabled(h, _run_context()) is False
        assert len(invocations) == 1


class TestEnabledCallableVariadic:
    """``*args`` callables are dispatched with the full 2-arg form."""

    async def test_var_positional_receives_context_and_target(self) -> None:
        target = _mock_agent("variadic")
        captured: list[Any] = []

        def gate(*args: Any) -> bool:
            captured.extend(args)
            return True

        h = Handoff(target=target, enabled=gate)
        ctx = _run_context()
        assert await is_handoff_enabled(h, ctx) is True
        assert len(captured) == 2
        assert captured[0] is ctx
        assert captured[1] is target


class TestEnabledMisconfiguration:
    """Misconfigured callables raise HandoffDefinitionError."""

    async def test_callable_without_context_raises(self) -> None:
        def gate(ctx: RunContext[Any], agent: Any) -> bool:
            return True

        h = Handoff(target=_mock_agent("billing"), enabled=gate)
        with pytest.raises(HandoffDefinitionError) as exc_info:
            await is_handoff_enabled(h, None)
        assert "billing" in str(exc_info.value)
        assert exc_info.value.handoff_name == "transfer_to_billing"

    async def test_non_bool_return_raises(self) -> None:
        def gate(ctx: RunContext[Any], agent: Any) -> Any:
            return "yes"

        h = Handoff(target=_mock_agent("oops"), enabled=gate)
        with pytest.raises(HandoffDefinitionError) as exc_info:
            await is_handoff_enabled(h, _run_context())
        msg = str(exc_info.value)
        assert "bool" in msg
        assert "str" in msg

    async def test_none_return_raises(self) -> None:
        # Forgetting to return is a common bug — surface it instead of
        # silently treating None as False.
        def gate(ctx: RunContext[Any], agent: Any) -> Any:
            return None

        h = Handoff(target=_mock_agent(), enabled=gate)
        with pytest.raises(HandoffDefinitionError):
            await is_handoff_enabled(h, _run_context())

    async def test_async_generator_return_raises(self) -> None:
        # An async generator function call returns an async-generator
        # object — truthy, never awaited. Surface it so the gate isn't
        # silently treated as enabled.
        async def gate(ctx: RunContext[Any], agent: Any):
            yield True

        # Intentionally passing an async-generator function to test the
        # rejection path — its return type does not satisfy
        # `MaybeAwaitable[bool]`, which is what we want to verify.
        h = Handoff(target=_mock_agent("gen"), enabled=gate)  # pyright: ignore[reportArgumentType]
        with pytest.raises(HandoffDefinitionError) as exc_info:
            await is_handoff_enabled(h, _run_context())
        assert "async generator" in str(exc_info.value)

    async def test_unintrospectable_signature_raises(self) -> None:
        # Some C-implemented callables (and certain builtins) don't
        # expose a Python signature. Surface with a clear error.
        # `dict.update` does not match `HandoffEnabledCallback`; we
        # pass it specifically to test the introspection failure path.
        h = Handoff(target=_mock_agent("c_callable"), enabled=dict.update)  # pyright: ignore[reportArgumentType]
        with pytest.raises(HandoffDefinitionError) as exc_info:
            await is_handoff_enabled(h, _run_context())
        assert "introspect" in str(exc_info.value)

    async def test_keyword_only_required_param_raises(self) -> None:
        # `def gate(*, flag: bool)` has no positional slot the
        # dispatcher can fill — must surface as misconfiguration
        # rather than the user seeing a raw TypeError.
        def gate(*, flag: bool) -> bool:
            return flag

        h = Handoff(target=_mock_agent("kw"), enabled=gate)
        with pytest.raises(HandoffDefinitionError) as exc_info:
            await is_handoff_enabled(h, _run_context())
        assert "keyword-only" in str(exc_info.value).lower()


class TestCallableDispatchIsActuallyInvoked:
    """A callable returning False MUST disable the handoff.

    Guards against the dead-branch regression where a callable
    ``enabled`` was unconditionally treated as True regardless of its
    return value.
    """

    async def test_callable_returning_false_actually_disables(self) -> None:
        def always_false(ctx: RunContext[Any], agent: Any) -> bool:
            return False

        h = Handoff(target=_mock_agent(), enabled=always_false)
        assert await is_handoff_enabled(h, _run_context()) is False

    async def test_callable_observes_context_state(self) -> None:
        def gate(ctx: RunContext[Any], agent: Any) -> bool:
            return ctx.context.get("flag") is True

        h = Handoff(target=_mock_agent(), enabled=gate)
        assert await is_handoff_enabled(h, _run_context({"flag": True})) is True
        assert await is_handoff_enabled(h, _run_context({"flag": False})) is False


class TestCodeOrchSharedContract:
    """LLM-orch and code-orch share one dispatch contract.

    Same arity rules, same guards, same error type. These tests pin
    the contract at the code-orch (HandoffRoute._is_enabled) entry
    point so the two paths can never drift apart silently.
    """

    async def test_code_orch_callable_returning_false_disables(self) -> None:
        from typing import Literal

        from troopai.adk.types.intents import Intent

        class RefundIntent(Intent):
            kind: Literal["refund"] = "refund"

        def gate(ctx: RunContext[Any], intent: Intent) -> bool:
            return False

        target = HandoffTarget(target=_mock_agent(), enabled=gate)
        assert await HandoffRoute._is_enabled(target, RefundIntent(), _run_context()) is False

    async def test_code_orch_callable_without_context_raises(self) -> None:
        from typing import Literal

        from troopai.adk.types.intents import Intent

        class RefundIntent(Intent):
            kind: Literal["refund"] = "refund"

        def gate(ctx: RunContext[Any], intent: Intent) -> bool:
            return True

        target = HandoffTarget(target=_mock_agent("refunds"), enabled=gate)
        with pytest.raises(HandoffDefinitionError):
            await HandoffRoute._is_enabled(target, RefundIntent(), None)

    async def test_code_orch_non_bool_return_raises(self) -> None:
        from typing import Literal

        from troopai.adk.types.intents import Intent

        class RefundIntent(Intent):
            kind: Literal["refund"] = "refund"

        def gate(ctx: RunContext[Any], intent: Intent) -> Any:
            return 1

        target = HandoffTarget(target=_mock_agent(), enabled=gate)
        with pytest.raises(HandoffDefinitionError):
            await HandoffRoute._is_enabled(target, RefundIntent(), _run_context())


class TestEvaluateEnabledDirect:
    """Direct exercise of the shared helper outside the Handoff /
    HandoffRoute callers — pins the public contract callers depend on.
    """

    async def test_callable_with_var_positional_uses_two_args(self) -> None:
        captured: list[Any] = []

        def gate(*args: Any) -> bool:
            captured.extend(args)
            return True

        marker = object()
        await evaluate_enabled(gate, _run_context(), marker, handoff_name="t")
        assert len(captured) == 2
        assert captured[1] is marker

    async def test_async_callable_awaited(self) -> None:
        async def gate(ctx: RunContext[Any], second: Any) -> bool:
            return True

        result = await evaluate_enabled(gate, _run_context(), object(), handoff_name="t")
        assert result is True
