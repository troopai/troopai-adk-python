"""Regression tests for ``handoff_target.invoke_on_handoff`` dispatch.

Pins the signature-detection logic that picks between the three supported
``on_handoff`` callback shapes — ``(ctx)``, ``(ctx, intent)``, and
``(ctx, data: HandoffInputData)``.

Focus: the NameError fallback path. When a callback's second-parameter
annotation is an unresolvable forward-reference string (``from __future__
import annotations`` plus a missing import), ``typing.get_type_hints``
raises ``NameError`` and dispatch falls back to inspecting the raw string
annotation. Only a *bare* ``HandoffInputData`` annotation is a data
callback; container/composite annotations (``list[HandoffInputData]``,
``HandoffInputData | None``) are intent callbacks — matching the resolved
path's ``ann is HandoffInputData`` behaviour.
"""

from __future__ import annotations

import typing
from typing import Any

from troopai.adk.handoffs.handoff_input_data import HandoffInputData
from troopai.adk.handoffs.handoff_target import invoke_on_handoff
from troopai.adk.run.context import RunContext


def _run_context() -> RunContext[dict[str, Any]]:
    return RunContext(context={})


def _callback_without_handoff_input_data_in_globals(
    annotation: str,
    received: list[Any],
) -> Any:
    """Build a 2-arg callback in a module namespace that lacks the
    ``HandoffInputData`` name, so ``get_type_hints`` raises ``NameError``
    and the string-annotation fallback path is exercised.
    """
    src = f"from __future__ import annotations\ndef cb(ctx, data: {annotation}):\n    received.append(data)\n"
    module_globals: dict[str, Any] = {"received": received}
    exec(src, module_globals)
    return module_globals["cb"]


class TestInvokeOnHandoffNameErrorFallback:
    """The string-annotation fallback must mirror the resolved-path check."""

    async def test_get_type_hints_raises_for_unresolvable_annotation(self) -> None:
        """Sanity check: the constructed callback really hits the NameError
        path (otherwise the regression below would pass trivially)."""
        cb = _callback_without_handoff_input_data_in_globals("list[HandoffInputData]", [])
        try:
            typing.get_type_hints(cb)
            raise AssertionError("expected get_type_hints to raise NameError")
        except NameError:
            pass

    async def test_container_annotation_dispatches_intent_not_data(self) -> None:
        """``list[HandoffInputData]`` contains the substring but is NOT a
        data callback — it must receive the intent, not the bare
        ``HandoffInputData``. (Substring match would wrongly pass the data.)"""
        received: list[Any] = []
        cb = _callback_without_handoff_input_data_in_globals("list[HandoffInputData]", received)

        ctx = _run_context()
        data = HandoffInputData(intent="my-intent", context=(), output=())
        await invoke_on_handoff(cb, ctx, intent="my-intent", handoff_data=data)

        assert received == ["my-intent"]
        assert not isinstance(received[0], HandoffInputData)

    async def test_optional_annotation_dispatches_intent_not_data(self) -> None:
        """``HandoffInputData | None`` is treated as an intent callback on the
        resolved path; the fallback must agree and pass the intent."""
        received: list[Any] = []
        cb = _callback_without_handoff_input_data_in_globals("HandoffInputData | None", received)

        ctx = _run_context()
        data = HandoffInputData(intent="my-intent", context=(), output=())
        await invoke_on_handoff(cb, ctx, intent="my-intent", handoff_data=data)

        assert received == ["my-intent"]

    async def test_bare_forward_ref_dispatches_data(self) -> None:
        """A bare unresolvable ``HandoffInputData`` annotation IS a data
        callback — the fallback must still pass the full handoff data."""
        received: list[Any] = []
        cb = _callback_without_handoff_input_data_in_globals("HandoffInputData", received)

        ctx = _run_context()
        data = HandoffInputData(intent="my-intent", context=(), output=())
        await invoke_on_handoff(cb, ctx, intent="my-intent", handoff_data=data)

        assert received == [data]
        assert isinstance(received[0], HandoffInputData)


class TestInvokeOnHandoffPositionalArity:
    """Only positional parameters count toward arity.

    Keyword-only params and ``**kwargs`` cannot receive the
    positionally-passed second argument, so a callback like
    ``(ctx, *, flag)`` or ``(ctx, **kw)`` is a one-positional ``(ctx)``
    callback. Counting them as positional makes dispatch pass a spurious
    second positional and raise ``TypeError``.
    """

    async def test_keyword_only_second_param_dispatches_context_only(self) -> None:
        """``(ctx, *, flag)`` has ONE positional; it must be called with
        just ``(ctx)`` — not ``(ctx, intent)`` (which raises TypeError)."""
        received: list[Any] = []

        def cb(ctx: Any, *, flag: bool = False) -> None:
            received.append((ctx, flag))

        ctx = _run_context()
        data = HandoffInputData(intent="i", context=(), output=())
        await invoke_on_handoff(cb, ctx, intent="i", handoff_data=data)

        assert len(received) == 1
        assert received[0][0] is ctx
        assert received[0][1] is False

    async def test_var_keyword_dispatches_context_only(self) -> None:
        """``(ctx, **kw)`` has ONE positional; passing a second positional
        raises TypeError, so it must be called with just ``(ctx)``."""
        received: list[Any] = []

        def cb(ctx: Any, **kw: Any) -> None:
            received.append(ctx)

        ctx = _run_context()
        await invoke_on_handoff(cb, ctx, intent="i")

        assert received == [ctx]

    async def test_var_positional_receives_intent(self) -> None:
        """``(ctx, *args)`` accepts a second positional; it must receive
        the intent as ``args[0]`` (parity with the prior behaviour)."""
        received: list[Any] = []

        def cb(ctx: Any, *args: Any) -> None:
            received.append(args)

        ctx = _run_context()
        await invoke_on_handoff(cb, ctx, intent="my-intent")

        assert received == [("my-intent",)]

    async def test_two_positionals_still_receive_intent(self) -> None:
        """Regression guard: a plain ``(ctx, intent)`` callback is
        unaffected by the kind-filtering refactor."""
        received: list[Any] = []

        def cb(ctx: Any, intent: Any) -> None:
            received.append(intent)

        ctx = _run_context()
        await invoke_on_handoff(cb, ctx, intent="my-intent")

        assert received == ["my-intent"]
