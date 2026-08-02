"""RestateLLM tests that run WITHOUT the restate SDK installed.

Cost delegation and the journaled ``_invoke`` closure's return annotation are
pure-Python concerns — neither touches the restate SDK — so these tests must
NOT ``importorskip('restate')`` (which is not installed here). ``RestateLLM``
imports fine without the SDK; only ``get_restate_context`` reaches for it
lazily, and it is patched out below.
"""

from __future__ import annotations

import inspect
from unittest.mock import AsyncMock, MagicMock, patch

from troopai.adk.llms.llm import LLM
from troopai.adk.types.responses.llm_response import LLMResponse
from troopai.adk.workflows.engine import ModelActivityConfig
from troopai.adk.workflows.restate.llm import RestateLLM


class TestRestateLLMDelegatesCost:
    """``RestateLLM`` must forward cost/estimate_cost to the wrapped LLM.

    Regression: the base ``LLM.cost`` returns ``None``, so without an override
    a durable run's tenant dollar budget is silently never charged.
    """

    def test_cost_delegates_to_wrapped(self) -> None:
        """``cost`` returns the wrapped provider's price for the call."""
        from troopai.adk.types.tokens.llm_usage import LLMUsage

        wrapped = MagicMock(spec=LLM)
        wrapped.cost.return_value = 1.5
        llm = RestateLLM(wrapped=wrapped, activity_config=ModelActivityConfig())
        usage = LLMUsage(input_tokens=100, output_tokens=50)

        assert llm.cost("gpt-4o", usage) == 1.5
        wrapped.cost.assert_called_once_with("gpt-4o", usage)

    def test_estimate_cost_delegates_to_wrapped(self) -> None:
        """``estimate_cost`` forwards args and returns the wrapped estimate."""
        wrapped = MagicMock(spec=LLM)
        sentinel = object()
        wrapped.estimate_cost.return_value = sentinel
        llm = RestateLLM(wrapped=wrapped, activity_config=ModelActivityConfig())

        result = llm.estimate_cost([], "gpt-4o", max_output_tokens=64)

        assert result is sentinel
        wrapped.estimate_cost.assert_called_once_with([], "gpt-4o", max_output_tokens=64)


class TestRestateLLMJournaledClosureAnnotation:
    """The journaled ``_invoke`` closure's return annotation must be resolvable
    under ``inspect.signature(eval_str=True)`` — Restate introspects the
    callable to type its journaled result.

    Regression: ``LLMResponse`` was a TYPE_CHECKING-only import, so evaluating
    the ``-> LLMResponse`` annotation raised ``NameError`` on the durable path.
    """

    async def test_invoke_return_annotation_resolves(self) -> None:
        """Capturing and introspecting the journaled closure must not NameError."""
        wrapped = MagicMock(spec=LLM)
        wrapped.acomplete = AsyncMock(
            return_value=LLMResponse(
                response_id="r",
                model="m",
                response=[],
                usage=None,
                finish_reason="stop",
                timestamp=None,
            )
        )
        llm = RestateLLM(wrapped=wrapped, activity_config=ModelActivityConfig())

        captured: dict = {}

        async def fake_run(name: str, fn: object) -> object:
            captured["fn"] = fn
            return await fn()  # type: ignore[operator]

        ctx = MagicMock()
        ctx.run = fake_run
        with patch("troopai.adk.workflows.restate.llm.get_restate_context", return_value=ctx):
            await llm.acomplete("hi", stream=False)

        # eval_str=True evaluates the "-> LLMResponse" annotation string in the
        # closure's globals; it must resolve rather than raise NameError.
        sig = inspect.signature(captured["fn"], eval_str=True)
        assert sig.return_annotation is LLMResponse
