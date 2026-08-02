"""RestateLLM — routes LLM calls through Restate's ctx.run() for journaled replay.

Wraps any :class:`~troopai.adk.llms.llm.LLM` implementation and transparently
routes :meth:`acomplete` through ``ctx.run()`` when called from inside a
Restate handler.  Outside a handler the call is forwarded directly to the
wrapped LLM, making ``RestateLLM`` safe to use in both durable and
non-durable contexts.

Usage::

    from troopai.adk.llms import LiteLLM
    from troopai.adk.workflows.engine import ModelActivityConfig
    from troopai.adk.workflows.restate.llm import RestateLLM

    llm = RestateLLM(
        wrapped=LiteLLM(model="gpt-4o"),
        activity_config=ModelActivityConfig(),
    )

References:
    Restate Python SDK ctx.run docs:
    https://docs.restate.dev/develop/python/durable-execution#journaling-results
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast, override

from troopai.adk.llms.llm import LLM

# LLMResponse is imported at runtime (not under TYPE_CHECKING) so the journaled
# ``_invoke`` closure's ``-> LLMResponse`` return annotation resolves under
# ``inspect.signature(eval_str=True)`` — Restate's ``ctx.run`` introspects the
# callable to type its journaled result, and a TYPE_CHECKING-only name would
# raise NameError there.
from troopai.adk.types.responses.llm_response import LLMResponse
from troopai.adk.workflows.engine import ModelActivityConfig

if TYPE_CHECKING:
    from troopai.adk.llms.cost import CostEstimate
    from troopai.adk.llms.llm_config import LLMConfig
    from troopai.adk.schemas import AgentOutputSchemaBase
    from troopai.adk.tools import Tool
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.responses.llm_response import LLMStreamEvent
    from troopai.adk.types.tokens.llm_usage import LLMUsage

logger = logging.getLogger(__name__)


def get_restate_context() -> Any | None:
    """Return the current Restate handler context, or ``None`` if unavailable.

    Tries to import the SDK and retrieve the current context via
    ``restate.extensions.current_context()``, which reads a ``ContextVar``
    populated by the Restate server for the duration of a handler
    invocation.  Returns ``None`` on ``ImportError`` (SDK not installed)
    or ``LookupError`` (the ``ContextVar`` is unset — called outside a
    handler).  Any other exception propagates: silently swallowing SDK
    errors here would fall back to un-journaled direct LLM calls and
    cause replay divergence.

    Returns:
        The active Restate context object when inside a handler, ``None``
        otherwise.
    """
    try:
        from restate.extensions import current_context  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return current_context()
    except LookupError:
        return None


@dataclasses.dataclass
class RestateLLM(LLM):
    """LLM bridge that routes ``acomplete`` through Restate's ``ctx.run()`` for journaled replay.

    When called from inside a Restate handler (detected via
    :func:`get_restate_context`), each LLM call is executed inside
    ``ctx.run()`` so that the result is journaled.  On replay, Restate
    returns the recorded result without re-executing the LLM call.

    Outside a handler the wrapped LLM is called directly — no overhead is
    added in non-durable paths.

    Attributes:
        wrapped: The real :class:`~troopai.adk.llms.llm.LLM` instance that
            handles provider communication.
        activity_config: Timeout and retry policy carried for compatibility
            with the :class:`~troopai.adk.workflows.engine.DurableEngine`
            Protocol.  Restate's retry behaviour is configured on the
            service/handler level, not per ``ctx.run()`` call.

    References:
        Restate Python SDK durable execution:
        https://docs.restate.dev/develop/python/durable-execution
        Restate ctx.run journaling:
        https://docs.restate.dev/develop/python/durable-execution#journaling-results
    """

    wrapped: LLM
    """The underlying LLM implementation that handles provider communication."""

    activity_config: ModelActivityConfig
    """Timeout and retry policy (carried for DurableEngine Protocol compatibility)."""

    # ------------------------------------------------------------------
    # Cost delegation
    # ------------------------------------------------------------------

    @override
    def cost(self, model: str, usage: LLMUsage) -> float | None:
        """Delegate USD costing to the wrapped LLM.

        Without this override the base :meth:`~troopai.adk.llms.llm.LLM.cost`
        returns ``None`` for every call, so a durable run's tenant dollar
        budget would silently never be charged.  Forwarding to the wrapped
        provider preserves the underlying cost table.

        Args:
            model: The model identifier that served the call.
            usage: Token usage for the call.

        Returns:
            The wrapped LLM's cost estimate, or ``None`` when it cannot price
            the call.
        """
        return self.wrapped.cost(model, usage)

    @override
    def estimate_cost(
        self,
        messages: list[LLMInputContentItem],
        model: str,
        *,
        max_output_tokens: int | None = None,
    ) -> CostEstimate:
        """Delegate the pre-call cost estimate to the wrapped LLM.

        Symmetric with :meth:`cost`: the wrapped provider owns token counting
        and pricing, so pre-call dollar gating stays correct under durable
        execution.

        Args:
            messages: The input messages to count tokens for.
            model: The model identifier.
            max_output_tokens: Optional output-side bound for the estimate.

        Returns:
            The wrapped LLM's :class:`~troopai.adk.llms.cost.CostEstimate`.
        """
        return self.wrapped.estimate_cost(messages, model, max_output_tokens=max_output_tokens)

    # ------------------------------------------------------------------
    # LLM ABC implementation
    # ------------------------------------------------------------------

    async def acomplete(  # type: ignore[override]  # overload narrowing lives on ABC; impl signature is wider
        self,
        messages: str | list[Any],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
        """Dispatch an LLM call, routing through Restate ctx.run() when inside a handler.

        Args:
            messages: Conversation input — plain string or a list of
                provider-agnostic content items.
            llm_config: Optional LLM parameters (temperature, max tokens, etc.).
            tools: Optional pre-filtered tool list.
            output_schema: Optional structured output schema.
            stream: When ``True``, returns an ``AsyncIterator[LLMStreamEvent]``.
                Streaming is not supported inside a Restate handler (``ctx.run``
                results are serialized snapshots); requesting ``stream=True``
                inside a handler raises ``NotImplementedError``.

        Returns:
            ``LLMResponse`` when ``stream=False``; ``AsyncIterator[LLMStreamEvent]``
            when ``stream=True`` and called outside a handler.

        References:
            Restate ctx.run docs:
            https://docs.restate.dev/develop/python/durable-execution#journaling-results
        """
        ctx = get_restate_context()

        if ctx is None:
            logger.debug(
                "RestateLLM: forwarding acomplete directly (outside Restate handler), wrapped=%r",
                self.wrapped,
            )
            if stream:
                return await self.wrapped.acomplete(
                    messages,
                    llm_config,
                    tools,
                    output_schema,
                    stream=True,
                )
            return await self.wrapped.acomplete(
                messages,
                llm_config,
                tools,
                output_schema,
                stream=False,
            )

        if stream:
            raise NotImplementedError(
                "RestateLLM does not support stream=True inside a Restate handler. "
                "Restate ctx.run() results are serialized journaled snapshots, so "
                "streaming semantics are not available in durable execution. "
                "Set stream=False."
            )

        logger.info(
            "RestateLLM: routing acomplete through Restate ctx.run(), wrapped=%r",
            self.wrapped,
        )

        wrapped = self.wrapped
        llm_config_captured = llm_config
        tools_captured = tools
        output_schema_captured = output_schema

        async def _invoke() -> LLMResponse:
            result = await wrapped.acomplete(
                messages,
                llm_config_captured,
                tools_captured,
                output_schema_captured,
                stream=False,
            )
            if not isinstance(result, LLMResponse):
                raise TypeError(
                    f"RestateLLM expected LLMResponse from wrapped={wrapped!r} but got {type(result).__name__}"
                )
            return result

        # ctx is Any (Restate SDK type); _invoke is annotated -> LLMResponse,
        # so the runtime result is always LLMResponse (isinstance-guarded inside _invoke).
        return cast(LLMResponse, await ctx.run("invoke_model", _invoke))
