"""Abstract base class for LLM implementations.

Defines the ``LLM`` interface that all LLM providers must implement.
The primary method is ``acomplete()`` — an async method that accepts
provider-agnostic input items and returns an ``LLMResponse``
(or streams ``LLMStreamEvent`` objects).

The ``LLM`` layer is responsible for:

- Converting provider-agnostic items to wire format (via converter)
- Translating ``LLMConfig`` to provider-specific parameters
- Handling structured output via ``response_format`` / JSON schema
- Parsing provider responses into ``LLMResponse``
- Streaming support via ``LLMStreamEvent``

The ``LLM`` layer does NOT handle:

- Tool enablement checks (``FunctionTool.enabled``) — Runner concern
- Handoff conversion to tools — Runner concern
- System prompt role detection — Runner concern
- Message history management — Runner concern

Example::

    from troopai.adk.llms import LiteLLM

    llm = LiteLLM(model="gpt-4o")
    response = await llm.acomplete(
        messages="Hello!",
    )
    logger.info(response.content)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Literal, overload

from troopai.adk.types.input import LLMInputContentItem

if TYPE_CHECKING:
    from troopai.adk.llms.cost import CostEstimate
    from troopai.adk.llms.llm_config import LLMConfig
    from troopai.adk.schemas import AgentOutputSchemaBase
    from troopai.adk.tools import Tool
    from troopai.adk.types.responses.llm_response import LLMResponse, LLMStreamEvent
    from troopai.adk.types.tokens.llm_usage import LLMUsage

logger = logging.getLogger(__name__)


class LLM(ABC):
    """Abstract base class for Large Language Models (LLMs) used in agents.

    Subclasses implement provider-specific communication while
    exposing a unified interface.  The ``Runner`` interacts with
    agents via this interface, never directly with provider SDKs.

    The two entry points are :meth:`acomplete` (async, primary) and
    :meth:`complete` (sync convenience wrapper).
    """

    @overload
    async def acomplete(
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        stream: Literal[False] = False,
    ) -> LLMResponse: ...

    @overload
    async def acomplete(
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        *,
        stream: Literal[True],
    ) -> AsyncIterator[LLMStreamEvent]: ...

    @abstractmethod
    async def acomplete(
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
        """Call the LLM asynchronously.

        This is the primary interface for LLM communication.  The
        ``Runner`` calls this method on each turn of the agent loop.

        Args:
            messages: Conversation input as a plain string or a list of
                provider-agnostic ``LLMInputContentItem`` items.  The
                LLM implementation converts these to wire format via
                ``ChatCompletionConverter.items_to_messages()`` before
                sending to the provider.
            llm_config: Optional LLM parameters (temperature, max tokens,
                tool choice, tool execution mode, etc.).  The implementation
                maps these to provider-specific parameters.
            tools: Optional list of ``Tool`` instances (``FunctionTool``,
                ``BuiltinTool``, etc.).  Pre-filtered by the Runner
                (``enabled`` checks, retry budget).  The LLM implementation
                converts these to wire format internally.
            output_schema: Optional structured output schema.  When provided,
                the implementation uses ``response_format`` (JSON schema mode)
                to constrain the model's output and validates the response.
            stream: If ``True``, returns an ``AsyncIterator[LLMStreamEvent]``
                instead of an ``LLMResponse``.  The iterator yields content
                deltas and tool call fragments, ending with a ``"done"``
                event containing the complete ``LLMResponse``.

        Returns:
            When ``stream=False``: An ``LLMResponse`` with content,
            tool calls, usage, and optional structured output.

            When ``stream=True``: An ``AsyncIterator[LLMStreamEvent]``
            yielding streaming events.

        Raises:
            Exception: Provider-specific errors (rate limits, auth failures,
                invalid requests, etc.).
        """

    def complete(
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
    ) -> LLMResponse:
        """Call the LLM synchronously.

        Convenience wrapper around :meth:`acomplete` for synchronous code.
        Uses ``asyncio.run()`` internally.

        .. warning::

            Cannot be called from within a running event loop.
            Use :meth:`acomplete` in async contexts instead.

        Args:
            messages: Conversation input (str or list of items).
            llm_config: Optional LLM parameters.
            tools: Optional tool definitions.
            output_schema: Optional structured output schema.

        Returns:
            An ``LLMResponse`` with the model's output.
        """
        import asyncio

        return asyncio.run(
            self.acomplete(
                messages=messages,
                llm_config=llm_config,
                tools=tools,
                output_schema=output_schema,
                stream=False,
            )
        )

    def cost(self, model: str, usage: LLMUsage) -> float | None:
        """Best-effort USD cost for one completed call, or ``None`` when
        unknown. The base implementation returns ``None``; providers that can
        price a call override it. Callers MUST treat ``None`` as "cost
        unavailable" and never fail the run on it.

        Args:
            model: The model identifier that served the call.
            usage: Token usage for the call (input and output counts).

        Returns:
            Estimated USD cost, or ``None`` when no cost table is
            available.
        """
        del model, usage
        return None

    def estimate_cost(
        self,
        messages: list[LLMInputContentItem],
        model: str,
        *,
        max_output_tokens: int | None = None,
    ) -> CostEstimate:
        """Best-effort pre-call cost estimate. Symmetric with :meth:`cost`.

        Input tokens are counted via :class:`TokenCounter`. Output is
        bounded by ``max_output_tokens`` when provided; absent it the
        estimate is input-only (a documented floor) — no token default is
        invented. ``estimated_cost_usd`` is ``None`` when :meth:`cost`
        returns ``None`` (no provider cost table); callers skip dollar
        gating in that case rather than failing the run.

        Args:
            messages: The input messages to count tokens for.
            model: The model identifier (used for token counting and
                cost lookup).
            max_output_tokens: When provided, bounds the output side of
                the estimate.  When absent the estimate covers input
                tokens only.

        Returns:
            A :class:`CostEstimate` with token counts and optional USD
            estimate.
        """
        from troopai.adk.context.token_counter import TokenCounter
        from troopai.adk.llms.cost import CostEstimate
        from troopai.adk.types.tokens.llm_usage import LLMUsage

        input_tokens = TokenCounter.count_messages(messages, model)
        out = max_output_tokens if max_output_tokens is not None else 0
        estimated = self.cost(model, LLMUsage(input_tokens=input_tokens, output_tokens=out))
        logger.debug(
            "estimate_cost model=%s input=%d output=%d usd=%s",
            model,
            input_tokens,
            out,
            estimated,
        )
        return CostEstimate(
            model=model,
            input_tokens=input_tokens,
            estimated_output_tokens=out,
            estimated_cost_usd=estimated,
            output_bounded=max_output_tokens is not None,
        )
