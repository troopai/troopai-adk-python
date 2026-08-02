"""Native OpenAI Responses-API LLM implementation.

Calls ``openai.AsyncOpenAI().responses.create()`` directly — no litellm
layer. Input items and output items round-trip through
``openai.types.responses.*`` via
:class:`OpenAIResponsesConverter`; wire types never leak outside this
module's call sites.

Provider-native hosted capabilities (web_search, file_search,
computer_use, image_generation, code_interpreter, hosted MCP servers,
…) are NOT wrapped in framework tool classes — pass the raw provider
JSON through :attr:`LLMConfig.extra_body` or :attr:`LLMConfig.extra_args`.

Usage::

    from troopai.adk.llms.openai import OpenAIResponsesLLM
    from troopai.adk.tools import function_tool


    @function_tool(name="lookup", description="Look up a record")
    def lookup(record_id: str) -> str:
        return f"Record {record_id}"


    agent = Agent(
        llm=OpenAIResponsesLLM(model="gpt-5.1"),
        tools=[lookup],
    )

Refs:
    - Responses API: https://platform.openai.com/docs/api-reference/responses
    - OpenAI SDK: https://github.com/openai/openai-python
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Literal, overload, override

from openai import NOT_GIVEN, AsyncStream, omit
from openai.types.responses import Response

from troopai.adk.llms.llm import LLM
from troopai.adk.llms.openai.openai_boundary import (
    headers_as_sdk,
    metadata_as_sdk,
    query_as_sdk,
    sanitize_for_log,
)
from troopai.adk.llms.openai.openai_responses_converter import OpenAIResponsesConverter
from troopai.adk.llms.openai.openai_retry import call_with_retry
from troopai.adk.llms.stream_error import stream_with_error_contract
from troopai.adk.types.input import LLMInputContentItem, LLMInputEasyMessage
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
    LLMResponsePart,
    LLMResponseProviderItem,
    LLMResponseReasoning,
    LLMResponseText,
    LLMStreamEvent,
)

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.responses import ResponseStreamEvent
    from openai.types.responses.response_output_item_added_event import (
        ResponseOutputItemAddedEvent,
    )

    from troopai.adk.llms.llm_config import LLMConfig
    from troopai.adk.llms.openai.openai_responses_config import OpenAIResponsesConfig
    from troopai.adk.schemas import AgentOutputSchemaBase
    from troopai.adk.tools import Tool


logger = logging.getLogger(__name__)


class OpenAIResponsesLLM(LLM):
    """Native OpenAI Responses-API ``LLM`` implementation.

    Calls ``client.responses.create()`` directly with explicit named
    parameters. ``OpenAIResponsesConfig`` fields
    (``reasoning``, ``include``, ``store``, ``previous_response_id``,
    ``truncation``, ``service_tier``, ``prompt_cache_key``,
    ``prompt_cache_retention``, ``max_tool_calls``, ``background``)
    are routed verbatim onto the corresponding SDK parameters when an
    :class:`OpenAIResponsesConfig` instance is supplied; the generic
    :class:`LLMConfig` base provides ``temperature`` / ``top_p`` /
    ``max_output_tokens`` / ``top_logprobs`` / ``tool_choice`` /
    ``tool_execution_mode`` and the ``extra_*`` escape hatches.

    Args:
        model: OpenAI model ID (e.g., ``"gpt-5.1"``, ``"o4"``). Do NOT
            prefix with ``"openai/"`` — this is not litellm.
        api_key: API key. Falls back to ``OPENAI_API_KEY`` env var.
        base_url: Optional custom base URL (Azure OpenAI, local proxy,
            etc.).
        organization: Optional OpenAI organization ID.
        project: Optional OpenAI project ID.
        max_retries: SDK-level retries for transient errors (default 0 —
            disabled). Set to a positive value to enable SDK-level retries.
            Independent of :attr:`LLMConfig.retry_policy`, which runs
            one layer up with framework-owned classification.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
        max_retries: int = 0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._organization = organization
        self._project = project
        self._max_retries = max_retries
        self._client: AsyncOpenAI | None = None

    @property
    def model(self) -> str:
        """The model identifier passed to ``client.responses.create()``."""
        return self._model

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-initialize the OpenAI async client.

        The client is constructed the first time ``acomplete`` needs
        it and cached for the lifetime of the ``LLM`` instance —
        mirrors the pattern used by ``AnthropicLLM``.
        """
        client = self._client
        if client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "The 'openai' package is required for OpenAIResponsesLLM. Install it with: pip install openai"
                ) from e

            client = AsyncOpenAI(
                api_key=self._api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=self._base_url,
                organization=self._organization,
                project=self._project,
                max_retries=self._max_retries,
            )
            self._client = client
        return client

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @overload
    async def acomplete(
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        stream: Literal[False] = False,
    ) -> LLMResponse: ...

    # PyCharm's PyMethodOverriding analyzer pairs each subclass @overload
    # against the base class implementation rather than the corresponding
    # base @overload, flagging the narrower Literal[True]/AsyncIterator
    # return as a "mismatch." Signature is byte-identical to LLM.acomplete;
    # mypy + pyright clear the site.
    # noinspection PyMethodOverriding
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

    @override
    async def acomplete(
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
        """Call the OpenAI Responses API.

        Orchestrates: convert → resolve params → API call →
        parse/stream. Non-streaming calls route through
        :func:`call_with_retry` when :attr:`LLMConfig.retry_policy` is
        set; streaming calls are never retried (mid-stream reconnect
        would silently drop or double-emit tokens).
        """
        from troopai.adk.llms.llm_config import LLMConfig
        from troopai.adk.llms.openai.openai_responses_config import OpenAIResponsesConfig

        config: LLMConfig = llm_config if llm_config is not None else LLMConfig()
        responses_config: OpenAIResponsesConfig | None = config if isinstance(config, OpenAIResponsesConfig) else None

        # 1. Normalise messages → Responses-API input items
        if isinstance(messages, str):
            easy: LLMInputEasyMessage = {"role": "user", "content": messages}
            normalised: list[LLMInputContentItem] = [easy]
        else:
            normalised = messages
        input_items = OpenAIResponsesConverter.items_to_input(normalised)

        # 2. Convert tools
        wire_tools = OpenAIResponsesConverter.convert_tools(tools) if tools is not None and len(tools) > 0 else None

        # 3. Resolve tool_choice (flat shape: {type: function, name: ...})
        tool_choice = OpenAIResponsesConverter.convert_tool_choice(
            config.tool_choice,
            tools_present=wire_tools is not None,
        )

        # 4. Resolve structured-output text config
        text_config = OpenAIResponsesConverter.resolve_response_format(output_schema)

        # 5. parallel_tool_calls from tool_execution_mode. Only send it when
        #    tools are actually present — the Responses API rejects
        #    ``parallel_tool_calls`` with a 400 when the request carries no
        #    ``tools``, so a bare execution-mode config must not leak the
        #    parameter onto a tool-less call.
        parallel_tool_calls: bool | None = None
        if config.tool_execution_mode is not None and wire_tools is not None:
            parallel_tool_calls = config.tool_execution_mode == "parallel"

        # 6. Merge extra_body + extra_args. ``LLMConfig.extra_body`` is
        #    typed as ``object`` (framework-wide); we only know how to
        #    merge when the caller passed a mapping. A non-mapping
        #    ``extra_body`` cannot be merged into the request body dict,
        #    so it is dropped with a warning rather than discarded
        #    silently — pass a dict to forward extra request-body fields.
        extra_body: dict[str, Any] | None
        if isinstance(config.extra_body, Mapping) or config.extra_args is not None:
            merged: dict[str, Any] = {}
            if isinstance(config.extra_body, Mapping):
                merged.update(config.extra_body)
            if config.extra_args is not None:
                merged.update(config.extra_args)
            extra_body = merged if len(merged) > 0 else None
        else:
            if config.extra_body is not None:
                logger.warning(
                    "extra_body value of type %s is not a Mapping and cannot be forwarded; "
                    "pass a dict to include extra request body fields",
                    type(config.extra_body).__name__,
                )
            extra_body = None

        client = self._get_client()

        # 7. Responses-specific fields — only populated when the caller
        #    passed an OpenAIResponsesConfig. Nullable fields pass
        #    through as ``None``; non-nullable SDK fields
        #    (``prompt_cache_key``) use the ``omit`` sentinel.
        reasoning = responses_config.reasoning if responses_config is not None else None
        include = responses_config.include if responses_config is not None else None
        store = responses_config.store if responses_config is not None else None
        previous_response_id = responses_config.previous_response_id if responses_config is not None else None
        truncation = responses_config.truncation if responses_config is not None else None
        service_tier = responses_config.service_tier if responses_config is not None else None
        prompt_cache_key = responses_config.prompt_cache_key if responses_config is not None else None
        prompt_cache_retention = responses_config.prompt_cache_retention if responses_config is not None else None
        max_tool_calls = responses_config.max_tool_calls if responses_config is not None else None
        background = responses_config.background if responses_config is not None else None

        logger.info(
            "OpenAI Responses API call: model=%s, items=%d, tools=%d, stream=%s",
            sanitize_for_log(self._model),
            len(input_items),
            len(wire_tools) if wire_tools is not None else 0,
            stream,
        )

        if stream:
            # The SDK types ``tools`` / ``tool_choice`` / ``text`` /
            # ``prompt_cache_key`` as non-nullable (``T | Omit``); every
            # other optional routes through as ``T | None | Omit`` so
            # ``None`` is acceptable and clearer than a conditional
            # kwargs-merge.
            stream_iter = await client.responses.create(
                model=self._model,
                input=input_items,
                tools=wire_tools if wire_tools is not None else omit,
                tool_choice=tool_choice if tool_choice is not None else omit,
                text=text_config if text_config is not None else omit,
                temperature=config.temperature,
                top_p=config.top_p,
                max_output_tokens=config.max_output_tokens,
                top_logprobs=config.top_logprobs,
                parallel_tool_calls=parallel_tool_calls,
                metadata=metadata_as_sdk(config.metadata),
                timeout=config.timeout if config.timeout is not None else NOT_GIVEN,
                extra_headers=headers_as_sdk(config.extra_headers),
                extra_query=query_as_sdk(config.extra_query),
                extra_body=extra_body,
                reasoning=reasoning,
                include=include,
                store=store,
                previous_response_id=previous_response_id,
                truncation=truncation,
                service_tier=service_tier,
                prompt_cache_key=prompt_cache_key if prompt_cache_key is not None else omit,
                prompt_cache_retention=prompt_cache_retention,
                max_tool_calls=max_tool_calls,
                background=background,
                stream=True,
            )
            # stream=True above guarantees AsyncStream, not Response. The SDK's
            # @overload chain on .create() returns Response | AsyncStream when
            # called with stream: bool; PyCharm/pyright cannot narrow True to
            # Literal[True] across the SDK boundary. Use an explicit raise
            # rather than `assert` so the guard survives `python -O`.
            if not isinstance(stream_iter, AsyncStream):
                raise TypeError(
                    f"OpenAI responses.create(stream=True) returned {type(stream_iter).__name__}, expected AsyncStream",
                )
            return stream_with_error_contract(self._stream(stream_iter), model=self._model, logger=logger)

        async def _call_responses() -> Response | AsyncStream[ResponseStreamEvent]:
            return await client.responses.create(
                model=self._model,
                input=input_items,
                tools=wire_tools if wire_tools is not None else omit,
                tool_choice=tool_choice if tool_choice is not None else omit,
                text=text_config if text_config is not None else omit,
                temperature=config.temperature,
                top_p=config.top_p,
                max_output_tokens=config.max_output_tokens,
                top_logprobs=config.top_logprobs,
                parallel_tool_calls=parallel_tool_calls,
                metadata=metadata_as_sdk(config.metadata),
                timeout=config.timeout if config.timeout is not None else NOT_GIVEN,
                extra_headers=headers_as_sdk(config.extra_headers),
                extra_query=query_as_sdk(config.extra_query),
                extra_body=extra_body,
                reasoning=reasoning,
                include=include,
                store=store,
                previous_response_id=previous_response_id,
                truncation=truncation,
                service_tier=service_tier,
                prompt_cache_key=prompt_cache_key if prompt_cache_key is not None else omit,
                prompt_cache_retention=prompt_cache_retention,
                max_tool_calls=max_tool_calls,
                background=background,
                stream=False,
            )

        if config.retry_policy is not None:
            response = await call_with_retry(
                _call_responses,
                config.retry_policy,
                model=self._model,
            )
        else:
            response = await _call_responses()

        # stream=False inside _call_responses guarantees Response, not
        # AsyncStream. The SDK's @overload chain on .create() returns
        # Response | AsyncStream when called with stream: bool; the union
        # propagates through call_with_retry and back here, where we narrow.
        # Explicit raise (not `assert`) so the guard survives `python -O`.
        if not isinstance(response, Response):
            raise TypeError(
                f"OpenAI responses.create(stream=False) returned {type(response).__name__}, expected Response",
            )
        parsed = OpenAIResponsesConverter.response_to_llm_response(response)
        if parsed.usage is not None:
            logger.info(
                "OpenAI Responses result: model=%s, input_tokens=%d, output_tokens=%d",
                sanitize_for_log(parsed.model),
                parsed.usage.input_tokens,
                parsed.usage.output_tokens,
            )
        return parsed

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def _stream(
        self,
        stream_iter: AsyncIterator[ResponseStreamEvent],
    ) -> AsyncIterator[LLMStreamEvent]:
        """Drain a Responses-API stream into framework ``LLMStreamEvent``s.

        Maps the SDK's 53-variant ``ResponseStreamEvent`` union onto the
        four-variant framework stream protocol:

        - ``ResponseOutputItemAddedEvent`` → ``"part_start"``
        - text/reasoning/function-arg delta events → ``"part_delta"``
        - ``ResponseOutputItemDoneEvent`` → ``"part_end"``
        - ``ResponseCompletedEvent`` → ``"done"`` (carries the full
          converted :class:`LLMResponse`)

        Unrecognised variants (audio, image-gen progress, MCP status,
        etc.) are ignored — hosted-tool outputs still land in the final
        ``Response`` via the ``"done"`` event, so no information is
        lost for replay purposes.
        """
        from openai.types.responses import (
            ResponseCompletedEvent,
            ResponseFailedEvent,
            ResponseFunctionCallArgumentsDeltaEvent,
            ResponseIncompleteEvent,
            ResponseOutputItemAddedEvent,
            ResponseOutputItemDoneEvent,
            ResponseReasoningSummaryTextDeltaEvent,
            ResponseReasoningTextDeltaEvent,
            ResponseRefusalDeltaEvent,
            ResponseTextDeltaEvent,
        )

        async for event in stream_iter:
            if isinstance(event, ResponseOutputItemAddedEvent):
                part = _item_added_to_placeholder_part(event)
                if part is not None:
                    yield LLMStreamEvent(
                        type="part_start",
                        index=event.output_index,
                        part=part,
                    )

            elif isinstance(
                event,
                (
                    ResponseTextDeltaEvent,
                    ResponseReasoningTextDeltaEvent,
                    ResponseReasoningSummaryTextDeltaEvent,
                    ResponseRefusalDeltaEvent,
                    ResponseFunctionCallArgumentsDeltaEvent,
                ),
            ):
                if len(event.delta) > 0:
                    yield LLMStreamEvent(
                        type="part_delta",
                        index=event.output_index,
                        delta=event.delta,
                    )

            elif isinstance(event, ResponseOutputItemDoneEvent):
                yield LLMStreamEvent(
                    type="part_end",
                    index=event.output_index,
                )

            elif isinstance(event, (ResponseCompletedEvent, ResponseIncompleteEvent, ResponseFailedEvent)):
                # All three terminal events carry the full ``Response``
                # (output + usage). An incomplete response (max_output_tokens,
                # content filter) or a failed response still holds partial
                # output and token usage; converting ``event.response`` surfaces
                # finish_reason=length/incomplete/error AND bills the tokens,
                # instead of discarding them via the empty-done fallback below.
                parsed = OpenAIResponsesConverter.response_to_llm_response(event.response)
                yield LLMStreamEvent(type="done", response=parsed)
                return

        # Stream closed without a ResponseCompletedEvent — emit an empty
        # "done" event so downstream consumers don't block, and surface
        # ``finish_reason="incomplete"`` so callers can distinguish this
        # from a successful zero-output response (refusal, all-hosted-tool
        # turn with no text, etc). Common causes: network interruption,
        # SSE connection dropped, or upstream aborted mid-stream.
        logger.warning(
            "OpenAI Responses stream closed without ResponseCompletedEvent "
            "(truncated stream for model=%s); emitting finish_reason=incomplete.",
            sanitize_for_log(self._model),
        )
        empty = LLMResponse(
            response_id="",
            model=self._model,
            response=[],
            usage=None,
            finish_reason="incomplete",
        )
        yield LLMStreamEvent(type="done", response=empty)


def _item_added_to_placeholder_part(
    event: ResponseOutputItemAddedEvent,
) -> LLMResponsePart | None:
    """Choose a placeholder ``LLMResponsePart`` for an added output item.

    The stream protocol expects a ``part`` on ``part_start`` so the
    consumer can branch its delta handler by type. We only need a
    discriminator here — the fully-populated part lands in the final
    ``LLMResponse`` via the converter when the stream closes.
    """
    from openai.types.responses import (
        ResponseFunctionToolCall,
        ResponseOutputMessage,
        ResponseReasoningItem,
    )

    item = event.item
    if isinstance(item, ResponseOutputMessage):
        return LLMResponseText(text="")
    if isinstance(item, ResponseReasoningItem):
        return LLMResponseReasoning(thinking="", signature="")
    if isinstance(item, ResponseFunctionToolCall):
        return LLMResponseFunctionToolCall(
            call_id=item.call_id,
            name=item.name,
            arguments="",
        )
    # Hosted tool items (web_search_call, file_search_call,
    # image_generation_call, etc.) surface as a provider_item placeholder —
    # the final payload lands in the ResponseCompletedEvent's converted
    # response, so the placeholder carries an empty `raw` dict.
    item_type = item.type
    if len(item_type) > 0:
        return LLMResponseProviderItem(item_type=item_type, raw={})
    return None
