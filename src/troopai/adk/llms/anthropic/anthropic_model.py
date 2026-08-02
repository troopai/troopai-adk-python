"""Anthropic native LLM implementation using the ``anthropic`` SDK directly.

Calls ``anthropic.AsyncAnthropic().messages.create()`` — no intermediate
litellm layer. Provider-native capabilities (web search, computer use,
bash, text editor) are not wrapped as framework tool classes; pass raw
provider JSON via ``LLMConfig.extra_body`` / ``extra_args``.

Usage::

    from troopai.adk.llms.anthropic import AnthropicLLM
    from troopai.adk.tools import function_tool


    @function_tool(name="lookup", description="Look up a record")
    def lookup(record_id: str) -> str:
        return f"Record {record_id}"


    agent = Agent(
        llm=AnthropicLLM(model="claude-sonnet-4-20250514"),
        tools=[lookup],
    )

Refs:
    - Anthropic Messages API: https://docs.anthropic.com/en/api/messages
    - Anthropic SDK: https://github.com/anthropics/anthropic-sdk-python
"""

from __future__ import annotations

import dataclasses
import logging
import os
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Literal, overload, override

from anthropic import NOT_GIVEN, AsyncStream, Omit, omit
from anthropic.types import (
    Message,
    OutputConfigParam,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    RawMessageStreamEvent,
)

from troopai.adk.llms.anthropic.anthropic_boundary import (
    headers_as_sdk,
    metadata_as_sdk,
    sanitize_for_log,
)
from troopai.adk.llms.anthropic.anthropic_cache_applicator import apply_cache_control
from troopai.adk.llms.anthropic.anthropic_config import AnthropicConfig
from troopai.adk.llms.anthropic.anthropic_converter import (
    STRUCTURED_OUTPUT_TOOL_NAME,
    AnthropicConverter,
)
from troopai.adk.llms.anthropic.anthropic_reasoning_resolver import resolve_thinking
from troopai.adk.llms.anthropic.anthropic_retry import call_with_retry
from troopai.adk.llms.llm import LLM
from troopai.adk.llms.stream_error import stream_with_error_contract
from troopai.adk.types.input import LLMInputContentItem
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
    LLMResponsePart,
    LLMResponseReasoning,
    LLMResponseText,
    LLMStreamEvent,
)

if TYPE_CHECKING:
    from anthropic import AsyncAnthropic
    from anthropic.types import (
        MessageParam,
        TextBlockParam,
        ThinkingConfigParam,
        ToolChoiceParam,
        ToolUnionParam,
    )

    from troopai.adk.llms.llm_config import LLMConfig
    from troopai.adk.schemas import AgentOutputSchemaBase
    from troopai.adk.tools import Tool
    from troopai.adk.types.llms import EffortLevel
    from troopai.adk.types.tokens import LLMUsage

logger = logging.getLogger(__name__)

# Default max_tokens for Anthropic (required parameter on every call).
_MID_SYSTEM_BETA = "mid-conversation-system-2026-04-07"
"""Beta header value gating in-place ``role:"system"`` messages."""

_DEFAULT_MAX_TOKENS = 8192


@dataclasses.dataclass
class _BlockAccumulator:
    """Per-content-block streaming accumulator.

    One instance per ``content_block_start`` event, keyed on the
    block's index. Buffers all text / tool-input / thinking /
    signature deltas until ``content_block_stop`` finalises the
    block into an :class:`LLMResponsePart`.
    """

    block_type: str
    text_buf: list[str] = dataclasses.field(default_factory=list)
    tool_id: str = ""
    tool_name: str = ""
    tool_input_buf: list[str] = dataclasses.field(default_factory=list)
    thinking_buf: list[str] = dataclasses.field(default_factory=list)
    signature_buf: list[str] = dataclasses.field(default_factory=list)
    redacted_data: str = ""


class AnthropicLLM(LLM):
    """Anthropic native LLM via the ``anthropic`` SDK.

    Calls ``client.messages.create()`` directly with explicit named
    parameters. Supports streaming via ``stream=True``, extended
    thinking, prompt caching, structured output (synthetic-tool
    pattern), and framework retry policies.

    Args:
        model: Anthropic model ID (e.g., ``"claude-sonnet-4-20250514"``).
            Do NOT include the ``"anthropic/"`` prefix — this is not litellm.
        api_key: API key. Falls back to ``ANTHROPIC_API_KEY`` env var.
        base_url: Optional custom base URL.
        max_retries: SDK-level retries inside the anthropic client
            (default 0 — disabled). Set to a positive value to enable
            SDK-level retries for transient errors. This is distinct from
            :attr:`LLMConfig.retry_policy`, which runs *outside* the
            SDK with explicit backoff and category filters.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 0,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._max_retries = max_retries
        self._client: AsyncAnthropic | None = None

    @property
    def model(self) -> str:
        """The model identifier."""
        return self._model

    def _get_client(self) -> AsyncAnthropic:
        """Lazy-initialize the Anthropic async client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError as e:
                raise ImportError(
                    "The 'anthropic' package is required for AnthropicLLM. Install it with: pip install anthropic"
                ) from e

            self._client = AsyncAnthropic(
                api_key=self._api_key or os.environ.get("ANTHROPIC_API_KEY"),
                base_url=self._base_url,
                max_retries=self._max_retries,
            )
        return self._client

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

    # PyCharm pairs each subclass overload against the base class
    # implementation rather than the corresponding base overload, flagging
    # the narrower Literal[True]/AsyncIterator return as an LSP mismatch.
    # mypy + pyright clear the site; the signature is byte-identical to
    # ``LLM.acomplete``, ``LiteLLM.acomplete``, and
    # ``OpenAIChatCompletionsLLM.acomplete``.
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
        """Call Anthropic Messages API.

        Orchestrates: convert → resolve params → optionally inject
        cache markers → API call (with retry on non-streaming) →
        parse / stream.

        When ``output_schema`` is supplied AND streaming is on, raises
        :class:`NotImplementedError` — structured output relies on
        post-hoc validation of the synthetic tool's input, which is
        only available once the full response is in hand.

        Args:
            messages: Conversation input as a plain string or a list of
                provider-agnostic ``LLMInputContentItem`` items.
            llm_config: Optional LLM parameters including ``temperature``,
                ``max_output_tokens``, ``tool_choice``, thinking budget,
                retry policy, and cache-control settings.
            tools: Optional pre-filtered tool list. When
                ``output_schema`` is set, this list is replaced by the
                synthetic structured-output tool.
            output_schema: Optional structured output schema enforced via
                the synthetic-tool pattern.  Cannot be combined with
                ``stream=True``.
            stream: When ``True``, returns an
                ``AsyncIterator[LLMStreamEvent]`` instead of an
                ``LLMResponse``.

        Returns:
            ``LLMResponse`` when ``stream=False``; an
            ``AsyncIterator[LLMStreamEvent]`` when ``stream=True``.

        Raises:
            NotImplementedError: When ``output_schema`` is supplied and
                ``stream=True``.
        """
        from troopai.adk.llms.llm_config import LLMConfig

        if output_schema is not None and stream is True:
            raise NotImplementedError(
                "AnthropicLLM does not support stream=True with output_schema. "
                "The synthetic-tool pattern needs the complete tool_use block "
                "before validation; call with stream=False or remove the schema."
            )

        config = llm_config or LLMConfig()

        # Plain-text output schemas behave like no schema at all.
        wants_structured = output_schema is not None and not output_schema.is_plain_text()

        # 1. Convert messages.
        mid_system = isinstance(config, AnthropicConfig) and config.mid_system_messages is True
        system_prompt, converted_messages = AnthropicConverter.items_to_messages(
            messages,
            preserve_mid_system=mid_system,
        )

        # 2. Resolve tools — synthetic tool replaces user tools when
        #    structured output is requested. The Runner can register
        #    function tools alongside structured output, but the
        #    Anthropic synthetic-tool pattern only works when the
        #    structured tool is the only one the model is allowed to
        #    call (otherwise the model may pick a different tool).
        wire_tools: list[ToolUnionParam] | None
        if wants_structured and output_schema is not None:
            wire_tools = [AnthropicConverter.build_structured_output_tool(output_schema)]
        elif tools is not None and len(tools) > 0:
            wire_tools = AnthropicConverter.convert_tools(tools)
        else:
            wire_tools = None

        # 3. Resolve tool_choice. Structured output forces the
        #    synthetic tool; otherwise honour the config. Annotate
        #    the union explicitly so the structured-output branch's
        #    narrower ``ToolChoiceToolParam`` doesn't lock the type.
        tool_choice: ToolChoiceParam | None
        if wants_structured:
            tool_choice = AnthropicConverter.build_structured_output_tool_choice()
        else:
            tool_choice = AnthropicConverter.convert_tool_choice(
                config.tool_choice,
                tools_present=wire_tools is not None,
            )

        # 4. Resolve thinking / extended-reasoning.
        thinking = resolve_thinking(config)

        # 5. Resolve max_tokens. Anthropic requires it; raise the
        #    floor when extended thinking is active so the answer
        #    has room beyond the thinking budget. Only None means
        #    "unset" — an explicit 0 is the developer's value and is
        #    not coerced to the default.
        max_tokens = config.max_output_tokens if config.max_output_tokens is not None else _DEFAULT_MAX_TOKENS
        if thinking is not None and thinking.get("type") == "enabled":
            budget = thinking.get("budget_tokens", 0)
            if isinstance(budget, int) and budget >= max_tokens:
                # Anthropic requires max_tokens > budget_tokens. Raise
                # by the smallest margin so the developer's explicit
                # cost ceiling is not silently overshot by a full
                # default's worth of tokens they never opted into.
                new_floor = budget + 1
                logger.warning(
                    "Anthropic requires max_tokens > thinking budget_tokens; "
                    "raising max_tokens from %d to %d. Set a higher "
                    "max_output_tokens to leave more room for the answer.",
                    max_tokens,
                    new_floor,
                )
                max_tokens = new_floor

        # 6. Inject prompt-cache markers when AnthropicConfig opts in.
        #    The applicator may convert a string system prompt into a
        #    list of TextBlockParam, so widen the local type.
        wire_system: str | list[TextBlockParam] | None
        wire_system, converted_messages, wire_tools = apply_cache_control(
            system_prompt, converted_messages, wire_tools, config
        )

        # 7. AnthropicConfig-specific typed fields. Read after the
        #    base LLMConfig fields so explicit settings beat extra_args.
        service_tier = config.service_tier if isinstance(config, AnthropicConfig) else None
        effort = config.effort if isinstance(config, AnthropicConfig) else None

        # Merge ``extra_args`` and ``extra_body`` into a single dict
        # passed via the SDK's ``extra_body=`` parameter, which is
        # typed as ``object | None`` and accepts arbitrary extra
        # request-body fields. Routing through ``extra_body``
        # preserves the ``messages.create`` overload's typed return
        # — spreading ``**dict[str, Any]`` would degrade it to ``Any``.
        # ``thinking`` is consumed above and dropped from extra_args.
        combined_extra_body: dict[str, Any] = {}
        if isinstance(config.extra_body, Mapping):
            # ``Body`` is typed as bare ``object`` upstream so the SDK
            # can accept arbitrary JSON-serialisable payloads. Narrow
            # to ``Mapping`` here so ``dict.update`` is well-defined;
            # non-Mapping values are simply ignored (the only sensible
            # update for a body dict).
            combined_extra_body.update(config.extra_body)
        if config.extra_args is not None:
            for key, value in config.extra_args.items():
                if key == "thinking":
                    continue
                combined_extra_body[key] = value

        logger.info(
            "Anthropic API call: model=%s, max_tokens=%d, tools=%d, stream=%s, structured=%s",
            sanitize_for_log(self._model),
            max_tokens,
            len(wire_tools) if wire_tools is not None else 0,
            stream,
            wants_structured,
        )

        if stream is True:
            inner_stream = self._stream(
                messages=converted_messages,
                max_tokens=max_tokens,
                system=wire_system,
                tools=wire_tools,
                tool_choice=tool_choice,
                thinking=thinking,
                config=config,
                service_tier=service_tier,
                effort=effort,
                mid_system=mid_system,
                extra_body=combined_extra_body,
            )
            # Apply the cross-provider streaming-error contract: a mid-stream
            # failure emits a terminal done(finish_reason="error") then re-raises.
            return stream_with_error_contract(inner_stream, model=self._model, logger=logger)

        # Non-streaming: optionally wrap in framework retry loop.
        async def _do_call() -> Message:
            result = await self._call_messages(
                stream=False,
                messages=converted_messages,
                max_tokens=max_tokens,
                system=wire_system,
                tools=wire_tools,
                tool_choice=tool_choice,
                thinking=thinking,
                config=config,
                service_tier=service_tier,
                effort=effort,
                mid_system=mid_system,
                extra_body=combined_extra_body,
            )
            # The just-passed ``stream=False`` literal narrows the SDK's
            # overload union to ``Message``. Raise (not ``assert``, which
            # ``-O`` strips) so the runtime invariant is enforced even
            # under optimization; mirrors the OpenAI Responses model.
            if not isinstance(result, Message):
                raise TypeError(f"messages.create(stream=False) returned {type(result).__name__}, expected Message")
            return result

        if config.retry_policy is not None:
            response = await call_with_retry(
                _do_call,
                config.retry_policy,
                model=self._model,
            )
        else:
            response = await _do_call()

        logger.info(
            "Anthropic response: model=%s, stop_reason=%s, input_tokens=%d, output_tokens=%d",
            sanitize_for_log(response.model),
            sanitize_for_log(str(response.stop_reason)),
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        return self._build_response(response, output_schema if wants_structured else None)

    # ------------------------------------------------------------------
    # SDK boundary helper
    # ------------------------------------------------------------------

    async def _call_messages(
        self,
        *,
        stream: bool,
        messages: list[MessageParam],
        max_tokens: int,
        system: str | list[TextBlockParam] | None,
        tools: list[ToolUnionParam] | None,
        tool_choice: ToolChoiceParam | None,
        thinking: ThinkingConfigParam | None,
        config: LLMConfig,
        service_tier: Literal["auto", "standard_only"] | None,
        effort: EffortLevel | None,
        mid_system: bool,
        extra_body: dict[str, Any],
    ) -> Message | AsyncStream[RawMessageStreamEvent]:
        """Invoke ``client.messages.create`` and return the SDK union.

        Every optional Anthropic param is passed as an explicit
        keyword argument with ``NOT_GIVEN`` for the unset case, so
        the SDK's ``@overload`` chain resolves cleanly under
        type-checkers — spread-kwargs would degrade the return type
        to ``Any``.

        The return type matches the SDK's overload union for the
        ``stream: bool`` fallback signature exactly. Callers narrow
        via :func:`isinstance` after passing a Literal ``stream``
        value — pre-narrowing here would silence a downstream warning
        but mask SDK-side variant changes on upgrade.
        """
        client = self._get_client()

        # Honour a per-call SDK retry count. ``LLMConfig.num_retries`` maps
        # to the anthropic client's ``max_retries``; when unset (None) the
        # client keeps its construction-time ``max_retries`` (default 0 — no
        # retries), so the developer never pays for retries they did not
        # request. An explicit per-call value wins over the constructor.
        if config.num_retries is not None:
            client = client.with_options(max_retries=config.num_retries)

        # Resolve common optional kwargs once. Anthropic's ``omit``
        # sentinel lets the SDK keep the field unset rather than
        # sending an explicit null. (The older ``NOT_GIVEN`` /
        # ``NotGiven`` is still exported but the new ``messages.create``
        # overloads expect the ``Omit`` type — pass ``omit``.)
        kwargs_system = system if system is not None else omit
        kwargs_tools = tools if tools is not None else omit
        kwargs_tool_choice = tool_choice if tool_choice is not None else omit
        kwargs_thinking = thinking if thinking is not None else omit
        kwargs_temperature = config.temperature if config.temperature is not None else omit
        kwargs_top_p = config.top_p if config.top_p is not None else omit
        # ``top_k`` is ``int | Omit`` on the SDK; framework carries float.
        kwargs_top_k = int(config.top_k) if config.top_k is not None else omit
        kwargs_stop = config.stop_sequences if config.stop_sequences is not None else omit
        # ``timeout`` predates the ``Omit`` migration on the
        # anthropic SDK and still expects ``NotGiven`` at the
        # call boundary — keep that sentinel for this single field.
        kwargs_timeout = config.timeout if config.timeout is not None else NOT_GIVEN
        if mid_system:
            # In-place role:"system" messages are gated behind a beta
            # header; merge with any caller-supplied beta values. The
            # coercing copy exists only on this path — the default path
            # hands the caller's mapping straight to headers_as_sdk.
            effective_headers: dict[str, str] = (
                {key: str(value) for key, value in config.extra_headers.items()}
                if config.extra_headers is not None
                else {}
            )
            existing_beta = effective_headers.get("anthropic-beta")
            effective_headers["anthropic-beta"] = (
                f"{existing_beta},{_MID_SYSTEM_BETA}" if existing_beta else _MID_SYSTEM_BETA
            )
            kwargs_extra_headers = headers_as_sdk(effective_headers)
        else:
            kwargs_extra_headers = headers_as_sdk(config.extra_headers)
        kwargs_output_config: OutputConfigParam | Omit = (
            OutputConfigParam(effort=effort) if effort is not None else omit
        )
        kwargs_service_tier = service_tier if service_tier is not None else omit
        # Metadata: framework's ``Metadata`` is ``Mapping[str, object]``,
        # SDK's ``MetadataParam`` is ``{user_id?: str}``. The boundary
        # helper extracts ``user_id`` and drops the rest.
        normalised_metadata = metadata_as_sdk(config.metadata)
        kwargs_metadata = normalised_metadata if normalised_metadata is not None else omit
        # ``extra_query`` on ``messages.create`` is ``Query | None``
        # (not Omit-typed); pass ``None`` for the unset case, mirroring
        # the ``extra_body`` handling below.
        kwargs_extra_query = dict(config.extra_query) if config.extra_query is not None else None

        if stream is True:
            return await client.messages.create(
                model=self._model,
                messages=messages,
                max_tokens=max_tokens,
                system=kwargs_system,
                tools=kwargs_tools,
                tool_choice=kwargs_tool_choice,
                thinking=kwargs_thinking,
                temperature=kwargs_temperature,
                top_p=kwargs_top_p,
                top_k=kwargs_top_k,
                stop_sequences=kwargs_stop,
                metadata=kwargs_metadata,
                service_tier=kwargs_service_tier,
                output_config=kwargs_output_config,
                timeout=kwargs_timeout,
                extra_headers=kwargs_extra_headers,
                extra_query=kwargs_extra_query,
                stream=True,
                extra_body=extra_body if len(extra_body) > 0 else None,
            )
        return await client.messages.create(
            model=self._model,
            messages=messages,
            max_tokens=max_tokens,
            system=kwargs_system,
            tools=kwargs_tools,
            tool_choice=kwargs_tool_choice,
            thinking=kwargs_thinking,
            temperature=kwargs_temperature,
            top_p=kwargs_top_p,
            top_k=kwargs_top_k,
            stop_sequences=kwargs_stop,
            metadata=kwargs_metadata,
            service_tier=kwargs_service_tier,
            output_config=kwargs_output_config,
            timeout=kwargs_timeout,
            extra_headers=kwargs_extra_headers,
            extra_query=kwargs_extra_query,
            stream=False,
            extra_body=extra_body if len(extra_body) > 0 else None,
        )

    # ------------------------------------------------------------------
    # Response assembly (non-streaming)
    # ------------------------------------------------------------------

    def _build_response(
        self,
        message: Message,
        structured_schema: AgentOutputSchemaBase | None,
    ) -> LLMResponse:
        """Build the framework :class:`LLMResponse` from a Message.

        When *structured_schema* is provided, the synthetic tool's
        ``ToolUseBlock.input`` is JSON-serialised, validated through
        the schema, and surfaced as an :class:`LLMResponseText` part
        carrying that JSON. This matches the Runner's expectation that
        structured output arrives as JSON text on a text part —
        identical to the litellm path's ``json_schema`` response
        format. The original tool_use part is dropped so it doesn't
        leak into the conversation history as an unresolved tool call.
        """
        response = AnthropicConverter.response_to_llm_response(message)
        if structured_schema is None:
            return response

        # Validate first — any failure surfaces immediately rather
        # than letting an unstructured response slip through.
        AnthropicConverter.parse_structured_output(message, structured_schema)

        # Replace the synthetic tool's function-call part with a
        # text part carrying the JSON. The Runner reads from
        # ``LLMResponse.content`` (which joins all text parts) and
        # validates that JSON via ``output_schema.validate_json``.
        rebuilt: list[LLMResponsePart] = []
        for part in response.response:
            if isinstance(part, LLMResponseFunctionToolCall) and part.name == STRUCTURED_OUTPUT_TOOL_NAME:
                rebuilt.append(LLMResponseText(text=part.arguments))
            else:
                rebuilt.append(part)
        response.response = rebuilt
        # The tool_use stop_reason is correct for the wire response
        # but misleading for a structured-output run — the model
        # produced its final answer, not a tool call awaiting
        # execution. Normalise to ``end_turn`` so the Runner does
        # not try to dispatch the synthetic tool.
        response.finish_reason = "end_turn"
        return response

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    async def _stream(
        self,
        *,
        messages: list[MessageParam],
        max_tokens: int,
        system: str | list[TextBlockParam] | None,
        tools: list[ToolUnionParam] | None,
        tool_choice: ToolChoiceParam | None,
        thinking: ThinkingConfigParam | None,
        config: LLMConfig,
        service_tier: Literal["auto", "standard_only"] | None,
        effort: EffortLevel | None,
        mid_system: bool,
        extra_body: dict[str, Any],
    ) -> AsyncIterator[LLMStreamEvent]:
        """Process Anthropic streaming events → ``LLMStreamEvent`` yields.

        Anthropic stream-event taxonomy:

        - ``message_start`` — initial Message metadata + input_tokens
        - ``content_block_start`` — new block (text / tool_use / thinking)
        - ``content_block_delta`` — text / json / thinking / signature / citation deltas
        - ``content_block_stop`` — block complete; flush the accumulator
        - ``message_delta`` — final stop_reason + output_tokens
        - ``message_stop`` — stream complete (no payload)

        The block-keyed accumulator preserves stable ordering and avoids
        the ``signature_delta`` race the prior implementation hit when
        signature deltas arrived before any thinking part had been
        appended to a list.

        Yields:
            ``LLMStreamEvent`` objects ending with a ``"done"`` event
            carrying the assembled :class:`LLMResponse`.
        """
        stream_iter = await self._call_messages(
            stream=True,
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            thinking=thinking,
            config=config,
            service_tier=service_tier,
            effort=effort,
            mid_system=mid_system,
            extra_body=extra_body,
        )
        # ``stream=True`` above guarantees AsyncStream, not Message —
        # narrow at the call site rather than pre-narrowing the helper's
        # return type. Raise (not ``assert``, which ``-O`` strips) so the
        # invariant holds under optimization.
        if not isinstance(stream_iter, AsyncStream):
            raise TypeError(f"messages.create(stream=True) returned {type(stream_iter).__name__}, expected AsyncStream")

        try:
            async for stream_event in self._assemble_stream(stream_iter):
                yield stream_event
        finally:
            # Always release the SDK stream's HTTP connection — on normal
            # completion, on an early consumer break (generator aclose), and
            # on a mid-stream error. ``AsyncStream.close`` is idempotent
            # (a fully-drained stream is already closed).
            await stream_iter.close()

    async def _assemble_stream(
        self,
        stream_iter: AsyncStream[RawMessageStreamEvent],
    ) -> AsyncIterator[LLMStreamEvent]:
        """Consume the Anthropic event stream into ``LLMStreamEvent`` yields.

        Split from :meth:`_stream` so the caller can guarantee the SDK
        stream is closed in a ``finally`` regardless of how iteration ends
        (normal completion, early consumer break, or mid-stream error).

        Yields:
            ``LLMStreamEvent`` objects ending with a ``"done"`` event
            carrying the assembled :class:`LLMResponse`.
        """
        accumulators: dict[int, _BlockAccumulator] = {}
        block_order: list[int] = []
        message_id = ""
        model_name = self._model
        start_input_tokens = 0
        start_cache_read = 0
        start_cache_creation = 0
        final_output_tokens = 0
        final_thinking_tokens = 0
        final_stop_reason: str | None = None

        async for event in stream_iter:
            if isinstance(event, RawMessageStartEvent):
                msg = event.message
                message_id = msg.id
                model_name = msg.model
                # Capture the prompt-side usage that the prior
                # implementation discarded — Anthropic delivers it
                # only on this event during streaming.
                start_input_tokens = msg.usage.input_tokens or 0
                start_cache_read = msg.usage.cache_read_input_tokens or 0
                start_cache_creation = msg.usage.cache_creation_input_tokens or 0
                continue

            if isinstance(event, RawContentBlockStartEvent):
                block = event.content_block
                block_type = getattr(block, "type", "")
                acc = _BlockAccumulator(block_type=block_type)
                if block_type == "tool_use":
                    acc.tool_id = getattr(block, "id", "")
                    acc.tool_name = getattr(block, "name", "")
                    yield LLMStreamEvent(
                        type="part_start",
                        index=event.index,
                        part=LLMResponseFunctionToolCall(
                            call_id=acc.tool_id,
                            name=acc.tool_name,
                        ),
                    )
                elif block_type == "thinking":
                    yield LLMStreamEvent(
                        type="part_start",
                        index=event.index,
                        part=LLMResponseReasoning(),
                    )
                elif block_type == "text":
                    yield LLMStreamEvent(
                        type="part_start",
                        index=event.index,
                        part=LLMResponseText(),
                    )
                elif block_type == "redacted_thinking":
                    acc.redacted_data = getattr(block, "data", "")
                accumulators[event.index] = acc
                block_order.append(event.index)
                continue

            if isinstance(event, RawContentBlockDeltaEvent):
                # Fetch the accumulator into a fresh local so mypy
                # narrows on ``Optional`` without colliding with the
                # ``acc`` introduced in the start-event branch.
                delta_acc: _BlockAccumulator | None = accumulators.get(event.index)
                if delta_acc is None:
                    # A delta with no matching start is a protocol
                    # violation; log and drop rather than crash.
                    logger.debug("Anthropic stream: orphan delta at index=%d", event.index)
                    continue
                delta = event.delta
                delta_type = getattr(delta, "type", "")
                if delta_type == "text_delta":
                    text = getattr(delta, "text", "")
                    if len(text) > 0:
                        delta_acc.text_buf.append(text)
                        yield LLMStreamEvent(
                            type="part_delta",
                            index=event.index,
                            delta=text,
                        )
                elif delta_type == "input_json_delta":
                    partial = getattr(delta, "partial_json", "")
                    if len(partial) > 0:
                        delta_acc.tool_input_buf.append(partial)
                        yield LLMStreamEvent(
                            type="part_delta",
                            index=event.index,
                            delta=partial,
                        )
                elif delta_type == "thinking_delta":
                    thinking_text = getattr(delta, "thinking", "")
                    if len(thinking_text) > 0:
                        delta_acc.thinking_buf.append(thinking_text)
                        yield LLMStreamEvent(
                            type="part_delta",
                            index=event.index,
                            delta=thinking_text,
                        )
                elif delta_type == "signature_delta":
                    # Bug fix: signature deltas land in a per-block
                    # buffer instead of indexing into a not-yet-populated
                    # global thinking-parts list.
                    signature = getattr(delta, "signature", "")
                    if len(signature) > 0:
                        delta_acc.signature_buf.append(signature)
                elif delta_type == "citations_delta":
                    # Citations on streaming text aren't currently
                    # surfaced by ``LLMResponseText.annotations`` — the
                    # framework's annotation type is OpenAI-shaped.
                    # Drop with a debug log so this remains visible to
                    # anyone investigating later. Annotation parity
                    # across providers is a separate workstream.
                    logger.debug(
                        "Anthropic stream: dropping citations_delta at index=%d "
                        "(framework annotation parity not yet wired)",
                        event.index,
                    )
                continue

            if isinstance(event, RawContentBlockStopEvent):
                # Block boundary — emit part_end so consumers can
                # render the closure even though the part itself is
                # built when we assemble the final response below.
                yield LLMStreamEvent(type="part_end", index=event.index)
                continue

            if isinstance(event, RawMessageDeltaEvent):
                _raw_stop_reason = event.delta.stop_reason
                final_output_tokens = event.usage.output_tokens or 0
                # Refresh the prompt-side counters from message_delta when
                # the API supplies finalized values — they supersede the
                # preliminary message_start counts. Dropping them would
                # under-report cache_read / cache_creation usage (and any
                # corrected input_tokens) on the streaming path.
                if event.usage.input_tokens is not None:
                    start_input_tokens = event.usage.input_tokens
                if event.usage.cache_read_input_tokens is not None:
                    start_cache_read = event.usage.cache_read_input_tokens
                if event.usage.cache_creation_input_tokens is not None:
                    start_cache_creation = event.usage.cache_creation_input_tokens
                # Capture extended-thinking output token count when available.
                _delta_out_details = getattr(event.usage, "output_tokens_details", None)
                final_thinking_tokens = (
                    getattr(_delta_out_details, "thinking_tokens", 0) or 0 if _delta_out_details is not None else 0
                )
                # Encode structured refusal details into finish_reason so
                # callers can distinguish safety categories — mirrors the
                # non-streaming path in AnthropicConverter.response_to_llm_response.
                if _raw_stop_reason == "refusal":
                    _stop_details = getattr(event.delta, "stop_details", None)
                    if _stop_details is not None:
                        _category = getattr(_stop_details, "category", None)
                        if _category:
                            final_stop_reason = f"refusal:{_category}"
                            continue
                final_stop_reason = _raw_stop_reason
                continue

            # ``message_stop`` and unknown future events fall through
            # — the loop naturally terminates when the stream closes.

        # Assemble parts in stable block-index order.
        parts: list[LLMResponsePart] = []
        for idx in block_order:
            acc = accumulators[idx]
            if acc.block_type == "text":
                parts.append(LLMResponseText(text="".join(acc.text_buf)))
            elif acc.block_type == "tool_use":
                parts.append(
                    LLMResponseFunctionToolCall(
                        call_id=acc.tool_id,
                        name=acc.tool_name,
                        arguments="".join(acc.tool_input_buf),
                    )
                )
            elif acc.block_type == "thinking":
                parts.append(
                    LLMResponseReasoning(
                        thinking="".join(acc.thinking_buf),
                        signature="".join(acc.signature_buf) if len(acc.signature_buf) > 0 else None,
                    )
                )
            elif acc.block_type == "redacted_thinking":
                parts.append(
                    LLMResponseReasoning(
                        thinking="",
                        encrypted_content=acc.redacted_data,
                        is_redacted=True,
                    )
                )
            else:
                logger.debug(
                    "Anthropic stream: dropping block_type=%s at index=%d",
                    acc.block_type,
                    idx,
                )

        usage = self._build_streaming_usage(
            input_tokens=start_input_tokens,
            output_tokens=final_output_tokens,
            cache_read=start_cache_read,
            cache_creation=start_cache_creation,
            thinking_tokens=final_thinking_tokens,
        )

        response = LLMResponse(
            response_id=message_id,
            model=model_name,
            response=parts,
            usage=usage,
            finish_reason=final_stop_reason,
        )

        yield LLMStreamEvent(type="done", response=response)

    @staticmethod
    def _build_streaming_usage(
        *,
        input_tokens: int,
        output_tokens: int,
        cache_read: int,
        cache_creation: int,
        thinking_tokens: int = 0,
    ) -> LLMUsage:
        """Construct the per-stream :class:`LLMUsage`.

        Anthropic streaming splits usage across two events:

        - ``message_start`` carries ``input_tokens`` (and the cache
          fields).
        - ``message_delta`` (final) carries ``output_tokens`` and
          ``output_tokens_details.thinking_tokens`` when extended
          thinking is enabled.

        The prior implementation collected only the latter, so every
        streamed response reported ``input_tokens=0``. This helper
        merges both halves.
        """
        from troopai.adk.types.tokens import (
            InputTokensDetails,
            LLMSingleRequestUsage,
            LLMUsage,
            OutputTokensDetails,
        )

        input_details = InputTokensDetails(
            cached_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
        )
        output_details = OutputTokensDetails(reasoning_tokens=thinking_tokens)
        # Anthropic reports the ``message_start`` / ``message_delta``
        # ``input_tokens`` EXCLUSIVE of cache tokens. Report the inclusive
        # total (raw + cache_read + cache_creation) so streamed runs match
        # the non-streaming path and the litellm path — otherwise token
        # limits and cost tracking under-count whenever caching is active.
        inclusive_input = input_tokens + cache_read + cache_creation
        total = inclusive_input + output_tokens
        single = LLMSingleRequestUsage(
            input_tokens=inclusive_input,
            output_tokens=output_tokens,
            total_tokens=total,
            input_tokens_details=input_details,
            output_tokens_details=output_details,
        )
        return LLMUsage(
            requests=1,
            input_tokens=inclusive_input,
            output_tokens=output_tokens,
            total_tokens=total,
            input_tokens_details=input_details,
            output_tokens_details=output_details,
            usage=[single],
        )
