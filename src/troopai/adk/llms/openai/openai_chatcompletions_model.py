"""Native OpenAI Chat Completions LLM implementation.

Calls ``openai.AsyncOpenAI().chat.completions.create()`` directly — no
litellm layer. Input items and output items round-trip through
``openai.types.chat.*`` via :class:`OpenAIChatCompletionsConverter`;
wire types never leak outside this module's call sites.

Provider-native hosted capabilities (web_search, file_search,
code_interpreter, image generation, hosted MCP servers, …) are NOT
wrapped in framework tool classes — pass the raw provider JSON through
:attr:`LLMConfig.extra_body` / :attr:`LLMConfig.extra_args`. The Chat
Completions API also offers ``web_search_options`` as a first-class
parameter on :class:`OpenAIChatCompletionsConfig`.

Usage::

    from troopai.adk.llms.openai import OpenAIChatCompletionsLLM
    from troopai.adk.tools import function_tool


    @function_tool(name="lookup", description="Look up a record")
    def lookup(record_id: str) -> str:
        return f"Record {record_id}"


    agent = Agent(
        llm=OpenAIChatCompletionsLLM(model="gpt-4o"),
        tools=[lookup],
    )

Refs:
    - Chat Completions API:
      https://platform.openai.com/docs/api-reference/chat
    - OpenAI Python SDK:
      https://github.com/openai/openai-python
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Mapping
from typing import TYPE_CHECKING, Any, Literal, overload, override

from openai import NOT_GIVEN, omit

from troopai.adk.llms.llm import LLM
from troopai.adk.llms.openai.openai_boundary import (
    headers_as_sdk,
    metadata_as_sdk,
    query_as_sdk,
    sanitize_for_log,
)
from troopai.adk.llms.openai.openai_chatcompletions_converter import (
    OpenAIChatCompletionsConverter,
)
from troopai.adk.llms.openai.openai_retry import call_with_retry
from troopai.adk.llms.stream_error import stream_with_error_contract
from troopai.adk.types.input import LLMInputContentItem, LLMInputEasyMessage
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
    LLMResponsePart,
    LLMResponseRefusal,
    LLMResponseText,
    LLMStreamEvent,
)
from troopai.adk.types.tokens.llm_usage import LLMUsage

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai.types.chat import ChatCompletion, ChatCompletionChunk

    from troopai.adk.llms.llm_config import LLMConfig
    from troopai.adk.llms.openai.openai_chatcompletions_config import (
        OpenAIChatCompletionsConfig,
    )
    from troopai.adk.schemas import AgentOutputSchemaBase
    from troopai.adk.tools import Tool


logger = logging.getLogger(__name__)


# Upper bound on the number of distinct tool_call indices a single
# streaming response may emit. The SDK's protocol is driven by the
# model — a buggy or adversarial upstream could, in theory, stream
# an unbounded number of unique ``tool_calls[i].index`` values and
# balloon the accumulator dict. 256 is well above any realistic
# agent turn (OpenAI's documented ``parallel_tool_calls`` ceiling is
# in the tens) and below anything that would materially affect
# memory. Excess entries are dropped with a WARNING so operators
# can see the upstream misbehaviour.
_MAX_TOOL_CALLS_PER_STREAM: int = 256


class OpenAIChatCompletionsLLM(LLM):
    """Native OpenAI Chat-Completions-API ``LLM`` implementation.

    Calls ``client.chat.completions.create()`` directly with explicit
    named parameters. ``OpenAIChatCompletionsConfig`` fields
    (``audio``, ``web_search_options``, ``prediction``, ``modalities``,
    ``store``, ``service_tier``, ``prompt_cache_key``,
    ``prompt_cache_retention``, ``verbosity``) are routed verbatim when
    an :class:`OpenAIChatCompletionsConfig` instance is supplied; the
    generic :class:`LLMConfig` base provides ``temperature`` / ``top_p``
    / ``max_output_tokens`` → ``max_completion_tokens`` /
    ``top_logprobs`` / ``stop_sequences`` → ``stop`` /
    ``frequency_penalty`` / ``presence_penalty`` / ``seed`` /
    ``tool_choice`` / ``tool_execution_mode`` plus the ``extra_*``
    escape hatches.

    ``max_output_tokens`` maps to ``max_completion_tokens``, which
    OpenAI's reasoning models require (they reject ``max_tokens``).

    When ``stream=True`` the wrapper always injects
    ``stream_options={"include_usage": True}`` so the final
    ``ChatCompletionChunk`` carries ``CompletionUsage``. Set
    :attr:`LLMConfig.include_usage` to ``False`` to opt out.

    Args:
        model: OpenAI model ID (e.g., ``"gpt-4o"``, ``"gpt-5.1"``). Do
            NOT prefix with ``"openai/"`` — this is not litellm.
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
        """The model identifier passed to ``client.chat.completions.create()``."""
        return self._model

    def _get_client(self) -> AsyncOpenAI:
        """Lazy-initialize the OpenAI async client.

        Constructed on first use and cached for the lifetime of the
        ``LLM`` instance — same pattern as ``OpenAIResponsesLLM``.
        """
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError as e:
                raise ImportError(
                    "The 'openai' package is required for OpenAIChatCompletionsLLM. Install it with: pip install openai"
                ) from e

            self._client = AsyncOpenAI(
                api_key=self._api_key or os.environ.get("OPENAI_API_KEY"),
                base_url=self._base_url,
                organization=self._organization,
                project=self._project,
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
        """Call the OpenAI Chat Completions API.

        Orchestrates: convert → resolve params → API call →
        parse/stream. Non-streaming calls route through
        :func:`call_with_retry` when :attr:`LLMConfig.retry_policy` is
        set; streaming calls are never retried (mid-stream reconnect
        would silently drop or double-emit tokens).
        """
        from troopai.adk.llms.llm_config import LLMConfig
        from troopai.adk.llms.openai.openai_chatcompletions_config import (
            OpenAIChatCompletionsConfig,
        )

        config: LLMConfig = llm_config if llm_config is not None else LLMConfig()
        cc_config: OpenAIChatCompletionsConfig | None = (
            config if isinstance(config, OpenAIChatCompletionsConfig) else None
        )

        # 1. Normalise messages → Chat-Completions messages list
        if isinstance(messages, str):
            easy: LLMInputEasyMessage = {"role": "user", "content": messages}
            normalised: list[LLMInputContentItem] = [easy]
        else:
            normalised = messages
        wire_messages = OpenAIChatCompletionsConverter.items_to_messages(normalised)

        # 2. Convert tools
        wire_tools = (
            OpenAIChatCompletionsConverter.convert_tools(tools) if tools is not None and len(tools) > 0 else None
        )

        # 3. Resolve tool_choice (NESTED shape: {type: function, function: {name: ...}})
        tool_choice = OpenAIChatCompletionsConverter.convert_tool_choice(
            config.tool_choice,
            tools_present=wire_tools is not None,
        )

        # 4. Resolve structured-output response_format
        response_format = OpenAIChatCompletionsConverter.resolve_response_format(output_schema)

        # 5. parallel_tool_calls from tool_execution_mode. Only sent when
        #    tools are actually present — the OpenAI API rejects
        #    ``parallel_tool_calls`` (even ``False``) with a 400 when no
        #    ``tools`` are specified, so gate on ``wire_tools`` the same
        #    way ``tool_choice`` does above.
        parallel_tool_calls: bool | None = None
        if config.tool_execution_mode is not None and wire_tools is not None:
            parallel_tool_calls = config.tool_execution_mode == "parallel"

        # 6. Merge extra_body + extra_args — same contract as the
        #    Responses wrapper; see that module for the rationale.
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

        # 7. ChatCompletions-specific fields — only populated when the
        #    caller passed an OpenAIChatCompletionsConfig. Nullable
        #    fields pass through as ``None``; non-nullable SDK fields
        #    (``prompt_cache_key``, ``web_search_options``) use the
        #    ``omit`` sentinel when unset.
        audio = cc_config.audio if cc_config is not None else None
        web_search_options = cc_config.web_search_options if cc_config is not None else None
        prediction = cc_config.prediction if cc_config is not None else None
        modalities = cc_config.modalities if cc_config is not None else None
        store = cc_config.store if cc_config is not None else None
        service_tier = cc_config.service_tier if cc_config is not None else None
        prompt_cache_key = cc_config.prompt_cache_key if cc_config is not None else None
        prompt_cache_retention = cc_config.prompt_cache_retention if cc_config is not None else None
        verbosity = cc_config.verbosity if cc_config is not None else None

        logger.info(
            "OpenAI Chat Completions API call: model=%s, messages=%d, tools=%d, stream=%s",
            sanitize_for_log(self._model),
            len(wire_messages),
            len(wire_tools) if wire_tools is not None else 0,
            stream,
        )

        if stream:
            # Default: include usage in the final chunk so downstream
            # consumers can report accurate token counts. Explicit
            # ``include_usage=False`` on the config overrides.
            include_usage_flag = config.include_usage if config.include_usage is not None else True
            stream_opts: Any = {"include_usage": True} if include_usage_flag else omit

            chunk_iter = await client.chat.completions.create(
                model=self._model,
                messages=wire_messages,
                tools=wire_tools if wire_tools is not None else omit,
                tool_choice=tool_choice if tool_choice is not None else omit,
                response_format=response_format if response_format is not None else omit,
                temperature=config.temperature,
                top_p=config.top_p,
                max_completion_tokens=config.max_output_tokens,
                logprobs=config.response_logprobs,
                top_logprobs=config.top_logprobs,
                stop=config.stop_sequences,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                seed=config.seed,
                parallel_tool_calls=parallel_tool_calls if parallel_tool_calls is not None else omit,
                metadata=metadata_as_sdk(config.metadata),
                timeout=config.timeout if config.timeout is not None else NOT_GIVEN,
                extra_headers=headers_as_sdk(config.extra_headers),
                extra_query=query_as_sdk(config.extra_query),
                extra_body=extra_body,
                audio=audio if audio is not None else omit,
                web_search_options=web_search_options if web_search_options is not None else omit,
                prediction=prediction if prediction is not None else omit,
                modalities=modalities if modalities is not None else omit,
                store=store if store is not None else omit,
                service_tier=service_tier if service_tier is not None else omit,
                prompt_cache_key=prompt_cache_key if prompt_cache_key is not None else omit,
                prompt_cache_retention=prompt_cache_retention if prompt_cache_retention is not None else omit,
                verbosity=verbosity if verbosity is not None else omit,
                stream=True,
                stream_options=stream_opts,
            )
            # Cross-provider streaming-error contract: a mid-stream failure
            # emits a terminal done(finish_reason="error") then re-raises.
            return stream_with_error_contract(self._stream(chunk_iter), model=self._model, logger=logger)

        async def _call_completions() -> ChatCompletion:
            return await client.chat.completions.create(
                model=self._model,
                messages=wire_messages,
                tools=wire_tools if wire_tools is not None else omit,
                tool_choice=tool_choice if tool_choice is not None else omit,
                response_format=response_format if response_format is not None else omit,
                temperature=config.temperature,
                top_p=config.top_p,
                max_completion_tokens=config.max_output_tokens,
                logprobs=config.response_logprobs,
                top_logprobs=config.top_logprobs,
                stop=config.stop_sequences,
                frequency_penalty=config.frequency_penalty,
                presence_penalty=config.presence_penalty,
                seed=config.seed,
                parallel_tool_calls=parallel_tool_calls if parallel_tool_calls is not None else omit,
                metadata=metadata_as_sdk(config.metadata),
                timeout=config.timeout if config.timeout is not None else NOT_GIVEN,
                extra_headers=headers_as_sdk(config.extra_headers),
                extra_query=query_as_sdk(config.extra_query),
                extra_body=extra_body,
                audio=audio if audio is not None else omit,
                web_search_options=web_search_options if web_search_options is not None else omit,
                prediction=prediction if prediction is not None else omit,
                modalities=modalities if modalities is not None else omit,
                store=store if store is not None else omit,
                service_tier=service_tier if service_tier is not None else omit,
                prompt_cache_key=prompt_cache_key if prompt_cache_key is not None else omit,
                prompt_cache_retention=prompt_cache_retention if prompt_cache_retention is not None else omit,
                verbosity=verbosity if verbosity is not None else omit,
                stream=False,
            )

        if config.retry_policy is not None:
            completion = await call_with_retry(
                _call_completions,
                config.retry_policy,
                model=self._model,
            )
        else:
            completion = await _call_completions()

        parsed = OpenAIChatCompletionsConverter.chat_completion_to_llm_response(completion)
        if parsed.usage is not None:
            logger.info(
                "OpenAI Chat Completions result: model=%s, input_tokens=%d, output_tokens=%d",
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
        chunk_iter: AsyncIterator[ChatCompletionChunk],
    ) -> AsyncIterator[LLMStreamEvent]:
        """Reconstruct framework ``LLMStreamEvent``s from Chat Completions chunks.

        Chat Completions streaming is raw-delta-based — it does NOT
        emit explicit "item added / done" events like the Responses
        API. This wrapper therefore tracks part boundaries itself:
        emit ``part_start`` the first time text content or a given
        tool-call index appears, emit ``part_delta`` for every
        subsequent delta, and emit ``part_end`` for each active part
        when the choice's ``finish_reason`` arrives. The terminal
        ``"done"`` event carries the fully reconstructed
        :class:`LLMResponse`.
        """
        response_id: str = ""
        model: str = self._model
        text_index: int | None = None
        text_buffer: list[str] = []
        refusal_index: int | None = None
        refusal_buffer: list[str] = []
        # tool_call_index (chunk-delta index) → (framework part index,
        # call_id, name, accumulated arguments)
        tool_calls: dict[int, _ToolCallAccumulator] = {}
        next_part_index: int = 0
        usage: LLMUsage | None = None
        finish_reason: str | None = None

        async for chunk in chunk_iter:
            if len(response_id) == 0 and len(chunk.id) > 0:
                response_id = chunk.id
            if len(chunk.model) > 0:
                model = chunk.model

            if chunk.usage is not None:
                usage = OpenAIChatCompletionsConverter.parse_usage(chunk.usage)

            if len(chunk.choices) == 0:
                # The final "usage-only" chunk emitted when
                # include_usage is on. No deltas here.
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content is not None and len(delta.content) > 0:
                if text_index is None:
                    text_index = next_part_index
                    next_part_index += 1
                    yield LLMStreamEvent(
                        type="part_start",
                        index=text_index,
                        part=LLMResponseText(text=""),
                    )
                text_buffer.append(delta.content)
                yield LLMStreamEvent(
                    type="part_delta",
                    index=text_index,
                    delta=delta.content,
                )

            if delta.refusal is not None and len(delta.refusal) > 0:
                if refusal_index is None:
                    refusal_index = next_part_index
                    next_part_index += 1
                    yield LLMStreamEvent(
                        type="part_start",
                        index=refusal_index,
                        part=LLMResponseRefusal(refusal=""),
                    )
                refusal_buffer.append(delta.refusal)
                yield LLMStreamEvent(
                    type="part_delta",
                    index=refusal_index,
                    delta=delta.refusal,
                )

            if delta.tool_calls is not None:
                for tc_delta in delta.tool_calls:
                    tc_idx = tc_delta.index
                    acc = tool_calls.get(tc_idx)
                    if acc is None:
                        if len(tool_calls) >= _MAX_TOOL_CALLS_PER_STREAM:
                            # Hard cap: drop new tool_call indices once
                            # the accumulator is saturated. Log once per
                            # overflow (subsequent new indices fall into
                            # the same branch and are also dropped).
                            logger.warning(
                                "Chat Completions stream exceeded %d distinct tool_call indices; dropping index=%d.",
                                _MAX_TOOL_CALLS_PER_STREAM,
                                tc_idx,
                            )
                            continue
                        call_id = tc_delta.id if tc_delta.id is not None else ""
                        fn_name = ""
                        if tc_delta.function is not None and tc_delta.function.name is not None:
                            fn_name = tc_delta.function.name
                        part_index = next_part_index
                        next_part_index += 1
                        acc = _ToolCallAccumulator(
                            part_index=part_index,
                            call_id=call_id,
                            name=fn_name,
                            arguments=[],
                        )
                        tool_calls[tc_idx] = acc
                        yield LLMStreamEvent(
                            type="part_start",
                            index=part_index,
                            part=LLMResponseFunctionToolCall(
                                call_id=call_id,
                                name=fn_name,
                                arguments="",
                            ),
                        )
                    else:
                        # ``id`` arrives on the first chunk for a given
                        # ``tool_calls[i]``; subsequent chunks may omit
                        # it. Same for ``function.name``. Be defensive.
                        if tc_delta.id is not None and len(acc.call_id) == 0:
                            acc.call_id = tc_delta.id
                        if tc_delta.function is not None and tc_delta.function.name is not None and len(acc.name) == 0:
                            acc.name = tc_delta.function.name

                    if tc_delta.function is not None and tc_delta.function.arguments is not None:
                        args_delta = tc_delta.function.arguments
                        if len(args_delta) > 0:
                            acc.arguments.append(args_delta)
                            yield LLMStreamEvent(
                                type="part_delta",
                                index=acc.part_index,
                                delta=args_delta,
                            )

            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
                if text_index is not None:
                    yield LLMStreamEvent(type="part_end", index=text_index)
                if refusal_index is not None:
                    yield LLMStreamEvent(type="part_end", index=refusal_index)
                for acc in tool_calls.values():
                    yield LLMStreamEvent(type="part_end", index=acc.part_index)

        # Build the final response from the accumulated state. Preserve
        # the order we assigned part indices (text first if present,
        # then tool calls by delta-index).
        parts: list[LLMResponsePart] = []
        if text_index is not None:
            parts.append(LLMResponseText(text="".join(text_buffer)))
        if refusal_index is not None:
            parts.append(LLMResponseRefusal(refusal="".join(refusal_buffer)))
        for _, acc in sorted(tool_calls.items()):
            parts.append(
                LLMResponseFunctionToolCall(
                    call_id=acc.call_id,
                    name=acc.name,
                    arguments="".join(acc.arguments),
                    id=acc.call_id,
                    status="completed",
                )
            )

        final = LLMResponse(
            response_id=response_id,
            model=model,
            response=parts,
            usage=usage,
            finish_reason=finish_reason if finish_reason is not None else "stop",
        )
        yield LLMStreamEvent(type="done", response=final)


# ---------------------------------------------------------------------
# Internal accumulators / helpers
# ---------------------------------------------------------------------


class _ToolCallAccumulator:
    """Per-tool-call-index scratch state used by :meth:`_stream`."""

    __slots__ = ("arguments", "call_id", "name", "part_index")

    def __init__(
        self,
        *,
        part_index: int,
        call_id: str,
        name: str,
        arguments: list[str],
    ) -> None:
        self.part_index = part_index
        self.call_id = call_id
        self.name = name
        self.arguments = arguments
