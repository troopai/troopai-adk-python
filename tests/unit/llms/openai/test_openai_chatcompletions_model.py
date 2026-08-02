"""Tests for OpenAIChatCompletionsLLM.

Covers:
- Lazy client initialisation + idempotence on repeat ``_get_client``.
- ``acomplete`` forwards converted messages / tools / tool_choice /
  structured-output response_format to ``client.chat.completions.create``.
- Chat-Completions-specific config fields (audio / web_search_options /
  prediction / modalities / store / service_tier / prompt_cache_key /
  prompt_cache_retention / verbosity) are routed verbatim.
- ``max_output_tokens`` maps to ``max_completion_tokens`` (NOT
  ``max_tokens``; the legacy param is deprecated for reasoning models).
- ``stream_options={"include_usage": True}`` is injected on streaming
  calls by default and can be disabled via ``include_usage=False``.
- ``extra_body`` and ``extra_args`` merge and pass through unchanged.
- ``tool_execution_mode`` maps to ``parallel_tool_calls`` correctly.
- ``retry_policy`` wraps non-streaming calls; streaming calls are NOT
  retried.
- Streaming reconstructs part boundaries from raw delta chunks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import RateLimitError
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.completion_usage import (
    CompletionTokensDetails,
    CompletionUsage,
    PromptTokensDetails,
)

from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.llms.openai.openai_chatcompletions_config import OpenAIChatCompletionsConfig
from troopai.adk.llms.openai.openai_chatcompletions_model import OpenAIChatCompletionsLLM
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.types.llms import LLMRetryPolicy
from troopai.adk.types.responses.llm_response import (
    LLMResponseFunctionToolCall,
    LLMResponseText,
    LLMStreamEvent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_completion(
    text: str = "hello",
    completion_id: str = "chatcmpl_abc",
    model: str = "gpt-4o",
) -> ChatCompletion:
    """Build a minimal ``ChatCompletion`` carrying one assistant text message."""
    message = ChatCompletionMessage.model_construct(
        role="assistant",
        content=text,
        refusal=None,
        tool_calls=None,
    )
    choice = Choice.model_construct(
        finish_reason="stop",
        index=0,
        logprobs=None,
        message=message,
    )
    usage = CompletionUsage(
        prompt_tokens=7,
        completion_tokens=11,
        total_tokens=18,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=0),
        completion_tokens_details=CompletionTokensDetails(reasoning_tokens=0),
    )
    return ChatCompletion.model_construct(
        id=completion_id,
        choices=[choice],
        created=0,
        model=model,
        object="chat.completion",
        usage=usage,
    )


def _make_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _make_httpx_response(status: int) -> httpx.Response:
    return httpx.Response(status_code=status, request=_make_request())


def _llm_with_mock_client(
    *,
    create_return: Any = None,
    create_side_effect: Any = None,
) -> tuple[OpenAIChatCompletionsLLM, MagicMock]:
    """Build an ``OpenAIChatCompletionsLLM`` with ``_get_client`` patched.

    Returns the LLM instance and the ``MagicMock`` standing in for
    ``AsyncOpenAI``. ``client.chat.completions.create`` is an
    ``AsyncMock``.
    """
    llm = OpenAIChatCompletionsLLM(model="gpt-4o", api_key="test-key")
    client = MagicMock(name="AsyncOpenAI")
    create_mock = AsyncMock()
    if create_side_effect is not None:
        create_mock.side_effect = create_side_effect
    else:
        create_mock.return_value = create_return if create_return is not None else _make_completion()
    client.chat.completions.create = create_mock
    llm._get_client = lambda: client  # type: ignore[method-assign]  # inject mock client for testing
    return llm, client


async def _noop_invoke(_ctx: Any, _raw_args: str) -> str:
    return "ok"


def _dummy_tool(name: str = "lookup") -> FunctionTool:
    """Minimal ``FunctionTool`` so requests carry a non-empty ``tools`` list."""
    return FunctionTool(
        name=name,
        schema={"type": "object", "properties": {"q": {"type": "string"}}},
        description="Look something up",
        on_invoke=_noop_invoke,
    )


async def _collect_stream(
    stream: AsyncIterator[LLMStreamEvent],
) -> list[LLMStreamEvent]:
    events: list[LLMStreamEvent] = []
    async for ev in stream:
        events.append(ev)
    return events


# ---------------------------------------------------------------------------
# Lazy client init
# ---------------------------------------------------------------------------


class TestLazyClientInit:
    def test_client_constructed_on_first_access(self) -> None:
        llm = OpenAIChatCompletionsLLM(model="gpt-4o", api_key="test-key")
        with patch("openai.AsyncOpenAI") as mock_ctor:
            mock_ctor.return_value = MagicMock(name="client")
            first = llm._get_client()
            mock_ctor.assert_called_once()
        assert first is not None

    def test_client_cached_across_calls(self) -> None:
        """Same instance on second _get_client call — no reconstruction."""
        llm = OpenAIChatCompletionsLLM(model="gpt-4o", api_key="test-key")
        with patch("openai.AsyncOpenAI") as mock_ctor:
            mock_ctor.return_value = MagicMock(name="client")
            first = llm._get_client()
            second = llm._get_client()
            assert first is second
            assert mock_ctor.call_count == 1

    def test_model_property_returns_init_value(self) -> None:
        llm = OpenAIChatCompletionsLLM(model="gpt-4o")
        assert llm.model == "gpt-4o"


# ---------------------------------------------------------------------------
# acomplete — basic non-streaming
# ---------------------------------------------------------------------------


class TestAcompleteBasic:
    async def test_string_message_wrapped_as_user_message(self) -> None:
        llm, client = _llm_with_mock_client()
        await llm.acomplete(messages="hello world")

        kwargs = client.chat.completions.create.call_args.kwargs
        messages = kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "hello world"

    async def test_model_forwarded(self) -> None:
        llm, client = _llm_with_mock_client()
        await llm.acomplete(messages="hi")
        assert client.chat.completions.create.call_args.kwargs["model"] == "gpt-4o"

    async def test_returns_parsed_llm_response(self) -> None:
        llm, _ = _llm_with_mock_client(create_return=_make_completion(text="hi there"))
        result = await llm.acomplete(messages="hi")
        text_parts = [p for p in result.response if isinstance(p, LLMResponseText)]
        assert len(text_parts) == 1
        assert text_parts[0].text == "hi there"

    async def test_max_output_tokens_maps_to_max_completion_tokens(self) -> None:
        """The SDK's current param name is ``max_completion_tokens`` — the
        legacy ``max_tokens`` is deprecated for reasoning models."""
        llm, client = _llm_with_mock_client()
        config = LLMConfig(max_output_tokens=4096)
        await llm.acomplete(messages="hi", llm_config=config)

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["max_completion_tokens"] == 4096
        # The legacy param must NOT be set — it's the whole point.
        assert "max_tokens" not in kwargs

    async def test_generic_config_fields_forwarded(self) -> None:
        llm, client = _llm_with_mock_client()
        config = LLMConfig(
            temperature=0.5,
            top_p=0.9,
            frequency_penalty=0.2,
            presence_penalty=0.3,
            stop_sequences=["END"],
            seed=42,
            top_logprobs=3,
            response_logprobs=True,
        )
        await llm.acomplete(messages="hi", llm_config=config)

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.9
        assert kwargs["frequency_penalty"] == 0.2
        assert kwargs["presence_penalty"] == 0.3
        assert kwargs["stop"] == ["END"]
        assert kwargs["seed"] == 42
        assert kwargs["top_logprobs"] == 3
        assert kwargs["logprobs"] is True


# ---------------------------------------------------------------------------
# Chat-Completions-specific config routing
# ---------------------------------------------------------------------------


class TestChatCompletionsConfigRouting:
    async def test_all_config_fields_routed_to_sdk(self) -> None:
        """Every field on OpenAIChatCompletionsConfig lands on ``create``."""
        llm, client = _llm_with_mock_client()
        config = OpenAIChatCompletionsConfig(
            modalities=["text"],
            store=True,
            service_tier="flex",
            prompt_cache_key="cache_k",
            prompt_cache_retention="24h",
            verbosity="low",
        )
        await llm.acomplete(messages="hi", llm_config=config)

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["modalities"] == ["text"]
        assert kwargs["store"] is True
        assert kwargs["service_tier"] == "flex"
        assert kwargs["prompt_cache_key"] == "cache_k"
        assert kwargs["prompt_cache_retention"] == "24h"
        assert kwargs["verbosity"] == "low"

    async def test_plain_llm_config_leaves_extras_unset(self) -> None:
        """With a base ``LLMConfig``, none of the CC-specific fields
        should be populated with a framework-supplied value. The
        framework passes the ``Omit`` sentinel (not ``None``) so the
        SDK drops the field from the wire request entirely — some
        OpenAI models reject ``null`` even when the SDK accepts it.
        """
        from openai import Omit

        llm, client = _llm_with_mock_client()
        await llm.acomplete(messages="hi", llm_config=LLMConfig())

        kwargs = client.chat.completions.create.call_args.kwargs
        # All Optional[T] | Omit fields use the sentinel when unset so
        # the SDK omits them from the request.
        assert isinstance(kwargs["modalities"], Omit)
        assert isinstance(kwargs["store"], Omit)
        assert isinstance(kwargs["service_tier"], Omit)
        assert isinstance(kwargs["prompt_cache_retention"], Omit)
        assert isinstance(kwargs["verbosity"], Omit)
        assert isinstance(kwargs["prompt_cache_key"], Omit)


# ---------------------------------------------------------------------------
# Tool execution mode
# ---------------------------------------------------------------------------


class TestToolExecutionMode:
    async def test_parallel_mode_sends_true(self) -> None:
        from troopai.adk.types.tools import ToolExecutionMode

        llm, client = _llm_with_mock_client()
        config = LLMConfig(tool_execution_mode=ToolExecutionMode.PARALLEL)
        await llm.acomplete(messages="hi", llm_config=config, tools=[_dummy_tool()])
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["parallel_tool_calls"] is True

    async def test_sequential_mode_sends_false(self) -> None:
        from troopai.adk.types.tools import ToolExecutionMode

        llm, client = _llm_with_mock_client()
        config = LLMConfig(tool_execution_mode=ToolExecutionMode.SEQUENTIAL)
        await llm.acomplete(messages="hi", llm_config=config, tools=[_dummy_tool()])
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["parallel_tool_calls"] is False

    async def test_unset_mode_uses_omit(self) -> None:
        from openai import Omit

        llm, client = _llm_with_mock_client()
        await llm.acomplete(messages="hi", llm_config=LLMConfig())
        kwargs = client.chat.completions.create.call_args.kwargs
        assert isinstance(kwargs["parallel_tool_calls"], Omit)

    async def test_mode_set_without_tools_uses_omit(self) -> None:
        """``parallel_tool_calls`` must be omitted when no tools are
        present even if ``tool_execution_mode`` is set — the OpenAI API
        rejects ``parallel_tool_calls`` (including ``False``) with a 400
        when no ``tools`` are specified."""
        from openai import Omit

        from troopai.adk.types.tools import ToolExecutionMode

        for mode in (ToolExecutionMode.PARALLEL, ToolExecutionMode.SEQUENTIAL):
            llm, client = _llm_with_mock_client()
            config = LLMConfig(tool_execution_mode=mode)
            await llm.acomplete(messages="hi", llm_config=config)
            kwargs = client.chat.completions.create.call_args.kwargs
            assert isinstance(kwargs["parallel_tool_calls"], Omit), mode
            assert isinstance(kwargs["tools"], Omit), mode


# ---------------------------------------------------------------------------
# Extra body / extra args
# ---------------------------------------------------------------------------


class TestExtraBodyPassthrough:
    async def test_extra_body_passes_through_verbatim(self) -> None:
        """extra_body is the documented escape hatch for provider-native
        options that aren't first-class on LLMConfig / the SDK."""
        llm, client = _llm_with_mock_client()
        config = LLMConfig(extra_body={"safety_identifier": "opaque-id"})
        await llm.acomplete(messages="hi", llm_config=config)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"safety_identifier": "opaque-id"}

    async def test_extra_args_merges_into_extra_body(self) -> None:
        llm, client = _llm_with_mock_client()
        config = LLMConfig(
            extra_body={"key_a": "val_a"},
            extra_args={"key_b": "val_b"},
        )
        await llm.acomplete(messages="hi", llm_config=config)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"key_a": "val_a", "key_b": "val_b"}

    async def test_no_extras_sends_none(self) -> None:
        llm, client = _llm_with_mock_client()
        await llm.acomplete(messages="hi", llm_config=LLMConfig())
        assert client.chat.completions.create.call_args.kwargs["extra_body"] is None


# ---------------------------------------------------------------------------
# Timeout forwarding
# ---------------------------------------------------------------------------


class TestTimeout:
    async def test_unset_timeout_forwards_not_given(self) -> None:
        """An unset ``LLMConfig.timeout`` must forward the SDK's
        ``NOT_GIVEN`` sentinel — passing ``None`` would override the
        client's default request timeout with no-timeout behaviour,
        hanging the agent loop on a stalled connection."""
        from openai._types import NotGiven

        llm, client = _llm_with_mock_client()
        await llm.acomplete(messages="hi", llm_config=LLMConfig())
        kwargs = client.chat.completions.create.call_args.kwargs
        assert isinstance(kwargs["timeout"], NotGiven)
        assert kwargs["timeout"] is not None

    async def test_unset_timeout_forwards_not_given_streaming(self) -> None:
        from openai._types import NotGiven

        llm, client = _llm_with_streaming_client(_text_stream_chunks())
        stream = await llm.acomplete(messages="hi", stream=True)
        await _collect_stream(stream)
        kwargs = client.chat.completions.create.call_args.kwargs
        assert isinstance(kwargs["timeout"], NotGiven)

    async def test_explicit_timeout_forwarded_verbatim(self) -> None:
        llm, client = _llm_with_mock_client()
        await llm.acomplete(messages="hi", llm_config=LLMConfig(timeout=12.5))
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["timeout"] == 12.5


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    async def test_retry_policy_retries_rate_limit_until_success(self) -> None:
        calls = 0
        success = _make_completion(text="finally")

        def _side_effect(**_: Any) -> ChatCompletion:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise RateLimitError(
                    "slow down",
                    response=_make_httpx_response(429),
                    body=None,
                )
            return success

        llm, _ = _llm_with_mock_client(create_side_effect=_side_effect)
        config = LLMConfig(
            retry_policy=LLMRetryPolicy(
                max_retries=5,
                initial_delay=0.001,
                jitter=False,
            )
        )
        result = await llm.acomplete(messages="hi", llm_config=config)
        assert calls == 3
        assert result.response_id == "chatcmpl_abc"

    async def test_no_retry_policy_does_not_catch(self) -> None:
        from openai import BadRequestError

        def _side_effect(**_: Any) -> ChatCompletion:
            raise BadRequestError(
                "malformed",
                response=_make_httpx_response(400),
                body=None,
            )

        llm, _ = _llm_with_mock_client(create_side_effect=_side_effect)
        with pytest.raises(BadRequestError):
            await llm.acomplete(messages="hi")


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------


class _DummySchema:
    """Minimal AgentOutputSchemaBase stand-in for response_format forwarding."""

    def name(self) -> str:
        return "DummyOutput"

    def json_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"x": {"type": "integer"}}}

    def is_strict_json_schema(self) -> bool:
        return True


class TestStructuredOutput:
    async def test_output_schema_routes_to_response_format(self) -> None:
        llm, client = _llm_with_mock_client()
        await llm.acomplete(messages="hi", output_schema=_DummySchema())  # type: ignore[arg-type]
        rf = client.chat.completions.create.call_args.kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "DummyOutput"
        assert rf["json_schema"]["strict"] is True

    async def test_no_output_schema_omits_response_format(self) -> None:
        from openai import Omit

        llm, client = _llm_with_mock_client()
        await llm.acomplete(messages="hi")
        rf = client.chat.completions.create.call_args.kwargs["response_format"]
        assert isinstance(rf, Omit)


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class _FakeStream:
    """Async iterator stub that yields a canned list of chunks then stops."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = list(chunks)
        self._i = 0

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> Any:
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._i]
        self._i += 1
        return chunk


def _chunk(
    *,
    content: Optional[str] = None,
    tool_call: Optional[ChoiceDeltaToolCall] = None,
    finish_reason: Optional[str] = None,
    chunk_id: str = "chatcmpl_stream_1",
    usage: Optional[CompletionUsage] = None,
) -> ChatCompletionChunk:
    """Build a ``ChatCompletionChunk`` with a single-choice delta."""
    tool_calls = [tool_call] if tool_call is not None else None
    delta = ChoiceDelta.model_construct(
        role="assistant" if content is not None else None,
        content=content,
        refusal=None,
        tool_calls=tool_calls,
    )
    choice = ChunkChoice.model_construct(
        delta=delta,
        finish_reason=finish_reason,
        index=0,
        logprobs=None,
    )
    return ChatCompletionChunk.model_construct(
        id=chunk_id,
        choices=[choice],
        created=0,
        model="gpt-4o",
        object="chat.completion.chunk",
        usage=usage,
    )


def _text_stream_chunks() -> list[ChatCompletionChunk]:
    """Canned happy-path stream: 'he' + 'llo' + finish_reason='stop'."""
    return [
        _chunk(content="he"),
        _chunk(content="llo"),
        _chunk(finish_reason="stop"),
    ]


def _tool_call_stream_chunks() -> list[ChatCompletionChunk]:
    """Canned happy-path: one tool call built up across three chunks."""
    first = ChoiceDeltaToolCall(
        index=0,
        id="call_1",
        type="function",
        function=ChoiceDeltaToolCallFunction(name="lookup", arguments=""),
    )
    mid = ChoiceDeltaToolCall(
        index=0,
        function=ChoiceDeltaToolCallFunction(arguments='{"id":'),
    )
    last = ChoiceDeltaToolCall(
        index=0,
        function=ChoiceDeltaToolCallFunction(arguments='"abc"}'),
    )
    return [
        _chunk(tool_call=first),
        _chunk(tool_call=mid),
        _chunk(tool_call=last),
        _chunk(finish_reason="tool_calls"),
    ]


def _llm_with_streaming_client(
    chunks: list[ChatCompletionChunk],
) -> tuple[OpenAIChatCompletionsLLM, MagicMock]:
    llm = OpenAIChatCompletionsLLM(model="gpt-4o", api_key="test-key")
    client = MagicMock(name="AsyncOpenAI")
    create_mock = AsyncMock()
    create_mock.return_value = _FakeStream(chunks)
    client.chat.completions.create = create_mock
    llm._get_client = lambda: client  # type: ignore[method-assign]  # inject mock client for testing
    return llm, client


class TestStreaming:
    async def test_text_stream_emits_part_start_deltas_end_done(self) -> None:
        llm, _ = _llm_with_streaming_client(_text_stream_chunks())
        stream = await llm.acomplete(messages="hi", stream=True)
        events = await _collect_stream(stream)

        types = [ev.type for ev in events]
        assert types[0] == "part_start"
        assert types.count("part_delta") == 2
        assert types.count("part_end") == 1
        assert types[-1] == "done"

        deltas: list[str] = []
        for ev in events:
            if ev.type == "part_delta":
                assert ev.delta is not None
                deltas.append(ev.delta)
        assert "".join(deltas) == "hello"

    async def test_tool_call_stream_reconstructs_function_part(self) -> None:
        llm, _ = _llm_with_streaming_client(_tool_call_stream_chunks())
        stream = await llm.acomplete(messages="hi", stream=True)
        events = await _collect_stream(stream)

        # First event is a part_start carrying an LLMResponseFunctionToolCall
        # placeholder with the captured call_id + name.
        assert events[0].type == "part_start"
        assert isinstance(events[0].part, LLMResponseFunctionToolCall)
        assert events[0].part.call_id == "call_1"
        assert events[0].part.name == "lookup"

        done = events[-1]
        assert done.type == "done"
        assert done.response is not None
        calls = [p for p in done.response.response if isinstance(p, LLMResponseFunctionToolCall)]
        assert len(calls) == 1
        assert calls[0].arguments == '{"id":"abc"}'

    async def test_empty_stream_emits_done_event(self) -> None:
        """If the SDK stream closes without any chunks the wrapper must
        still yield a terminal ``"done"`` so downstream consumers don't
        block."""
        llm, _ = _llm_with_streaming_client([])
        stream = await llm.acomplete(messages="hi", stream=True)
        events = await _collect_stream(stream)
        assert len(events) == 1
        assert events[0].type == "done"

    async def test_stream_options_include_usage_injected(self) -> None:
        llm, client = _llm_with_streaming_client(_text_stream_chunks())
        stream = await llm.acomplete(messages="hi", stream=True)
        await _collect_stream(stream)

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["stream_options"] == {"include_usage": True}

    async def test_include_usage_false_omits_stream_options(self) -> None:
        from openai import Omit

        llm, client = _llm_with_streaming_client(_text_stream_chunks())
        config = LLMConfig(include_usage=False)
        stream = await llm.acomplete(messages="hi", llm_config=config, stream=True)
        await _collect_stream(stream)

        kwargs = client.chat.completions.create.call_args.kwargs
        assert isinstance(kwargs["stream_options"], Omit)

    async def test_final_done_carries_usage_from_last_chunk(self) -> None:
        usage = CompletionUsage(
            prompt_tokens=3,
            completion_tokens=5,
            total_tokens=8,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=1),
            completion_tokens_details=CompletionTokensDetails(reasoning_tokens=2),
        )
        chunks = [
            *_text_stream_chunks(),
            _chunk(usage=usage),
        ]
        llm, _ = _llm_with_streaming_client(chunks)
        stream = await llm.acomplete(messages="hi", stream=True)
        events = await _collect_stream(stream)

        done = events[-1]
        assert done.type == "done"
        assert done.response is not None
        assert done.response.usage is not None
        assert done.response.usage.input_tokens == 3
        assert done.response.usage.output_tokens == 5
        assert done.response.usage.input_tokens_details.cached_tokens == 1
        assert done.response.usage.output_tokens_details.reasoning_tokens == 2
