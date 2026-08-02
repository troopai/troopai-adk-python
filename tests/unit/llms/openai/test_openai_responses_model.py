"""Tests for OpenAIResponsesLLM.

Covers:
- Lazy client initialisation + idempotence on repeat ``_get_client``.
- ``acomplete`` forwards converted input / tools / tool_choice /
  structured-output text to ``client.responses.create``.
- Responses-specific config fields (reasoning / include / store /
  previous_response_id / truncation / service_tier /
  prompt_cache_key / prompt_cache_retention / max_tool_calls /
  background) are routed verbatim.
- ``extra_body`` and ``extra_args`` merge and pass through unchanged.
- ``tool_execution_mode`` maps to ``parallel_tool_calls`` correctly.
- ``retry_policy`` wraps non-streaming calls; streaming calls are NOT
  retried.
- Streaming dispatches through ``_stream`` into framework events.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai import RateLimitError
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseOutputItemAddedEvent,
    ResponseOutputItemDoneEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseRefusalDeltaEvent,
    ResponseTextDeltaEvent,
    ResponseUsage,
)
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.llms.openai.openai_responses_config import OpenAIResponsesConfig
from troopai.adk.llms.openai.openai_responses_model import OpenAIResponsesLLM
from troopai.adk.tools import Tool, function_tool
from troopai.adk.types.llms import LLMRetryPolicy
from troopai.adk.types.responses.llm_response import LLMResponseText, LLMStreamEvent


def _one_tool() -> list[Tool]:
    """A single trivial function tool so ``parallel_tool_calls`` is legal."""

    @function_tool
    def ping() -> str:
        """Ping."""
        return "pong"

    return [ping]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    text: str = "hello",
    response_id: str = "resp_abc",
    model: str = "gpt-5.1",
) -> Response:
    """Build a minimal ``Response`` carrying one assistant text message."""
    message = ResponseOutputMessage.model_construct(
        id="msg_1",
        type="message",
        role="assistant",
        status="completed",
        content=[
            ResponseOutputText.model_construct(
                type="output_text",
                text=text,
                annotations=[],
            )
        ],
    )
    usage = ResponseUsage(
        input_tokens=5,
        output_tokens=10,
        total_tokens=15,
        input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )
    return Response.model_construct(
        id=response_id,
        object="response",
        created_at=0,
        model=model,
        status="completed",
        output=[message],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=usage,
        error=None,
        incomplete_details=None,
        instructions=None,
        metadata=None,
        temperature=None,
        top_p=None,
        background=None,
    )


def _make_status_response(
    status: Literal["completed", "failed", "in_progress", "cancelled", "queued", "incomplete"],
    *,
    text: str = "partial",
    incomplete_reason: str | None = None,
) -> Response:
    """Build a ``Response`` with a non-completed status carrying usage + output.

    ``ResponseIncompleteEvent`` / ``ResponseFailedEvent`` still carry the full
    ``Response`` (partial output + token usage); the streaming path must
    convert it rather than emit an empty done that discards the tokens.
    """
    message = ResponseOutputMessage.model_construct(
        id="msg_1",
        type="message",
        role="assistant",
        status="incomplete",
        content=[ResponseOutputText.model_construct(type="output_text", text=text, annotations=[])],
    )
    usage = ResponseUsage(
        input_tokens=7,
        output_tokens=3,
        total_tokens=10,
        input_tokens_details=InputTokensDetails(cached_tokens=0, cache_write_tokens=0),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
    )
    # SimpleNamespace duck-types the ``incomplete_details.reason`` the converter
    # reads via getattr; typed Any so model_construct's stricter param accepts it.
    details: Any = SimpleNamespace(reason=incomplete_reason) if incomplete_reason is not None else None
    return Response.model_construct(
        id="resp_x",
        object="response",
        created_at=0,
        model="gpt-5.1",
        status=status,
        output=[message],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        usage=usage,
        error=None,
        incomplete_details=details,
        instructions=None,
        metadata=None,
        temperature=None,
        top_p=None,
        background=None,
    )


def _make_request() -> httpx.Request:
    return httpx.Request("POST", "https://api.openai.com/v1/responses")


def _make_httpx_response(status: int) -> httpx.Response:
    return httpx.Response(status_code=status, request=_make_request())


def _llm_with_mock_client(
    *,
    create_return: Any = None,
    create_side_effect: Any = None,
) -> tuple[OpenAIResponsesLLM, MagicMock]:
    """Build an ``OpenAIResponsesLLM`` with ``_get_client`` patched.

    Returns the LLM instance and the ``MagicMock`` standing in for
    ``AsyncOpenAI``. ``client.responses.create`` is an ``AsyncMock``.
    """
    llm = OpenAIResponsesLLM(model="gpt-5.1", api_key="test-key")
    client = MagicMock(name="AsyncOpenAI")
    create_mock = AsyncMock()
    if create_side_effect is not None:
        create_mock.side_effect = create_side_effect
    else:
        create_mock.return_value = create_return if create_return is not None else _make_response()
    client.responses.create = create_mock
    llm._get_client = lambda: client  # type: ignore[method-assign]  # inject mock client for testing
    return llm, client


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
        llm = OpenAIResponsesLLM(model="gpt-5.1", api_key="test-key")
        with patch("openai.AsyncOpenAI") as mock_ctor:
            mock_ctor.return_value = MagicMock(name="client")
            first = llm._get_client()
            mock_ctor.assert_called_once()
        assert first is not None

    def test_client_cached_across_calls(self) -> None:
        """Same instance on second _get_client call — no reconstruction."""
        llm = OpenAIResponsesLLM(model="gpt-5.1", api_key="test-key")
        with patch("openai.AsyncOpenAI") as mock_ctor:
            mock_ctor.return_value = MagicMock(name="client")
            first = llm._get_client()
            second = llm._get_client()
            assert first is second
            # Constructor called exactly once across both accesses.
            assert mock_ctor.call_count == 1

    def test_model_property_returns_init_value(self) -> None:
        llm = OpenAIResponsesLLM(model="gpt-5.1")
        assert llm.model == "gpt-5.1"


# ---------------------------------------------------------------------------
# acomplete — basic non-streaming
# ---------------------------------------------------------------------------


class TestAcompleteBasic:
    async def test_string_message_wrapped_as_user_input(self) -> None:
        llm, client = _llm_with_mock_client()
        await llm.acomplete(messages="hello world")

        create_kwargs = client.responses.create.call_args.kwargs
        input_items = create_kwargs["input"]
        assert len(input_items) == 1
        assert input_items[0]["role"] == "user"

    async def test_model_forwarded(self) -> None:
        llm, client = _llm_with_mock_client()
        await llm.acomplete(messages="hi")
        assert client.responses.create.call_args.kwargs["model"] == "gpt-5.1"

    async def test_returns_parsed_llm_response(self) -> None:
        llm, _ = _llm_with_mock_client(create_return=_make_response(text="hi there"))
        result = await llm.acomplete(messages="hi")
        text_parts = [p for p in result.response if isinstance(p, LLMResponseText)]
        assert len(text_parts) == 1
        assert text_parts[0].text == "hi there"

    async def test_generic_config_fields_forwarded(self) -> None:
        llm, client = _llm_with_mock_client()
        config = LLMConfig(
            temperature=0.5,
            top_p=0.9,
            max_output_tokens=1024,
            top_logprobs=3,
        )
        await llm.acomplete(messages="hi", llm_config=config)

        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.9
        assert kwargs["max_output_tokens"] == 1024
        assert kwargs["top_logprobs"] == 3


# ---------------------------------------------------------------------------
# Responses-specific config routing
# ---------------------------------------------------------------------------


class TestResponsesConfigRouting:
    async def test_all_responses_fields_routed_to_sdk(self) -> None:
        """Every field on OpenAIResponsesConfig lands on ``create``."""
        llm, client = _llm_with_mock_client()
        config = OpenAIResponsesConfig(
            reasoning={"effort": "medium"},
            include=["reasoning.encrypted_content"],
            store=True,
            previous_response_id="resp_prev_123",
            truncation="auto",
            service_tier="priority",
            prompt_cache_key="cache_key_1",
            prompt_cache_retention="24h",
            max_tool_calls=5,
            background=False,
        )
        await llm.acomplete(messages="hi", llm_config=config)

        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["reasoning"] == {"effort": "medium"}
        assert kwargs["include"] == ["reasoning.encrypted_content"]
        assert kwargs["store"] is True
        assert kwargs["previous_response_id"] == "resp_prev_123"
        assert kwargs["truncation"] == "auto"
        assert kwargs["service_tier"] == "priority"
        assert kwargs["prompt_cache_key"] == "cache_key_1"
        assert kwargs["prompt_cache_retention"] == "24h"
        assert kwargs["max_tool_calls"] == 5
        assert kwargs["background"] is False

    async def test_generic_config_leaves_responses_fields_omitted(self) -> None:
        """A plain LLMConfig sends the omit sentinel for prompt_cache_key."""
        llm, client = _llm_with_mock_client()
        await llm.acomplete(messages="hi", llm_config=LLMConfig())

        kwargs = client.responses.create.call_args.kwargs
        # Nullable fields pass None through.
        assert kwargs["reasoning"] is None
        assert kwargs["store"] is None
        assert kwargs["max_tool_calls"] is None
        # Non-nullable prompt_cache_key uses ``omit`` — not a str or None.
        assert kwargs["prompt_cache_key"] is not None
        assert not isinstance(kwargs["prompt_cache_key"], str)


# ---------------------------------------------------------------------------
# Tool execution mode → parallel_tool_calls
# ---------------------------------------------------------------------------


class TestToolExecutionMode:
    async def test_parallel_mode_sets_parallel_tool_calls_true(self) -> None:
        llm, client = _llm_with_mock_client()
        from troopai.adk.types.tools import ToolExecutionMode

        config = LLMConfig(tool_execution_mode=ToolExecutionMode.PARALLEL)
        await llm.acomplete(messages="hi", llm_config=config, tools=_one_tool())
        assert client.responses.create.call_args.kwargs["parallel_tool_calls"] is True

    async def test_sequential_mode_sets_parallel_tool_calls_false(self) -> None:
        llm, client = _llm_with_mock_client()
        from troopai.adk.types.tools import ToolExecutionMode

        config = LLMConfig(tool_execution_mode=ToolExecutionMode.SEQUENTIAL)
        await llm.acomplete(messages="hi", llm_config=config, tools=_one_tool())
        assert client.responses.create.call_args.kwargs["parallel_tool_calls"] is False

    async def test_unset_mode_leaves_parallel_tool_calls_none(self) -> None:
        llm, client = _llm_with_mock_client()
        await llm.acomplete(messages="hi", llm_config=LLMConfig(), tools=_one_tool())
        assert client.responses.create.call_args.kwargs["parallel_tool_calls"] is None

    async def test_execution_mode_without_tools_omits_parallel_tool_calls(self) -> None:
        # The Responses API 400s on parallel_tool_calls when no tools are
        # present; a bare execution-mode config must not leak the parameter.
        llm, client = _llm_with_mock_client()
        from troopai.adk.types.tools import ToolExecutionMode

        config = LLMConfig(tool_execution_mode=ToolExecutionMode.PARALLEL)
        await llm.acomplete(messages="hi", llm_config=config)
        assert client.responses.create.call_args.kwargs["parallel_tool_calls"] is None


# ---------------------------------------------------------------------------
# extra_body + extra_args merge
# ---------------------------------------------------------------------------


class TestExtraBodyPassthrough:
    async def test_extra_body_dict_passes_through(self) -> None:
        llm, client = _llm_with_mock_client()
        config = LLMConfig(extra_body={"web_search_options": {"search_context_size": "low"}})
        await llm.acomplete(messages="hi", llm_config=config)
        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["extra_body"] == {"web_search_options": {"search_context_size": "low"}}

    async def test_extra_args_merges_into_extra_body(self) -> None:
        llm, client = _llm_with_mock_client()
        config = LLMConfig(
            extra_body={"key_a": "val_a"},
            extra_args={"key_b": "val_b"},
        )
        await llm.acomplete(messages="hi", llm_config=config)
        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["extra_body"] == {"key_a": "val_a", "key_b": "val_b"}

    async def test_no_extras_sends_none(self) -> None:
        llm, client = _llm_with_mock_client()
        await llm.acomplete(messages="hi", llm_config=LLMConfig())
        assert client.responses.create.call_args.kwargs["extra_body"] is None

    async def test_non_mapping_extra_body_dropped_with_warning(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A non-Mapping ``extra_body`` cannot be merged into the body dict.

        Regression: the wrapper used to discard it silently. It must now
        send ``extra_body=None`` AND log a warning so the misconfiguration
        is visible rather than producing invisible data loss.
        """
        llm, client = _llm_with_mock_client()
        # ``Body`` is typed as ``object``; a bytes payload is a permitted
        # non-Mapping value that the merge step cannot handle.
        config = LLMConfig(extra_body=b"raw-non-dict-body")
        with caplog.at_level("WARNING", logger="troopai.adk.llms.openai.openai_responses_model"):
            await llm.acomplete(messages="hi", llm_config=config)

        assert client.responses.create.call_args.kwargs["extra_body"] is None
        warnings = [r for r in caplog.records if r.levelname == "WARNING"]
        assert any("extra_body" in r.getMessage() and "not a Mapping" in r.getMessage() for r in warnings)


# ---------------------------------------------------------------------------
# Retry policy
# ---------------------------------------------------------------------------


class TestRetryPolicy:
    async def test_retry_policy_retries_rate_limit_until_success(self) -> None:
        calls = 0
        success = _make_response(text="finally")

        def _side_effect(**_: Any) -> Response:
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
        assert result.response_id == "resp_abc"

    async def test_no_retry_policy_does_not_catch(self) -> None:
        """Without a policy, permanent errors must surface untouched."""
        from openai import BadRequestError

        def _side_effect(**_: Any) -> Response:
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
    """Minimal AgentOutputSchemaBase stand-in for text=… forwarding."""

    def name(self) -> str:
        return "DummyOutput"

    def json_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"x": {"type": "integer"}}}

    def is_strict_json_schema(self) -> bool:
        return True


class TestStructuredOutput:
    async def test_output_schema_routes_to_text_field(self) -> None:
        llm, client = _llm_with_mock_client()
        await llm.acomplete(messages="hi", output_schema=_DummySchema())  # type: ignore[arg-type]
        text_arg = client.responses.create.call_args.kwargs["text"]
        assert text_arg["format"]["type"] == "json_schema"
        assert text_arg["format"]["name"] == "DummyOutput"
        assert text_arg["format"]["strict"] is True

    async def test_no_output_schema_omits_text(self) -> None:
        llm, client = _llm_with_mock_client()
        await llm.acomplete(messages="hi")
        text_arg = client.responses.create.call_args.kwargs["text"]
        # text=omit is neither a dict nor None.
        assert not isinstance(text_arg, dict)
        assert text_arg is not None


# ---------------------------------------------------------------------------
# Streaming dispatch
# ---------------------------------------------------------------------------


class _FakeStream:
    """Async-iterable stand-in for the SDK's AsyncStream[ResponseStreamEvent]."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    def __aiter__(self) -> _FakeStream:
        return self

    async def __anext__(self) -> Any:
        if len(self._events) == 0:
            raise StopAsyncIteration
        return self._events.pop(0)


def _stream_events_happy_path() -> list[Any]:
    """Build a minimal text-only stream ending with ResponseCompletedEvent."""
    message_placeholder = ResponseOutputMessage.model_construct(
        id="msg_1",
        type="message",
        role="assistant",
        status="in_progress",
        content=[],
    )
    added = ResponseOutputItemAddedEvent.model_construct(
        type="response.output_item.added",
        output_index=0,
        sequence_number=1,
        item=message_placeholder,
    )
    delta_a = ResponseTextDeltaEvent.model_construct(
        type="response.output_text.delta",
        content_index=0,
        delta="hel",
        item_id="msg_1",
        logprobs=[],
        output_index=0,
        sequence_number=2,
    )
    delta_b = ResponseTextDeltaEvent.model_construct(
        type="response.output_text.delta",
        content_index=0,
        delta="lo",
        item_id="msg_1",
        logprobs=[],
        output_index=0,
        sequence_number=3,
    )
    done_item = ResponseOutputItemDoneEvent.model_construct(
        type="response.output_item.done",
        output_index=0,
        sequence_number=4,
        item=message_placeholder,
    )
    completed = ResponseCompletedEvent.model_construct(
        type="response.completed",
        sequence_number=5,
        response=_make_response(text="hello"),
    )
    return [added, delta_a, delta_b, done_item, completed]


def _llm_with_streaming_client(
    events: list[Any],
) -> tuple[OpenAIResponsesLLM, MagicMock]:
    """Build an ``OpenAIResponsesLLM`` whose SDK boundary is faked.

    The matching ``isinstance(stream_iter, AsyncStream)`` assertion in
    the model code is bypassed by the ``_patch_responses_async_stream``
    fixture — every test using this helper MUST also request that
    fixture (see :func:`_patch_responses_async_stream` below).
    """
    llm = OpenAIResponsesLLM(model="gpt-5.1", api_key="test-key")
    client = MagicMock(name="AsyncOpenAI")
    create_mock = AsyncMock(return_value=_FakeStream(events))
    client.responses.create = create_mock
    llm._get_client = lambda: client  # type: ignore[method-assign]  # inject mock client for testing
    return llm, client


@pytest.fixture
def _patch_responses_async_stream(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace ``openai_responses_model.AsyncStream`` with the test stand-in.

    Restores the original symbol after the test, preventing
    cross-test bleed. The fake type is the same ``_FakeStream``
    helper defined above — it is duck-typed for async iteration but
    does not inherit from the real ``openai.AsyncStream``.
    """
    from troopai.adk.llms.openai import openai_responses_model

    monkeypatch.setattr(openai_responses_model, "AsyncStream", _FakeStream)
    yield


@pytest.mark.usefixtures("_patch_responses_async_stream")
class TestStreaming:
    async def test_streaming_emits_part_start_delta_end_done(self) -> None:
        llm, _ = _llm_with_streaming_client(_stream_events_happy_path())
        stream = await llm.acomplete(messages="hi", stream=True)
        events = await _collect_stream(stream)

        types = [ev.type for ev in events]
        assert "part_start" in types
        assert types.count("part_delta") == 2
        assert types[-1] == "done"

        # Deltas concatenate to the original text.
        deltas: list[str] = []
        for ev in events:
            if ev.type == "part_delta":
                assert ev.delta is not None
                deltas.append(ev.delta)
        assert "".join(deltas) == "hello"

    async def test_done_event_carries_parsed_response(self) -> None:
        llm, _ = _llm_with_streaming_client(_stream_events_happy_path())
        stream = await llm.acomplete(messages="hi", stream=True)
        events = await _collect_stream(stream)

        done = events[-1]
        assert done.type == "done"
        assert done.response is not None
        text_parts = [p for p in done.response.response if isinstance(p, LLMResponseText)]
        assert len(text_parts) == 1
        assert text_parts[0].text == "hello"

    async def test_empty_stream_emits_done_event(self) -> None:
        """If the SDK stream closes without a ``response.completed`` event
        the wrapper must still yield a ``"done"`` so downstream consumers
        don't block waiting for a terminator."""
        llm, _ = _llm_with_streaming_client([])
        stream = await llm.acomplete(messages="hi", stream=True)
        events = await _collect_stream(stream)
        assert len(events) == 1
        assert events[0].type == "done"

    async def test_incomplete_event_surfaces_response_and_usage(self) -> None:
        """A ``response.incomplete`` terminal event carries partial output and
        usage — it must be converted, not discarded as an empty done."""
        incomplete = ResponseIncompleteEvent.model_construct(
            type="response.incomplete",
            sequence_number=1,
            response=_make_status_response("incomplete", text="cut off", incomplete_reason="max_output_tokens"),
        )
        llm, _ = _llm_with_streaming_client([incomplete])
        stream = await llm.acomplete(messages="hi", stream=True)
        events = await _collect_stream(stream)

        assert events[-1].type == "done"
        response = events[-1].response
        assert response is not None
        assert response.finish_reason == "length"
        assert response.usage is not None
        assert response.usage.output_tokens == 3
        text_parts = [p for p in response.response if isinstance(p, LLMResponseText)]
        assert [p.text for p in text_parts] == ["cut off"]

    async def test_failed_event_surfaces_response_with_error_reason(self) -> None:
        """A ``response.failed`` terminal event carries usage and must be
        surfaced with ``finish_reason="error"`` rather than an empty done."""
        failed = ResponseFailedEvent.model_construct(
            type="response.failed",
            sequence_number=1,
            response=_make_status_response("failed", text=""),
        )
        llm, _ = _llm_with_streaming_client([failed])
        stream = await llm.acomplete(messages="hi", stream=True)
        events = await _collect_stream(stream)

        assert events[-1].type == "done"
        response = events[-1].response
        assert response is not None
        assert response.finish_reason == "error"
        assert response.usage is not None
        assert response.usage.input_tokens == 7

    async def test_stream_true_forwarded_to_create(self) -> None:
        llm, client = _llm_with_streaming_client(_stream_events_happy_path())
        stream = await llm.acomplete(messages="hi", stream=True)
        # Drain so the mock sees the full call.
        await _collect_stream(stream)

        kwargs = client.responses.create.call_args.kwargs
        assert kwargs["stream"] is True

    async def test_reasoning_summary_deltas_stream_incrementally(self) -> None:
        """o-series reasoning summaries stream via
        ``response.reasoning_summary_text.delta``. Regression: these were
        dropped, so a reasoning ``part_start`` was followed by zero deltas."""
        reasoning_placeholder = ResponseReasoningItem.model_construct(
            id="rs_1",
            type="reasoning",
            summary=[],
            content=None,
            encrypted_content=None,
            status="in_progress",
        )
        added = ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added",
            output_index=0,
            sequence_number=1,
            item=reasoning_placeholder,
        )
        delta_a = ResponseReasoningSummaryTextDeltaEvent.model_construct(
            type="response.reasoning_summary_text.delta",
            delta="step ",
            item_id="rs_1",
            output_index=0,
            summary_index=0,
            sequence_number=2,
        )
        delta_b = ResponseReasoningSummaryTextDeltaEvent.model_construct(
            type="response.reasoning_summary_text.delta",
            delta="one",
            item_id="rs_1",
            output_index=0,
            summary_index=0,
            sequence_number=3,
        )
        completed = ResponseCompletedEvent.model_construct(
            type="response.completed",
            sequence_number=4,
            response=_make_response(text="done"),
        )
        llm, _ = _llm_with_streaming_client([added, delta_a, delta_b, completed])
        stream = await llm.acomplete(messages="hi", stream=True)
        events = await _collect_stream(stream)

        deltas = [ev.delta for ev in events if ev.type == "part_delta"]
        assert deltas == ["step ", "one"]

    async def test_refusal_deltas_stream_incrementally(self) -> None:
        """Refusal text streams via ``response.refusal.delta``. Regression:
        these were dropped, breaking incremental refusal streaming."""
        message_placeholder = ResponseOutputMessage.model_construct(
            id="msg_1",
            type="message",
            role="assistant",
            status="in_progress",
            content=[],
        )
        added = ResponseOutputItemAddedEvent.model_construct(
            type="response.output_item.added",
            output_index=0,
            sequence_number=1,
            item=message_placeholder,
        )
        delta_a = ResponseRefusalDeltaEvent.model_construct(
            type="response.refusal.delta",
            content_index=0,
            delta="I cannot ",
            item_id="msg_1",
            output_index=0,
            sequence_number=2,
        )
        delta_b = ResponseRefusalDeltaEvent.model_construct(
            type="response.refusal.delta",
            content_index=0,
            delta="help with that.",
            item_id="msg_1",
            output_index=0,
            sequence_number=3,
        )
        completed = ResponseCompletedEvent.model_construct(
            type="response.completed",
            sequence_number=4,
            response=_make_response(text="done"),
        )
        llm, _ = _llm_with_streaming_client([added, delta_a, delta_b, completed])
        stream = await llm.acomplete(messages="hi", stream=True)
        events = await _collect_stream(stream)

        deltas = [ev.delta for ev in events if ev.type == "part_delta"]
        assert deltas == ["I cannot ", "help with that."]
