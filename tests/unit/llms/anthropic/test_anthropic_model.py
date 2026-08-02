"""Tests for ``AnthropicLLM`` end-to-end paths.

Mocks ``anthropic.AsyncAnthropic`` via ``AsyncMock``. Covers:

- Non-streaming text response
- Tool-use response
- Structured output via the synthetic-tool path
- ``output_schema`` + ``stream=True`` raises NotImplementedError
- ``LLMConfig.retry_policy`` retries on rate-limit, then succeeds
- ``AnthropicConfig.thinking`` flows through to ``client.messages.create``
- ``AnthropicConfig.auto_cache_control`` injects markers
- ``max_output_tokens`` resolution edge cases (explicit 0, thinking floor)
- Streaming usage refresh from ``message_delta`` cache/input counters
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from anthropic import RateLimitError
from anthropic.types import (
    Message,
    MessageDeltaUsage,
    RawContentBlockDeltaEvent,
    RawContentBlockStartEvent,
    RawContentBlockStopEvent,
    RawMessageDeltaEvent,
    RawMessageStartEvent,
    StopReason,
    TextBlock,
    TextDelta,
    ToolUseBlock,
    Usage,
)
from anthropic.types.raw_message_delta_event import Delta as RawMessageDeltaEventDelta
from pydantic import BaseModel

from troopai.adk.llms.anthropic import AnthropicConfig, AnthropicLLM
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.schemas import AgentOutputSchema
from troopai.adk.types.llms import LLMRetryPolicy


def _make_message(
    content: list[Any],
    *,
    stop_reason: StopReason = "end_turn",
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> Message:
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-sonnet-4",
        content=content,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def _install_mock_client(llm: AnthropicLLM, response: Message | list[Any]) -> AsyncMock:
    """Wire a mocked ``AsyncAnthropic`` client into the LLM.

    Returns the ``messages.create`` AsyncMock so callers can assert
    against the call args.
    """
    create_mock = AsyncMock()
    if isinstance(response, list):
        create_mock.side_effect = response
    else:
        create_mock.return_value = response
    fake_client = MagicMock()
    fake_client.messages.create = create_mock
    llm._client = fake_client  # bypass lazy init for tests
    return create_mock


class TestNonStreamingTextResponse:
    async def test_returns_llm_response(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        msg = _make_message([TextBlock(type="text", text="Hi!", citations=None)])
        _install_mock_client(llm, msg)

        response = await llm.acomplete(messages="hello")

        assert response.content == "Hi!"
        assert response.finish_reason == "end_turn"
        assert response.usage is not None
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5

    async def test_passes_temperature_and_max_tokens(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        msg = _make_message([TextBlock(type="text", text="ok", citations=None)])
        create_mock = _install_mock_client(llm, msg)

        await llm.acomplete(
            messages="test",
            llm_config=LLMConfig(temperature=0.7, max_output_tokens=4096),
        )

        assert create_mock.await_args is not None
        kwargs = create_mock.await_args.kwargs
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 4096


class TestToolUseResponse:
    async def test_returns_tool_call_part(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        msg = _make_message(
            [ToolUseBlock(type="tool_use", id="t_1", name="lookup", input={"id": 7})],
            stop_reason="tool_use",
        )
        _install_mock_client(llm, msg)

        response = await llm.acomplete(messages="find 7")

        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "lookup"
        assert response.finish_reason == "tool_use"


class TestStructuredOutputSyntheticTool:
    async def test_validates_synthetic_tool_input(self) -> None:
        class Answer(BaseModel):
            value: str

        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        msg = _make_message(
            [
                ToolUseBlock(
                    type="tool_use",
                    id="t_1",
                    name="structured_output",
                    input={"value": "ok"},
                )
            ],
            stop_reason="tool_use",
        )
        create_mock = _install_mock_client(llm, msg)

        # Should NOT raise — schema validates against tool input.
        response = await llm.acomplete(
            messages="answer",
            output_schema=AgentOutputSchema(Answer),
        )

        # Verify the synthetic tool was registered + tool_choice forced.
        assert create_mock.await_args is not None
        kwargs = create_mock.await_args.kwargs
        assert kwargs["tools"] != []
        assert kwargs["tools"][0]["name"] == "structured_output"
        assert kwargs["tool_choice"] == {"type": "tool", "name": "structured_output"}
        # The synthetic tool's tool_use block is replaced with an
        # LLMResponseText carrying the JSON; the tool_use stop_reason
        # is normalised to "end_turn" so the Runner doesn't try to
        # dispatch the synthetic tool.
        assert response.finish_reason == "end_turn"
        assert response.content == '{"value": "ok"}'
        assert len(response.tool_calls) == 0

    async def test_streaming_with_output_schema_raises(self) -> None:
        class Answer(BaseModel):
            value: str

        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        with pytest.raises(NotImplementedError, match="stream=True"):
            await llm.acomplete(
                messages="answer",
                output_schema=AgentOutputSchema(Answer),
                stream=True,
            )

    async def test_plain_text_schema_skips_synthetic_tool(self) -> None:
        # AgentOutputSchema(str) reports is_plain_text() — should NOT
        # trigger the synthetic-tool path even when output_schema is set.
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        msg = _make_message([TextBlock(type="text", text="freeform", citations=None)])
        create_mock = _install_mock_client(llm, msg)

        await llm.acomplete(
            messages="say hi",
            output_schema=AgentOutputSchema(str),
        )

        assert create_mock.await_args is not None
        kwargs = create_mock.await_args.kwargs
        # No synthetic tool should be registered.
        assert kwargs["tools"] == kwargs.get("tools") and (
            "tools" not in kwargs or not isinstance(kwargs["tools"], list) or len(kwargs["tools"]) == 0
        )


class TestRetryPolicy:
    async def test_retries_on_rate_limit_then_succeeds(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        rate_limit = RateLimitError(
            message="429",
            response=httpx.Response(429, request=request),
            body=None,
        )
        success = _make_message([TextBlock(type="text", text="ok", citations=None)])
        create_mock = _install_mock_client(llm, [rate_limit, success])

        # Use a tight policy — initial_delay tiny, jitter off — so the
        # test runs near-instantly.
        config = LLMConfig(
            retry_policy=LLMRetryPolicy(
                max_retries=2,
                initial_delay=0.0,
                jitter=False,
            )
        )

        response = await llm.acomplete(messages="hi", llm_config=config)

        assert response.content == "ok"
        assert create_mock.await_count == 2

    async def test_raises_when_budget_exhausted(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        rate_limit = RateLimitError(
            message="429",
            response=httpx.Response(429, request=request),
            body=None,
        )
        _install_mock_client(llm, [rate_limit, rate_limit, rate_limit])
        config = LLMConfig(
            retry_policy=LLMRetryPolicy(
                max_retries=1,
                initial_delay=0.0,
                jitter=False,
            )
        )
        with pytest.raises(RateLimitError):
            await llm.acomplete(messages="hi", llm_config=config)


class TestConfigNumRetries:
    """``LLMConfig.num_retries`` maps to the SDK client's ``max_retries``."""

    async def test_num_retries_applied_via_with_options(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        msg = _make_message([TextBlock(type="text", text="ok", citations=None)])
        create_mock = AsyncMock(return_value=msg)
        # ``with_options(max_retries=...)`` returns a derived client that
        # shares the create mock, mirroring the anthropic SDK contract.
        derived = MagicMock()
        derived.messages.create = create_mock
        fake_client = MagicMock()
        fake_client.with_options.return_value = derived
        llm._client = fake_client

        await llm.acomplete(messages="hi", llm_config=LLMConfig(num_retries=3))

        fake_client.with_options.assert_called_once_with(max_retries=3)
        assert create_mock.await_count == 1

    async def test_num_retries_unset_does_not_override_client(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        msg = _make_message([TextBlock(type="text", text="ok", citations=None)])
        create_mock = AsyncMock(return_value=msg)
        fake_client = MagicMock()
        fake_client.messages.create = create_mock
        llm._client = fake_client

        await llm.acomplete(messages="hi", llm_config=LLMConfig())

        # ``num_retries`` defaults to None → the developer never opts into
        # retries → the client is used as constructed (cost-conservative).
        fake_client.with_options.assert_not_called()
        assert create_mock.await_count == 1


class TestAnthropicConfigPropagation:
    async def test_thinking_field_passed_to_sdk(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        msg = _make_message([TextBlock(type="text", text="ok", citations=None)])
        create_mock = _install_mock_client(llm, msg)
        cfg = AnthropicConfig(
            thinking={"type": "enabled", "budget_tokens": 2048},
        )

        await llm.acomplete(messages="reason", llm_config=cfg)

        assert create_mock.await_args is not None
        kwargs = create_mock.await_args.kwargs
        assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}

    async def test_thinking_raises_floor_for_max_tokens(self) -> None:
        # If budget_tokens >= max_output_tokens, max_tokens must rise.
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        msg = _make_message([TextBlock(type="text", text="ok", citations=None)])
        create_mock = _install_mock_client(llm, msg)
        cfg = AnthropicConfig(
            thinking={"type": "enabled", "budget_tokens": 10000},
            max_output_tokens=4096,  # smaller than budget
        )

        await llm.acomplete(messages="reason", llm_config=cfg)

        assert create_mock.await_args is not None
        kwargs = create_mock.await_args.kwargs
        assert kwargs["max_tokens"] > 10000

    async def test_service_tier_field_passed_to_sdk(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        msg = _make_message([TextBlock(type="text", text="ok", citations=None)])
        create_mock = _install_mock_client(llm, msg)
        cfg = AnthropicConfig(service_tier="standard_only")

        await llm.acomplete(messages="hi", llm_config=cfg)

        assert create_mock.await_args is not None
        kwargs = create_mock.await_args.kwargs
        assert kwargs["service_tier"] == "standard_only"

    async def test_auto_cache_control_marks_system_block(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        msg = _make_message([TextBlock(type="text", text="ok", citations=None)])
        create_mock = _install_mock_client(llm, msg)
        cfg = AnthropicConfig(auto_cache_control=True)

        from troopai.adk.types.input import LLMInputEasyMessage

        sys_msg: LLMInputEasyMessage = {"role": "system", "content": "Be very helpful."}
        user_msg: LLMInputEasyMessage = {"role": "user", "content": "hi"}
        await llm.acomplete(messages=[sys_msg, user_msg], llm_config=cfg)

        assert create_mock.await_args is not None
        kwargs = create_mock.await_args.kwargs
        # System should be promoted to a list with the cache marker.
        assert isinstance(kwargs["system"], list)
        assert kwargs["system"][0]["cache_control"]["type"] == "ephemeral"


class TestEffortParameter:
    async def test_effort_forwarded_as_output_config(self) -> None:
        llm = AnthropicLLM(model="claude-opus-4-8", api_key="test")
        msg = _make_message([TextBlock(type="text", text="ok", citations=None)])
        create_mock = _install_mock_client(llm, msg)

        await llm.acomplete(messages="hello", llm_config=AnthropicConfig(effort="xhigh"))

        kwargs = create_mock.call_args.kwargs
        assert kwargs["output_config"] == {"effort": "xhigh"}

    async def test_effort_omitted_by_default(self) -> None:
        from anthropic import Omit

        llm = AnthropicLLM(model="claude-opus-4-8", api_key="test")
        msg = _make_message([TextBlock(type="text", text="ok", citations=None)])
        create_mock = _install_mock_client(llm, msg)

        await llm.acomplete(messages="hello", llm_config=AnthropicConfig())

        kwargs = create_mock.call_args.kwargs
        assert isinstance(kwargs["output_config"], Omit), "unset effort must keep output_config unset"


class TestMidSystemMessages:
    _ITEMS: list[Any] = [
        {"type": "message", "role": "user", "content": "hi"},
        {"type": "message", "role": "system", "content": "Terse mode."},
    ]

    async def test_opt_in_sends_beta_header_and_in_place_message(self) -> None:
        llm = AnthropicLLM(model="claude-opus-4-8", api_key="test")
        msg = _make_message([TextBlock(type="text", text="ok", citations=None)])
        create_mock = _install_mock_client(llm, msg)

        await llm.acomplete(messages=self._ITEMS, llm_config=AnthropicConfig(mid_system_messages=True))

        kwargs = create_mock.call_args.kwargs
        headers = kwargs["extra_headers"]
        assert headers is not None
        assert "mid-conversation-system-2026-04-07" in str(headers.get("anthropic-beta"))
        roles = [m["role"] for m in kwargs["messages"]]
        assert "system" in roles

    async def test_default_hoists_and_sends_no_beta_header(self) -> None:
        llm = AnthropicLLM(model="claude-opus-4-8", api_key="test")
        msg = _make_message([TextBlock(type="text", text="ok", citations=None)])
        create_mock = _install_mock_client(llm, msg)

        await llm.acomplete(messages=self._ITEMS, llm_config=AnthropicConfig())

        kwargs = create_mock.call_args.kwargs
        roles = [m["role"] for m in kwargs["messages"]]
        assert "system" not in roles
        assert kwargs["system"] is not None
        headers = kwargs["extra_headers"]
        beta_value = headers.get("anthropic-beta") if isinstance(headers, dict) else None
        assert beta_value is None or "mid-conversation-system" not in beta_value

    async def test_beta_header_merges_with_existing(self) -> None:
        llm = AnthropicLLM(model="claude-opus-4-8", api_key="test")
        msg = _make_message([TextBlock(type="text", text="ok", citations=None)])
        create_mock = _install_mock_client(llm, msg)

        config = AnthropicConfig(
            mid_system_messages=True,
            extra_headers={"anthropic-beta": "some-other-beta"},
        )
        await llm.acomplete(messages=self._ITEMS, llm_config=config)

        beta = create_mock.call_args.kwargs["extra_headers"]["anthropic-beta"]
        assert "some-other-beta" in beta
        assert "mid-conversation-system-2026-04-07" in beta


class TestMaxTokensResolution:
    """``max_output_tokens`` mapping must honour the developer's cost intent.

    Two cost-conservative invariants:

    1. Only ``None`` means "unset"; an explicit ``0`` is the developer's
       value and must not be coerced to the 8192 default.
    2. When extended thinking forces ``max_tokens > budget_tokens``, the
       raise is by the smallest margin (budget + 1), not a full default's
       worth of answer headroom the developer never opted into.
    """

    async def test_explicit_zero_is_not_coerced_to_default(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        msg = _make_message([TextBlock(type="text", text="ok", citations=None)])
        create_mock = _install_mock_client(llm, msg)

        await llm.acomplete(
            messages="hi",
            llm_config=LLMConfig(max_output_tokens=0),
        )

        assert create_mock.await_args is not None
        # Before the fix, ``0 or _DEFAULT_MAX_TOKENS`` silently became
        # 8192; the developer's explicit value must be preserved.
        assert create_mock.await_args.kwargs["max_tokens"] == 0

    async def test_thinking_floor_raises_by_smallest_margin(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        msg = _make_message([TextBlock(type="text", text="ok", citations=None)])
        create_mock = _install_mock_client(llm, msg)
        cfg = AnthropicConfig(
            thinking={"type": "enabled", "budget_tokens": 5000},
            max_output_tokens=2000,  # smaller than budget
        )

        await llm.acomplete(messages="reason", llm_config=cfg)

        assert create_mock.await_args is not None
        # Anthropic requires max_tokens > budget_tokens, so the raise to
        # budget + 1 is unavoidable — but the previous code added a full
        # _DEFAULT_MAX_TOKENS (8192) on top, overshooting the developer's
        # 2000-token ceiling by ~6.6x. The minimal floor is 5001.
        assert create_mock.await_args.kwargs["max_tokens"] == 5001


class _FakeAsyncStream:
    """Minimal AsyncStream stand-in that yields a fixed list of events.

    ``_call_messages`` is patched directly so the model's
    ``isinstance(stream_iter, AsyncStream)`` guard is satisfied via the
    monkeypatched module symbol rather than the real SDK stream.
    """

    def __init__(self, events: list[Any]) -> None:
        self._events = events
        self.closed = False

    def __aiter__(self) -> _FakeAsyncStream:
        self._iter = iter(self._events)
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._iter)
        except StopIteration as e:
            raise StopAsyncIteration from e

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def _patch_async_stream(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
    """Point ``anthropic_model.AsyncStream`` at the test stand-in."""
    from troopai.adk.llms.anthropic import anthropic_model

    monkeypatch.setattr(anthropic_model, "AsyncStream", _FakeAsyncStream)
    yield


def _install_streaming_mock(llm: AnthropicLLM, events: list[Any]) -> AsyncMock:
    """Patch ``llm._call_messages`` to return a fake stream."""
    fake_stream = _FakeAsyncStream(events)
    call_mock = AsyncMock(return_value=fake_stream)
    llm._call_messages = call_mock  # type: ignore[method-assign]
    return call_mock


@pytest.mark.usefixtures("_patch_async_stream")
class TestStreamingUsageRefreshFromMessageDelta:
    """``message_delta`` carries finalized prompt-side usage.

    Anthropic's own stream accumulator overrides ``input_tokens``,
    ``cache_read_input_tokens``, and ``cache_creation_input_tokens`` from
    the ``message_delta`` event when present — the ``message_start``
    counts are preliminary. Dropping them under-reports cache usage on
    the streaming path relative to the non-streaming path.
    """

    async def test_cache_and_input_tokens_refreshed_from_delta(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        # message_start carries preliminary prompt-side counts; the
        # finalized values arrive on message_delta and must win.
        start = RawMessageStartEvent(
            type="message_start",
            message=Message(
                id="msg_stream",
                type="message",
                role="assistant",
                model="claude-sonnet-4",
                content=[],
                stop_reason=None,
                stop_sequence=None,
                usage=Usage(
                    input_tokens=10,
                    output_tokens=0,
                    cache_read_input_tokens=0,
                    cache_creation_input_tokens=0,
                ),
            ),
        )
        delta = RawMessageDeltaEvent(
            type="message_delta",
            delta=RawMessageDeltaEventDelta(stop_reason="end_turn", stop_sequence=None),
            usage=MessageDeltaUsage(
                output_tokens=7,
                input_tokens=50,
                cache_read_input_tokens=40,
                cache_creation_input_tokens=10,
            ),
        )
        events = [
            start,
            RawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=TextBlock(type="text", text="", citations=None),
            ),
            RawContentBlockDeltaEvent(
                type="content_block_delta",
                index=0,
                delta=TextDelta(type="text_delta", text="hi"),
            ),
            RawContentBlockStopEvent(type="content_block_stop", index=0),
            delta,
        ]
        _install_streaming_mock(llm, events)

        stream = await llm.acomplete(messages="hi", stream=True)
        done_event = None
        async for event in stream:
            if event.type == "done":
                done_event = event

        assert done_event is not None
        assert done_event.response is not None
        usage = done_event.response.usage
        assert usage is not None
        # Finalized message_delta counts override the preliminary
        # message_start values — before the fix these were dropped.
        # ``input_tokens`` is the inclusive total (raw + cache_read +
        # cache_creation) so limits/cost see cached prompt tokens.
        assert usage.input_tokens == 100  # 50 + 40 + 10
        assert usage.input_tokens_details.cached_tokens == 40
        assert usage.input_tokens_details.cache_creation_input_tokens == 10
        assert usage.output_tokens == 7
        assert usage.total_tokens == 107

    async def test_absent_delta_counts_preserve_message_start(self) -> None:
        # When message_delta omits the prompt-side fields (None), the
        # preliminary message_start values must be kept, not zeroed.
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        start = RawMessageStartEvent(
            type="message_start",
            message=Message(
                id="msg_stream",
                type="message",
                role="assistant",
                model="claude-sonnet-4",
                content=[],
                stop_reason=None,
                stop_sequence=None,
                usage=Usage(
                    input_tokens=33,
                    output_tokens=0,
                    cache_read_input_tokens=20,
                    cache_creation_input_tokens=5,
                ),
            ),
        )
        delta = RawMessageDeltaEvent(
            type="message_delta",
            delta=RawMessageDeltaEventDelta(stop_reason="end_turn", stop_sequence=None),
            usage=MessageDeltaUsage(output_tokens=4),
        )
        events = [
            start,
            RawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=TextBlock(type="text", text="", citations=None),
            ),
            RawContentBlockStopEvent(type="content_block_stop", index=0),
            delta,
        ]
        _install_streaming_mock(llm, events)

        stream = await llm.acomplete(messages="hi", stream=True)
        done_event = None
        async for event in stream:
            if event.type == "done":
                done_event = event

        assert done_event is not None
        assert done_event.response is not None
        usage = done_event.response.usage
        assert usage is not None
        # Inclusive total: 33 raw + 20 cache_read + 5 cache_creation.
        assert usage.input_tokens == 58
        assert usage.input_tokens_details.cached_tokens == 20
        assert usage.input_tokens_details.cache_creation_input_tokens == 5
