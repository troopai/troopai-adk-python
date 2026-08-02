"""Tests for ``AnthropicConfig`` fields reaching the Messages API.

Regression coverage for ``AnthropicConfig.extra_query``: the field is
documented on both the base ``LLMConfig`` and ``AnthropicConfig``, and
``client.messages.create`` accepts ``extra_query=``, but the SDK
boundary (``AnthropicLLM._call_messages``) previously never read it —
silently dropping explicit developer intent. These tests assert the
field flows through to both the non-streaming and streaming
``messages.create`` calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

from anthropic.types import Message, TextBlock, Usage

from troopai.adk.llms.anthropic import AnthropicConfig, AnthropicLLM


def _make_message(text: str = "ok") -> Message:
    return Message(
        id="msg_test",
        type="message",
        role="assistant",
        model="claude-sonnet-4",
        content=[TextBlock(type="text", text=text, citations=None)],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def _install_mock_client(llm: AnthropicLLM, response: Message) -> AsyncMock:
    """Wire a mocked ``AsyncAnthropic`` client; return the create mock."""
    create_mock = AsyncMock(return_value=response)
    fake_client = MagicMock()
    fake_client.messages.create = create_mock
    llm._client = fake_client  # bypass lazy init for tests
    return create_mock


class TestExtraQueryForwarding:
    async def test_extra_query_reaches_non_streaming_create(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        create_mock = _install_mock_client(llm, _make_message())

        await llm.acomplete(
            messages="hello",
            llm_config=AnthropicConfig(extra_query={"my-param": "value"}),
        )

        assert create_mock.await_args is not None
        kwargs = create_mock.await_args.kwargs
        assert kwargs["extra_query"] == {"my-param": "value"}

    async def test_extra_query_none_when_unset(self) -> None:
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        create_mock = _install_mock_client(llm, _make_message())

        await llm.acomplete(messages="hello", llm_config=AnthropicConfig())

        assert create_mock.await_args is not None
        kwargs = create_mock.await_args.kwargs
        assert kwargs["extra_query"] is None

    async def test_extra_query_is_copied_not_aliased(self) -> None:
        # The boundary copies the mapping so later mutation of the
        # caller's dict cannot reach into an already-issued request.
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        create_mock = _install_mock_client(llm, _make_message())

        original: dict[str, str] = {"k": "v"}
        await llm.acomplete(
            messages="hello",
            llm_config=AnthropicConfig(extra_query=original),
        )
        original["k"] = "mutated"

        assert create_mock.await_args is not None
        forwarded = create_mock.await_args.kwargs["extra_query"]
        assert forwarded == {"k": "v"}

    async def test_extra_query_reaches_streaming_create(self) -> None:
        # ``_call_messages(stream=True, ...)`` is the streaming SDK
        # boundary; assert against its kwargs directly so the test does
        # not need a real ``AsyncStream`` to iterate.
        llm = AnthropicLLM(model="claude-sonnet-4-20250514", api_key="test")
        create_mock = AsyncMock(return_value=MagicMock())
        fake_client = MagicMock()
        fake_client.messages.create = create_mock
        llm._client = fake_client

        await llm._call_messages(
            stream=True,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=16,
            system=None,
            tools=None,
            tool_choice=None,
            thinking=None,
            config=AnthropicConfig(extra_query={"beta": "x"}),
            service_tier=None,
            effort=None,
            mid_system=False,
            extra_body={},
        )

        assert create_mock.await_args is not None
        kwargs: dict[str, Any] = create_mock.await_args.kwargs
        assert kwargs["stream"] is True
        assert kwargs["extra_query"] == {"beta": "x"}
