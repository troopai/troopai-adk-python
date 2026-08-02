"""Streaming-path regression tests for the LiteLLM provider.

Covers:
- Mid-stream error contract: done(finish_reason="error") then re-raise.
- finish_reason survives when include_usage=True sends a usage sentinel chunk
  with empty choices.
- part_end events are emitted for every started part (API contract parity with
  other provider implementations).
- Gemini __thought__ suffix stripped from streaming tool-call IDs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from troopai.adk.llms.litellm.litellm_model import LiteLLM


def _chunk(text: str) -> SimpleNamespace:
    """A minimal litellm-shaped streaming chunk carrying a content delta."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


def _stop_chunk(finish_reason: str = "stop", model: str = "gpt-4o-mini") -> SimpleNamespace:
    """A chunk carrying a finish_reason (last real content chunk)."""
    return SimpleNamespace(
        id="resp1",
        model=model,
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                delta=SimpleNamespace(content=None, tool_calls=None, reasoning_content=None),
            )
        ],
        usage=None,
    )


def _usage_sentinel(model: str = "gpt-4o-mini") -> SimpleNamespace:
    """Final sentinel chunk with empty choices and usage data (include_usage=True)."""
    return SimpleNamespace(
        id="resp1",
        model=model,
        choices=[],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=None,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
            completion_tokens_details=None,
        ),
    )


async def _failing_chunks():
    yield _chunk("hel")
    yield _chunk("lo")
    raise RuntimeError("provider exploded mid-stream")


async def test_stream_emits_done_error_then_reraises_on_midstream_failure() -> None:
    model = LiteLLM(model="gpt-4o-mini")

    events = []
    with pytest.raises(RuntimeError, match="provider exploded mid-stream"):
        async for event in model._stream(_failing_chunks()):
            events.append(event)

    # A terminal done event was emitted BEFORE the exception propagated.
    done_events = [e for e in events if e.type == "done"]
    assert len(done_events) == 1, f"expected exactly one terminal done; got {[e.type for e in events]}"
    done = done_events[0]
    assert done.response is not None
    # The contract: finish_reason='error' marks the abnormal termination.
    assert done.response.finish_reason == "error"


async def test_finish_reason_preserved_when_usage_sentinel_follows() -> None:
    """finish_reason must not become None when include_usage=True sends a usage sentinel.

    Regression: ``last_chunk = chunk`` was unconditional; the empty-choices usage
    sentinel overwrote the finish-reason chunk, causing finish_reason=None.
    """

    async def _chunks():
        yield _chunk("hello")
        yield _stop_chunk("stop")
        yield _usage_sentinel()

    model = LiteLLM(model="gpt-4o-mini")
    events = []
    async for event in model._stream(_chunks()):
        events.append(event)

    done_events = [e for e in events if e.type == "done"]
    assert len(done_events) == 1
    assert done_events[0].response is not None
    assert done_events[0].response.finish_reason == "stop", (
        f"Expected 'stop', got {done_events[0].response.finish_reason!r}"
    )


async def test_part_end_emitted_for_text_part() -> None:
    """part_end must be emitted for every started part (API contract parity).

    Regression: litellm _stream() never emitted part_end, while every other
    provider implementation (anthropic, openai_chatcompletions, openai_responses,
    gemini) does.
    """

    async def _chunks():
        yield _chunk("hello")
        yield _stop_chunk("stop")

    model = LiteLLM(model="gpt-4o-mini")
    events = []
    async for event in model._stream(_chunks()):
        events.append(event)

    event_types = [e.type for e in events]
    assert "part_start" in event_types, "part_start missing"
    assert "part_end" in event_types, f"part_end missing from {event_types}"
    # Ordering: part_end must come before done.
    part_end_idx = next(i for i, e in enumerate(events) if e.type == "part_end")
    done_idx = next(i for i, e in enumerate(events) if e.type == "done")
    assert part_end_idx < done_idx, "part_end must precede done"


async def test_gemini_thought_suffix_stripped_in_streaming_tool_call() -> None:
    """Gemini __thought__ suffix must be stripped from streaming tool-call IDs.

    Regression: the non-streaming path stripped the suffix at _parse_response()
    but the streaming path returned the raw ID from accumulated chunks.
    """

    def _tool_chunk(tc_id: str, tc_name: str, tc_args: str, model: str = "gemini/gemini-2.0-flash") -> SimpleNamespace:
        return SimpleNamespace(
            id="resp1",
            model=model,
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(
                        content=None,
                        reasoning_content=None,
                        tool_calls=[
                            SimpleNamespace(
                                index=0,
                                id=tc_id,
                                function=SimpleNamespace(name=tc_name, arguments=tc_args),
                            )
                        ],
                    ),
                )
            ],
            usage=None,
        )

    def _finish_chunk(model: str = "gemini/gemini-2.0-flash") -> SimpleNamespace:
        return SimpleNamespace(
            id="resp1",
            model=model,
            choices=[
                SimpleNamespace(
                    finish_reason="tool_calls",
                    delta=SimpleNamespace(content=None, tool_calls=None, reasoning_content=None),
                )
            ],
            usage=None,
        )

    async def _chunks():
        yield _tool_chunk("call_abc123__thought__xyz", "my_func", '{"x": 1}')
        yield _finish_chunk()

    model = LiteLLM(model="gpt-4o-mini")  # model attr; Gemini is detected from chunk.model
    events = []
    async for event in model._stream(_chunks()):
        events.append(event)

    done_events = [e for e in events if e.type == "done"]
    assert len(done_events) == 1
    response = done_events[0].response
    assert response is not None
    tool_calls = [p for p in response.response if hasattr(p, "call_id")]
    assert len(tool_calls) == 1
    assert tool_calls[0].call_id == "call_abc123", f"__thought__ suffix was not stripped; got {tool_calls[0].call_id!r}"


def _thinking_chunk(thinking: str | None = None, signature: str | None = None) -> SimpleNamespace:
    """A chunk carrying a fragment of a structured thinking block."""
    block: dict[str, str] = {"type": "thinking"}
    if thinking is not None:
        block["thinking"] = thinking
    if signature is not None:
        block["signature"] = signature
    return SimpleNamespace(
        id="resp1",
        model="claude-sonnet-4-20250514",
        choices=[
            SimpleNamespace(
                finish_reason=None,
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=None,
                    reasoning_content=thinking,
                    thinking_blocks=[block],
                ),
            )
        ],
        usage=None,
    )


async def test_streamed_thinking_blocks_with_signature_reconstructed() -> None:
    """Streamed Anthropic thinking blocks (with signatures) must survive.

    Regression: the signature was read off the final chunk's message/delta, but
    under include_usage=True the final chunk is a usage sentinel with
    ``choices=[]`` — so the structured thinking blocks (and their signatures,
    which Anthropic requires on replay) were always lost.
    """
    from troopai.adk.types.responses.llm_response import LLMResponseReasoning

    async def _chunks():
        yield _thinking_chunk(thinking="Let me think")
        yield _thinking_chunk(thinking=" it through")
        yield _thinking_chunk(signature="sig-xyz")  # signature closes the block
        yield _chunk("the answer")
        yield _usage_sentinel(model="claude-sonnet-4-20250514")  # choices=[]

    model = LiteLLM(model="claude-sonnet-4-20250514")
    events = [e async for e in model._stream(_chunks())]

    done_events = [e for e in events if e.type == "done"]
    assert len(done_events) == 1
    response = done_events[0].response
    assert response is not None
    reasoning = [p for p in response.response if isinstance(p, LLMResponseReasoning)]
    assert len(reasoning) == 1
    assert reasoning[0].thinking == "Let me think it through"
    assert reasoning[0].signature == "sig-xyz", "streamed thinking signature was lost"


async def test_streamed_redacted_thinking_block_reconstructed() -> None:
    """A streamed redacted_thinking block must round-trip with its opaque data."""
    from troopai.adk.types.responses.llm_response import LLMResponseReasoning

    def _redacted_chunk() -> SimpleNamespace:
        return SimpleNamespace(
            id="resp1",
            model="claude-sonnet-4-20250514",
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(
                        content=None,
                        tool_calls=None,
                        reasoning_content=None,
                        thinking_blocks=[{"type": "redacted_thinking", "data": "opaque-abc"}],
                    ),
                )
            ],
            usage=None,
        )

    async def _chunks():
        yield _redacted_chunk()
        yield _chunk("answer")
        yield _usage_sentinel(model="claude-sonnet-4-20250514")

    model = LiteLLM(model="claude-sonnet-4-20250514")
    events = [e async for e in model._stream(_chunks())]

    response = next(e.response for e in events if e.type == "done")
    assert response is not None
    reasoning = [p for p in response.response if isinstance(p, LLMResponseReasoning)]
    assert len(reasoning) == 1
    assert reasoning[0].thinking == ""
    assert reasoning[0].signature == "opaque-abc"
