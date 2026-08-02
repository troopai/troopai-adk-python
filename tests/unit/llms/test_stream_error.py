"""Cross-provider streaming-error contract (stream_with_error_contract)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

import pytest

from troopai.adk.llms.stream_error import stream_with_error_contract
from troopai.adk.types.responses.llm_response import LLMResponse, LLMStreamEvent

_LOG = logging.getLogger("test_stream_error")


async def test_midstream_error_emits_done_error_then_reraises() -> None:
    async def _inner() -> AsyncIterator[LLMStreamEvent]:
        yield LLMStreamEvent(type="part_delta", index=0, delta="hi")
        raise RuntimeError("provider blew up")

    events: list[LLMStreamEvent] = []
    with pytest.raises(RuntimeError, match="provider blew up"):
        async for ev in stream_with_error_contract(_inner(), model="m", logger=_LOG):
            events.append(ev)

    # The forwarded delta, then a terminal done(finish_reason="error") BEFORE the raise.
    assert events[0].type == "part_delta"
    done = [e for e in events if e.type == "done"]
    assert len(done) == 1
    assert done[0].response is not None
    assert done[0].response.finish_reason == "error"


async def test_success_forwards_inner_done_unchanged() -> None:
    async def _inner() -> AsyncIterator[LLMStreamEvent]:
        yield LLMStreamEvent(type="part_delta", index=0, delta="hi")
        yield LLMStreamEvent(
            type="done",
            response=LLMResponse(response_id="r", model="m", response=[], finish_reason="stop"),
        )

    events: list[LLMStreamEvent] = []
    async for ev in stream_with_error_contract(_inner(), model="m", logger=_LOG):
        events.append(ev)

    # The inner's own done (finish_reason="stop") is forwarded; no extra error done.
    assert [e.type for e in events] == ["part_delta", "done"]
    done = [e for e in events if e.type == "done"]
    assert len(done) == 1
    assert done[0].response is not None
    assert done[0].response.finish_reason == "stop"


async def test_cancellation_propagates_without_error_done() -> None:
    async def _inner() -> AsyncIterator[LLMStreamEvent]:
        yield LLMStreamEvent(type="part_delta", index=0, delta="hi")
        raise asyncio.CancelledError

    events: list[LLMStreamEvent] = []
    with pytest.raises(asyncio.CancelledError):
        async for ev in stream_with_error_contract(_inner(), model="m", logger=_LOG):
            events.append(ev)

    # Cancellation is BaseException, not Exception → never masked, no done emitted.
    assert [e.type for e in events] == ["part_delta"]
