"""Regression tests for ``GeminiLLM`` schema cleaning and streaming.

Covers two confirmed defects in ``gemini_model.py``:

1. ``_clean_schema_for_gemini`` deleted ``$ref`` / ``$defs`` outright,
   reducing every nested-model property to an unconstrained ``{}`` and
   discarding the nested shape. The fix resolves each ``$ref`` inline
   before dropping the ``$defs`` block.
2. The streaming accumulator keyed parts on the chunk-local ``enumerate``
   position alone. Gemini reuses position 0 across chunks (a thought delta
   then the answer-text delta), so the answer was folded into the reasoning
   slot and dropped from the response. The fix keys on
   ``(position, part_kind)`` and emits monotonic integer indices.
"""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
from typing import Any

import httpx
from google.genai.types import (
    Candidate,
    Content,
    FinishReason,
    FunctionCall,
    GenerateContentConfig,
    GenerateContentResponse,
    Part,
)
from pydantic import BaseModel

from troopai.adk.llms.gemini.gemini_model import GeminiLLM, _clean_schema_for_gemini
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.schemas import AgentOutputSchema
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
    LLMResponseReasoning,
    LLMResponseText,
    LLMStreamEvent,
)

# ----------------------------------------------------------------------
# Finding 1 — nested-model $ref / $defs must be inlined, not deleted
# ----------------------------------------------------------------------


class _Address(BaseModel):
    street: str
    city: str


class _Person(BaseModel):
    name: str
    address: _Address


class _Coord(BaseModel):
    lat: float
    lon: float


class _DeepAddress(BaseModel):
    street: str
    coord: _Coord


class _DeepPerson(BaseModel):
    name: str
    address: _DeepAddress
    tags: list[str]


class TestCleanSchemaForGemini:
    def test_nested_model_ref_is_inlined_not_dropped(self) -> None:
        cleaned = _clean_schema_for_gemini(AgentOutputSchema(_Person).json_schema())

        address = cleaned["properties"]["address"]
        # The bug produced ``{}`` here; the fix inlines the Address shape.
        assert address.get("type") == "object", f"address not resolved: {address!r}"
        assert set(address["properties"].keys()) == {"street", "city"}
        assert address["properties"]["street"]["type"] == "string"

    def test_defs_and_ref_and_additional_properties_removed(self) -> None:
        cleaned = _clean_schema_for_gemini(AgentOutputSchema(_Person).json_schema())

        def _walk(node: Any) -> None:
            if isinstance(node, dict):
                assert "$ref" not in node
                assert "$defs" not in node
                assert "additionalProperties" not in node
                for value in node.values():
                    _walk(value)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(cleaned)

    def test_doubly_nested_model_resolved_through_lists(self) -> None:
        cleaned = _clean_schema_for_gemini(AgentOutputSchema(_DeepPerson).json_schema())

        coord = cleaned["properties"]["address"]["properties"]["coord"]
        assert coord["type"] == "object"
        assert coord["properties"]["lat"]["type"] == "number"
        # List item schema also carries through unchanged.
        assert cleaned["properties"]["tags"]["items"]["type"] == "string"

    def test_flat_schema_unchanged_apart_from_strip(self) -> None:
        class _Flat(BaseModel):
            a: str
            b: int

        cleaned = _clean_schema_for_gemini(AgentOutputSchema(_Flat).json_schema())
        assert cleaned["properties"]["a"]["type"] == "string"
        assert cleaned["properties"]["b"]["type"] == "integer"

    def test_self_referential_schema_terminates(self) -> None:
        # Gemini cannot express recursion; the cleaner must not loop forever.
        class _Node(BaseModel):
            value: int
            children: list[_Node]

        _Node.model_rebuild()
        cleaned = _clean_schema_for_gemini(AgentOutputSchema(_Node).json_schema())
        # One level of children is inlined; the back-reference is dropped.
        children = cleaned["properties"]["children"]
        assert children["type"] == "array"
        inner = children["items"]
        assert inner["type"] == "object"
        assert "value" in inner["properties"]


# ----------------------------------------------------------------------
# Finding 2 — streaming must not fold answer text into the reasoning slot
# ----------------------------------------------------------------------


def _thought_chunk(text: str) -> GenerateContentResponse:
    """A streaming chunk carrying a single thought part at parts[0]."""
    return GenerateContentResponse(
        candidates=[Candidate(content=Content(role="model", parts=[Part(thought=True, text=text)]))],
        response_id="resp_stream",
        model_version="gemini-2.5-pro",
    )


def _text_chunk(text: str, *, finish: bool = False) -> GenerateContentResponse:
    """A streaming chunk carrying a single answer-text part at parts[0]."""
    candidate = Candidate(
        content=Content(role="model", parts=[Part.from_text(text=text)]),
        finish_reason=FinishReason.STOP if finish else None,
    )
    return GenerateContentResponse(candidates=[candidate])


class _FakeStream:
    """Minimal async iterator over pre-baked Gemini streaming chunks."""

    def __init__(self, chunks: list[GenerateContentResponse]) -> None:
        self._chunks = chunks

    def __aiter__(self) -> AsyncIterator[GenerateContentResponse]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[GenerateContentResponse]:
        for chunk in self._chunks:
            yield chunk


class _FakeModels:
    def __init__(self, chunks: list[GenerateContentResponse]) -> None:
        self._chunks = chunks

    async def generate_content_stream(self, **_kwargs: Any) -> _FakeStream:
        return _FakeStream(self._chunks)


class _FakeAio:
    def __init__(self, chunks: list[GenerateContentResponse]) -> None:
        self.models = _FakeModels(chunks)


class _FakeClient:
    def __init__(self, chunks: list[GenerateContentResponse]) -> None:
        self.aio = _FakeAio(chunks)


async def _collect(llm: GeminiLLM, chunks: list[GenerateContentResponse]) -> list[LLMStreamEvent]:
    llm._client = _FakeClient(chunks)  # type: ignore[assignment]
    return [event async for event in llm._stream([], GenerateContentConfig())]


class TestStreamingThoughtAnswerSplit:
    async def test_answer_after_thought_at_same_position_not_lost(self) -> None:
        # Gemini streams the thought first, then the answer — both at parts[0].
        chunks = [
            _thought_chunk("Let me think. "),
            _thought_chunk("Considering options. "),
            _text_chunk("The answer is "),
            _text_chunk("42.", finish=True),
        ]
        events = await _collect(GeminiLLM(model="gemini-2.5-pro"), chunks)

        done = [e for e in events if e.type == "done"]
        assert len(done) == 1
        response = done[0].response
        assert isinstance(response, LLMResponse)

        # The visible answer must survive as a distinct text part.
        assert response.content == "The answer is 42."
        # The reasoning must NOT swallow the answer text.
        assert response.thinking == "Let me think. Considering options. "

        parts = response.response
        assert any(isinstance(p, LLMResponseReasoning) for p in parts)
        assert any(isinstance(p, LLMResponseText) for p in parts)

    async def test_stream_event_indices_are_distinct_integers(self) -> None:
        chunks = [
            _thought_chunk("thinking "),
            _text_chunk("answer", finish=True),
        ]
        events = await _collect(GeminiLLM(model="gemini-2.5-pro"), chunks)

        starts = [e for e in events if e.type == "part_start"]
        assert len(starts) == 2
        indices = [e.index for e in starts]
        # Distinct integer indices so downstream int-keyed consumers don't collide.
        assert all(isinstance(i, int) for i in indices)
        assert len(set(indices)) == 2

        # Text deltas must be routed under the text part's index, never the
        # reasoning part's index.
        text_start = next(e for e in starts if isinstance(e.part, LLMResponseText))
        reasoning_start = next(e for e in starts if isinstance(e.part, LLMResponseReasoning))
        assert text_start.index != reasoning_start.index

        text_deltas = [e for e in events if e.type == "part_delta" and e.index == text_start.index]
        assert "".join(d.delta or "" for d in text_deltas) == "answer"
        reasoning_deltas = [e for e in events if e.type == "part_delta" and e.index == reasoning_start.index]
        assert "".join(d.delta or "" for d in reasoning_deltas) == "thinking "


# ----------------------------------------------------------------------
# Streamed parallel function calls must not collide on (0, "call")
# ----------------------------------------------------------------------


def _call_chunk(
    call_id: str,
    name: str,
    args: dict[str, Any],
    *,
    finish: bool = False,
) -> GenerateContentResponse:
    """A streaming chunk carrying one complete function call at parts[0]."""
    candidate = Candidate(
        content=Content(
            role="model",
            parts=[Part(function_call=FunctionCall(id=call_id, name=name, args=args))],
        ),
        finish_reason=FinishReason.STOP if finish else None,
    )
    return GenerateContentResponse(candidates=[candidate])


class TestStreamingParallelFunctionCalls:
    async def test_parallel_calls_in_separate_chunks_both_survive(self) -> None:
        # Gemini streams each parallel call complete in its own chunk at
        # parts[0]; keying on (position, kind) alone folded them all onto
        # (0, "call") so only the last call survived.
        chunks = [
            _call_chunk("c1", "search", {"q": "a"}),
            _call_chunk("c2", "lookup", {"id": 7}, finish=True),
        ]
        events = await _collect(GeminiLLM(model="gemini-2.5-pro"), chunks)

        done = [e for e in events if e.type == "done"]
        assert len(done) == 1
        response = done[0].response
        assert isinstance(response, LLMResponse)

        calls = response.tool_calls
        assert len(calls) == 2
        assert [c.name for c in calls] == ["search", "lookup"]
        assert [c.call_id for c in calls] == ["c1", "c2"]
        import json

        assert [json.loads(c.arguments) for c in calls] == [{"q": "a"}, {"id": 7}]

    async def test_each_parallel_call_emits_distinct_part_start(self) -> None:
        chunks = [
            _call_chunk("c1", "search", {"q": "a"}),
            _call_chunk("c2", "lookup", {"id": 7}, finish=True),
        ]
        events = await _collect(GeminiLLM(model="gemini-2.5-pro"), chunks)

        call_starts = [e for e in events if e.type == "part_start" and isinstance(e.part, LLMResponseFunctionToolCall)]
        assert len(call_starts) == 2
        assert len({e.index for e in call_starts}) == 2


# ----------------------------------------------------------------------
# Streamed thought signature must round-trip losslessly via base64
# ----------------------------------------------------------------------


def _thought_chunk_with_sig(text: str, sig: bytes) -> GenerateContentResponse:
    return GenerateContentResponse(
        candidates=[
            Candidate(content=Content(role="model", parts=[Part(thought=True, text=text, thought_signature=sig)]))
        ],
        response_id="resp_sig_stream",
        model_version="gemini-2.5-pro",
    )


class TestStreamingThoughtSignatureBase64:
    async def test_non_utf8_signature_carried_as_base64(self) -> None:
        raw_sig = bytes([0x80, 0x00, 0xFF, 0xC3, 0x28])  # invalid utf-8
        chunks = [
            _thought_chunk_with_sig("thinking", raw_sig),
            _text_chunk("done", finish=True),
        ]
        events = await _collect(GeminiLLM(model="gemini-2.5-pro"), chunks)

        done = [e for e in events if e.type == "done"]
        response = done[0].response
        assert isinstance(response, LLMResponse)
        reasoning = next(p for p in response.response if isinstance(p, LLMResponseReasoning))
        assert reasoning.encrypted_content is not None
        # Lossless: decodes back to the exact original bytes.
        assert base64.b64decode(reasoning.encrypted_content) == raw_sig


# ----------------------------------------------------------------------
# HttpOptions: timeout / extra_body / extra_args must be forwarded
# ----------------------------------------------------------------------


class TestHttpOptionsForwarding:
    def test_none_when_nothing_set(self) -> None:
        llm = GeminiLLM(model="gemini-2.5-flash")
        assert llm._build_http_options(LLMConfig()) is None

    def test_timeout_seconds_forwarded_as_millis(self) -> None:
        llm = GeminiLLM(model="gemini-2.5-flash")
        opts = llm._build_http_options(LLMConfig(timeout=30.0))
        assert opts is not None
        # Gemini HttpOptions.timeout is milliseconds.
        assert opts.timeout == 30000

    def test_httpx_timeout_uses_read_bound(self) -> None:
        llm = GeminiLLM(model="gemini-2.5-flash")
        opts = llm._build_http_options(LLMConfig(timeout=httpx.Timeout(None, read=12.0, connect=5.0)))
        assert opts is not None
        assert opts.timeout == 12000

    def test_extra_body_and_extra_args_merged_into_extra_body(self) -> None:
        llm = GeminiLLM(model="gemini-2.5-flash")
        opts = llm._build_http_options(LLMConfig(extra_body={"a": 1}, extra_args={"b": 2}))
        assert opts is not None
        assert opts.extra_body == {"a": 1, "b": 2}

    def test_forwarded_onto_generate_content_config(self) -> None:
        llm = GeminiLLM(model="gemini-2.5-flash")
        gen_config = llm._build_generate_content_config(
            config=LLMConfig(timeout=15.0, extra_args={"foo": "bar"}),
            system_instruction=None,
            wire_tools=None,
            tool_config=None,
            thinking_config=None,
            wants_structured=False,
            output_schema=None,
        )
        assert gen_config.http_options is not None
        assert gen_config.http_options.timeout == 15000
        assert gen_config.http_options.extra_body == {"foo": "bar"}
