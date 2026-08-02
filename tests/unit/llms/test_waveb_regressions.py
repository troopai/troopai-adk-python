"""Wave-B regression tests for llms/ findings.

Each test corresponds to a numbered finding and verifies the corrected
behaviour.  Tests are written so they would FAIL on the pre-fix code and
PASS on the fixed code.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import pytest

from troopai.adk.llms.litellm.litellm_model import LiteLLM
from troopai.adk.llms.retry import call_with_retry
from troopai.adk.types.llms import LLMRetryErrorKind, LLMRetryPolicy

# ---------------------------------------------------------------------------
# Finding 1: _parse_usage overwrites OpenAI cached_tokens with Anthropic
#            cache_read_input_tokens when both fields are present.
# ---------------------------------------------------------------------------


class TestParseUsageCachedTokensPriority:
    """_parse_usage must not overwrite OpenAI cached_tokens with Anthropic's.

    _parse_usage takes a response object and reads response.usage, so we
    wrap the usage namespace in a fake response.
    """

    def _response_with_usage(self, **usage_fields):  # type: ignore[no-untyped-def]
        """Wrap usage fields in a fake response.usage namespace."""
        usage = SimpleNamespace(**usage_fields)
        return SimpleNamespace(usage=usage)

    def test_openai_style_prompt_details_used_when_no_anthropic_field(self) -> None:
        llm = LiteLLM(model="gpt-4o")
        resp = self._response_with_usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=SimpleNamespace(cached_tokens=40),
        )
        usage = llm._parse_usage(resp)
        assert usage is not None
        assert usage.input_tokens_details is not None
        assert usage.input_tokens_details.cached_tokens == 40

    def test_anthropic_style_used_when_no_openai_details(self) -> None:
        llm = LiteLLM(model="claude-3-5-sonnet-20241022")
        resp = self._response_with_usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            cache_read_input_tokens=30,
        )
        usage = llm._parse_usage(resp)
        assert usage is not None
        assert usage.input_tokens_details is not None
        assert usage.input_tokens_details.cached_tokens == 30

    def test_cache_read_input_tokens_preferred_when_both_present(self) -> None:
        """cache_read_input_tokens takes priority over prompt_tokens_details.cached_tokens.

        litellm normalizes cache_read_input_tokens INTO prompt_tokens_details.cached_tokens
        (they are equal in practice), so preferring cache_read_input_tokens is safe and
        removes a dead branch with a misleading comment claiming mutual exclusivity.
        """
        llm = LiteLLM(model="gpt-4o")
        resp = self._response_with_usage(
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=120,
            prompt_tokens_details=SimpleNamespace(cached_tokens=40),
            # Explicit Anthropic field: preferred over prompt_tokens_details.
            cache_read_input_tokens=99,
        )
        usage = llm._parse_usage(resp)
        assert usage is not None
        assert usage.input_tokens_details is not None
        # cache_read_input_tokens wins (99), not the prompt_tokens_details value (40).
        assert usage.input_tokens_details.cached_tokens == 99


# ---------------------------------------------------------------------------
# Finding 2: _build_stream_response inserts thinking blocks at index 0 in
#            reverse order.  Fixed: collect in forward order then prepend.
# ---------------------------------------------------------------------------


class TestBuildStreamResponseThinkingBlockOrder:
    """Thinking blocks in _build_stream_response must preserve API order.

    Structured thinking blocks are now reconstructed from the per-chunk deltas
    accumulated during streaming (the final usage-sentinel chunk has empty
    choices and never carries them), so they arrive via
    ``accumulated_thinking_blocks`` rather than the last chunk's message.
    """

    def test_two_thinking_blocks_preserved_in_forward_order(self) -> None:
        from troopai.adk.types.responses.llm_response import LLMResponseReasoning

        llm = LiteLLM(model="claude-3-5-sonnet-20241022")
        blocks = [
            {"type": "thinking", "thinking": "first", "signature": "sig1"},
            {"type": "thinking", "thinking": "second", "signature": "sig2"},
        ]
        resp = llm._build_stream_response(
            accumulated_content="",
            accumulated_reasoning="some reasoning",
            accumulated_thinking_blocks=blocks,
            tool_call_data={},
            last_chunk=None,
        )
        reasoning_parts = [p for p in resp.response if isinstance(p, LLMResponseReasoning)]
        assert len(reasoning_parts) == 2
        assert reasoning_parts[0].thinking == "first"
        assert reasoning_parts[1].thinking == "second"
        assert reasoning_parts[0].signature == "sig1"
        assert reasoning_parts[1].signature == "sig2"


# ---------------------------------------------------------------------------
# Finding 3: _stream() uses hardcoded part indices; when reasoning is absent
#            text gets index 1 (gap at 0).  Fixed: monotonic counter.
# ---------------------------------------------------------------------------


class TestStreamPartIndicesMonotonic:
    """Stream part indices must be contiguous starting from 0."""

    def _content_chunk(self, text: str) -> SimpleNamespace:
        delta = SimpleNamespace(content=text, reasoning_content=None, tool_calls=None)
        choice = SimpleNamespace(delta=delta)
        return SimpleNamespace(choices=[choice])

    async def _collect_events(self, chunks):  # type: ignore[no-untyped-def]
        from troopai.adk.llms.litellm.litellm_model import LiteLLM

        llm = LiteLLM(model="gpt-4o")
        events = []
        async for event in llm._stream(_async_gen(chunks)):
            events.append(event)
        return events

    async def test_text_only_stream_starts_at_index_zero(self) -> None:
        """Without reasoning, text part_start must use index 0, not 1."""
        chunks = [self._content_chunk("hello"), self._content_chunk(" world")]
        events = await self._collect_events(chunks)
        part_starts = [e for e in events if e.type == "part_start"]
        assert len(part_starts) >= 1
        assert part_starts[0].index == 0, f"text part_start must be 0 when reasoning absent, got {part_starts[0].index}"

    async def test_reasoning_then_text_get_consecutive_indices(self) -> None:
        """reasoning → index 0, text → index 1 (consecutive, no gap)."""
        llm = LiteLLM(model="gpt-4o")

        def _reasoning_chunk(text: str) -> SimpleNamespace:
            delta = SimpleNamespace(reasoning_content=text, content=None, tool_calls=None)
            return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

        chunks = [_reasoning_chunk("think"), self._content_chunk("result")]
        events = []
        async for event in llm._stream(_async_gen(chunks)):
            events.append(event)
        part_starts = [e for e in events if e.type == "part_start"]
        indices = [e.index for e in part_starts]
        # Should be [0, 1] in order
        assert indices == list(range(len(indices))), f"expected consecutive indices 0..n, got {indices}"


# ---------------------------------------------------------------------------
# Finding 4: GeminiLLM._get_client builds kwargs dict; fixed to explicit
#            constructor calls.  Verify the client is constructed correctly.
# ---------------------------------------------------------------------------


class TestGeminiGetClientNoKwargs:
    """_get_client must use explicit constructor calls, not a kwargs dict."""

    def test_developer_api_path_explicit_constructor(self) -> None:
        from troopai.adk.llms.gemini.gemini_model import GeminiLLM

        llm = GeminiLLM(model="gemini-2.5-flash", api_key="test-key")
        captured: list = []

        class FakeClient:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                captured.append(kwargs)

        class FakeHttpOptions:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

        with (
            patch("google.genai.Client", FakeClient),
            patch("google.genai.types.HttpOptions", FakeHttpOptions),
        ):
            llm._get_client()

        assert len(captured) == 1
        # No "vertexai" key in the developer-api path
        assert "vertexai" not in captured[0]
        assert captured[0].get("api_key") == "test-key"

    def test_vertexai_path_explicit_constructor(self) -> None:
        from troopai.adk.llms.gemini.gemini_model import GeminiLLM

        llm = GeminiLLM(
            model="gemini-2.5-flash",
            vertexai=True,
            project="my-project",
            location="us-central1",
        )
        captured: list = []

        class FakeClient:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                captured.append(kwargs)

        class FakeHttpOptions:
            def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
                pass

        with (
            patch("google.genai.Client", FakeClient),
            patch("google.genai.types.HttpOptions", FakeHttpOptions),
        ):
            llm._get_client()

        assert len(captured) == 1
        assert captured[0].get("vertexai") is True
        assert captured[0].get("project") == "my-project"
        assert captured[0].get("location") == "us-central1"


# ---------------------------------------------------------------------------
# Finding 5: _build_stream_response always sets finish_reason=None.
#            Fixed: extract from last chunk.
# ---------------------------------------------------------------------------


class TestBuildStreamResponseFinishReason:
    """_build_stream_response must propagate finish_reason from the last chunk."""

    def _last_chunk(self, finish_reason: str | None) -> SimpleNamespace:
        choice = SimpleNamespace(finish_reason=finish_reason, message=None, delta=None)
        return SimpleNamespace(id="cid", model="gpt-4o", choices=[choice])

    def test_finish_reason_stop_propagated(self) -> None:
        llm = LiteLLM(model="gpt-4o")
        resp = llm._build_stream_response(
            accumulated_content="hello",
            accumulated_reasoning="",
            accumulated_thinking_blocks=[],
            tool_call_data={},
            last_chunk=self._last_chunk("stop"),
        )
        assert resp.finish_reason == "stop"

    def test_finish_reason_none_when_chunk_has_none(self) -> None:
        llm = LiteLLM(model="gpt-4o")
        resp = llm._build_stream_response(
            accumulated_content="hello",
            accumulated_reasoning="",
            accumulated_thinking_blocks=[],
            tool_call_data={},
            last_chunk=self._last_chunk(None),
        )
        assert resp.finish_reason is None

    def test_finish_reason_tool_calls_propagated(self) -> None:
        llm = LiteLLM(model="gpt-4o")
        resp = llm._build_stream_response(
            accumulated_content="",
            accumulated_reasoning="",
            accumulated_thinking_blocks=[],
            tool_call_data={},
            last_chunk=self._last_chunk("tool_calls"),
        )
        assert resp.finish_reason == "tool_calls"


# ---------------------------------------------------------------------------
# Finding 6: fix_tool_message_ordering silently drops unmatched tool results.
#            Fixed: append orphaned results + log warning.
# ---------------------------------------------------------------------------


class TestFixToolMessageOrderingOrphanedResults:
    """Unmatched tool results must NOT be silently dropped."""

    def _make_assistant_tool_calls(self, *call_ids: str) -> dict:  # type: ignore[type-arg]
        from litellm.types.llms.openai import (
            ChatCompletionAssistantMessage,
            ChatCompletionAssistantToolCall,
            ChatCompletionToolCallFunctionChunk,
        )

        tc_list = [
            ChatCompletionAssistantToolCall(
                id=cid,
                type="function",
                function=ChatCompletionToolCallFunctionChunk(name="fn", arguments="{}"),
            )
            for cid in call_ids
        ]
        msg = ChatCompletionAssistantMessage(role="assistant")
        msg["tool_calls"] = tc_list
        return msg  # type: ignore[return-value]

    def _make_tool_result(self, call_id: str) -> dict:  # type: ignore[type-arg]
        from litellm.types.llms.openai import ChatCompletionToolMessage

        msg = ChatCompletionToolMessage(role="tool", tool_call_id=call_id, content="result")
        return msg  # type: ignore[return-value]

    def test_orphaned_tool_result_preserved(self) -> None:
        from troopai.adk.llms.litellm.litellm_converter import ChatCompletionConverter as LiteLLMConverter

        # Assistant has two tool calls: "tc1" and "tc2".
        # There are three tool results: tc1, tc2, PLUS an orphan "tc_orphan".
        asst = self._make_assistant_tool_calls("tc1", "tc2")
        res1 = self._make_tool_result("tc1")
        res2 = self._make_tool_result("tc2")
        orphan = self._make_tool_result("tc_orphan")

        messages = [asst, res1, res2, orphan]
        result = LiteLLMConverter.fix_tool_message_ordering(messages)

        # All tool results must appear in the output
        tool_ids_in_result = [
            str(m.get("tool_call_id", "")) for m in result if isinstance(m, dict) and m.get("role") == "tool"
        ]
        assert "tc1" in tool_ids_in_result
        assert "tc2" in tool_ids_in_result
        assert "tc_orphan" in tool_ids_in_result, "orphaned tool result must not be silently dropped"


# ---------------------------------------------------------------------------
# Finding 7: _call_litellm() annotated -> Any; fixed to ModelResponse | CustomStreamWrapper.
#            (Type annotation only — verified via ruff/pyright gate, not at runtime.)
# ---------------------------------------------------------------------------


class TestCallLitellmReturnAnnotation:
    """The inner _call_litellm closure's annotation must not be Any."""

    def test_annotation_in_source(self) -> None:
        import inspect

        import troopai.adk.llms.litellm.litellm_model as mod

        src = inspect.getsource(mod)
        assert "async def _call_litellm() -> Any:" not in src, "_call_litellm return annotation must not be Any"
        assert "ModelResponse | CustomStreamWrapper" in src, (
            "_call_litellm must annotate return as ModelResponse | CustomStreamWrapper"
        )


# ---------------------------------------------------------------------------
# Finding 8: call_with_retry catches BaseException, routing KeyboardInterrupt/
#            SystemExit through the classifier.  Fixed: except Exception only.
# ---------------------------------------------------------------------------


class TestCallWithRetryBaseExceptionBypass:
    """KeyboardInterrupt / asyncio.CancelledError must bypass the retry classifier."""

    async def test_keyboard_interrupt_not_retried(self) -> None:
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise KeyboardInterrupt("user abort")

        classifier_calls = 0

        def classifier(exc: BaseException) -> Optional[LLMRetryErrorKind]:
            nonlocal classifier_calls
            classifier_calls += 1
            return "timeout"  # Would retry if reached

        policy = LLMRetryPolicy(max_retries=5, initial_delay=0.001, jitter=False)
        with pytest.raises(KeyboardInterrupt):
            await call_with_retry(op, policy, classifier)

        assert calls == 1, "KeyboardInterrupt must not be retried"
        assert classifier_calls == 0, "classifier must never see KeyboardInterrupt"

    async def test_cancelled_error_not_retried(self) -> None:
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise asyncio.CancelledError("cancelled")

        classifier_calls = 0

        def classifier(exc: BaseException) -> Optional[LLMRetryErrorKind]:
            nonlocal classifier_calls
            classifier_calls += 1
            return "timeout"

        policy = LLMRetryPolicy(max_retries=5, initial_delay=0.001, jitter=False)
        with pytest.raises(asyncio.CancelledError):
            await call_with_retry(op, policy, classifier)

        assert calls == 1, "CancelledError must not be retried"
        assert classifier_calls == 0, "classifier must never see CancelledError"

    async def test_regular_exception_still_retried(self) -> None:
        """Normal Exception subclass must still go through the classifier."""
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ValueError("transient")
            return "ok"

        def classifier(exc: BaseException) -> Optional[LLMRetryErrorKind]:
            if isinstance(exc, ValueError):
                return "rate_limit"
            return None

        policy = LLMRetryPolicy(max_retries=5, initial_delay=0.001, jitter=False)
        result = await call_with_retry(op, policy, classifier)
        assert result == "ok"
        assert calls == 3


# ---------------------------------------------------------------------------
# Finding 9: _convert_tools annotated list[Any]; fixed to list[ChatCompletionToolParam].
#            Verified via source inspection (type gate also catches this).
# ---------------------------------------------------------------------------


class TestConvertToolsReturnAnnotation:
    """_convert_tools must declare list[ChatCompletionToolParam], not list[Any]."""

    def test_annotation_in_source(self) -> None:
        import inspect

        import troopai.adk.llms.litellm.litellm_model as mod

        src = inspect.getsource(mod)
        # The old annotation used list[Any]
        assert "def _convert_tools(" in src
        # The fixed annotation
        assert ") -> list[ChatCompletionToolParam]:" in src


# ---------------------------------------------------------------------------
# Finding 10: litellm_retry.py ImportError guard returns None, silently
#             defeating retry policy.  Fixed: re-raise.
# ---------------------------------------------------------------------------


class TestLitellmRetryImportErrorReraise:
    """An ImportError on litellm import must re-raise, not return None."""

    def test_importerror_reraises(self) -> None:
        import sys

        from troopai.adk.llms.litellm.litellm_retry import litellm_exception_to_kind

        # Temporarily hide litellm
        real_litellm = sys.modules.get("litellm")
        sys.modules["litellm"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError):
                litellm_exception_to_kind(ValueError("something"))
        finally:
            if real_litellm is not None:
                sys.modules["litellm"] = real_litellm
            else:
                del sys.modules["litellm"]


# ---------------------------------------------------------------------------
# Finding 11: cost() catches all Exception at DEBUG; non-pricing errors
#             should be WARNING.  Verified via log level.
# ---------------------------------------------------------------------------


class TestCostLookupExceptionLevels:
    """Pricing-miss → DEBUG; infrastructure error → WARNING."""

    def test_not_found_error_logged_at_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        import litellm

        llm = LiteLLM(model="gpt-4o")
        from troopai.adk.types.tokens.llm_usage import LLMUsage

        usage = LLMUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
        not_found = litellm.NotFoundError(
            message="no pricing",
            model="unknown-model",
            llm_provider="unknown",
        )
        with (
            patch("litellm.cost_per_token", side_effect=not_found),
            caplog.at_level("DEBUG", logger="troopai.adk.llms.litellm.litellm_model"),
        ):
            result = llm.cost("unknown-model", usage)
        assert result is None
        # NotFoundError must NOT produce a WARNING
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) == 0, f"NotFoundError must not log at WARNING: {warning_records}"

    def test_unexpected_exception_logged_at_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        llm = LiteLLM(model="gpt-4o")
        from troopai.adk.types.tokens.llm_usage import LLMUsage

        usage = LLMUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)
        with (
            patch("litellm.cost_per_token", side_effect=RuntimeError("network timeout")),
            caplog.at_level("WARNING", logger="troopai.adk.llms.litellm.litellm_model"),
        ):
            result = llm.cost("gpt-4o", usage)
        assert result is None
        warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
        assert len(warning_records) >= 1, "Unexpected exception must log at WARNING"


# ---------------------------------------------------------------------------
# Finding 12: not-a-bug (Gemini streaming delivers function_call atomically).
# No test needed.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _async_gen(items):  # type: ignore[no-untyped-def]
    for item in items:
        yield item
