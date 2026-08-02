"""Regression tests for the LiteLLM provider model.

Covers confirmed defects in ``llms/litellm/litellm_model.py``:

- Named-tool ``tool_choice`` ("my_tool") must be converted to the wire-format
  named-tool shape before reaching ``litellm.acompletion`` — a bare string is
  rejected by providers.
- ``ToolExecutionMode.SEQUENTIAL`` must send ``parallel_tool_calls=False``;
  sending ``None`` lets providers default to parallel tool calls.
- ``_parse_usage`` must synthesize ``total_tokens`` from prompt+completion when
  a provider omits it, so the per-request usage breakdown is recorded.
- Streaming must suppress the empty priming content/reasoning delta so no
  phantom empty part (and its part index) is emitted.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

from troopai.adk.llms.litellm.litellm_model import LiteLLM
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.llms.llm_usage import LLMUsage
from troopai.adk.types.input import LLMInputContentItem
from troopai.adk.types.tools import ToolExecutionMode


def _model_response() -> SimpleNamespace:
    """A minimal litellm ModelResponse-shaped object for acompletion mocks."""
    return SimpleNamespace(
        id="r1",
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="hello",
                    tool_calls=None,
                    reasoning_content=None,
                    thinking_blocks=None,
                    refusal=None,
                    annotations=None,
                    provider_specific_fields=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=5,
            completion_tokens=3,
            total_tokens=8,
            prompt_tokens_details=None,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
            completion_tokens_details=None,
        ),
        created=None,
    )


async def _capture_acompletion_kwargs(config: LLMConfig) -> dict:
    """Run acomplete with a captured acompletion and return the call kwargs."""
    captured: dict = {}

    async def fake_acompletion(**kwargs):  # type: ignore[misc]
        captured.update(kwargs)
        return _model_response()

    llm = LiteLLM(model="gpt-4o-mini")
    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        await llm.acomplete(messages="hi", llm_config=config)
    return captured


async def _capture_kwargs_for(model: str, messages: list[LLMInputContentItem]) -> dict:
    """Capture acompletion kwargs for a given model + message history."""
    captured: dict = {}

    async def fake_acompletion(**kwargs):  # type: ignore[misc]
        captured.update(kwargs)
        return _model_response()

    llm = LiteLLM(model=model)
    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        await llm.acomplete(messages=messages)
    return captured


# A history with tool messages but no tools defined on the request — the
# trigger for the Anthropic-only placeholder-tool workaround.
_TOOL_HISTORY: list[LLMInputContentItem] = [
    {"type": "function_call", "call_id": "c1", "name": "fn", "arguments": "{}"},
    {"type": "function_call_output", "call_id": "c1", "output": "ok"},
]


# ---------------------------------------------------------------------------
# Placeholder tool must be injected ONLY for Anthropic-family models.
# ---------------------------------------------------------------------------


async def test_placeholder_tool_injected_for_anthropic() -> None:
    """Anthropic 400s without tools when the history has tool messages."""
    captured = await _capture_kwargs_for("claude-sonnet-4-20250514", _TOOL_HISTORY)
    tools = captured["tools"]
    assert tools is not None
    assert any(t["function"]["name"] == "_placeholder" for t in tools)


async def test_no_placeholder_tool_for_openai() -> None:
    """OpenAI accepts historical tool messages without re-declaring tools.

    Regression: the placeholder tool was injected for every provider, adding an
    un-opted-in tool the model could call spuriously.
    """
    captured = await _capture_kwargs_for("gpt-4o-mini", _TOOL_HISTORY)
    assert captured["tools"] is None


async def test_no_placeholder_tool_for_gemini() -> None:
    """Gemini does not require the placeholder workaround either."""
    captured = await _capture_kwargs_for("gemini/gemini-2.0-flash", _TOOL_HISTORY)
    assert captured["tools"] is None


# ---------------------------------------------------------------------------
# extra_args / extra_body keys colliding with mapped named params must not
# crash the acompletion call with "multiple values for keyword argument".
# ---------------------------------------------------------------------------


async def test_extra_args_colliding_named_param_does_not_crash() -> None:
    captured = await _capture_acompletion_kwargs(LLMConfig(temperature=0.3, extra_args={"temperature": 0.9}))
    # No TypeError; the explicit LLMConfig value wins over the extra copy.
    assert captured["temperature"] == 0.3


async def test_extra_body_colliding_named_param_does_not_crash() -> None:
    captured = await _capture_acompletion_kwargs(LLMConfig(extra_body={"stop": ["STOP"]}))
    # The colliding 'stop' copy is dropped; the mapped (unset) value is sent.
    assert captured["stop"] is None


async def test_non_colliding_extra_args_still_forwarded() -> None:
    # A genuinely provider-specific key (not a mapped param) still passes through.
    captured = await _capture_acompletion_kwargs(LLMConfig(extra_args={"custom_param": "v"}))
    assert captured["custom_param"] == "v"


# ---------------------------------------------------------------------------
# _convert_gemini_thought_signatures must not mutate the caller's history.
# ---------------------------------------------------------------------------


def test_convert_gemini_thought_signatures_does_not_mutate_input() -> None:
    tool_call: dict[str, Any] = {
        "id": "c1",
        "type": "function",
        "function": {"name": "fn", "arguments": "{}"},
        "provider_data": {"google": {"thought_signature": "sig-1"}},
    }
    messages: list[Any] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "tool_calls": [tool_call]},
    ]
    original = copy.deepcopy(messages)

    result = LiteLLM._convert_gemini_thought_signatures(messages)

    # Input is untouched: provider_data still present, no injected fields.
    assert messages == original
    assert "provider_data" in messages[1]["tool_calls"][0]
    assert "provider_specific_fields" not in messages[1]["tool_calls"][0]
    # The returned copy carries the converted signature.
    converted_call = result[1]["tool_calls"][0]
    assert converted_call["provider_specific_fields"] == {"thought_signature": "sig-1"}
    assert "provider_data" not in converted_call


# ---------------------------------------------------------------------------
# tool_choice: named tool must be converted to the wire-format shape
# ---------------------------------------------------------------------------


async def test_named_tool_choice_converted_to_wire_shape() -> None:
    """A bare tool-name string becomes the nested named-tool dict.

    Regression: the litellm path forwarded config.tool_choice unchanged, so a
    legitimate ``tool_choice="my_tool"`` reached the provider as a bare string,
    which OpenAI-schema providers reject.
    """
    captured = await _capture_acompletion_kwargs(LLMConfig(tool_choice="my_tool"))

    assert captured["tool_choice"] == {
        "type": "function",
        "function": {"name": "my_tool"},
    }


async def test_auto_tool_choice_passes_through_unchanged() -> None:
    """The sentinel values round-trip identically through the converter."""
    captured = await _capture_acompletion_kwargs(LLMConfig(tool_choice="auto"))
    assert captured["tool_choice"] == "auto"


async def test_none_tool_choice_omitted() -> None:
    """No tool_choice → None (omitted from the wire call)."""
    captured = await _capture_acompletion_kwargs(LLMConfig())
    assert captured["tool_choice"] is None


# ---------------------------------------------------------------------------
# tool_execution_mode → parallel_tool_calls three-state mapping
# ---------------------------------------------------------------------------


async def test_sequential_mode_sends_parallel_false() -> None:
    """SEQUENTIAL must send parallel_tool_calls=False.

    Regression: SEQUENTIAL mapped to None, letting providers default to
    parallel tool calls and silently ignoring the one-tool-per-turn request.
    """
    captured = await _capture_acompletion_kwargs(LLMConfig(tool_execution_mode=ToolExecutionMode.SEQUENTIAL))
    assert captured["parallel_tool_calls"] is False


async def test_parallel_mode_sends_parallel_true() -> None:
    """PARALLEL must send parallel_tool_calls=True."""
    captured = await _capture_acompletion_kwargs(LLMConfig(tool_execution_mode=ToolExecutionMode.PARALLEL))
    assert captured["parallel_tool_calls"] is True


async def test_unset_mode_omits_parallel_tool_calls() -> None:
    """No tool_execution_mode → None (provider default)."""
    captured = await _capture_acompletion_kwargs(LLMConfig())
    assert captured["parallel_tool_calls"] is None


# ---------------------------------------------------------------------------
# _parse_usage: synthesize total_tokens when the provider omits it
# ---------------------------------------------------------------------------


def test_parse_usage_synthesizes_total_tokens_when_omitted() -> None:
    """total_tokens=0 with non-zero prompt/completion is synthesized.

    Regression: litellm leaves total_tokens at 0 when a provider omits it; the
    zero caused LLMUsage.__add__ to drop the per-request breakdown entry and
    contribute 0 to the cumulative total.
    """
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=7,
            total_tokens=0,  # provider omitted total
            prompt_tokens_details=None,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
            completion_tokens_details=None,
        )
    )

    usage = LiteLLM._parse_usage(response)
    assert usage is not None
    assert usage.total_tokens == 17

    # The synthesized total is non-zero, so the per-request breakdown survives
    # accumulation via LLMUsage.__add__.
    accumulated = LLMUsage() + usage
    assert accumulated.total_tokens == 17
    assert len(accumulated.usage) == 1


def test_parse_usage_keeps_provider_total_when_present() -> None:
    """An explicit total_tokens is never overwritten by the synthesis."""
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=7,
            total_tokens=20,  # provider supplied an explicit (non-sum) total
            prompt_tokens_details=None,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
            completion_tokens_details=None,
        )
    )

    usage = LiteLLM._parse_usage(response)
    assert usage is not None
    assert usage.total_tokens == 20


def test_parse_usage_total_stays_zero_when_no_token_counts() -> None:
    """An all-zero usage frame keeps total_tokens at 0 (nothing to synthesize)."""
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=None,
            cache_read_input_tokens=None,
            cache_creation_input_tokens=None,
            completion_tokens_details=None,
        )
    )

    usage = LiteLLM._parse_usage(response)
    assert usage is not None
    assert usage.total_tokens == 0


# ---------------------------------------------------------------------------
# Streaming: empty priming content/reasoning deltas emit no phantom part
# ---------------------------------------------------------------------------


def _content_chunk(text: str | None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=None,
                delta=SimpleNamespace(content=text, tool_calls=None, reasoning_content=None),
            )
        ]
    )


def _reasoning_chunk(reasoning: str | None, content: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=None,
                delta=SimpleNamespace(content=content, tool_calls=None, reasoning_content=reasoning),
            )
        ]
    )


def _stop_chunk() -> SimpleNamespace:
    return SimpleNamespace(
        id="resp1",
        model="gpt-4o-mini",
        choices=[
            SimpleNamespace(
                finish_reason="stop",
                delta=SimpleNamespace(content=None, tool_calls=None, reasoning_content=None),
            )
        ],
        usage=None,
    )


async def test_empty_priming_content_delta_emits_no_phantom_text_part() -> None:
    """An empty content delta ("") must not start a text part.

    Regression: the ``is not None``-only guard fired on the empty priming
    chunk, emitting a spurious part_start + empty part_delta and consuming the
    index-0 slot.
    """

    async def _chunks():
        yield _content_chunk("")  # priming chunk
        yield _content_chunk("hi")
        yield _stop_chunk()

    model = LiteLLM(model="gpt-4o-mini")
    events = [e async for e in model._stream(_chunks())]

    deltas = [e for e in events if e.type == "part_delta"]
    assert len(deltas) == 1
    assert deltas[0].delta == "hi"

    starts = [e for e in events if e.type == "part_start"]
    assert len(starts) == 1
    # The real text part keeps index 0 — the empty chunk did not steal a slot.
    assert starts[0].index == 0
    assert deltas[0].index == 0


async def test_empty_priming_reasoning_delta_emits_no_phantom_part() -> None:
    """An empty reasoning delta ("") must not start a reasoning part.

    Regression: same empty-priming bug in the reasoning branch shifted the
    text part to index 1 by consuming index 0 with a phantom reasoning part.
    """

    async def _chunks():
        yield _reasoning_chunk("")  # empty priming reasoning
        yield _content_chunk("answer")
        yield _stop_chunk()

    model = LiteLLM(model="gpt-4o-mini")
    events = [e async for e in model._stream(_chunks())]

    starts = [e for e in events if e.type == "part_start"]
    # Only the text part should start; the empty reasoning chunk is suppressed.
    assert len(starts) == 1
    assert starts[0].index == 0

    deltas = [e for e in events if e.type == "part_delta"]
    assert len(deltas) == 1
    assert deltas[0].delta == "answer"
    assert deltas[0].index == 0
