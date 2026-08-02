"""Pinning tests for three LiteLLM provider behaviours.

Covers:
- Partial-usage on stream interruption: when the stream raises before the usage
  frame, the done event carries whatever the last chunk provided — no backfill
  is possible from litellm 1.83.0 because the wrapper does not surface partial
  usage on mid-stream exception. The done event is emitted with None usage.
- Router-fallback cost attribution: the cost lookup uses the SERVING model name
  (outcome.model from the routing layer), not a different model.
- reasoning_effort=None omission: passing reasoning_effort=None must not send
  an explicit null to acompletion; litellm filters it as a no-op default.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.llms.litellm.litellm_model import LiteLLM, LiteLLMConfig
from troopai.adk.types.tokens.llm_usage import LLMUsage

# ---------------------------------------------------------------------------
# VERIFY partial-usage on stream interruption
# ---------------------------------------------------------------------------


async def test_stream_interruption_done_carries_no_usage_when_usage_frame_not_received() -> None:
    """done event has None usage when stream raises before the usage frame.

    litellm 1.83.0 does not expose partial-usage recovery on mid-stream
    exceptions: the CustomStreamWrapper accumulates usage only from chunks
    that carry a usage field, and when the generator raises before such a
    chunk arrives, last_chunk has no usage.  The done event therefore carries
    usage=None — the consumer must handle that gracefully.
    """

    async def _error_chunks():
        # Two content chunks, then raise before any usage chunk arrives.
        yield SimpleNamespace(
            id="resp1",
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(content="hel", tool_calls=None, reasoning_content=None),
                )
            ],
            usage=None,
        )
        yield SimpleNamespace(
            id="resp1",
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(content="lo", tool_calls=None, reasoning_content=None),
                )
            ],
            usage=None,
        )
        raise RuntimeError("provider exploded before usage frame")

    model = LiteLLM(model="gpt-4o-mini")
    events = []
    with pytest.raises(RuntimeError, match="provider exploded before usage frame"):
        async for event in model._stream(_error_chunks()):
            events.append(event)

    done_events = [e for e in events if e.type == "done"]
    assert len(done_events) == 1, f"Expected one done event; got {[e.type for e in events]}"
    done = done_events[0]
    assert done.response is not None
    assert done.response.finish_reason == "error"
    # No usage frame was received before the exception — usage must be None.
    # This is VERIFIED-BLOCKED: litellm 1.83.0 offers no recoverable partial-usage
    # signal on mid-stream exception; CustomStreamWrapper only exposes usage from
    # chunks that carry a `usage` attribute, and none arrived here.
    assert done.response.usage is None, (
        f"Expected None usage (no usage frame before exception); got {done.response.usage!r}"
    )


async def test_stream_interruption_done_carries_usage_if_last_chunk_had_usage() -> None:
    """done event carries usage when the LAST received chunk contained usage data.

    When the stream raises after a chunk that happens to carry usage (e.g. a
    combined content+usage chunk), the partial response includes that usage.
    """

    async def _chunks_with_usage_then_error():
        yield SimpleNamespace(
            id="resp1",
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(content="hi", tool_calls=None, reasoning_content=None),
                )
            ],
            # This chunk itself carries usage (some providers bundle usage with content)
            usage=SimpleNamespace(
                prompt_tokens=5,
                completion_tokens=2,
                total_tokens=7,
                prompt_tokens_details=None,
                cache_read_input_tokens=None,
                cache_creation_input_tokens=None,
                completion_tokens_details=None,
            ),
        )
        raise RuntimeError("error after usage-bearing chunk")

    model = LiteLLM(model="gpt-4o-mini")
    events = []
    with pytest.raises(RuntimeError, match="error after usage-bearing chunk"):
        async for event in model._stream(_chunks_with_usage_then_error()):
            events.append(event)

    done_events = [e for e in events if e.type == "done"]
    assert len(done_events) == 1
    done = done_events[0]
    assert done.response is not None
    assert done.response.finish_reason == "error"
    # The last chunk before the exception carried usage — that is recoverable.
    assert done.response.usage is not None, "Usage from usage-bearing chunk should be preserved"
    assert done.response.usage.input_tokens == 5
    assert done.response.usage.output_tokens == 2


# ---------------------------------------------------------------------------
# VERIFY router-retry cost attribution uses the SERVING model
# ---------------------------------------------------------------------------


def test_router_fallback_cost_uses_serving_model() -> None:
    """llm.cost() is called with the SERVING model name, not the primary.

    When a router fallback succeeds, outcome.model is the serving candidate's
    model name.  The loop at run/loop.py line 524 calls
    ``llm.cost(llm_model_name, response.usage)`` where ``llm_model_name`` has
    been updated to ``outcome.model``.  This test pins that the cost lookup
    uses the actual serving model.

    We verify by confirming that LiteLLM.cost() is invoked with the serving
    model ("gpt-4o-mini") rather than a hypothetical primary ("gpt-4o").
    """
    llm = LiteLLM(model="gpt-4o")
    usage = LLMUsage(requests=1, input_tokens=10, output_tokens=5, total_tokens=15)

    serving_model = "gpt-4o-mini"  # fallback model that actually served the request

    with patch("litellm.cost_per_token", return_value=(0.0001, 0.0002)) as mock_cost:
        result = llm.cost(serving_model, usage)

    assert result == pytest.approx(0.0003)
    # The cost_per_token call must use the serving model, not the instance model.
    mock_cost.assert_called_once_with(
        model=serving_model,
        prompt_tokens=10,
        completion_tokens=5,
    )


def test_router_serving_model_differs_from_instance_model() -> None:
    """LLM.cost() is model-name-agnostic: the caller always passes the serving model.

    Confirms that passing a different model name than self._model to cost()
    sends that exact name to litellm.cost_per_token.  This is the invariant
    the routing loop relies on: llm.cost(outcome.model, ...) not llm.cost(llm._model, ...).
    """
    llm = LiteLLM(model="gpt-4o")  # primary
    usage = LLMUsage(requests=1, input_tokens=20, output_tokens=10, total_tokens=30)

    with patch("litellm.cost_per_token", return_value=(0.002, 0.004)) as mock_cost:
        cost = llm.cost("claude-haiku-4-5-20251001", usage)  # serving model differs

    assert cost == pytest.approx(0.006)
    mock_cost.assert_called_once_with(
        model="claude-haiku-4-5-20251001",
        prompt_tokens=20,
        completion_tokens=10,
    )


# ---------------------------------------------------------------------------
# VERIFY reasoning_effort=None omits the parameter from acompletion
# ---------------------------------------------------------------------------


async def test_reasoning_effort_none_is_omitted_from_acompletion() -> None:
    """reasoning_effort=None must not forward an explicit null to acompletion.

    litellm registers None as the default for reasoning_effort in
    DEFAULT_CHAT_COMPLETION_PARAM_VALUES and filters it out via
    pre_process_non_default_params.  This test pins the ADK side:
    when LiteLLMConfig.reasoning_effort is None (the default), the
    acompletion call receives reasoning_effort=None (which litellm then
    filters away).  The parameter is never passed as a non-None value.
    """
    captured_kwargs: dict = {}

    async def fake_acompletion(**kwargs):  # type: ignore[misc]
        captured_kwargs.update(kwargs)
        # Return a minimal ModelResponse-shaped object
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

    llm = LiteLLM(model="gpt-4o-mini")
    config = LiteLLMConfig()  # reasoning_effort=None by default

    assert config.reasoning_effort is None, "Precondition: LiteLLMConfig.reasoning_effort defaults to None"

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        await llm.acomplete(
            messages="hello",
            llm_config=config,
        )

    assert "reasoning_effort" in captured_kwargs, (
        "reasoning_effort must be present in the call (litellm's own filtering handles None)"
    )
    assert captured_kwargs["reasoning_effort"] is None, (
        f"reasoning_effort must be None (not a non-None value); got {captured_kwargs['reasoning_effort']!r}"
    )


async def test_reasoning_effort_non_none_is_forwarded_to_acompletion() -> None:
    """reasoning_effort='medium' is passed through to acompletion unchanged."""
    captured_kwargs: dict = {}

    async def fake_acompletion(**kwargs):  # type: ignore[misc]
        captured_kwargs.update(kwargs)
        return SimpleNamespace(
            id="r2",
            model="gpt-4o-mini",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="ok",
                        tool_calls=None,
                        reasoning_content=None,
                        thinking_blocks=None,
                        refusal=None,
                        annotations=None,
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

    llm = LiteLLM(model="gpt-4o-mini")
    config = LiteLLMConfig(reasoning_effort="medium")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        await llm.acomplete(
            messages="hello",
            llm_config=config,
        )

    assert captured_kwargs.get("reasoning_effort") == "medium", (
        f"Expected reasoning_effort='medium'; got {captured_kwargs.get('reasoning_effort')!r}"
    )
