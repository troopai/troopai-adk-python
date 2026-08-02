"""Cross-provider streaming-error contract.

When a provider stream errors mid-stream, a consumer awaiting the terminal
``done`` event (usage flush, transcript finalization) must not be left
hanging while the exception surfaces elsewhere.
:func:`stream_with_error_contract` wraps a provider ``_stream`` generator so a
mid-stream error emits a terminal ``done`` event with ``finish_reason="error"``
and THEN re-raises — the error is never swallowed.

The native-provider implementations (Anthropic / Gemini / OpenAI Chat
Completions) compose this wrapper at their ``acomplete`` streaming-return site.
The LiteLLM provider implements the same observable contract inline so it can
also preserve the partial response on error; this wrapper's synthesized error
``done`` carries an empty parts list (the terminal signal is what the contract
requires — the success path still yields the inner generator's rich ``done``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from troopai.adk.types.responses.llm_response import LLMResponse, LLMStreamEvent

if TYPE_CHECKING:
    import logging

__all__ = ["stream_with_error_contract"]


async def stream_with_error_contract(
    inner: AsyncIterator[LLMStreamEvent],
    *,
    model: str,
    logger: logging.Logger,
) -> AsyncIterator[LLMStreamEvent]:
    """Forward ``inner``'s events; on a mid-stream error emit a terminal ``done`` then re-raise.

    Args:
        inner: The provider ``_stream`` async generator to forward.
        model: Model identifier, recorded on the synthesized error ``done``
            response and in the log line.
        logger: The provider module's logger.

    Yields:
        Every event from ``inner`` unchanged. On a mid-stream ``Exception``,
        one additional terminal ``LLMStreamEvent(type="done")`` whose response
        carries ``finish_reason="error"`` — emitted before the exception is
        re-raised.

    Raises:
        Exception: Re-raised after the terminal ``done`` is emitted, so the
            failure still propagates. ``CancelledError`` / ``KeyboardInterrupt``
            / ``SystemExit`` (``BaseException``, not ``Exception``) are NOT
            caught, so cancellation propagates untouched and is never masked.
    """
    try:
        async for event in inner:
            yield event
    except Exception as exc:
        logger.error("LLM stream failed mid-stream: model=%s: %s", model, exc)
        yield LLMStreamEvent(
            type="done",
            response=LLMResponse(
                response_id="",
                model=model,
                response=[],
                usage=None,
                finish_reason="error",
            ),
        )
        raise
