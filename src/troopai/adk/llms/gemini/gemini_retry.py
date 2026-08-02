"""Gemini-specific exception classification + retry wrapper.

Thin layer over :func:`troopai.adk.llms.retry.call_with_retry` that
supplies the Gemini SDK's exception classifier. The generic loop
(backoff, jitter, budget accounting) lives in
:mod:`troopai.adk.llms.retry` — this module only translates the
``google.genai.errors`` exception hierarchy onto the framework's
three retryable categories (``"rate_limit"`` / ``"timeout"`` /
``"server_error"``).

Classification rules (per the Gemini API troubleshooting doc —
https://ai.google.dev/gemini-api/docs/troubleshooting):

- :class:`google.genai.errors.ClientError` with HTTP 429 → ``"rate_limit"``
- :class:`google.genai.errors.ClientError` with other 4xx → ``None`` (permanent)
- :class:`google.genai.errors.ServerError` with HTTP 504 → ``"timeout"``
- :class:`google.genai.errors.ServerError` with other 5xx → ``"server_error"``

Any other exception maps to ``None`` — the caller lets it propagate.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from troopai.adk.llms.retry import call_with_retry as _generic_call_with_retry

if TYPE_CHECKING:
    from troopai.adk.types.llms import LLMRetryErrorKind, LLMRetryPolicy


def gemini_exception_to_kind(exc: BaseException) -> LLMRetryErrorKind | None:
    """Map a ``google.genai.errors`` exception to a framework retry kind.

    Returns ``None`` for permanent errors (authentication, bad
    request, content filter, invalid model). The generic retry loop
    treats ``None`` as "do not retry — re-raise".
    """
    from google.genai.errors import ClientError, ServerError

    if isinstance(exc, ClientError):
        # 429 RESOURCE_EXHAUSTED — rate limit. Other 4xx are
        # permanent (auth, bad request, model access denied).
        status = getattr(exc, "code", None)
        if status == 429:
            return "rate_limit"
        return None
    if isinstance(exc, ServerError):
        # 504 DEADLINE_EXCEEDED → timeout; everything else 5xx is a
        # generic server error eligible for backoff retry.
        status = getattr(exc, "code", None)
        if status == 504:
            return "timeout"
        return "server_error"
    return None


async def call_with_retry[T](
    coro_fn: Callable[[], Awaitable[T]],
    policy: LLMRetryPolicy,
    *,
    model: str | None = None,
) -> T:
    """Thin Gemini-specific shim over the generic retry loop.

    See :func:`troopai.adk.llms.retry.call_with_retry` for the loop
    contract; this wrapper only injects
    :func:`gemini_exception_to_kind` as the classifier so provider
    code can stay classifier-agnostic.
    """
    return await _generic_call_with_retry(
        coro_fn,
        policy,
        gemini_exception_to_kind,
        model=model,
    )
