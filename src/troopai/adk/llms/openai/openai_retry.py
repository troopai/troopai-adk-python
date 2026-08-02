"""OpenAI-specific exception classification + retry wrapper.

Thin layer over :func:`troopai.adk.llms.retry.call_with_retry` that
supplies the OpenAI SDK's exception classifier. The generic loop
(backoff, jitter, budget accounting) lives in
:mod:`troopai.adk.llms.retry` — this module only translates the
``openai`` SDK's exception hierarchy onto the framework's three
retryable categories (``"rate_limit"`` / ``"timeout"`` /
``"server_error"``).

Classification rules (match the OpenAI SDK's public exception types —
https://github.com/openai/openai-python#handling-errors):

- :class:`openai.RateLimitError` → ``"rate_limit"``
- :class:`openai.APITimeoutError` → ``"timeout"``
- :class:`openai.APIConnectionError` → ``"server_error"``
  (network-level — TCP reset, TLS handshake failure, etc.)
- :class:`openai.APIStatusError` — classified by HTTP status:

  =====================  =====================
  Status code            Kind
  =====================  =====================
  ``429``                ``"rate_limit"``
  ``408``                ``"timeout"``
  ``500`` / ``502`` / ``503`` / ``504`` / ``529``  ``"server_error"``
  other ``4xx``          ``None`` (permanent)
  =====================  =====================

Any other exception maps to ``None`` — the caller lets it propagate.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Optional

from troopai.adk.llms.retry import call_with_retry as _generic_call_with_retry

if TYPE_CHECKING:
    from troopai.adk.types.llms import LLMRetryErrorKind, LLMRetryPolicy


def openai_exception_to_kind(exc: BaseException) -> Optional[LLMRetryErrorKind]:
    """Map an ``openai`` SDK exception to a framework retry kind.

    Returns ``None`` for permanent errors (authentication, bad request,
    content filter, …). The generic retry loop treats ``None`` as
    "do not retry — re-raise".
    """
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        RateLimitError,
    )

    if isinstance(exc, RateLimitError):
        return "rate_limit"
    if isinstance(exc, APITimeoutError):
        return "timeout"
    if isinstance(exc, APIConnectionError):
        return "server_error"
    if isinstance(exc, APIStatusError):
        status = getattr(exc, "status_code", None)
        if status == 429:
            return "rate_limit"
        if status == 408:
            return "timeout"
        if status in (500, 502, 503, 504, 529):
            return "server_error"
        return None
    return None


async def call_with_retry[T](
    coro_fn: Callable[[], Awaitable[T]],
    policy: LLMRetryPolicy,
    *,
    model: Optional[str] = None,
) -> T:
    """Thin OpenAI-specific shim over the generic retry loop.

    See :func:`troopai.adk.llms.retry.call_with_retry` for the loop
    contract; this wrapper only injects :func:`openai_exception_to_kind`
    as the classifier so provider code can stay classifier-agnostic.
    """
    return await _generic_call_with_retry(
        coro_fn,
        policy,
        openai_exception_to_kind,
        model=model,
    )
