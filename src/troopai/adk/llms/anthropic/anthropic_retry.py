"""Anthropic-specific exception classification + retry wrapper.

Thin layer over :func:`troopai.adk.llms.retry.call_with_retry` that
supplies the Anthropic SDK's exception classifier. The generic loop
(backoff, jitter, budget accounting) lives in
:mod:`troopai.adk.llms.retry` — this module only translates the
``anthropic`` SDK's exception hierarchy onto the framework's three
retryable categories (``"rate_limit"`` / ``"timeout"`` /
``"server_error"``).

Classification rules (match the Anthropic SDK's public exception types —
https://github.com/anthropics/anthropic-sdk-python#error-handling):

- :class:`anthropic.RateLimitError` (HTTP 429) → ``"rate_limit"``
- :class:`anthropic.APITimeoutError` → ``"timeout"``
- :class:`anthropic.APIConnectionError` → ``"server_error"``
  (network-level — TCP reset, TLS handshake failure, etc.)
- :class:`anthropic.APIStatusError` — classified by HTTP status:

  =====================  =====================
  Status code            Kind
  =====================  =====================
  ``429``                ``"rate_limit"``
  ``529``                ``"rate_limit"``  (Anthropic overload)
  ``408`` / ``504``      ``"timeout"``
  ``500`` / ``502`` / ``503``  ``"server_error"``
  other ``4xx``          ``None`` (permanent)
  =====================  =====================

Anthropic's :class:`OverloadedError` (HTTP 529) is a subclass of
:class:`APIStatusError`, so the status-code branch handles it without
a dedicated import — the ``OverloadedError`` symbol is not re-exported
at the ``anthropic`` package root and would force an
``anthropic._exceptions`` private import to reach it.

Any other exception maps to ``None`` — the caller lets it propagate.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from troopai.adk.llms.retry import call_with_retry as _generic_call_with_retry

if TYPE_CHECKING:
    from troopai.adk.types.llms import LLMRetryErrorKind, LLMRetryPolicy


def anthropic_exception_to_kind(exc: BaseException) -> LLMRetryErrorKind | None:
    """Map an ``anthropic`` SDK exception to a framework retry kind.

    Returns ``None`` for permanent errors (authentication, bad
    request, content filter, …). The generic retry loop treats
    ``None`` as "do not retry — re-raise".
    """
    from anthropic import (
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
        # 529 = Anthropic's overload signal. OverloadedError is a
        # subclass of APIStatusError; its symbol is not re-exported
        # at the anthropic package root, so we route it via the
        # status code rather than reach into anthropic._exceptions.
        if status in (429, 529):
            return "rate_limit"
        if status in (408, 504):
            return "timeout"
        if status in (500, 502, 503):
            return "server_error"
        return None
    return None


async def call_with_retry[T](
    coro_fn: Callable[[], Awaitable[T]],
    policy: LLMRetryPolicy,
    *,
    model: str | None = None,
) -> T:
    """Thin Anthropic-specific shim over the generic retry loop.

    See :func:`troopai.adk.llms.retry.call_with_retry` for the loop
    contract; this wrapper only injects
    :func:`anthropic_exception_to_kind` as the classifier so provider
    code can stay classifier-agnostic.
    """
    return await _generic_call_with_retry(
        coro_fn,
        policy,
        anthropic_exception_to_kind,
        model=model,
    )
