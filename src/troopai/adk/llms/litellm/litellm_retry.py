"""LiteLLM-specific retry wrapper.

Glue between the framework-level :class:`LLMRetryPolicy` and the
exceptions raised by the ``litellm`` package. Two public symbols:

- :func:`litellm_exception_to_kind` — classifies a caught exception
  into a framework-level :data:`LLMRetryErrorKind`, or returns
  ``None`` when the exception is permanent.
- :func:`call_with_retry` — thin wrapper that delegates to
  :func:`troopai.adk.llms.retry.call_with_retry` with the litellm
  classifier. Kept as a public re-export so existing call sites
  (``runner.py``, tests) do not have to pass the classifier
  themselves.

Streaming calls are **not** retried by this module. Reconnecting
mid-stream would silently lose tokens or double-emit events, so
streaming errors are left to surface immediately.

Provider docs:
    LiteLLM exceptions —
    https://docs.litellm.ai/docs/exception_mapping
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Optional

from troopai.adk.llms.retry import call_with_retry as _generic_call_with_retry
from troopai.adk.types.llms import LLMRetryErrorKind, LLMRetryPolicy


def litellm_exception_to_kind(exc: BaseException) -> Optional[LLMRetryErrorKind]:
    """Map a ``litellm`` exception onto an :data:`LLMRetryErrorKind`.

    Returns ``None`` for permanent failures that MUST NOT be retried
    (authentication, bad request, not found, content filter, etc.).
    """
    try:
        import litellm
    except ImportError:
        # litellm is a required dependency; if it is missing the LiteLLM
        # model class cannot be instantiated at all. Surfacing the import
        # error here makes the missing-dependency root cause visible instead
        # of silently defeating the retry policy for the entire session.
        raise

    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"

    # litellm raises typed exceptions for common HTTP failure modes.
    # Classify by class so upstream version bumps don't silently
    # change our behaviour.
    rate_limit_cls = getattr(litellm, "RateLimitError", None)
    if rate_limit_cls is not None and isinstance(exc, rate_limit_cls):
        return "rate_limit"

    service_unavailable_cls = getattr(litellm, "ServiceUnavailableError", None)
    if service_unavailable_cls is not None and isinstance(exc, service_unavailable_cls):
        return "server_error"

    internal_cls = getattr(litellm, "InternalServerError", None)
    if internal_cls is not None and isinstance(exc, internal_cls):
        return "server_error"

    api_error_cls = getattr(litellm, "APIError", None)
    if api_error_cls is not None and isinstance(exc, api_error_cls):
        status = getattr(exc, "status_code", None)
        if status is not None:
            if status == 408 or status == 504:
                return "timeout"
            if status == 429:
                return "rate_limit"
            if 500 <= status < 600:
                return "server_error"

    api_connection_cls = getattr(litellm, "APIConnectionError", None)
    if api_connection_cls is not None and isinstance(exc, api_connection_cls):
        return "server_error"

    timeout_cls = getattr(litellm, "Timeout", None)
    if timeout_cls is not None and isinstance(exc, timeout_cls):
        return "timeout"

    return None


async def call_with_retry[T](
    coro_fn: Callable[[], Awaitable[T]],
    policy: LLMRetryPolicy,
    *,
    model: Optional[str] = None,
) -> T:
    """Invoke *coro_fn* with exponential-backoff retries.

    Thin delegator that binds the litellm classifier to the generic
    retry loop in :mod:`troopai.adk.llms.retry`. Kept as a public
    symbol so existing callers (``runner``, tests) do not have to
    pass the classifier themselves.

    Args:
        coro_fn: Zero-argument async callable that produces one LLM
            response.
        policy: Retry policy controlling backoff, jitter, and the
            set of error kinds that are retried.
        model: Optional model identifier for log messages.

    Returns:
        The awaited result of *coro_fn*.

    Raises:
        Any exception raised by *coro_fn* that cannot be classified
        as retryable, or whose retryable budget has been exhausted.
    """
    return await _generic_call_with_retry(
        coro_fn,
        policy,
        litellm_exception_to_kind,
        model=model,
    )
