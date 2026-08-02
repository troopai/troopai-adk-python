"""Provider-agnostic retry loop for LLM calls.

Shared between :mod:`troopai.adk.llms.litellm.litellm_retry` and
:mod:`troopai.adk.llms.openai.openai_retry` (and any future provider
module). The loop logic — backoff, jitter, budget accounting, logging —
is identical across providers; only the classifier differs, because
each SDK raises a distinct exception hierarchy.

This module exposes a single function:

- :func:`call_with_retry` — wraps an async callable in an
  exponential-backoff retry loop governed by an
  :class:`LLMRetryPolicy` and a caller-supplied classifier.

Streaming calls are **not** retried by this helper. Reconnecting
mid-stream would silently lose tokens or double-emit events, so
streaming errors are left to surface immediately.

Provider-specific classifiers live alongside their provider module:

- ``llms/litellm/litellm_retry.litellm_exception_to_kind``
- ``llms/openai/openai_retry.openai_exception_to_kind``
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Optional

from troopai.adk.types.llms import LLMRetryErrorKind, LLMRetryPolicy

logger = logging.getLogger(__name__)


async def call_with_retry[T](
    coro_fn: Callable[[], Awaitable[T]],
    policy: LLMRetryPolicy,
    classifier: Callable[[BaseException], Optional[LLMRetryErrorKind]],
    *,
    model: Optional[str] = None,
) -> T:
    """Invoke *coro_fn* with exponential-backoff retries.

    Args:
        coro_fn: Zero-argument async callable producing one LLM
            response. The wrapper awaits it, classifies any raised
            exception via *classifier*, and retries according to
            *policy*.
        policy: Retry policy controlling backoff, jitter, and the
            set of error kinds that are retried.
        classifier: Maps a caught exception onto a framework-level
            :data:`LLMRetryErrorKind`, or ``None`` when the exception
            is permanent (authentication, bad request, content filter,
            etc.). Each provider module supplies its own.
        model: Optional model identifier for log messages.

    Returns:
        The awaited result of *coro_fn*.

    Raises:
        Any exception raised by *coro_fn* that the classifier maps to
        ``None``, or whose retryable budget has been exhausted.
    """
    attempt = 0
    while True:
        try:
            return await coro_fn()
        except Exception as exc:
            # `Exception` excludes KeyboardInterrupt, SystemExit, and
            # asyncio.CancelledError (Python 3.8+: CancelledError is a
            # BaseException, not Exception) — those propagate immediately
            # without entering the classifier or retry-policy logic.
            kind = classifier(exc)
            if kind is None or not policy.should_retry(kind, attempt):
                raise
            delay = policy.compute_delay(attempt)
            logger.warning(
                "LLM call failed (model=%s, kind=%s, attempt=%d/%d). Sleeping %.2fs before retry.",
                model if model is not None else "?",
                kind,
                attempt + 1,
                policy.max_retries,
                delay,
            )
            await asyncio.sleep(delay)
            attempt += 1
