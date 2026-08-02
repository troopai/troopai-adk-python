"""Async retry decorator with configurable backoff + exception filtering.

Used by every backend that talks to a remote service: REST
hosted bridges, Docker daemon, Kubernetes API, cloud snapshot
stores. The chained-exception inspection covers cases where the
real cause is wrapped — ``urllib3``-via-``requests``-via-``boto3``
re-raises with ``__cause__``.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from collections.abc import Callable, Coroutine, Iterable
from enum import StrEnum
from typing import ParamSpec, TypeVar, cast

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_TRANSIENT_RETRY_BACKOFF",
    "DEFAULT_TRANSIENT_RETRY_INTERVAL_S",
    "DEFAULT_TRANSIENT_RETRY_MAX_ATTEMPT",
    "TRANSIENT_HTTP_STATUS_CODES",
    "BackoffStrategy",
    "exception_chain_contains_type",
    "exception_chain_has_status_code",
    "iter_exception_chain",
    "retry_async",
]

P = ParamSpec("P")
T = TypeVar("T")


class BackoffStrategy(StrEnum):
    """How a retry decorator computes the delay between attempts."""

    FIXED = "fixed"
    """Constant delay equal to ``interval`` for every retry."""

    LINEAR = "linear"
    """Delay grows linearly: ``interval * attempt``."""

    EXPONENTIAL = "exponential"
    """Delay doubles on each retry: ``interval * 2 ** (attempt - 1)``."""


DEFAULT_TRANSIENT_RETRY_INTERVAL_S: float = 0.25
"""Default first-retry delay in seconds."""

DEFAULT_TRANSIENT_RETRY_MAX_ATTEMPT: int = 3
"""Default maximum number of attempts (first call + 2 retries)."""

DEFAULT_TRANSIENT_RETRY_BACKOFF: BackoffStrategy = BackoffStrategy.EXPONENTIAL
"""Default backoff curve for transient retries."""

TRANSIENT_HTTP_STATUS_CODES: frozenset[int] = frozenset({500, 502, 503, 504})
"""HTTP status codes the framework treats as retryable by default."""


def iter_exception_chain(exc: BaseException) -> Iterable[BaseException]:
    """Walk ``exc`` and its ``__cause__`` / ``__context__`` chain.

    Mirrors CPython's traceback printer: prefer ``__cause__``; fall
    back to ``__context__`` only when ``__suppress_context__`` is
    False, so a cause/context explicitly hidden by ``raise ... from
    None`` is not crossed.

    Stops once an exception is revisited so a cyclic chain (rare
    but possible if the user calls ``raise ... from e`` on ``e``
    itself) cannot hang the iterator.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        yield current
        seen.add(id(current))
        # CPython exposes __cause__ and __context__ as Optional[BaseException]
        # on every BaseException, but getattr-with-default returns Any so the
        # cast pins the invariant the static checker can't see.
        cause = cast(BaseException | None, getattr(current, "__cause__", None))
        if cause is not None:
            current = cause
        elif not getattr(current, "__suppress_context__", False):
            current = cast(BaseException | None, getattr(current, "__context__", None))
        else:
            current = None


def exception_chain_contains_type(
    exc: BaseException,
    error_types: tuple[type[BaseException], ...],
) -> bool:
    """Return True iff any link in ``exc``'s chain is an instance of ``error_types``."""
    if len(error_types) == 0:
        return False
    return any(isinstance(candidate, error_types) for candidate in iter_exception_chain(exc))


def exception_chain_has_status_code(
    exc: BaseException,
    status_codes: set[int] | frozenset[int],
) -> bool:
    """Return True iff any chain link exposes one of ``status_codes``.

    Inspects ``.status_code``, ``.http_code``, and
    ``.response.status_code`` to cover the three conventions used
    by ``requests``, ``httpx``, and ``aiohttp``.
    """
    for candidate in iter_exception_chain(exc):
        for value in (
            getattr(candidate, "status_code", None),
            getattr(candidate, "http_code", None),
            getattr(getattr(candidate, "response", None), "status_code", None),
        ):
            if isinstance(value, int) and value in status_codes:
                return True
    return False


def retry_async(
    *,
    interval: float = DEFAULT_TRANSIENT_RETRY_INTERVAL_S,
    max_attempt: int = DEFAULT_TRANSIENT_RETRY_MAX_ATTEMPT,
    backoff: BackoffStrategy = DEFAULT_TRANSIENT_RETRY_BACKOFF,
    retry_if: Callable[..., bool],
    on_retry: Callable[..., object] | None = None,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> Callable[
    [Callable[P, Coroutine[object, object, T]]],
    Callable[P, Coroutine[object, object, T]],
]:
    """Decorator: retry an async function while ``retry_if`` says so.

    Args:
        interval: First-retry delay in seconds; subsequent delays
            scale per ``backoff``.
        max_attempt: Total attempts including the first one. ``1``
            disables retries.
        backoff: ``FIXED`` / ``LINEAR`` / ``EXPONENTIAL``.
        retry_if: ``(exc, *args, **kwargs) -> bool``. Return
            ``False`` to give up and re-raise.
        on_retry: Optional ``(exc, attempt, max_attempt, delay,
            *args, **kwargs) -> object`` hook. Awaited if the
            return value is awaitable.
        retry_on: Exception classes to consider for retry. Defaults
            to all ``Exception`` subclasses; callers SHOULD narrow
            this to specific transport / status-code errors so
            programmer bugs (TypeError, KeyError) propagate
            immediately instead of being retried.

    Raises:
        ValueError: ``max_attempt < 1`` or ``interval < 0``.
    """
    if max_attempt < 1:
        raise ValueError("retry_async: max_attempt must be >= 1")
    if interval < 0:
        raise ValueError("retry_async: interval must be >= 0")
    if backoff not in {
        BackoffStrategy.FIXED,
        BackoffStrategy.LINEAR,
        BackoffStrategy.EXPONENTIAL,
    }:
        raise ValueError(f"retry_async: backoff must be a BackoffStrategy member, got {backoff!r}")

    def _safe_retry_if(exc: BaseException, *args: object, **kwargs: object) -> bool:
        """Wrap the predicate so its own failures don't bury ``exc``."""
        try:
            return retry_if(exc, *args, **kwargs)
        except Exception as predicate_exc:
            logger.error(
                "retry_async: retry_if predicate raised %s while evaluating %s; treating as non-retryable",
                type(predicate_exc).__name__,
                type(exc).__name__,
            )
            return False

    def decorator(
        fn: Callable[P, Coroutine[object, object, T]],
    ) -> Callable[P, Coroutine[object, object, T]]:
        @functools.wraps(fn)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            for attempt in range(1, max_attempt + 1):
                try:
                    return await fn(*args, **kwargs)
                except retry_on as exc:
                    if attempt >= max_attempt or not _safe_retry_if(exc, *args, **kwargs):
                        raise

                    if backoff is BackoffStrategy.EXPONENTIAL:
                        delay_s = interval * (2 ** (attempt - 1))
                    elif backoff is BackoffStrategy.LINEAR:
                        delay_s = interval * attempt
                    else:
                        delay_s = interval

                    if on_retry is not None:
                        hook_result = on_retry(exc, attempt, max_attempt, delay_s, *args, **kwargs)
                        if inspect.isawaitable(hook_result):
                            await hook_result

                    logger.debug(
                        "retry_async: %s attempt %d/%d failed (%s); sleeping %.3fs",
                        fn.__name__,
                        attempt,
                        max_attempt,
                        exc,
                        delay_s,
                    )
                    await asyncio.sleep(delay_s)

            # Defensive: the loop must either return or re-raise above.
            # Use RuntimeError (not AssertionError): `assert` is stripped
            # under `python -O`, and this check needs to fire.
            raise RuntimeError(
                f"retry_async: wrapped function {fn.__name__} did not return or raise "
                f"after {max_attempt} attempts — framework invariant violation."
            )

        # functools.wraps preserves the wrapped callable's signature at
        # runtime but pyright cannot follow ParamSpec/TypeVar through the
        # decorator — the cast pins the invariant we just established
        # by construction (wrapped's signature mirrors fn's by P/T).
        return cast(Callable[P, Coroutine[object, object, T]], wrapped)

    return decorator
