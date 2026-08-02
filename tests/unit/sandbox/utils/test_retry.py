"""Tests for ``troopai.adk.sandbox.utils.retry``."""

from __future__ import annotations

import pytest

from troopai.adk.sandbox.utils.retry import (
    TRANSIENT_HTTP_STATUS_CODES,
    BackoffStrategy,
    exception_chain_contains_type,
    exception_chain_has_status_code,
    iter_exception_chain,
    retry_async,
)


class TransientError(Exception):
    pass


def _always(_exc: BaseException, *_args: object, **_kwargs: object) -> bool:
    return True


class TestRetryAsync:
    async def test_returns_first_success(self) -> None:
        calls = {"n": 0}

        @retry_async(max_attempt=3, interval=0, retry_if=_always)
        async def op() -> str:
            calls["n"] += 1
            return "ok"

        assert await op() == "ok"
        assert calls["n"] == 1

    async def test_retries_then_succeeds(self) -> None:
        calls = {"n": 0}

        @retry_async(max_attempt=3, interval=0, retry_if=_always)
        async def op() -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                raise TransientError("nope")
            return "ok"

        assert await op() == "ok"
        assert calls["n"] == 3

    async def test_gives_up_after_max_attempt(self) -> None:
        calls = {"n": 0}

        @retry_async(max_attempt=2, interval=0, retry_if=_always)
        async def op() -> str:
            calls["n"] += 1
            raise TransientError("never works")

        with pytest.raises(TransientError):
            await op()
        assert calls["n"] == 2

    async def test_skips_retry_when_predicate_false(self) -> None:
        calls = {"n": 0}

        @retry_async(max_attempt=5, interval=0, retry_if=lambda *_a, **_k: False)
        async def op() -> str:
            calls["n"] += 1
            raise TransientError("bad")

        with pytest.raises(TransientError):
            await op()
        assert calls["n"] == 1

    async def test_on_retry_hook_fires(self) -> None:
        observed: list[int] = []

        def hook(_exc: BaseException, attempt: int, _max: int, _delay: float, *_a: object, **_k: object) -> None:
            observed.append(attempt)

        @retry_async(max_attempt=3, interval=0, retry_if=_always, on_retry=hook)
        async def op() -> str:
            raise TransientError("bad")

        with pytest.raises(TransientError):
            await op()
        assert observed == [1, 2]

    async def test_async_on_retry_hook_awaited(self) -> None:
        observed: list[int] = []

        async def hook(_exc: BaseException, attempt: int, _max: int, _delay: float, *_a: object, **_k: object) -> None:
            observed.append(attempt)

        @retry_async(max_attempt=2, interval=0, retry_if=_always, on_retry=hook)
        async def op() -> str:
            raise TransientError("bad")

        with pytest.raises(TransientError):
            await op()
        assert observed == [1]

    def test_invalid_max_attempt(self) -> None:
        with pytest.raises(ValueError, match="max_attempt"):
            retry_async(max_attempt=0, retry_if=_always)

    def test_invalid_interval(self) -> None:
        with pytest.raises(ValueError, match="interval"):
            retry_async(interval=-1.0, retry_if=_always)


def test_iter_exception_chain_walks_cause_and_context() -> None:
    e1 = ValueError("root")
    try:
        try:
            raise e1
        except ValueError as orig:
            raise RuntimeError("middle") from orig
    except RuntimeError as outer:
        chain = list(iter_exception_chain(outer))

    types = [type(e) for e in chain]
    assert types == [RuntimeError, ValueError]


def test_iter_exception_chain_stops_at_suppressed_context() -> None:
    """``raise ... from None`` sets __suppress_context__; the walker must
    not cross into the suppressed context (mirrors CPython's printer)."""
    try:
        try:
            raise TransientError("transient")
        except TransientError:
            raise RuntimeError("wrapped") from None
    except RuntimeError as outer:
        chain = list(iter_exception_chain(outer))

    types = [type(e) for e in chain]
    assert types == [RuntimeError]


def test_exception_chain_contains_type_skips_suppressed_context() -> None:
    """A type hidden by ``from None`` must not be matched in the chain."""
    try:
        try:
            raise TransientError("transient")
        except TransientError:
            raise RuntimeError("wrapped") from None
    except RuntimeError as outer:
        assert not exception_chain_contains_type(outer, (TransientError,))
        assert exception_chain_contains_type(outer, (RuntimeError,))


def test_exception_chain_has_status_code_skips_suppressed_context() -> None:
    """A status-bearing exc hidden by ``from None`` must not trigger a match."""

    class HttpEx(Exception):
        status_code = 503

    try:
        try:
            raise HttpEx("transient")
        except HttpEx:
            raise RuntimeError("wrapped") from None
    except RuntimeError as outer:
        assert not exception_chain_has_status_code(outer, TRANSIENT_HTTP_STATUS_CODES)


def test_exception_chain_contains_type() -> None:
    e = RuntimeError("x")
    assert exception_chain_contains_type(e, (RuntimeError,))
    assert not exception_chain_contains_type(e, (KeyError,))
    assert not exception_chain_contains_type(e, ())


def test_exception_chain_has_status_code() -> None:
    class HttpEx(Exception):
        status_code = 503

    assert exception_chain_has_status_code(HttpEx(), TRANSIENT_HTTP_STATUS_CODES)
    assert not exception_chain_has_status_code(HttpEx(), frozenset({400}))


def test_backoff_strategy_enum_values() -> None:
    assert BackoffStrategy.FIXED.value == "fixed"
    assert BackoffStrategy.LINEAR.value == "linear"
    assert BackoffStrategy.EXPONENTIAL.value == "exponential"
    assert str(BackoffStrategy.EXPONENTIAL) == "exponential"
