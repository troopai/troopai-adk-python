"""Tests for LLMRetryPolicy and litellm_retry wrapper."""

import asyncio
from typing import Any

import pytest

from troopai.adk.llms.litellm.litellm_retry import (
    call_with_retry,
    litellm_exception_to_kind,
)
from troopai.adk.types.llms import LLMRetryPolicy

# ── Policy dataclass ─────────────────────────────────────────────────


class TestLLMRetryPolicyDefaults:
    def test_defaults(self) -> None:
        """Defaults must be cost-conservative: max_retries=1, retry_on=rate_limit only."""
        policy = LLMRetryPolicy()
        assert policy.max_retries == 1
        assert policy.initial_delay == 0.5
        assert policy.max_delay == 30.0
        assert policy.multiplier == 2.0
        assert policy.jitter is True
        # Default retry_on must be the narrowest useful set, not None (all kinds).
        assert policy.retry_on == frozenset(["rate_limit"]), (
            f"Expected frozenset(['rate_limit']), got {policy.retry_on!r}"
        )

    def test_is_frozen(self) -> None:
        policy = LLMRetryPolicy()
        try:
            policy.max_retries = 99  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError("LLMRetryPolicy should be frozen")


class TestShouldRetry:
    def test_retries_rate_limit_only_by_default(self) -> None:
        """Bare LLMRetryPolicy() retries only rate_limit errors (cost-conservative default)."""
        policy = LLMRetryPolicy()
        assert policy.should_retry("rate_limit", 0) is True
        assert policy.should_retry("server_error", 0) is False
        assert policy.should_retry("timeout", 0) is False

    def test_retries_all_kinds_when_retry_on_is_none(self) -> None:
        """Explicitly passing retry_on=None retries all error kinds."""
        policy = LLMRetryPolicy(retry_on=None)
        assert policy.should_retry("rate_limit", 0) is True
        assert policy.should_retry("server_error", 0) is True
        assert policy.should_retry("timeout", 0) is True

    def test_honours_retry_on_whitelist(self) -> None:
        policy = LLMRetryPolicy(retry_on=frozenset(["rate_limit"]))
        assert policy.should_retry("rate_limit", 0) is True
        assert policy.should_retry("server_error", 0) is False
        assert policy.should_retry("timeout", 0) is False

    def test_stops_at_max_retries(self) -> None:
        policy = LLMRetryPolicy(max_retries=2)
        assert policy.should_retry("rate_limit", 0) is True
        assert policy.should_retry("rate_limit", 1) is True
        assert policy.should_retry("rate_limit", 2) is False  # budget used

    def test_max_retries_zero_disables(self) -> None:
        policy = LLMRetryPolicy(max_retries=0)
        assert policy.should_retry("rate_limit", 0) is False


class TestComputeDelay:
    def test_no_jitter_exponential(self) -> None:
        policy = LLMRetryPolicy(initial_delay=1.0, multiplier=2.0, max_delay=100.0, jitter=False)
        assert policy.compute_delay(0) == pytest.approx(1.0)
        assert policy.compute_delay(1) == pytest.approx(2.0)
        assert policy.compute_delay(2) == pytest.approx(4.0)
        assert policy.compute_delay(3) == pytest.approx(8.0)

    def test_no_jitter_respects_max_delay(self) -> None:
        policy = LLMRetryPolicy(initial_delay=1.0, multiplier=10.0, max_delay=5.0, jitter=False)
        assert policy.compute_delay(0) == pytest.approx(1.0)
        assert policy.compute_delay(1) == pytest.approx(5.0)  # capped
        assert policy.compute_delay(2) == pytest.approx(5.0)  # still capped

    def test_jitter_bounded(self) -> None:
        policy = LLMRetryPolicy(initial_delay=2.0, multiplier=2.0, max_delay=10.0, jitter=True)
        for attempt in range(5):
            delay = policy.compute_delay(attempt)
            assert delay >= 0.0
            assert delay <= 10.0


# ── Exception → kind mapping ────────────────────────────────────────


class TestLitellmExceptionToKind:
    def test_asyncio_timeout_is_timeout_kind(self) -> None:
        err = TimeoutError("slow")
        assert litellm_exception_to_kind(err) == "timeout"

    def test_unknown_exception_returns_none(self) -> None:
        assert litellm_exception_to_kind(ValueError("nope")) is None

    def test_litellm_rate_limit_error(self) -> None:
        try:
            import litellm  # type: ignore[import-untyped]
        except ImportError:
            pytest.skip("litellm not installed")
        rate_limit_cls = getattr(litellm, "RateLimitError", None)
        if rate_limit_cls is None:
            pytest.skip("RateLimitError not exposed by installed litellm")
        # Minimal constructor that works across litellm versions.
        try:
            err = rate_limit_cls(message="429", model="gpt-4o-mini", llm_provider="openai")
        except TypeError:
            pytest.skip("RateLimitError constructor signature changed")
        assert litellm_exception_to_kind(err) == "rate_limit"


# ── call_with_retry loop ─────────────────────────────────────────────


class TestCallWithRetry:
    @pytest.mark.asyncio
    async def test_success_no_retries(self) -> None:
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        policy = LLMRetryPolicy(max_retries=3, initial_delay=0.001, jitter=False)
        result = await call_with_retry(op, policy)
        assert result == "ok"
        assert calls == 1

    @pytest.mark.asyncio
    async def test_retries_on_retryable_error(self) -> None:
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise TimeoutError("slow")
            return "ok"

        # Explicitly include timeout in retry_on since the default is rate_limit-only.
        policy = LLMRetryPolicy(max_retries=5, initial_delay=0.001, max_delay=0.01, jitter=False, retry_on=None)
        result = await call_with_retry(op, policy)
        assert result == "ok"
        assert calls == 3

    @pytest.mark.asyncio
    async def test_does_not_retry_unknown_error(self) -> None:
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise ValueError("permanent")

        policy = LLMRetryPolicy(max_retries=5, initial_delay=0.001, jitter=False)
        with pytest.raises(ValueError):
            await call_with_retry(op, policy)
        assert calls == 1  # no retries

    @pytest.mark.asyncio
    async def test_exhausts_budget_then_raises(self) -> None:
        calls = 0

        async def op() -> Any:
            nonlocal calls
            calls += 1
            raise TimeoutError("slow")

        # Explicitly allow timeout so the budget is exhausted (default is rate_limit-only).
        policy = LLMRetryPolicy(max_retries=2, initial_delay=0.001, jitter=False, retry_on=None)
        with pytest.raises(asyncio.TimeoutError):
            await call_with_retry(op, policy)
        # 1 initial + 2 retries = 3 total calls.
        assert calls == 3

    @pytest.mark.asyncio
    async def test_retry_on_filter_skips_unlisted_kinds(self) -> None:
        calls = 0

        async def op() -> Any:
            nonlocal calls
            calls += 1
            raise TimeoutError("slow")

        # Whitelist only rate_limit → timeout errors become terminal.
        policy = LLMRetryPolicy(
            max_retries=5,
            initial_delay=0.001,
            jitter=False,
            retry_on=frozenset(["rate_limit"]),
        )
        with pytest.raises(asyncio.TimeoutError):
            await call_with_retry(op, policy)
        assert calls == 1
