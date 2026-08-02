"""Tests for the provider-agnostic retry loop in llms/retry.py.

Covers the contract the generic loop adds on top of the policy
dataclass: classifier injection, delegation of should-retry decisions
to the policy, and budget-exhaustion re-raising.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import pytest

from troopai.adk.llms.retry import call_with_retry
from troopai.adk.types.llms import LLMRetryErrorKind, LLMRetryPolicy


def _timeout_classifier(exc: BaseException) -> Optional[LLMRetryErrorKind]:
    """Trivial classifier: TimeoutError → 'timeout', everything else → None."""
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    return None


def _rate_limit_classifier(exc: BaseException) -> Optional[LLMRetryErrorKind]:
    """Trivial classifier: ValueError → 'rate_limit'."""
    if isinstance(exc, ValueError):
        return "rate_limit"
    return None


class TestGenericCallWithRetry:
    async def test_success_no_retries(self) -> None:
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        policy = LLMRetryPolicy(max_retries=3, initial_delay=0.001, jitter=False)
        result = await call_with_retry(op, policy, _timeout_classifier)
        assert result == "ok"
        assert calls == 1

    async def test_classifier_drives_retry_decision(self) -> None:
        """Loop consults the classifier each attempt, not a hardcoded table."""
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ValueError("hit rate limit")
            return "ok"

        policy = LLMRetryPolicy(max_retries=5, initial_delay=0.001, jitter=False)
        # With the rate_limit classifier, ValueError is retryable.
        result = await call_with_retry(op, policy, _rate_limit_classifier)
        assert result == "ok"
        assert calls == 3

    async def test_classifier_none_does_not_retry(self) -> None:
        """A classifier returning None makes the exception terminal."""
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise ValueError("permanent for timeout classifier")

        policy = LLMRetryPolicy(max_retries=5, initial_delay=0.001, jitter=False)
        # Timeout classifier returns None for ValueError → no retry.
        with pytest.raises(ValueError):
            await call_with_retry(op, policy, _timeout_classifier)
        assert calls == 1

    async def test_budget_exhaustion_raises_last_exception(self) -> None:
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise TimeoutError("slow")

        # Explicitly include timeout so the budget is exhausted.
        policy = LLMRetryPolicy(max_retries=2, initial_delay=0.001, jitter=False, retry_on=frozenset(["timeout"]))
        with pytest.raises(asyncio.TimeoutError):
            await call_with_retry(op, policy, _timeout_classifier)
        # 1 initial + 2 retries = 3 total.
        assert calls == 3

    async def test_model_name_appears_in_log(self, caplog: pytest.LogCaptureFixture) -> None:
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("slow")
            return "ok"

        # Explicitly include timeout so the retry is attempted.
        policy = LLMRetryPolicy(max_retries=2, initial_delay=0.001, jitter=False, retry_on=frozenset(["timeout"]))
        with caplog.at_level("WARNING", logger="troopai.adk.llms.retry"):
            await call_with_retry(op, policy, _timeout_classifier, model="gpt-5.1")
        assert any("gpt-5.1" in record.getMessage() for record in caplog.records)

    async def test_retry_on_filter_skips_unlisted_kinds(self) -> None:
        """Loop respects policy.retry_on even when classifier returns a kind."""
        calls = 0

        async def op() -> str:
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
            await call_with_retry(op, policy, _timeout_classifier)
        assert calls == 1
