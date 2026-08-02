"""Feature 3: Timeout retryability.

Tests that:
- Without a retry policy (max_attempts=1), a timeout raises
  GraphNodeTimeoutError immediately — no behaviour change.
- With max_attempts>1 and an intermediate timeout, the attempt is retried.
- After all retry attempts are exhausted by timeouts, GraphNodeTimeoutError
  is raised with attempts == max_attempts.
- retry_on filtering: when retry_on excludes TimeoutError, intermediate
  timeouts are NOT retried (terminal immediately).
- Non-timeout retryable errors still retry as before.
- The existing test_non_retryable_timeout_still_wraps contract is preserved:
  a timeout on the FINAL attempt always raises GraphNodeTimeoutError.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator

import pytest

from troopai.adk.exceptions import GraphNodeTimeoutError
from troopai.adk.graphs.config import NodeRetryPolicy
from troopai.adk.run.node_reliability import run_node_with_reliability


@contextlib.asynccontextmanager
async def _fake_timeout(_seconds: float) -> AsyncGenerator[None, None]:
    """Replacement for asyncio.timeout that always raises TimeoutError."""
    raise TimeoutError("simulated timeout")
    yield  # pragma: no cover — makes this an async generator


class _NoSleep:
    async def __call__(self, _d: float) -> None:
        return None


class TestTimeoutRetryability:
    async def test_no_retry_policy_timeout_is_terminal(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """max_attempts=1: timeout raises GraphNodeTimeoutError immediately."""
        monkeypatch.setattr("asyncio.sleep", _NoSleep())
        monkeypatch.setattr("asyncio.timeout", _fake_timeout)

        call_count = 0

        async def invoke() -> None:
            nonlocal call_count
            call_count += 1

        with pytest.raises(GraphNodeTimeoutError) as ei:
            await run_node_with_reliability(
                node_id="n",
                policy=NodeRetryPolicy(max_attempts=1),
                timeout=0.01,
                invoke=invoke,  # type: ignore[arg-type]
            )
        assert ei.value.node_id == "n"
        assert ei.value.attempts == 1
        # Only one attempt — no retries with max_attempts=1
        assert call_count == 0  # timeout fires before invoke body runs

    async def test_with_retry_policy_timeout_is_retried(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """max_attempts=3: intermediate timeouts are retried; success on 3rd."""
        sleeps: list[float] = []

        async def record_sleep(d: float) -> None:
            sleeps.append(d)

        monkeypatch.setattr("asyncio.sleep", record_sleep)

        attempt_n = 0

        @contextlib.asynccontextmanager
        async def fake_timeout_succeeds_on_third(
            _seconds: float,
        ) -> AsyncGenerator[None, None]:
            nonlocal attempt_n
            attempt_n += 1
            if attempt_n < 3:
                raise TimeoutError("simulated")
            yield  # third attempt: do not timeout

        monkeypatch.setattr("asyncio.timeout", fake_timeout_succeeds_on_third)

        from troopai.adk.orchestration.executable import NodeResult

        async def invoke() -> NodeResult:
            return NodeResult(output="ok")

        result = await run_node_with_reliability(
            node_id="n",
            policy=NodeRetryPolicy(max_attempts=3, initial_backoff=0.1),
            timeout=0.01,
            invoke=invoke,
        )
        assert result.output == "ok"
        assert len(sleeps) == 2  # retried twice before succeeding

    async def test_all_timeouts_raises_graph_node_timeout_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When every attempt times out, GraphNodeTimeoutError is raised with
        attempts == max_attempts."""
        monkeypatch.setattr("asyncio.sleep", _NoSleep())
        monkeypatch.setattr("asyncio.timeout", _fake_timeout)

        async def invoke() -> None:
            pass  # pragma: no cover

        with pytest.raises(GraphNodeTimeoutError) as ei:
            await run_node_with_reliability(
                node_id="n",
                policy=NodeRetryPolicy(max_attempts=3, initial_backoff=0.001),
                timeout=0.01,
                invoke=invoke,  # type: ignore[arg-type]
            )
        assert ei.value.attempts == 3

    async def test_retry_on_excludes_timeout_no_intermediate_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """retry_on=(ValueError,): TimeoutError is not retryable → terminal immediately."""
        monkeypatch.setattr("asyncio.sleep", _NoSleep())
        monkeypatch.setattr("asyncio.timeout", _fake_timeout)

        async def invoke() -> None:
            pass  # pragma: no cover

        with pytest.raises(GraphNodeTimeoutError) as ei:
            await run_node_with_reliability(
                node_id="n",
                policy=NodeRetryPolicy(max_attempts=3, retry_on=(ValueError,)),
                timeout=0.01,
                invoke=invoke,  # type: ignore[arg-type]
            )
        # Only attempt 1 ran (non-retryable timeout → break immediately)
        assert ei.value.attempts == 1

    async def test_timeout_on_final_attempt_always_wraps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On the final attempt a timeout ALWAYS raises GraphNodeTimeoutError,
        even when retry_on would not include TimeoutError."""
        monkeypatch.setattr("asyncio.sleep", _NoSleep())

        attempt_n = 0

        @contextlib.asynccontextmanager
        async def timeout_on_last(_seconds: float) -> AsyncGenerator[None, None]:
            nonlocal attempt_n
            attempt_n += 1
            if attempt_n == 2:  # only the second (= last) attempt times out
                raise TimeoutError("final timeout")
            yield  # first attempt: raise a retryable non-timeout

        monkeypatch.setattr("asyncio.timeout", timeout_on_last)

        async def invoke() -> None:
            raise ValueError("non-timeout retryable")

        with pytest.raises(GraphNodeTimeoutError) as ei:
            await run_node_with_reliability(
                node_id="n",
                policy=NodeRetryPolicy(max_attempts=2, initial_backoff=0.001),
                timeout=0.01,
                invoke=invoke,  # type: ignore[arg-type]
            )
        assert ei.value.attempts == 2

    async def test_non_timeout_retries_still_work(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Non-timeout retryable errors still retry as before."""
        monkeypatch.setattr("asyncio.sleep", _NoSleep())

        call_count = 0

        async def invoke():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            from troopai.adk.orchestration.executable import NodeResult

            return NodeResult(output="ok")

        result = await run_node_with_reliability(
            node_id="n",
            policy=NodeRetryPolicy(max_attempts=3, initial_backoff=0.001),
            timeout=None,
            invoke=invoke,
        )
        assert result.output == "ok"
        assert call_count == 3
