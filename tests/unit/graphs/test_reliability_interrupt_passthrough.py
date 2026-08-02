"""node_reliability: InterruptException propagates unchanged — not retried, not wrapped."""

from __future__ import annotations

import pytest

from troopai.adk.exceptions import GraphNodeTimeoutError, NodeRetriesExhaustedError
from troopai.adk.graphs import Interrupt, InterruptException
from troopai.adk.graphs.config import NodeRetryPolicy
from troopai.adk.run.node_reliability import run_node_with_reliability


async def test_interrupt_passthrough_not_retried_not_wrapped() -> None:
    """A node whose attempt raises InterruptException must surface that
    exception verbatim — no retry, no GraphNodeTimeoutError wrap, no
    NodeRetriesExhaustedError wrap — even when the policy permits retries
    and a timeout. Attempts == 1 (no retries consumed)."""
    attempts = 0
    interrupt_obj = Interrupt(node_id="n1", question="approve?", kind="tool_approval")

    async def invoke() -> None:
        nonlocal attempts
        attempts += 1
        raise InterruptException(interrupt_obj)

    # Policy: retryable enabled, max_attempts=3, per-attempt timeout set.
    policy = NodeRetryPolicy(max_attempts=3, initial_backoff=0.001)
    timeout = 5.0

    with pytest.raises(InterruptException) as exc_info:
        await run_node_with_reliability(
            node_id="n1",
            policy=policy,
            timeout=timeout,
            invoke=invoke,  # type: ignore[arg-type]
        )
    assert exc_info.value.interrupt is interrupt_obj
    # Must NOT be wrapped in a reliability-layer error type
    assert not isinstance(exc_info.value, GraphNodeTimeoutError)
    assert not isinstance(exc_info.value, NodeRetriesExhaustedError)
    # Not retried — the wrapper let the very first InterruptException escape immediately
    assert attempts == 1


async def test_normal_failing_node_still_retries_unchanged() -> None:
    """Non-interrupt failures still consume retries and end with
    NodeRetriesExhaustedError when max_attempts is reached."""
    attempts = 0

    async def invoke() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("boom")

    policy = NodeRetryPolicy(max_attempts=2, initial_backoff=0.001)
    timeout = None

    with pytest.raises(NodeRetriesExhaustedError):
        await run_node_with_reliability(
            node_id="n2",
            policy=policy,
            timeout=timeout,
            invoke=invoke,  # type: ignore[arg-type]
        )
    assert attempts == 2  # retry behavior unchanged
