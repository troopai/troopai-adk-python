"""Unit tests for graph node reliability — exception types, policy
resolution, and the retry/timeout wrapper."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator

import pytest

from troopai.adk.exceptions import (
    GraphNodeTimeoutError,
    NodeRetriesExhaustedError,
    TroopAIError,
)


class _NoSleep:
    async def __call__(self, _d: float) -> None:
        return None


class TestReliabilityExceptions:
    def test_timeout_error_attrs_and_subclass(self) -> None:
        err = GraphNodeTimeoutError(node_id="a", timeout=2.5, attempts=3)
        assert isinstance(err, TroopAIError)
        assert err.node_id == "a"
        assert err.timeout == 2.5
        assert err.attempts == 3
        assert "a" in str(err) and "2.5" in str(err)

    def test_retries_exhausted_attrs_and_subclass(self) -> None:
        cause = ValueError("boom")
        err = NodeRetriesExhaustedError(node_id="b", attempts=4, last_error=cause)
        assert isinstance(err, TroopAIError)
        assert err.node_id == "b"
        assert err.attempts == 4
        assert err.last_error is cause
        assert "b" in str(err) and "4" in str(err)

    def test_retries_exhausted_chains_cause_when_raised_from(self) -> None:
        cause = ValueError("boom")
        with pytest.raises(NodeRetriesExhaustedError) as ei:
            raise NodeRetriesExhaustedError(node_id="c", attempts=2, last_error=cause) from cause
        assert ei.value.__cause__ is cause
        assert ei.value.last_error is cause


class TestResolveNodeReliability:
    def _graph(self, *, default_retry=None, per_node_timeout=None):
        from troopai.adk.graphs.config import GraphConfig, NodeRetryPolicy
        from troopai.adk.graphs.graph import Graph

        config = GraphConfig(
            default_retry=default_retry if default_retry is not None else NodeRetryPolicy(),
            per_node_timeout=per_node_timeout,
        )
        return (
            Graph.new("resolve-test")
            .node("a", lambda: "a")
            .node("b", lambda: "b")
            .edge("a", "b")
            .entry("a")
            .terminal("b")
            .with_config(config)
            .compile()
        )

    def test_inherits_graph_defaults_when_node_unset(self) -> None:
        from troopai.adk.graphs.config import NodeRetryPolicy
        from troopai.adk.run.node_reliability import resolve_node_reliability

        gdef = NodeRetryPolicy(max_attempts=2)
        g = self._graph(default_retry=gdef, per_node_timeout=7.0)
        policy, timeout = resolve_node_reliability(g, g.get_node("a"))
        assert policy is gdef
        assert timeout == 7.0

    def test_node_field_overrides_graph_default(self) -> None:
        import dataclasses

        from troopai.adk.graphs.config import NodeRetryPolicy
        from troopai.adk.run.node_reliability import resolve_node_reliability

        gdef = NodeRetryPolicy(max_attempts=2)
        g = self._graph(default_retry=gdef, per_node_timeout=7.0)
        node_pol = NodeRetryPolicy(max_attempts=5)
        overridden = dataclasses.replace(g.get_node("a"), retry=node_pol, timeout=1.5)
        policy, timeout = resolve_node_reliability(g, overridden)
        assert policy is node_pol
        assert timeout == 1.5

    def test_per_field_independence(self) -> None:
        import dataclasses

        from troopai.adk.graphs.config import NodeRetryPolicy
        from troopai.adk.run.node_reliability import resolve_node_reliability

        gdef = NodeRetryPolicy(max_attempts=2)
        g = self._graph(default_retry=gdef, per_node_timeout=7.0)
        node = dataclasses.replace(g.get_node("a"), timeout=3.0)  # retry unset
        policy, timeout = resolve_node_reliability(g, node)
        assert policy is gdef  # inherited
        assert timeout == 3.0  # overridden


class TestRunNodeWithReliability:
    async def test_success_first_try_no_sleep(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from troopai.adk.graphs.config import NodeRetryPolicy
        from troopai.adk.orchestration.executable import NodeResult
        from troopai.adk.run.node_reliability import run_node_with_reliability

        sleeps: list[float] = []

        async def record_sleep(d: float) -> None:
            sleeps.append(d)

        monkeypatch.setattr("asyncio.sleep", record_sleep)

        call_count = 0

        async def invoke() -> NodeResult:
            nonlocal call_count
            call_count += 1
            return NodeResult(output="good")

        result = await run_node_with_reliability(
            node_id="a",
            policy=NodeRetryPolicy(max_attempts=3),
            timeout=None,
            invoke=invoke,
        )
        assert result.output == "good"
        assert call_count == 1
        assert len(sleeps) == 0

    async def test_retries_then_succeeds_with_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from troopai.adk.graphs.config import NodeRetryPolicy
        from troopai.adk.orchestration.executable import NodeResult
        from troopai.adk.run.node_reliability import run_node_with_reliability

        sleeps: list[float] = []

        async def record_sleep(d: float) -> None:
            sleeps.append(d)

        monkeypatch.setattr("asyncio.sleep", record_sleep)

        call_count = 0

        async def invoke() -> NodeResult:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("transient")
            return NodeResult(output="recovered")

        result = await run_node_with_reliability(
            node_id="a",
            policy=NodeRetryPolicy(max_attempts=5, initial_backoff=1.0, max_backoff=10.0),
            timeout=None,
            invoke=invoke,
        )
        assert result.output == "recovered"
        assert call_count == 3
        assert sleeps == [1.0, 2.0]

    async def test_backoff_capped_at_max(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from troopai.adk.graphs.config import NodeRetryPolicy
        from troopai.adk.run.node_reliability import run_node_with_reliability

        sleeps: list[float] = []

        async def record_sleep(d: float) -> None:
            sleeps.append(d)

        monkeypatch.setattr("asyncio.sleep", record_sleep)

        async def invoke() -> None:
            raise ValueError("always fails")

        with pytest.raises(NodeRetriesExhaustedError):
            await run_node_with_reliability(
                node_id="a",
                policy=NodeRetryPolicy(max_attempts=5, initial_backoff=4.0, max_backoff=10.0),
                timeout=None,
                invoke=invoke,  # type: ignore[arg-type]
            )
        assert sleeps == [4.0, 8.0, 10.0, 10.0]

    async def test_retry_on_filters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from troopai.adk.graphs.config import NodeRetryPolicy
        from troopai.adk.run.node_reliability import run_node_with_reliability

        sleeps: list[float] = []

        async def record_sleep(d: float) -> None:
            sleeps.append(d)

        monkeypatch.setattr("asyncio.sleep", record_sleep)

        call_count = 0

        async def invoke() -> None:
            nonlocal call_count
            call_count += 1
            raise KeyError("not retryable")

        with pytest.raises(KeyError):
            await run_node_with_reliability(
                node_id="a",
                policy=NodeRetryPolicy(max_attempts=4, retry_on=(TypeError,)),
                timeout=None,
                invoke=invoke,  # type: ignore[arg-type]
            )
        assert call_count == 1
        assert len(sleeps) == 0

    async def test_max_attempts_one_no_timeout_reraises_original(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from troopai.adk.graphs.config import NodeRetryPolicy
        from troopai.adk.run.node_reliability import run_node_with_reliability

        monkeypatch.setattr("asyncio.sleep", _NoSleep())

        sentinel = RuntimeError("original")

        async def invoke() -> None:
            raise sentinel

        with pytest.raises(RuntimeError) as ei:
            await run_node_with_reliability(
                node_id="a",
                policy=NodeRetryPolicy(max_attempts=1),
                timeout=None,
                invoke=invoke,  # type: ignore[arg-type]
            )
        assert ei.value is sentinel

    async def test_retries_exhausted_wraps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from troopai.adk.graphs.config import NodeRetryPolicy
        from troopai.adk.run.node_reliability import run_node_with_reliability

        monkeypatch.setattr("asyncio.sleep", _NoSleep())

        boom = ValueError("boom")

        async def invoke() -> None:
            raise boom

        with pytest.raises(NodeRetriesExhaustedError) as ei:
            await run_node_with_reliability(
                node_id="a",
                policy=NodeRetryPolicy(max_attempts=3),
                timeout=None,
                invoke=invoke,  # type: ignore[arg-type]
            )
        err = ei.value
        assert err.node_id == "a"
        assert err.attempts == 3
        assert err.last_error is boom
        assert err.__cause__ is boom

    async def test_timeout_wraps_and_is_retryable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from troopai.adk.graphs.config import NodeRetryPolicy
        from troopai.adk.run.node_reliability import run_node_with_reliability

        monkeypatch.setattr("asyncio.sleep", _NoSleep())

        @contextlib.asynccontextmanager
        async def fake_timeout(_seconds: float) -> AsyncGenerator[None, None]:
            raise TimeoutError("timed out")
            yield  # makes it an async generator (unreachable but required)

        monkeypatch.setattr("asyncio.timeout", fake_timeout)

        async def invoke() -> None:
            pass  # never reached — timeout fires first

        with pytest.raises(GraphNodeTimeoutError) as ei:
            await run_node_with_reliability(
                node_id="a",
                policy=NodeRetryPolicy(max_attempts=2),
                timeout=0.01,
                invoke=invoke,  # type: ignore[arg-type]
            )
        err = ei.value
        assert err.node_id == "a"
        assert err.timeout == 0.01
        assert err.attempts == 2

    async def test_non_retryable_timeout_still_wraps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from troopai.adk.exceptions import GraphNodeTimeoutError
        from troopai.adk.graphs.config import NodeRetryPolicy
        from troopai.adk.run.node_reliability import run_node_with_reliability

        monkeypatch.setattr("asyncio.sleep", _NoSleep())

        @contextlib.asynccontextmanager
        async def fake_timeout(_t: float) -> AsyncGenerator[None, None]:
            raise TimeoutError
            yield  # pragma: no cover

        monkeypatch.setattr("asyncio.timeout", fake_timeout)

        async def invoke() -> None:  # never reached; timeout fires on context enter
            raise AssertionError("unreachable")

        # retry_on excludes TimeoutError → NOT retryable, but a timeout
        # must STILL wrap as GraphNodeTimeoutError (spec: timeout always wraps).
        with pytest.raises(GraphNodeTimeoutError) as ei:
            await run_node_with_reliability(
                node_id="a",
                policy=NodeRetryPolicy(max_attempts=3, retry_on=(ValueError,)),
                timeout=0.01,
                invoke=invoke,  # type: ignore[arg-type]
            )
        assert ei.value.node_id == "a"
        assert ei.value.timeout == 0.01
        assert ei.value.attempts == 1
        assert isinstance(ei.value.__cause__, TimeoutError)

    async def test_cancelled_not_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from troopai.adk.graphs.config import NodeRetryPolicy
        from troopai.adk.run.node_reliability import run_node_with_reliability

        monkeypatch.setattr("asyncio.sleep", _NoSleep())

        call_count = 0

        async def invoke() -> None:
            nonlocal call_count
            call_count += 1
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await run_node_with_reliability(
                node_id="a",
                policy=NodeRetryPolicy(max_attempts=5),
                timeout=None,
                invoke=invoke,  # type: ignore[arg-type]
            )
        assert call_count == 1
