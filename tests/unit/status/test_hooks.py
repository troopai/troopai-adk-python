"""Tests for StatusTrackingHooks — integration with RunHooks lifecycle."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.exceptions import QuotaExceeded
from troopai.adk.llms.llm_usage import LLMUsage
from troopai.adk.status import (
    AgentQuota,
    AgentStatusStore,
    StatusTrackingHooks,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _mock_agent(name: str = "test-agent") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


def _mock_context(
    requests: int = 3,
    input_tokens: int = 300,
    output_tokens: int = 200,
) -> MagicMock:
    ctx = MagicMock()
    ctx.usage = LLMUsage(
        requests=requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )
    ctx.tenant_id = None
    ctx.cost_usd = 0.0
    return ctx


# ── TestOnAgentEnd ───────────────────────────────────────────────────


class TestOnAgentEnd:
    """Test that on_agent_end records runs correctly."""

    @pytest.mark.asyncio
    async def test_records_run_on_agent_end(self) -> None:
        """on_agent_end creates a record with correct usage data."""
        store = AgentStatusStore()
        hooks: StatusTrackingHooks[object] = StatusTrackingHooks(store=store)
        agent = _mock_agent("my-agent")
        ctx = _mock_context(requests=5, input_tokens=400, output_tokens=100)
        result = MagicMock()

        await hooks.on_agent_start(ctx, agent)
        await hooks.on_agent_end(ctx, agent, result)

        status = await store.get_status("my-agent")
        assert status.total_runs == 1
        assert status.successful_runs == 1
        assert status.total_requests == 5
        assert status.total_tokens == 500
        assert status.total_input_tokens == 400
        assert status.total_output_tokens == 100

    @pytest.mark.asyncio
    async def test_duration_calculated(self) -> None:
        """Duration is calculated from on_agent_start to on_agent_end."""
        store = AgentStatusStore()
        hooks: StatusTrackingHooks[object] = StatusTrackingHooks(store=store)
        agent = _mock_agent()
        ctx = _mock_context()
        result = MagicMock()

        await hooks.on_agent_start(ctx, agent)
        await hooks.on_agent_end(ctx, agent, result)

        records = await store.get_records("test-agent")
        assert len(records) == 1
        assert records[0].duration_ms >= 0
        assert records[0].status == "success"


# ── TestQuotaEnforcement ─────────────────────────────────────────────


class TestQuotaEnforcement:
    """Test that on_agent_start enforces quotas."""

    @pytest.mark.asyncio
    async def test_quota_blocks_run(self) -> None:
        """QuotaExceeded raised in on_agent_start when over limit."""
        store = AgentStatusStore()

        # Pre-populate with usage
        from troopai.adk.status.types import AgentRunRecord

        now = time.time()
        record = AgentRunRecord(
            id="pre-1",
            agent_name="expensive",
            status="success",
            started_at=now - 100,
            ended_at=now - 99,
            duration_ms=1000.0,
            requests=10,
            input_tokens=5000,
            output_tokens=5000,
            total_tokens=10_000,
            error=None,
        )
        await store.record(record)

        quota = AgentQuota(
            agent_name="expensive",
            window_seconds=3600,
            max_total_tokens=5_000,
        )
        hooks: StatusTrackingHooks[object] = StatusTrackingHooks(store=store, quotas=[quota])
        agent = _mock_agent("expensive")
        ctx = _mock_context()

        with pytest.raises(QuotaExceeded) as exc_info:
            await hooks.on_agent_start(ctx, agent)

        assert exc_info.value.agent_name == "expensive"
        assert exc_info.value.resource == "total_tokens"

    @pytest.mark.asyncio
    async def test_no_quota_passes(self) -> None:
        """No quotas means on_agent_start passes without checking."""
        store = AgentStatusStore()
        hooks: StatusTrackingHooks[object] = StatusTrackingHooks(store=store)
        agent = _mock_agent()
        ctx = _mock_context()

        await hooks.on_agent_start(ctx, agent)  # Should not raise


# ── TestRecordError ──────────────────────────────────────────────────


class TestRecordError:
    """Test manual error recording."""

    @pytest.mark.asyncio
    async def test_record_error_creates_error_record(self) -> None:
        """record_error() creates a record with status='error'."""
        store = AgentStatusStore()
        hooks: StatusTrackingHooks[object] = StatusTrackingHooks(store=store)

        # Simulate a run that started but failed (key is (tenant_id, agent_name, context_id))
        sentinel_ctx_id = 12345
        hooks._run_starts[(None, "my-agent", sentinel_ctx_id)] = time.time() - 1.0
        await hooks.record_error("my-agent", "Connection timeout")

        status = await store.get_status("my-agent")
        assert status.total_runs == 1
        assert status.failed_runs == 1
        assert status.successful_runs == 0

        records = await store.get_records("my-agent")
        assert records[0].status == "error"
        assert records[0].error == "Connection timeout"


# ── TestConcurrentCollision ──────────────────────────────────────────


class TestConcurrentCollision:
    """Concurrent same-agent+tenant runs must not collide on _run_starts."""

    @pytest.mark.asyncio
    async def test_stale_start_entries_evicted(self) -> None:
        """Starts abandoned by runs that raised without record_error are swept.

        Without eviction, a long-lived hooks instance grows _run_starts
        unboundedly when callers forget the record_error contract.
        """
        store = MagicMock(spec=AgentStatusStore)
        store.check_quotas = AsyncMock(return_value=None)
        hooks: StatusTrackingHooks[object] = StatusTrackingHooks(store=store, stale_run_seconds=60.0)

        stale_key = (None, "ghost-agent", 12345)
        hooks._run_starts[stale_key] = time.time() - 3600.0  # an hour-old orphan
        fresh_key = (None, "live-agent", 67890)
        hooks._run_starts[fresh_key] = time.time() - 1.0  # in-flight run

        await hooks.on_agent_start(_mock_context(), _mock_agent("next-agent"))

        assert stale_key not in hooks._run_starts, "hour-old orphan must be evicted"
        assert fresh_key in hooks._run_starts, "in-flight run must be kept"

    async def test_concurrent_same_agent_tenant_timing(self) -> None:
        """Two concurrent runs for the same agent+tenant get independent durations."""
        import asyncio

        store = AgentStatusStore()
        hooks: StatusTrackingHooks[object] = StatusTrackingHooks(store=store)
        agent = _mock_agent("shared-agent")
        result = MagicMock()

        ctx_a = _mock_context()
        ctx_a.tenant_id = "tenant-1"
        ctx_b = _mock_context()
        ctx_b.tenant_id = "tenant-1"

        # Both runs start concurrently (same agent name and tenant_id).
        await asyncio.gather(
            hooks.on_agent_start(ctx_a, agent),
            hooks.on_agent_start(ctx_b, agent),
        )
        # Two entries must coexist — second start must not overwrite first.
        assert len(hooks._run_starts) == 2

        # Both runs end successfully; neither should see duration_ms=0.
        await asyncio.gather(
            hooks.on_agent_end(ctx_a, agent, result),
            hooks.on_agent_end(ctx_b, agent, result),
        )

        # Both entries are consumed.
        assert len(hooks._run_starts) == 0

        records = await store.get_records("shared-agent")
        assert len(records) == 2
        for rec in records:
            assert rec.duration_ms >= 0
            assert rec.status == "success"

    async def test_record_error_with_context_does_not_steal_sibling_start(self) -> None:
        """record_error(context=...) pops only the failing run's start entry.

        Regression: with the old two-element (tenant_id, agent_name) scan +
        max-timestamp heuristic, when run A fails while sibling run B (same
        agent + tenant, started later) is still in flight, record_error pops
        B's start entry. B's later on_agent_end then finds nothing and records
        duration_ms=0, and A's error record is mis-attributed B's start time.
        Passing the failing run's own context makes the lookup exact.
        """
        store = AgentStatusStore()
        hooks: StatusTrackingHooks[object] = StatusTrackingHooks(store=store)
        agent = _mock_agent("shared-agent")
        result = MagicMock()

        ctx_a = _mock_context()
        ctx_a.tenant_id = "tenant-1"
        ctx_b = _mock_context()
        ctx_b.tenant_id = "tenant-1"

        # Run A starts first, run B (same agent+tenant) starts later.
        await hooks.on_agent_start(ctx_a, agent)
        await hooks.on_agent_start(ctx_b, agent)
        # Pin deterministic, distinct start times: A is older, B is newer.
        now = time.time()
        hooks._run_starts[("tenant-1", "shared-agent", id(ctx_a))] = now - 0.5
        hooks._run_starts[("tenant-1", "shared-agent", id(ctx_b))] = now - 0.1

        # Run A fails. Passing ctx_a must pop A's entry, leaving B's intact.
        await hooks.record_error("shared-agent", "boom", context=ctx_a)

        # B's start entry survives — the failure did not steal it.
        assert ("tenant-1", "shared-agent", id(ctx_b)) in hooks._run_starts
        assert ("tenant-1", "shared-agent", id(ctx_a)) not in hooks._run_starts

        # B completes successfully and gets its OWN (non-zero) duration.
        await hooks.on_agent_end(ctx_b, agent, result)

        records = await store.get_records("shared-agent", tenant_id="tenant-1")
        by_status = {r.status: r for r in records}
        assert len(records) == 2
        # A's error record reflects A's ~500 ms start, not B's ~100 ms one.
        assert by_status["error"].duration_ms >= 400, (
            f"error duration_ms={by_status['error'].duration_ms:.1f} — "
            "expected >= 400 ms (A's own start); near-100 ms means B's start was stolen"
        )
        # B's success record reflects its own ~100 ms start, not a zero fallback.
        assert by_status["success"].duration_ms < 400
        assert by_status["success"].duration_ms > 0, (
            "B's duration is 0 — its start entry was consumed by A's record_error"
        )
