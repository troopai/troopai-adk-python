"""Tests for AgentStatusStore — persistence, aggregation, and quota enforcement."""

import time

import pytest

from troopai.adk.exceptions import QuotaExceeded
from troopai.adk.status import AgentQuota, AgentRunRecord, AgentStatusStore

# ── Helpers ──────────────────────────────────────────────────────────


def _record(
    agent_name: str = "test-agent",
    record_id: str = "rec-1",
    status: str = "success",
    started_at: float = 1000.0,
    total_tokens: int = 500,
    requests: int = 2,
    error: str | None = None,
) -> AgentRunRecord:
    return AgentRunRecord(
        id=record_id,
        agent_name=agent_name,
        status=status,
        started_at=started_at,
        ended_at=started_at + 1.0,
        duration_ms=1000.0,
        requests=requests,
        input_tokens=total_tokens // 2,
        output_tokens=total_tokens // 2,
        total_tokens=total_tokens,
        error=error,
    )


@pytest.fixture
def store() -> AgentStatusStore:
    return AgentStatusStore()  # :memory:


# ── Record + Get Status ─────────────────────────────────────────────


class TestRecordAndGetStatus:
    """Test recording runs and querying aggregated status."""

    @pytest.mark.asyncio
    async def test_record_and_get_status(self, store: AgentStatusStore) -> None:
        """Single record produces correct aggregated status."""
        await store.record(_record(total_tokens=700, requests=3))
        status = await store.get_status("test-agent")

        assert status.agent_name == "test-agent"
        assert status.total_runs == 1
        assert status.successful_runs == 1
        assert status.failed_runs == 0
        assert status.total_tokens == 700
        assert status.total_requests == 3
        assert status.error_rate == 0.0

    @pytest.mark.asyncio
    async def test_multiple_records_aggregate(self, store: AgentStatusStore) -> None:
        """Multiple records are aggregated correctly."""
        await store.record(_record(record_id="r1", total_tokens=300, requests=1))
        await store.record(_record(record_id="r2", total_tokens=500, requests=2))
        await store.record(
            _record(
                record_id="r3",
                status="error",
                total_tokens=100,
                requests=1,
                error="timeout",
            )
        )

        status = await store.get_status("test-agent")
        assert status.total_runs == 3
        assert status.successful_runs == 2
        assert status.failed_runs == 1
        assert status.total_tokens == 900
        assert status.total_requests == 4
        assert abs(status.error_rate - 1 / 3) < 0.01

    @pytest.mark.asyncio
    async def test_empty_status_returns_zeros(self, store: AgentStatusStore) -> None:
        """Querying agent with no records returns zero-filled status."""
        status = await store.get_status("nonexistent")
        assert status.total_runs == 0
        assert status.total_tokens == 0
        assert status.first_run_at is None
        assert status.last_run_at is None
        assert status.error_rate == 0.0

    @pytest.mark.asyncio
    async def test_time_windowed_query(self, store: AgentStatusStore) -> None:
        """since parameter filters records by start time."""
        await store.record(_record(record_id="old", started_at=100.0, total_tokens=1000))
        await store.record(_record(record_id="new", started_at=200.0, total_tokens=500))

        # Query only recent
        status = await store.get_status("test-agent", since=150.0)
        assert status.total_runs == 1
        assert status.total_tokens == 500

        # Query all
        status_all = await store.get_status("test-agent")
        assert status_all.total_runs == 2
        assert status_all.total_tokens == 1500

    @pytest.mark.asyncio
    async def test_different_agents_isolated(self, store: AgentStatusStore) -> None:
        """Records for different agents don't cross-contaminate."""
        await store.record(_record(agent_name="agent-a", record_id="a1", total_tokens=100))
        await store.record(_record(agent_name="agent-b", record_id="b1", total_tokens=200))

        status_a = await store.get_status("agent-a")
        assert status_a.total_tokens == 100
        assert status_a.total_runs == 1

        status_b = await store.get_status("agent-b")
        assert status_b.total_tokens == 200


# ── Get Records ──────────────────────────────────────────────────────


class TestGetRecords:
    """Test retrieving individual run records."""

    @pytest.mark.asyncio
    async def test_get_records_ordered_by_most_recent(self, store: AgentStatusStore) -> None:
        """Records returned most recent first."""
        await store.record(_record(record_id="r1", started_at=100.0))
        await store.record(_record(record_id="r2", started_at=200.0))
        await store.record(_record(record_id="r3", started_at=150.0))

        records = await store.get_records("test-agent")
        assert len(records) == 3
        assert records[0].id == "r2"  # most recent
        assert records[2].id == "r1"  # oldest

    @pytest.mark.asyncio
    async def test_get_records_with_limit(self, store: AgentStatusStore) -> None:
        """limit parameter caps the number of records returned."""
        for i in range(5):
            await store.record(_record(record_id=f"r{i}", started_at=float(i)))

        records = await store.get_records("test-agent", limit=2)
        assert len(records) == 2


# ── Quota Enforcement ────────────────────────────────────────────────


class TestQuotaEnforcement:
    """Test check_quota and check_quotas methods."""

    @pytest.mark.asyncio
    async def test_quota_within_limit_passes(self, store: AgentStatusStore) -> None:
        """No exception when usage is within quota."""
        now = time.time()
        await store.record(_record(record_id="r1", started_at=now - 100, total_tokens=500))

        quota = AgentQuota(
            agent_name="test-agent",
            window_seconds=3600,
            max_total_tokens=10_000,
        )
        await store.check_quota("test-agent", quota)  # Should not raise

    @pytest.mark.asyncio
    async def test_quota_exceeds_tokens(self, store: AgentStatusStore) -> None:
        """QuotaExceeded raised when token limit breached."""
        now = time.time()
        await store.record(_record(record_id="r1", started_at=now - 100, total_tokens=8000))
        await store.record(_record(record_id="r2", started_at=now - 50, total_tokens=3000))

        quota = AgentQuota(
            agent_name="test-agent",
            window_seconds=3600,
            max_total_tokens=10_000,
        )
        with pytest.raises(QuotaExceeded) as exc_info:
            await store.check_quota("test-agent", quota)

        assert exc_info.value.resource == "total_tokens"
        assert exc_info.value.current_value == 11_000
        assert exc_info.value.limit == 10_000

    @pytest.mark.asyncio
    async def test_quota_exceeds_requests(self, store: AgentStatusStore) -> None:
        """QuotaExceeded raised when request limit breached."""
        now = time.time()
        for i in range(5):
            await store.record(
                _record(
                    record_id=f"r{i}",
                    started_at=now - 100 + i,
                    requests=10,
                )
            )

        quota = AgentQuota(
            agent_name="test-agent",
            window_seconds=3600,
            max_requests=40,
        )
        with pytest.raises(QuotaExceeded) as exc_info:
            await store.check_quota("test-agent", quota)

        assert exc_info.value.resource == "requests"
        assert exc_info.value.current_value == 50

    @pytest.mark.asyncio
    async def test_quota_exceeds_runs(self, store: AgentStatusStore) -> None:
        """QuotaExceeded raised when run count limit breached."""
        now = time.time()
        for i in range(3):
            await store.record(_record(record_id=f"r{i}", started_at=now - 100 + i))

        quota = AgentQuota(
            agent_name="test-agent",
            window_seconds=3600,
            max_runs=3,
        )
        with pytest.raises(QuotaExceeded):
            await store.check_quota("test-agent", quota)

    @pytest.mark.asyncio
    async def test_wildcard_quota_applies_to_all(self, store: AgentStatusStore) -> None:
        """agent_name='*' checks tokens across ALL agents."""
        now = time.time()
        await store.record(
            _record(
                agent_name="agent-a",
                record_id="a1",
                started_at=now - 100,
                total_tokens=6000,
            )
        )
        await store.record(
            _record(
                agent_name="agent-b",
                record_id="b1",
                started_at=now - 50,
                total_tokens=5000,
            )
        )

        quota = AgentQuota(
            agent_name="*",
            window_seconds=3600,
            max_total_tokens=10_000,
        )
        with pytest.raises(QuotaExceeded) as exc_info:
            await store.check_quota("agent-a", quota)

        assert exc_info.value.current_value == 11_000

    @pytest.mark.asyncio
    async def test_old_records_outside_window_ignored(self, store: AgentStatusStore) -> None:
        """Records outside the quota window are not counted."""
        now = time.time()
        # Old record (outside 1-hour window)
        await store.record(
            _record(
                record_id="old",
                started_at=now - 7200,
                total_tokens=50_000,
            )
        )
        # Recent record (inside window)
        await store.record(
            _record(
                record_id="new",
                started_at=now - 100,
                total_tokens=500,
            )
        )

        quota = AgentQuota(
            agent_name="test-agent",
            window_seconds=3600,
            max_total_tokens=10_000,
        )
        await store.check_quota("test-agent", quota)  # Should not raise

    @pytest.mark.asyncio
    async def test_check_quotas_matches_by_name(self, store: AgentStatusStore) -> None:
        """check_quotas only enforces quotas matching the agent name."""
        now = time.time()
        await store.record(
            _record(
                agent_name="expensive",
                record_id="r1",
                started_at=now - 100,
                total_tokens=5000,
            )
        )

        quotas = [
            AgentQuota(agent_name="cheap", window_seconds=3600, max_total_tokens=100),
            AgentQuota(agent_name="expensive", window_seconds=3600, max_total_tokens=10_000),
        ]
        # "cheap" quota should not apply to "expensive" agent
        await store.check_quotas("expensive", quotas)  # Should not raise


# ── Prune ────────────────────────────────────────────────────────────


class TestPrune:
    """Test record pruning."""

    @pytest.mark.asyncio
    async def test_prune_deletes_old_records(self, store: AgentStatusStore) -> None:
        """Records older than cutoff are deleted."""
        await store.record(_record(record_id="old", started_at=100.0))
        await store.record(_record(record_id="new", started_at=200.0))

        deleted = await store.prune(older_than=150.0)
        assert deleted == 1

        records = await store.get_records("test-agent")
        assert len(records) == 1
        assert records[0].id == "new"
