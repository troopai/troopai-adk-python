import sqlite3
import tempfile

import pytest

from troopai.adk.exceptions import QuotaExceeded
from troopai.adk.status.store import AgentStatusStore
from troopai.adk.status.types import AgentQuota, AgentRunRecord


def _record(
    agent: str,
    *,
    tenant: str | None,
    cost: float,
    tokens: int,
    rid: str,
) -> AgentRunRecord:
    return AgentRunRecord(
        id=rid,
        agent_name=agent,
        status="success",
        started_at=1000.0,
        ended_at=1001.0,
        duration_ms=1000.0,
        requests=1,
        input_tokens=tokens,
        output_tokens=0,
        total_tokens=tokens,
        error=None,
        tenant_id=tenant,
        cost_usd=cost,
    )


async def test_record_and_get_status_carry_tenant_and_cost() -> None:
    store = AgentStatusStore(path=":memory:")
    await store.record(_record("a", tenant="acme", cost=0.5, tokens=100, rid="r1"))
    await store.record(_record("a", tenant="globex", cost=0.2, tokens=50, rid="r2"))
    acme = await store.get_status("a", tenant_id="acme")
    assert acme.total_runs == 1
    assert acme.total_cost_usd == 0.5
    await store.close()


async def test_quota_is_tenant_scoped() -> None:
    store = AgentStatusStore(path=":memory:")
    await store.record(_record("a", tenant="acme", cost=0.0, tokens=100, rid="r1"))
    quota = AgentQuota(agent_name="a", window_seconds=10_000_000_000, max_total_tokens=100)
    await store.check_quota("a", quota, tenant_id="globex")  # no raise
    with pytest.raises(QuotaExceeded):
        await store.check_quota("a", quota, tenant_id="acme")
    await store.close()


async def test_self_heal_old_schema_file_db() -> None:
    """Self-heal path: open a file-backed DB missing tenant_id/cost_usd columns.

    Simulates a database created before tenant_id and cost_usd existed.
    Verifies that _ensure_ready adds the missing columns (no exception),
    that the new record with tenant/cost is persisted correctly, and that
    the pre-existing old row reads back with tenant_id=None and cost_usd=None.
    """
    old_schema = """
    CREATE TABLE agent_run_records (
        id TEXT PRIMARY KEY,
        agent_name TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'success',
        started_at REAL NOT NULL,
        ended_at REAL NOT NULL,
        duration_ms REAL NOT NULL,
        requests INTEGER NOT NULL DEFAULT 0,
        input_tokens INTEGER NOT NULL DEFAULT 0,
        output_tokens INTEGER NOT NULL DEFAULT 0,
        total_tokens INTEGER NOT NULL DEFAULT 0,
        error TEXT
    )
    """
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    # Create old-schema DB and insert a legacy row (no tenant/cost columns).
    with sqlite3.connect(db_path) as legacy_conn:
        legacy_conn.execute(old_schema)
        legacy_conn.execute(
            "INSERT INTO agent_run_records "
            "(id, agent_name, status, started_at, ended_at, duration_ms, "
            " requests, input_tokens, output_tokens, total_tokens, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("old-1", "legacy-agent", "success", 500.0, 501.0, 1000.0, 1, 10, 5, 15, None),
        )
        legacy_conn.commit()

    # Open with AgentStatusStore — should self-heal without exception.
    store = AgentStatusStore(path=db_path)
    new_rec = AgentRunRecord(
        id="new-1",
        agent_name="legacy-agent",
        status="success",
        started_at=1000.0,
        ended_at=1001.0,
        duration_ms=1000.0,
        requests=2,
        input_tokens=20,
        output_tokens=10,
        total_tokens=30,
        error=None,
        tenant_id="acme",
        cost_usd=0.3,
    )
    await store.record(new_rec)  # (a) no exception — self-heal added columns

    # (b) get_status with tenant filter reflects only the new row's cost.
    acme_status = await store.get_status("legacy-agent", tenant_id="acme")
    assert acme_status.total_cost_usd == 0.3

    # (c) pre-existing old row reads back with tenant_id=None and cost_usd=None.
    old_records = await store.get_records("legacy-agent", tenant_id=None)
    old_rows = [r for r in old_records if r.id == "old-1"]
    assert len(old_rows) == 1
    assert old_rows[0].tenant_id is None
    assert old_rows[0].cost_usd is None

    await store.close()


async def test_get_records_tenant_filter() -> None:
    """get_records with tenant_id returns only records for that tenant."""
    store = AgentStatusStore(path=":memory:")
    await store.record(_record("b", tenant="acme", cost=0.1, tokens=10, rid="b1"))
    await store.record(_record("b", tenant="globex", cost=0.2, tokens=20, rid="b2"))
    await store.record(_record("b", tenant="acme", cost=0.3, tokens=30, rid="b3"))

    acme_records = await store.get_records("b", tenant_id="acme")
    assert len(acme_records) == 2
    assert all(r.tenant_id == "acme" for r in acme_records)
    acme_ids = {r.id for r in acme_records}
    assert acme_ids == {"b1", "b3"}

    globex_records = await store.get_records("b", tenant_id="globex")
    assert len(globex_records) == 1
    assert globex_records[0].id == "b2"

    await store.close()
