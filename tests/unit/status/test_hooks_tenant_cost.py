"""Tests for StatusTrackingHooks tenant/cost wiring and multi-tenant bug fixes."""

import logging
import time
from unittest.mock import MagicMock

import pytest

from troopai.adk.run.context import RunContext
from troopai.adk.status.hooks import StatusTrackingHooks
from troopai.adk.status.store import AgentStatusStore


def _mock_agent(name: str = "a") -> MagicMock:
    agent = MagicMock()
    agent.name = name
    return agent


async def test_hooks_record_tenant_and_cost_from_context() -> None:
    """tenant_id and cost_usd from RunContext land in the recorded AgentRunRecord."""
    store = AgentStatusStore(path=":memory:")
    hooks: StatusTrackingHooks[object] = StatusTrackingHooks(store=store)
    ctx = RunContext.make(None, tenant_id="acme")
    ctx.cost_usd = 0.42
    agent = _mock_agent()
    result = MagicMock()

    await hooks.on_agent_start(ctx, agent)
    await hooks.on_agent_end(ctx, agent, result)

    status = await store.get_status("a", tenant_id="acme")
    assert status.total_runs == 1
    assert status.total_cost_usd == 0.42
    await store.close()


async def test_run_starts_keyed_by_tenant_no_collision() -> None:
    """Concurrent same-name agents for different tenants don't collide in _run_starts.

    Regression test: with the old string-key implementation (keyed by agent_name
    only), on_agent_start for globex would overwrite acme's start time.  When
    on_agent_end fires for acme it would pop globex's (later) start, producing a
    near-zero duration for acme and a zero-fallback duration for globex.  With
    the tuple key (tenant_id, agent_name) each tenant's start time is preserved.

    White-box: injects known, deterministic start times into _run_starts so the
    assertions are not subject to real-clock timing.  The injected keys match the
    tuple form used by the correct implementation; the old string-key form would
    never see these entries, causing both durations to fall back to the
    ended_at sentinel (duration_ms == 0), which the assertions catch.
    """
    store = AgentStatusStore(path=":memory:")
    hooks: StatusTrackingHooks[object] = StatusTrackingHooks(store=store)
    agent = _mock_agent()
    result = MagicMock()
    a = RunContext.make(None, tenant_id="acme")
    b = RunContext.make(None, tenant_id="globex")

    await hooks.on_agent_start(a, agent)
    await hooks.on_agent_start(b, agent)  # same agent name, different tenant

    # Overwrite with known, distinct start times so duration assertions are
    # deterministic.  acme started 0.5 s ago, globex started 0.1 s ago.
    # Keys are (tenant_id, agent_name, id(context)) — use id(a) and id(b).
    now = time.time()
    hooks._run_starts[("acme", "a", id(a))] = now - 0.5  # white-box regression test
    hooks._run_starts[("globex", "a", id(b))] = now - 0.1  # white-box regression test

    await hooks.on_agent_end(a, agent, result)
    await hooks.on_agent_end(b, agent, result)

    # Each tenant must have exactly one record.
    assert (await store.get_status("a", tenant_id="acme")).total_runs == 1
    assert (await store.get_status("a", tenant_id="globex")).total_runs == 1

    # Per-tenant duration integrity: each recorded duration must reflect the
    # tenant's OWN start time, not the other tenant's.
    # acme started ~500 ms ago → duration_ms must be clearly >= 400.
    # globex started ~100 ms ago → duration_ms must be clearly < 400.
    # The old string-key bug leaves both durations at 0 (start time clobbered /
    # missing), so either assertion would fail against the buggy implementation.
    acme_records = await store.get_records("a", tenant_id="acme")
    globex_records = await store.get_records("a", tenant_id="globex")
    assert len(acme_records) == 1
    assert len(globex_records) == 1
    assert acme_records[0].duration_ms >= 400, (
        f"acme duration_ms={acme_records[0].duration_ms:.1f} — "
        "expected >= 400 ms (own start preserved); got near-zero → collision bug"
    )
    assert globex_records[0].duration_ms < 400, (
        f"globex duration_ms={globex_records[0].duration_ms:.1f} — "
        "expected < 400 ms (own start preserved); got large value → collision bug"
    )

    await store.close()


async def test_record_error_tenant_scoped() -> None:
    """record_error with tenant_id records the error record under the correct tenant."""
    store = AgentStatusStore(path=":memory:")
    hooks: StatusTrackingHooks[object] = StatusTrackingHooks(store=store)
    ctx = RunContext.make(None, tenant_id="acme")
    agent = _mock_agent()

    # Simulate a run that started but failed
    await hooks.on_agent_start(ctx, agent)
    await hooks.record_error("a", "Connection timeout", tenant_id="acme")

    status = await store.get_status("a", tenant_id="acme")
    assert status.total_runs == 1
    assert status.failed_runs == 1

    records = await store.get_records("a", tenant_id="acme")
    assert len(records) == 1
    assert records[0].status == "error"
    assert records[0].tenant_id == "acme"
    assert records[0].error == "Connection timeout"
    await store.close()


async def test_record_error_cost_usd_included_in_status() -> None:
    """record_error with cost_usd=0.05 stores the cost; AgentStatus.total_cost_usd reflects it."""
    store = AgentStatusStore(path=":memory:")
    hooks: StatusTrackingHooks[object] = StatusTrackingHooks(store=store)
    ctx = RunContext.make(None, tenant_id="acme")
    agent = _mock_agent()

    # Simulate a run that started, accrued cost, then failed before on_agent_end
    await hooks.on_agent_start(ctx, agent)
    await hooks.record_error("a", "boom", tenant_id="acme", cost_usd=0.05)

    status = await store.get_status("a", tenant_id="acme")
    assert status.total_runs == 1
    assert status.failed_runs == 1
    assert status.total_cost_usd == pytest.approx(0.05)

    records = await store.get_records("a", tenant_id="acme")
    assert len(records) == 1
    assert records[0].cost_usd == pytest.approx(0.05)
    await store.close()


async def test_on_agent_end_without_start_warns_and_records(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """on_agent_end with no matching on_agent_start emits a warning and records duration_ms=0."""
    store = AgentStatusStore(path=":memory:")
    hooks: StatusTrackingHooks[object] = StatusTrackingHooks(store=store)
    ctx = RunContext.make(None, tenant_id="acme")
    agent = _mock_agent()
    result = MagicMock()

    # Call on_agent_end WITHOUT a prior on_agent_start — no entry in _run_starts
    with caplog.at_level(logging.WARNING, logger="troopai.adk.status.hooks"):
        await hooks.on_agent_end(ctx, agent, result)

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) >= 1
    assert any("no start time" in r.message for r in warning_records)

    records = await store.get_records("a", tenant_id="acme")
    assert len(records) == 1
    assert records[0].duration_ms == pytest.approx(0.0, abs=5.0)
    await store.close()


async def test_record_error_without_start_warns_and_records(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """record_error with no matching on_agent_start emits a warning and records duration_ms=0."""
    store = AgentStatusStore(path=":memory:")
    hooks: StatusTrackingHooks[object] = StatusTrackingHooks(store=store)

    # Call record_error WITHOUT a prior on_agent_start
    with caplog.at_level(logging.WARNING, logger="troopai.adk.status.hooks"):
        await hooks.record_error("a", "boom", tenant_id="acme", cost_usd=0.01)

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warning_records) >= 1
    assert any("no start time" in r.message for r in warning_records)

    records = await store.get_records("a", tenant_id="acme")
    assert len(records) == 1
    assert records[0].status == "error"
    assert records[0].duration_ms == pytest.approx(0.0, abs=5.0)
    await store.close()
