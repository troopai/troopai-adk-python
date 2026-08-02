"""Tests for opt-in live-billing retrieval in the sandbox lifecycle teardown."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.sandbox.runner_integration.lifecycle import sandbox_run_context
from troopai.adk.types.sandbox.cost import (
    SandboxBackendCapabilities,
    SandboxBillingRecord,
    SandboxCostDescriptor,
)


def _client(billing: SandboxBillingRecord | None) -> MagicMock:
    session = AsyncMock()
    session.session_id = "s"
    session.start = AsyncMock()
    session.apply_manifest = AsyncMock()
    session.aclose = AsyncMock()
    c = MagicMock()
    c.backend_id = "e2b"
    c.cost = SandboxCostDescriptor(usd_per_minute=0.06)
    c.capabilities = SandboxBackendCapabilities(network=True)
    c.create = AsyncMock(return_value=session)
    c.fetch_billing = AsyncMock(return_value=billing)
    return c


async def test_capture_live_cost_sets_billed() -> None:
    client = _client(SandboxBillingRecord(cost_usd=0.42))
    cfg = SandboxRunConfig(client=client, capture_live_cost=True)
    async with sandbox_run_context(
        config=cfg,
        capabilities=[],
        run_as=None,
        concurrency_guard=None,
        agent=MagicMock(),
        run_context=MagicMock(),
        hooks=AsyncMock(),
        tracing_enabled=False,
    ) as handle:
        assert handle.observability is not None
        usage = handle.observability.usage
    assert usage.billed_cost_usd == 0.42
    client.fetch_billing.assert_awaited_once_with(client.create.return_value)


async def test_no_capture_skips_billing() -> None:
    client = _client(SandboxBillingRecord(cost_usd=0.42))
    cfg = SandboxRunConfig(client=client, capture_live_cost=False)
    async with sandbox_run_context(
        config=cfg,
        capabilities=[],
        run_as=None,
        concurrency_guard=None,
        agent=MagicMock(),
        run_context=MagicMock(),
        hooks=AsyncMock(),
        tracing_enabled=False,
    ) as handle:
        assert handle.observability is not None
        usage = handle.observability.usage
    assert usage.billed_cost_usd is None
    client.fetch_billing.assert_not_awaited()


async def test_capture_live_cost_none_record_leaves_billed_none() -> None:
    # The E2B default: fetch_billing returns None (no raise) — the call is
    # made because capture opted in, but billed_cost_usd stays None.
    client = _client(billing=None)
    cfg = SandboxRunConfig(client=client, capture_live_cost=True)
    async with sandbox_run_context(
        config=cfg,
        capabilities=[],
        run_as=None,
        concurrency_guard=None,
        agent=MagicMock(),
        run_context=MagicMock(),
        hooks=AsyncMock(),
        tracing_enabled=False,
    ) as handle:
        assert handle.observability is not None
        usage = handle.observability.usage
    assert usage.billed_cost_usd is None
    client.fetch_billing.assert_awaited_once_with(client.create.return_value)


async def test_billing_endpoint_failure_leaves_run_clean() -> None:
    client = _client(billing=None)
    client.fetch_billing = AsyncMock(side_effect=RuntimeError("billing endpoint down"))
    cfg = SandboxRunConfig(client=client, capture_live_cost=True)
    async with sandbox_run_context(
        config=cfg,
        capabilities=[],
        run_as=None,
        concurrency_guard=None,
        agent=MagicMock(),
        run_context=MagicMock(),
        hooks=AsyncMock(),
        tracing_enabled=False,
    ) as handle:
        assert handle.observability is not None
        usage = handle.observability.usage
    # The billing-endpoint error is suppressed: the run completes and
    # billed_cost_usd stays None.
    assert usage.billed_cost_usd is None
