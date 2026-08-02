"""Tests for the E2B fetch_billing reference override."""

from __future__ import annotations

from unittest.mock import MagicMock

from troopai.adk.sandbox.clients.hosted.e2b.e2b_client import E2bSandboxClient


async def test_e2b_fetch_billing_returns_none() -> None:
    # E2B reports usage at the account level, not per sandbox, so the
    # reference override returns None (computed_cost_usd is the estimate).
    client = E2bSandboxClient()
    record = await client.fetch_billing(MagicMock())
    assert record is None
