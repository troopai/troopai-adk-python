from __future__ import annotations

import pytest

from troopai.adk.audit import InMemoryAuditSink
from troopai.adk.exceptions import ToolNotPermittedForTenant
from troopai.adk.run.config import RunConfig
from troopai.adk.run.governance import enforce_tenant_allowlist


async def test_permitted_returns_none() -> None:
    config = RunConfig(tenant_tool_allowlist={"t1": {"search"}})
    out = await enforce_tenant_allowlist(
        config, tenant_id="t1", agent_name="a", tool_name="search", call_id="c1", raw_args="{}"
    )
    assert out is None


async def test_hard_deny_raises_and_audits() -> None:
    sink = InMemoryAuditSink()
    config = RunConfig(tenant_tool_allowlist={"t1": {"search"}}, audit_sink=sink)
    with pytest.raises(ToolNotPermittedForTenant) as exc_info:
        await enforce_tenant_allowlist(
            config, tenant_id="t1", agent_name="a", tool_name="delete", call_id="c1", raw_args="{}"
        )
    err = exc_info.value
    assert err.tenant_id == "t1"
    assert err.tool_name == "delete"
    assert err.agent_name == "a"
    assert len(sink.events) == 1
    assert sink.events[0].outcome == "denied"


async def test_soft_deny_returns_message_and_audits() -> None:
    sink = InMemoryAuditSink()
    config = RunConfig(
        tenant_tool_allowlist={"t1": {"search"}},
        tenant_allowlist_soft_deny=True,
        audit_sink=sink,
    )
    out = await enforce_tenant_allowlist(
        config, tenant_id="t1", agent_name="a", tool_name="delete", call_id="c1", raw_args="{}"
    )
    assert isinstance(out, str) and len(out) > 0
    assert sink.events[0].outcome == "denied"


async def test_default_deny_absent_tenant_raises_and_audits() -> None:
    sink = InMemoryAuditSink()
    config = RunConfig(
        tenant_tool_allowlist={"t1": {"search"}},
        tenant_allowlist_default_deny=True,
        audit_sink=sink,
    )
    # t2 is absent from the map; default_deny denies it.
    with pytest.raises(ToolNotPermittedForTenant):
        await enforce_tenant_allowlist(
            config, tenant_id="t2", agent_name="a", tool_name="search", call_id="c1", raw_args="{}"
        )
    assert sink.events[0].outcome == "denied"
