from __future__ import annotations

import pytest

from troopai.adk.audit import InMemoryAuditSink
from troopai.adk.audit.event import AuditEvent
from troopai.adk.run.config import RunConfig
from troopai.adk.run.governance import emit_audit


async def test_emit_records_to_sink() -> None:
    sink = InMemoryAuditSink()
    config = RunConfig(audit_sink=sink)
    await emit_audit(
        config,
        tenant_id="t1",
        agent_name="a",
        tool_name="tool",
        call_id="c1",
        args={"q": 1},
        outcome="ok",
        result="hi",
    )
    assert len(sink.events) == 1
    event = sink.events[0]
    assert event.outcome == "ok"
    assert event.result_hash is not None  # result provided -> hashed


async def test_emit_without_result_leaves_result_hash_none() -> None:
    sink = InMemoryAuditSink()
    await emit_audit(
        RunConfig(audit_sink=sink),
        tenant_id="t1",
        agent_name="a",
        tool_name="tool",
        call_id="c1",
        args="{}",
        outcome="denied",
    )
    assert sink.events[0].result_hash is None


async def test_emit_is_noop_without_sink() -> None:
    await emit_audit(RunConfig(), tenant_id="t1", agent_name="a", tool_name="t", call_id="c", args="{}", outcome="ok")


async def test_emit_best_effort_swallows_sink_failure() -> None:
    class _Boom:
        async def record(self, event: AuditEvent) -> None:
            raise RuntimeError("sink down")

    await emit_audit(
        RunConfig(audit_sink=_Boom()),
        tenant_id="t1",
        agent_name="a",
        tool_name="t",
        call_id="c",
        args="{}",
        outcome="ok",
    )


async def test_emit_strict_reraises_sink_failure() -> None:
    class _Boom:
        async def record(self, event: AuditEvent) -> None:
            raise RuntimeError("sink down")

    with pytest.raises(RuntimeError, match="sink down"):
        await emit_audit(
            RunConfig(audit_sink=_Boom(), audit_strict=True),
            tenant_id="t1",
            agent_name="a",
            tool_name="t",
            call_id="c",
            args="{}",
            outcome="ok",
        )
