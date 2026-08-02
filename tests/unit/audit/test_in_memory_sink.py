from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from troopai.adk.audit.event import AuditEvent
from troopai.adk.audit.sink import AuditSink, InMemoryAuditSink


def _event(outcome: Literal["ok", "denied", "error"] = "ok") -> AuditEvent:
    return AuditEvent(
        tenant_id="t1",
        agent_name="a",
        tool_name="tool",
        tool_call_id="c1",
        args_hash="x",
        result_hash=None,
        outcome=outcome,
        timestamp=datetime.now(UTC),
    )


def test_in_memory_is_an_audit_sink() -> None:
    assert isinstance(InMemoryAuditSink(), AuditSink)


async def test_records_events_in_order() -> None:
    sink = InMemoryAuditSink()
    await sink.record(_event("denied"))
    await sink.record(_event("ok"))
    assert [e.outcome for e in sink.events] == ["denied", "ok"]
