from __future__ import annotations


def test_public_exports() -> None:
    from troopai.adk.audit import (
        AuditEvent,
        AuditSink,
        InMemoryAuditSink,
        JsonlFileAuditSink,
        hash_payload,
    )

    assert isinstance(InMemoryAuditSink(), AuditSink)
    assert callable(hash_payload)
    assert AuditEvent is not None
    assert JsonlFileAuditSink is not None
