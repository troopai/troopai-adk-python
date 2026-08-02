"""Append-only tool-call audit logging with pluggable sinks.

Heavy backends live under ``sinks/`` and are imported directly by the
caller (they require optional extras):
``from troopai.adk.audit.sinks.s3 import S3AuditSink``.
"""

from __future__ import annotations

from troopai.adk.audit.event import AuditEvent, hash_payload
from troopai.adk.audit.sink import AuditSink, InMemoryAuditSink, JsonlFileAuditSink

__all__ = [
    "AuditEvent",
    "AuditSink",
    "InMemoryAuditSink",
    "JsonlFileAuditSink",
    "hash_payload",
]
