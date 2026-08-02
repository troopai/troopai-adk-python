"""Sandbox observability: audit sink + span factory wiring.

- ``AuditSink`` ABC (+ ``NullAuditSink``, ``LoggingAuditSink``)
  receives structured events for every sandbox lifecycle transition.
- ``sandbox_span`` lives in ``tracing/spans.py`` and is
  re-exported here for ergonomic access.
- ``SandboxObservability`` is the run-scoped handle the capability tools emit through.
"""

from __future__ import annotations

from troopai.adk.sandbox.observability.audit_sink import (
    AuditSink,
    LoggingAuditSink,
    NullAuditSink,
    SandboxAuditEvent,
)
from troopai.adk.sandbox.observability.observability import SandboxObservability

__all__ = [
    "AuditSink",
    "LoggingAuditSink",
    "NullAuditSink",
    "SandboxAuditEvent",
    "SandboxObservability",
]
