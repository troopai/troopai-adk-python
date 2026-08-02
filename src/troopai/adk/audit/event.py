"""Audit event type + payload hashing.

The audit log records privacy-preserving hashes of tool arguments and
results, never the raw payloads.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class AuditEvent:
    """One tool-call resolution recorded to an :class:`AuditSink`.

    Frozen + keyword-only: an audit record is immutable once created
    (append-only semantics) and is always constructed by name.

    Attributes:
        tenant_id: Tenant the run belonged to, or ``None`` if untenanted.
        agent_name: Name of the agent that issued the call.
        tool_name: Name of the tool.
        tool_call_id: Provider tool-call id (correlates with the turn).
        args_hash: sha256 hex of the canonicalised arguments.
        result_hash: sha256 hex of the result. ``None`` for denied calls
            and for errors that re-raise; set to the hash of the
            error-message string when an error is returned as the result.
        outcome: ``"ok"`` (executed), ``"denied"`` (allowlist), or
            ``"error"`` (tool raised or timed out).
        timestamp: UTC time the event was created.
    """

    tenant_id: str | None
    """Tenant the run belonged to, or ``None`` if untenanted."""
    agent_name: str
    """Name of the agent that issued the call."""
    tool_name: str
    """Name of the tool."""
    tool_call_id: str
    """Provider tool-call id (correlates with the turn)."""
    args_hash: str
    """sha256 hex of the canonicalised arguments."""
    result_hash: str | None
    """sha256 hex of the result. ``None`` for denied calls and re-raised
    errors; set to the hash of the error message when an error is returned
    as the result."""
    outcome: Literal["ok", "denied", "error"]
    """Resolution outcome."""
    timestamp: datetime
    """UTC time the event was created."""


def hash_payload(value: Any) -> str:
    """Return a stable sha256 hex digest of ``value``.

    Canonical JSON (sorted keys) when serialisable, else ``str(value)``.
    Never raises: an unrepresentable value yields the sentinel
    ``"<unhashable>"``. Used to record arguments/results without storing
    the raw (possibly sensitive) payloads.

    Args:
        value: The value to hash. Any JSON-serialisable type is accepted;
            non-serialisable values fall back to ``str()``.

    Returns:
        A lowercase sha256 hex digest string, or ``"<unhashable>"`` if
        the value cannot be represented at all.
    """
    try:
        serialized = json.dumps(value, sort_keys=True, default=str)
    except Exception:
        logger.debug("hash_payload: canonical JSON failed; falling back to str()")
        try:
            serialized = str(value)
        except Exception:
            logger.debug("hash_payload: value is unrepresentable; using sentinel")
            return "<unhashable>"
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = ["AuditEvent", "hash_payload"]
