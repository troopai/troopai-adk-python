"""Helpers for serializing sandbox session events + probing streams.

``event_to_json_line`` emits one JSONL line per event (sorted keys
+ separator-tight serialization so the audit log is grep-friendly
and byte-stable across runs).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.sandbox.session.events import SandboxSessionEvent

logger = logging.getLogger(__name__)

__all__ = ["event_to_json_line", "safe_decode_with_max_chars"]


def safe_decode_with_max_chars(payload: bytes, *, max_chars: int) -> str:
    """Decode ``payload`` as UTF-8 with replacement; cap output length at ``max_chars``.

    Used by backend event-builders that opt into
    ``EventPayloadPolicy.include_exec_output`` — they call this to
    cap the stdout / stderr text attached to finish events at the
    policy's per-stream char ceiling. The ellipsis ``…`` is
    INCLUDED in the ``max_chars`` budget: when truncation happens,
    the returned string is at most ``max_chars`` characters long
    (the slice is sized at ``max_chars - 1`` to leave room for the
    marker). ``max_chars == 0`` produces ``""`` regardless of the
    payload, matching downstream length-bound consumers.
    """
    decoded = payload.decode("utf-8", errors="replace")
    if len(decoded) <= max_chars:
        return decoded
    if max_chars <= 0:
        return ""
    if max_chars == 1:
        return "…"
    return decoded[: max_chars - 1] + "…"


def event_to_json_line(event: SandboxSessionEvent) -> str:
    """Serialize ``event`` to one JSONL line (newline-terminated).

    Sorted keys and tight separators give a byte-stable rendering
    so diffs across runs surface only real changes. When the
    payload contains a non-JSON-serializable object inside
    ``event.data`` (e.g. a ``Path``), the function logs the
    event identity at ERROR and re-raises — silent loss of an
    audit record is forbidden by the contract.
    """
    payload = event.model_dump(mode="json")
    try:
        return json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
    except TypeError:
        logger.exception(
            "event_to_json_line: serialization failed for event_id=%s op=%s phase=%s",
            event.event_id,
            event.op,
            event.phase,
        )
        raise
