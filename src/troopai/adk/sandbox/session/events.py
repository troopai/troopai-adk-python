"""Typed audit events for sandbox sessions.

Every backend operation (start / run / read / write / ...) emits a
``SandboxSessionStartEvent`` before invocation and a
``SandboxSessionFinishEvent`` afterwards. Sinks (file, HTTP, in-memory)
consume the discriminated ``SandboxSessionEvent`` union to record audit
trails without backend-specific knowledge.

The event payload policy controls how much potentially sensitive or
large data is included. By default exec stdout/stderr is OFF —
sandboxed commands can emit any payload, so opt-in is the cost-
conservative default.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from troopai.adk.sandbox.session.op_codes import ErrorCode, OpName

logger = logging.getLogger(__name__)

__all__ = [
    "EventPayloadPolicy",
    "EventPhase",
    "SandboxSessionEvent",
    "SandboxSessionEventBase",
    "SandboxSessionFinishEvent",
    "SandboxSessionStartEvent",
    "validate_sandbox_session_event",
]

EventPhase = Literal["start", "finish"]
"""Two-phase event lifecycle keyed by ``phase``."""


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class EventPayloadPolicy(BaseModel):
    """How much sensitive / large data an event records.

    Attributes:
        include_exec_output: When True, finish events for ``run``
            carry truncated stdout/stderr text. Default OFF — exec
            output can be noisy or sensitive.
        max_stdout_chars: Decoded-string ceiling on stdout when
            ``include_exec_output`` is True. ``0`` disables.
        max_stderr_chars: Mirror of ``max_stdout_chars`` for stderr.
        include_write_len: When True, write events record a
            best-effort byte length (NEVER file bytes).

    Frozen on purpose: the instrumentation layer merges policies
    using Pydantic ``model_fields_set``, which only tracks fields
    supplied through construction / ``model_validate``. A
    post-construction attribute assignment would NOT join
    ``model_fields_set`` and would be silently dropped from the
    override set — a security-relevant redaction mis-merge.
    Freezing makes that misuse raise instead of silently no-op.
    Build a new policy (or use ``model_copy(update=...)``) to
    change a field.
    """

    model_config = ConfigDict(frozen=True)

    include_exec_output: bool = Field(default=False)
    """Default OFF — exec output is opt-in."""

    max_stdout_chars: int = Field(default=8_000, ge=0)
    """Decoded-string ceiling for stdout."""

    max_stderr_chars: int = Field(default=8_000, ge=0)
    """Decoded-string ceiling for stderr."""

    include_write_len: bool = Field(default=True)
    """Best-effort byte length on write events."""


class SandboxSessionEventBase(BaseModel):
    """Shared fields for all sandbox audit events.

    Attributes:
        event_id: Unique per-event UUID generated at emission time.
        ts: UTC datetime at which the event was emitted.
        session_id: UUID identifying the sandbox session that emitted
            the event.
        seq: Per-session monotonic sequence number; used to detect gaps
            in an audit trail.
        op: Operation this event belongs to (e.g. ``"run"``,
            ``"read"``, ``"write"``).
        phase: ``"start"`` or ``"finish"`` — discriminator for the
            union type.
        span_id: Correlates the start and finish records for one
            operation; matches the ADK tracing span id when tracing is
            enabled.
        parent_span_id: Optional parent span id when traces nest.
        trace_id: Optional trace identifier that groups spans from a
            single agent run.
        data: Operation-specific metadata bag (paths, argv, timings,
            exit codes, …).
    """

    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    """Unique event identifier."""

    ts: datetime = Field(default_factory=_utcnow)
    """UTC timestamp at emission."""

    session_id: uuid.UUID
    """Identifier of the emitting sandbox session."""

    seq: int
    """Per-session monotonic sequence number."""

    op: OpName
    """Operation this event belongs to."""

    phase: EventPhase
    """``"start"`` or ``"finish"`` — discriminator for the union."""

    span_id: str
    """Correlates start/finish records — ADK tracing span id when available."""

    parent_span_id: str | None = None
    """Optional parent span id when traces nest."""

    trace_id: str | None = None
    """Optional trace identifier."""

    data: dict[str, object] = Field(default_factory=dict)
    """Operation-specific metadata bag (paths, argv, timings, ...)."""


class SandboxSessionStartEvent(SandboxSessionEventBase):
    """Emitted before backend code touches the operation's primitives."""

    phase: Literal["start"] = Field(default="start")
    """Constant discriminator."""


class SandboxSessionFinishEvent(SandboxSessionEventBase):
    """Emitted after backend code returns or raises.

    Inherits all fields from ``SandboxSessionEventBase``. Additional
    fields below describe the operation outcome.

    Attributes:
        event_id: Unique per-event UUID (inherited).
        ts: UTC datetime at emission (inherited).
        session_id: Sandbox session identifier (inherited).
        seq: Per-session monotonic sequence number (inherited).
        op: Operation name (inherited).
        phase: Constant ``"finish"`` — discriminator for the union.
        span_id: Span correlation id (inherited).
        parent_span_id: Optional parent span id (inherited).
        trace_id: Optional trace identifier (inherited).
        data: Operation-specific metadata bag (inherited).
        ok: True iff the operation completed without raising.
        duration_ms: Wall-clock duration of the operation in milliseconds.
        error_code: Coarse error classification when ``ok=False``.
        error_type: Fully-qualified exception class name when
            ``ok=False``; ``None`` on success.
        error_message: One-line exception summary when ``ok=False``;
            ``None`` on success.
        stdout: Truncated decoded stdout; populated only when
            ``EventPayloadPolicy.include_exec_output`` is True.
        stderr: Truncated decoded stderr; populated only when
            ``EventPayloadPolicy.include_exec_output`` is True.
        stdout_bytes: Raw stdout bytes for per-sink policy application.
            Excluded from JSON serialization.
        stderr_bytes: Raw stderr bytes for per-sink policy application.
            Excluded from JSON serialization.
    """

    phase: Literal["finish"] = Field(default="finish")
    """Constant discriminator."""

    ok: bool
    """True iff the operation completed without raising."""

    duration_ms: float
    """Wall-clock duration in milliseconds."""

    error_code: ErrorCode | None = None
    """Coarse error classification when ``ok=False``."""

    error_type: str | None = None
    """Fully-qualified exception class name when ``ok=False``."""

    error_message: str | None = None
    """One-line exception summary when ``ok=False``."""

    stdout: str | None = None
    """Truncated decoded stdout (opt-in via ``EventPayloadPolicy``)."""

    stderr: str | None = None
    """Truncated decoded stderr (opt-in via ``EventPayloadPolicy``)."""

    stdout_bytes: bytes | None = Field(default=None, exclude=True)
    """Raw stdout for per-sink policy application — excluded from JSON serialization."""

    stderr_bytes: bytes | None = Field(default=None, exclude=True)
    """Raw stderr for per-sink policy application — excluded from JSON serialization."""


SandboxSessionEvent = Annotated[
    SandboxSessionStartEvent | SandboxSessionFinishEvent,
    Field(discriminator="phase"),
]
"""Discriminated union of start / finish events keyed by ``phase``."""

_SANDBOX_SESSION_EVENT_ADAPTER: TypeAdapter[SandboxSessionEvent] = TypeAdapter(SandboxSessionEvent)


def validate_sandbox_session_event(obj: object) -> SandboxSessionEvent:
    """Parse a serialized event payload into the correct phase-specific model.

    Useful when reading a JSONL audit trail back into memory: the
    discriminator drives the union resolution so each line gets the
    right model type. On parse failure we log the partial-parse
    ``session_id`` (when visible) at ERROR before re-raising — an
    operator forensicly walking a multi-MB audit log can grep out
    the offending session without having to bisect by hand.
    """
    try:
        return _SANDBOX_SESSION_EVENT_ADAPTER.validate_python(obj)
    except ValidationError:
        partial_session_id = obj.get("session_id") if isinstance(obj, dict) else None
        logger.exception(
            "validate_sandbox_session_event: parse failed (session_id=%s)",
            partial_session_id,
        )
        raise
