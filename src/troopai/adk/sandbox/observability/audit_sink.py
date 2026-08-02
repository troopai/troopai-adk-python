"""``AuditSink`` — pluggable destination for sandbox lifecycle events.

The Runner emits one ``SandboxAuditEvent`` per lifecycle transition
(session start/stop, command start/end, snapshot persist, policy
violation, runtime error). Sinks fan out to logging / SIEM / OTel /
custom destinations.
"""

from __future__ import annotations

import abc
import dataclasses
import logging
from datetime import UTC, datetime
from typing import Any, Literal, override

__all__ = [
    "AuditSink",
    "LoggingAuditSink",
    "NullAuditSink",
    "SandboxAuditEvent",
]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SandboxAuditEvent:
    """Structured audit event emitted on every sandbox lifecycle transition.

    Attributes:
        event_type: Discriminator (``"start"`` / ``"stop"`` / ``"exec"``
            / ``"snapshot"`` / ``"violation"`` / ``"error"``).
        agent_name: Agent the run belongs to.
        session_id: Backend-assigned session identifier, or ``None``
            before the session is started.
        backend_id: Backend identifier (``"unix_local"`` / ``"docker"``
            / hosted-provider name).
        command: Command associated with the event (truncated /
            redacted), or ``None`` for non-exec events.
        exit_code: Process exit code for exec events; ``None`` otherwise.
        usage: Optional usage dict (cpu_ms, memory_peak_mb, …).
        error: Optional human-readable error reason for violation /
            error events.
        timestamp_iso: ISO-8601 UTC timestamp the event was emitted.
        extra: Provider-specific extras (opaque to the framework).
    """

    event_type: Literal["start", "stop", "exec", "snapshot", "violation", "error"]
    """Discriminator."""

    agent_name: str
    """Agent the run belongs to."""

    backend_id: str
    """Backend identifier."""

    session_id: str | None = None
    """Backend-assigned session identifier."""

    command: str | None = None
    """Command associated with the event (truncated)."""

    exit_code: int | None = None
    """Process exit code for exec events."""

    usage: dict[str, int] | None = None
    """Optional usage dict (cpu_ms, memory_peak_mb, ...)."""

    error: str | None = None
    """Optional error reason for violation/error events."""

    timestamp_iso: str = dataclasses.field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
    )
    """ISO-8601 UTC timestamp the event was emitted."""

    extra: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Provider-specific extras (opaque to the framework)."""


class AuditSink(abc.ABC):
    """Abstract destination for sandbox audit events."""

    @abc.abstractmethod
    async def emit(self, event: SandboxAuditEvent) -> None:
        """Record ``event``. Concrete sinks decide format + destination."""


class NullAuditSink(AuditSink):
    """No-op sink. The framework default — zero overhead."""

    @override
    async def emit(self, event: SandboxAuditEvent) -> None:
        del event


class LoggingAuditSink(AuditSink):
    """Sink that forwards events to a Python logger.

    Attributes:
        logger: Logger instance (default: this module's logger).
        level_map: Maps event_type → logging level (defaults below).
    """

    _DEFAULT_LEVELS: dict[str, int] = {
        "start": logging.INFO,
        "stop": logging.INFO,
        "exec": logging.DEBUG,
        "snapshot": logging.INFO,
        "violation": logging.WARNING,
        "error": logging.ERROR,
    }

    def __init__(
        self,
        logger: logging.Logger | None = None,
        *,
        level_map: dict[str, int] | None = None,
    ) -> None:
        self._logger = logger or logging.getLogger(__name__)
        self._level_map = dict(self._DEFAULT_LEVELS)
        if level_map is not None:
            self._level_map.update(level_map)

    @override
    async def emit(self, event: SandboxAuditEvent) -> None:
        level = self._level_map.get(event.event_type, logging.INFO)
        self._logger.log(
            level,
            "sandbox.%s backend=%s agent=%s session=%s command=%r exit_code=%s error=%s",
            event.event_type,
            event.backend_id,
            event.agent_name,
            event.session_id,
            event.command,
            event.exit_code,
            event.error,
        )
