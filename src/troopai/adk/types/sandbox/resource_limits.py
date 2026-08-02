"""Resource limits for sandbox sessions.

``SandboxResourceLimits`` is a backend-agnostic declaration of CPU,
memory, disk, time, process-count, and egress caps. Each backend
translates the settable fields to its wire format:

- Docker: ``--cpus``, ``--memory``, ``--pids-limit``, ``--tmpfs`` size.
- K8s: container ``resources.limits`` + ``ephemeralStorage`` limit.
- Hosted: provider-specific REST fields.
- LocalSubprocess: best-effort via ``resource`` module rlimits.

All fields default to ``None``: "no ADK-enforced limit; respect
backend default." Developers opt INTO specific caps — never opting
OUT of a cap they did not choose.
"""

from __future__ import annotations

import dataclasses

__all__ = ["SandboxResourceLimits"]


@dataclasses.dataclass(frozen=True, kw_only=True)
class SandboxResourceLimits:
    """Caps applied to a sandbox session by its backend.

    Attributes:
        cpu_cores: Maximum CPU cores the session may consume.
            Fractional (e.g. ``0.5`` = half a core).
        memory_mb: Maximum resident memory in MiB.
        disk_mb: Maximum ephemeral disk in MiB.
        exec_timeout: Per-command wall-clock cap in seconds.
        session_timeout: Whole-session wall-clock cap in seconds.
        max_processes: Maximum number of processes (pids limit).
        max_egress_bytes: Maximum cumulative outbound network bytes
            before the backend kills or throttles the session.
    """

    cpu_cores: float | None = None
    """Maximum CPU cores; fractional allowed."""

    memory_mb: int | None = None
    """Maximum resident memory in MiB."""

    disk_mb: int | None = None
    """Maximum ephemeral disk in MiB."""

    exec_timeout: float | None = None
    """Per-command wall-clock cap (seconds)."""

    session_timeout: float | None = None
    """Whole-session wall-clock cap (seconds)."""

    max_processes: int | None = None
    """Maximum number of processes / threads."""

    max_egress_bytes: int | None = None
    """Maximum cumulative outbound network bytes."""

    def __post_init__(self) -> None:
        for name, value in (
            ("cpu_cores", self.cpu_cores),
            ("memory_mb", self.memory_mb),
            ("disk_mb", self.disk_mb),
            ("exec_timeout", self.exec_timeout),
            ("session_timeout", self.session_timeout),
            ("max_processes", self.max_processes),
            ("max_egress_bytes", self.max_egress_bytes),
        ):
            if value is None:
                continue
            if value <= 0:
                raise ValueError(f"SandboxResourceLimits.{name} must be positive when set, got {value}")

    def is_unbounded(self) -> bool:
        """True iff every field is ``None`` (backend default behavior)."""
        return all(
            v is None
            for v in (
                self.cpu_cores,
                self.memory_mb,
                self.disk_mb,
                self.exec_timeout,
                self.session_timeout,
                self.max_processes,
                self.max_egress_bytes,
            )
        )
