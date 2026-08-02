"""Operation-name + error-code identifiers for sandbox audit events.

These are framework-internal string enums consumed by
``SandboxSessionEvent``. Adding a new ``OpName`` is a one-line
change; backends that emit events MUST pick a name from this
enum so consumers (audit sinks, tracing exporters) don't have
to discover names by string interpolation.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["ErrorCode", "OpName"]


class OpName(StrEnum):
    """Canonical sandbox-session operation names emitted in events."""

    START = "start"
    STOP = "stop"
    SHUTDOWN = "shutdown"
    RUN = "run"
    READ = "read"
    WRITE = "write"
    LS = "ls"
    MKDIR = "mkdir"
    RM = "rm"
    EXTRACT = "extract"
    APPLY_PATCH = "apply_patch"
    APPLY_MANIFEST = "apply_manifest"
    PERSIST_WORKSPACE = "persist_workspace"
    HYDRATE_WORKSPACE = "hydrate_workspace"
    RESOLVE_EXPOSED_PORT = "resolve_exposed_port"
    PTY_START = "pty_start"
    PTY_WRITE_STDIN = "pty_write_stdin"
    PTY_TERMINATE_ALL = "pty_terminate_all"


class ErrorCode(StrEnum):
    """Canonical error-code labels surfaced on finish events.

    Backends pick the most specific code that fits the failure;
    consumers can filter audit streams by code without parsing
    free-form error messages.
    """

    CONFIGURATION = "configuration"
    """Bad input / configuration — operation can never succeed as called."""

    RUNTIME = "runtime"
    """Operation failed at runtime (backend-side); retryable per provider."""

    TIMEOUT = "timeout"
    """Wall-clock cap exceeded."""

    TRANSPORT = "transport"
    """Network / transport error reaching the backend."""

    NON_ZERO_EXIT = "non_zero_exit"
    """Exec command returned a non-zero exit code."""

    WORKSPACE_IO = "workspace_io"
    """File operation failed (read/write/ls/mkdir/rm/extract)."""

    WORKSPACE_NOT_FOUND = "workspace_not_found"
    """Read/rm targeted a path that does not exist."""

    SNAPSHOT = "snapshot"
    """Snapshot persist / restore error."""

    APPLY_PATCH = "apply_patch"
    """apply_patch failed (decode / diff / file-not-found)."""

    GUARDRAIL = "guardrail"
    """A guardrail tripped on input or output."""

    CONCURRENCY = "concurrency"
    """Concurrency guard refused the operation."""

    UNEXPECTED = "unexpected"
    """Anything not covered above — fallback for surprise paths."""
