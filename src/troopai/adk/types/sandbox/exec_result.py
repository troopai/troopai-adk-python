"""Command-execution result types for sandbox sessions.

These types describe the OUTPUT shape of a sandbox session's
run-a-command primitive. The session's input shape (command + timeout
+ user + shell) lives on ``BaseSandboxSession``; the result
shape lives here so capability tools and observers can depend on
Layer-1 types alone.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar, override

__all__ = [
    "ExecResult",
    "ExposedPortEndpoint",
    "PtyHandle",
]


_KNOWN_URL_SCHEMES: frozenset[str] = frozenset({"http", "https", "ws", "wss"})
"""URL schemes the framework's ``url_for`` helper can render safely.

Other schemes (ftp, ssh, redis, http+unix, …) have different port and
authority conventions; rendering them generically would silently
produce wrong URLs. Callers needing a non-listed scheme MUST format
the URL themselves.
"""


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExecResult:
    """Outcome of a single command run inside a sandbox session.

    Backends produce one ``ExecResult`` per non-PTY command invocation.
    The PTY path streams events instead; PTY teardown does NOT produce
    an ``ExecResult``.

    Attributes:
        stdout: Captured standard output bytes.
        stderr: Captured standard error bytes.
        exit_code: Process exit code. ``0`` indicates clean exit; any
            non-zero value MUST be surfaced to the caller — non-zero
            exits are NOT treated as exceptions at this layer (the
            tool's guardrail or the caller decides).
        duration_ms: Wall-clock duration in milliseconds. ``None``
            when the backend cannot measure (rare).
    """

    stdout: bytes
    """Captured standard output bytes."""

    stderr: bytes
    """Captured standard error bytes."""

    exit_code: int
    """Process exit code; ``0`` indicates clean exit."""

    duration_ms: int | None = None
    """Wall-clock duration in milliseconds (``None`` if unmeasured)."""

    @property
    def ok(self) -> bool:
        """True iff ``exit_code == 0``."""
        return self.exit_code == 0

    def decoded_stdout(self, *, encoding: str = "utf-8", errors: str = "strict") -> str:
        """Return ``stdout`` decoded with ``encoding`` (default UTF-8 strict).

        Defaults to ``errors="strict"``: malformed bytes raise
        ``UnicodeDecodeError`` rather than being silently replaced.
        Sandboxed code can emit arbitrary bytes (binary leaks, locale
        mismatches, attacker-injected payloads); silently masking them
        loses forensic signal. Pass ``errors="replace"`` to opt INTO
        lossy decoding.
        """
        return self.stdout.decode(encoding, errors=errors)

    def decoded_stderr(self, *, encoding: str = "utf-8", errors: str = "strict") -> str:
        """Return ``stderr`` decoded with ``encoding`` (default UTF-8 strict).

        Same fail-loud default as ``decoded_stdout`` — see that method's
        docstring for the rationale.
        """
        return self.stderr.decode(encoding, errors=errors)


@dataclasses.dataclass(frozen=True, kw_only=True)
class ExposedPortEndpoint:
    """Network endpoint where a sandbox-exposed port is reachable.

    Returned by ``BaseSandboxSession.resolve_exposed_port(port)``.
    Backends translate sandbox-internal port numbers to host- or
    tunnel-routable endpoints.

    Attributes:
        host: Hostname or IP address the endpoint listens on.
        port: TCP port number.
        tls: Whether the endpoint is TLS-terminated.
        query: Optional query-string fragment (no leading ``?``) the
            caller should append. Hosted providers occasionally bake
            auth tokens here.
    """

    host: str
    """Hostname or IP address the endpoint listens on."""

    port: int
    """TCP port number (1..65535)."""

    tls: bool = False
    """Whether the endpoint is TLS-terminated."""

    query: str = ""
    """Optional query-string fragment (no leading ``?``)."""

    DEFAULT_HTTP_PORT: ClassVar[int] = 80
    """Conventional HTTP port (used by ``url_for`` defaults)."""

    DEFAULT_HTTPS_PORT: ClassVar[int] = 443
    """Conventional HTTPS port (used by ``url_for`` defaults)."""

    def __post_init__(self) -> None:
        if len(self.host) == 0:
            raise ValueError("ExposedPortEndpoint.host must be non-empty")
        if self.port < 1 or self.port > 65535:
            raise ValueError(f"ExposedPortEndpoint.port must be in 1..65535, got {self.port}")

    def url_for(self, scheme: str = "http") -> str:
        """Render a URL for this endpoint (HTTP/HTTPS/WS/WSS only).

        Pass ``scheme="https"`` to mark the URL as secure; the
        ``:port`` suffix is omitted when the port matches the
        scheme's conventional default. Schemes outside
        ``{http, https, ws, wss}`` are rejected because their
        authority/port conventions differ — callers needing FTP,
        Redis, http+unix, etc., MUST format the URL themselves.
        """
        if len(scheme) == 0:
            raise ValueError("ExposedPortEndpoint.url_for: scheme must be non-empty")
        if scheme not in _KNOWN_URL_SCHEMES:
            raise ValueError(
                f"ExposedPortEndpoint.url_for: unsupported scheme {scheme!r}; "
                f"expected one of {sorted(_KNOWN_URL_SCHEMES)}"
            )
        default_port = self.DEFAULT_HTTPS_PORT if scheme in {"https", "wss"} else self.DEFAULT_HTTP_PORT
        host_part = f"{self.host}:{self.port}" if self.port != default_port else self.host
        query_part = f"?{self.query}" if len(self.query) > 0 else ""
        return f"{scheme}://{host_part}{query_part}"


@dataclasses.dataclass(frozen=True, kw_only=True)
class PtyHandle:
    """Opaque reference to an active PTY session inside a sandbox.

    Backends that support interactive command execution
    (``BaseSandboxSession.supports_pty() == True``) return a
    ``PtyHandle`` from ``pty_exec_start``; callers pass it back to
    ``pty_write_stdin(handle, data)`` and ``pty_terminate_all()``.

    The ``backend_payload`` is provider-opaque; framework code MUST
    treat it as a black box and never introspect.

    Attributes:
        session_id: Stable identifier scoped to the sandbox session.
        command: The shell command the PTY was opened for. Stored
            for tracing + audit; backends are free to truncate.
        backend_payload: Provider-specific routing data. Opaque.
    """

    session_id: str
    """Stable identifier scoped to the sandbox session."""

    command: str
    """The shell command the PTY was opened for."""

    backend_payload: object
    """Provider-specific routing data; framework code MUST NOT introspect."""

    @override
    def __repr__(self) -> str:
        # Redact backend_payload so provider tokens, exec IDs, or socket
        # paths cannot leak into tracebacks or log captures. Callers
        # that legitimately need the payload (e.g. backend impls)
        # access ``self.backend_payload`` directly.
        return f"PtyHandle(session_id={self.session_id!r}, command={self.command!r}, backend_payload=<opaque>)"
