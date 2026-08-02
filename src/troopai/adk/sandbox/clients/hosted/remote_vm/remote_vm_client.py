"""``RemoteVMSandboxClient`` — intermediate base for hosted bridges.

Shared machinery for every hosted-provider bridge: HTTP client
construction, retry policy, transport-error mapping. Concrete
hosted clients (E2B, Vercel, Modal, ...) extend this and override
provider-specific create / resume / delete / auth.
"""

from __future__ import annotations

import abc
import logging
import random
from typing import TYPE_CHECKING, Any, Literal, override

from troopai.adk.exceptions.exceptions import (
    ExecTimeoutError,
    ExecTransportError,
    SandboxConfigurationError,
    SandboxRuntimeError,
    UnsupportedSandboxClientError,
)
from troopai.adk.sandbox.clients.base import BaseSandboxClient, BaseSandboxClientOptions
from troopai.adk.sandbox.clients.session import BaseSandboxSession
from troopai.adk.types.sandbox.exec_result import ExecResult
from troopai.adk.types.sandbox.session_state import SandboxSessionState

if TYPE_CHECKING:
    from troopai.adk.types.sandbox.manifest import Manifest
    from troopai.adk.types.sandbox.snapshot import SnapshotSpec

logger = logging.getLogger(__name__)

__all__ = [
    "RemoteVMSandboxClient",
    "RemoteVMSandboxClientOptions",
    "build_httpx_client",
    "map_http_error_to_sandbox_error",
    "parse_exec_result",
    "retry_async_call",
]


def _require_httpx() -> None:
    try:
        import httpx  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ImportError as exc:
        raise UnsupportedSandboxClientError(
            "RemoteVMSandboxClient (and hosted bridges) require the optional "
            "'httpx' package: pip install 'troopai-adk-python[sandbox-remote-vm]'",
        ) from exc


# ---------------------------------------------------------------------------
# TRV.1 — HTTP client construction
# ---------------------------------------------------------------------------


def build_httpx_client(
    *,
    base_url: str | None,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
) -> Any:
    """Construct a configured ``httpx.AsyncClient``.

    Hosted-bridge subclasses call this in their ``create()`` to get a
    client pre-configured with the provider's base URL, auth headers,
    and per-request timeout. The returned client is the caller's
    responsibility to close (typically via the session's ``aclose``).
    """
    _require_httpx()
    import httpx

    return httpx.AsyncClient(
        base_url=base_url or "",
        headers=headers or {},
        timeout=httpx.Timeout(timeout),
    )


# ---------------------------------------------------------------------------
# TRV.2 — Retry policy
# ---------------------------------------------------------------------------


async def retry_async_call(
    func: Any,
    *args: Any,
    max_retries: int = 0,
    initial_wait: float = 1.0,
    max_wait: float = 30.0,
    **kwargs: Any,
) -> Any:
    """Call ``func(*args, **kwargs)`` with exponential-backoff retry.

    Retries on:
    - ``httpx.TransportError`` (network / DNS / TLS / connect failures)
    - HTTP 5xx responses (when the wrapped function returns a response
      object with a ``status_code`` >= 500)
    - HTTP 429 (rate limited)

    Does NOT retry on:
    - HTTP 4xx (other than 429) — these surface as
      ``SandboxConfigurationError`` via ``map_http_error_to_sandbox_error``.
    - Non-transport exceptions raised inside ``func``.

    Returns the function's return value on success. Raises the last
    exception (wrapped to ``ExecTransportError``) when the budget is
    exhausted.
    """
    _require_httpx()
    import asyncio

    import httpx

    attempt = 0
    last_exc: BaseException | None = None
    while attempt <= max_retries:
        try:
            result = await func(*args, **kwargs)
            status = getattr(result, "status_code", None)
            if isinstance(status, int) and (status == 429 or status >= 500):
                # A returned (non-raising) response carrying a retriable
                # status. Honour the documented retry contract: back off
                # and retry while budget remains, otherwise return the
                # last response so the caller maps it via
                # map_http_error_to_sandbox_error.
                if attempt >= max_retries:
                    return result
            else:
                return result
        except httpx.TransportError as exc:
            last_exc = exc
            if attempt >= max_retries:
                raise ExecTransportError(
                    f"RemoteVM transport error after {max_retries} retries: {exc}",
                ) from exc
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429 or exc.response.status_code >= 500:
                last_exc = exc
                if attempt >= max_retries:
                    raise ExecTransportError(
                        f"RemoteVM HTTP {exc.response.status_code} after {max_retries} retries: {exc}",
                    ) from exc
            else:
                # Non-retriable 4xx — let the caller map via
                # map_http_error_to_sandbox_error.
                raise
        # Exponential backoff with jitter.
        wait = min(initial_wait * (2**attempt), max_wait)
        jitter = wait * random.uniform(0.0, 0.3)
        await asyncio.sleep(wait + jitter)
        attempt += 1
    # Should be unreachable; defensive raise.
    if last_exc is not None:
        raise ExecTransportError(
            f"RemoteVM retry budget exhausted: {last_exc}",
        ) from last_exc
    raise ExecTransportError("RemoteVM retry budget exhausted (no exception captured)")


# ---------------------------------------------------------------------------
# TRV.3 — HTTP error → SandboxError mapping
# ---------------------------------------------------------------------------


def map_http_error_to_sandbox_error(
    status_code: int,
    body_text: str,
    *,
    backend_id: str = "remote_vm",
) -> Exception:
    """Translate an HTTP status code + body into a typed SandboxError.

    Mappings:
    - 401, 403 → ``SandboxConfigurationError`` (auth / permission).
    - 404 → ``SandboxConfigurationError`` (resource missing).
    - 408 → ``ExecTimeoutError``.
    - 409, 422 → ``SandboxConfigurationError`` (invalid input shape).
    - 429 → ``ExecTransportError`` (caller should retry).
    - Any other 4xx → ``SandboxConfigurationError``.
    - 5xx → ``SandboxRuntimeError`` (server-side failure).
    """
    snippet = body_text[:500] if len(body_text) > 0 else "<empty>"
    if status_code in {401, 403}:
        return SandboxConfigurationError(
            f"RemoteVM {backend_id}: HTTP {status_code} auth/permission denied: {snippet}",
        )
    if status_code == 404:
        return SandboxConfigurationError(
            f"RemoteVM {backend_id}: HTTP 404 resource not found: {snippet}",
        )
    if status_code == 408:
        return ExecTimeoutError(
            f"RemoteVM {backend_id}: HTTP 408 request timeout: {snippet}",
        )
    if status_code in {409, 422}:
        return SandboxConfigurationError(
            f"RemoteVM {backend_id}: HTTP {status_code} invalid request: {snippet}",
        )
    if status_code == 429:
        return ExecTransportError(
            f"RemoteVM {backend_id}: HTTP 429 rate limited: {snippet}",
        )
    if 400 <= status_code < 500:
        return SandboxConfigurationError(
            f"RemoteVM {backend_id}: HTTP {status_code} client error: {snippet}",
        )
    if status_code >= 500:
        return SandboxRuntimeError(
            f"RemoteVM {backend_id}: HTTP {status_code} server error: {snippet}",
        )
    return SandboxRuntimeError(
        f"RemoteVM {backend_id}: unexpected HTTP {status_code}: {snippet}",
    )


# ---------------------------------------------------------------------------
# TRV.4 — ExecResult normalization
# ---------------------------------------------------------------------------


def parse_exec_result(
    raw: dict[str, Any],
    *,
    stdout_key: str = "stdout",
    stderr_key: str = "stderr",
    exit_code_key: str = "exit_code",
    duration_ms_key: str = "duration_ms",
    encoding: Literal["text", "base64"] = "text",
) -> ExecResult:
    """Build an ``ExecResult`` from a provider's JSON exec response.

    Hosted bridges with different JSON keys (e.g. Vercel uses
    ``exitCode`` camelCase, Modal returns nested) pass per-provider
    key overrides. Stdout/stderr are decoded as bytes according to the
    provider-declared ``encoding``:

    - ``"text"`` (default): the string is UTF-8 encoded back to bytes.
    - ``"base64"``: the string is base64-decoded.

    The encoding is provider-declared, never guessed: a heuristic
    "try base64 first" silently corrupts plain-text output whose length
    and alphabet happen to satisfy base64 (e.g. ``"test"`` would decode
    to ``b"\\xb5\\xeb-"``). Bridges whose exec endpoint returns
    base64-encoded streams pass ``encoding="base64"``.
    """
    import base64

    def _to_bytes(value: Any) -> bytes:
        if value is None:
            return b""
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            if encoding == "base64":
                return base64.b64decode(value)
            return value.encode("utf-8", errors="replace")
        raise TypeError(
            f"parse_exec_result: stdout/stderr must be bytes/str/None, got {type(value).__name__}",
        )

    stdout = _to_bytes(raw.get(stdout_key))
    stderr = _to_bytes(raw.get(stderr_key))
    exit_code = raw.get(exit_code_key)
    if exit_code is None:
        exit_code = raw.get("exitCode")  # camelCase fallback
    if exit_code is None:
        exit_code = -1
    duration_ms = raw.get(duration_ms_key)
    if duration_ms is None:
        duration_ms = raw.get("durationMs")
    if isinstance(duration_ms, float):
        duration_ms = int(duration_ms)
    return ExecResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=int(exit_code),
        duration_ms=duration_ms if isinstance(duration_ms, int) else None,
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class RemoteVMSandboxClientOptions(BaseSandboxClientOptions):
    """Base options for any RemoteVM-backed bridge.

    Concrete provider option classes (E2B, Vercel, Modal, ...)
    inherit from this and add their own auth fields.

    Attributes:
        base_url: Optional base URL override. ``None`` uses the
            provider's default endpoint.
        max_retries: Retry budget for transient transport failures
            (``httpx.TransportError``, HTTP 429, HTTP 5xx). Defaults to
            ``0`` (no retries) — each retry is a real billable provider
            call, so the developer opts into the budget explicitly.
        request_timeout: Per-request wall-clock timeout in seconds.
    """

    base_url: str | None = None
    """Optional base URL override. None means provider default."""

    max_retries: int = 0
    """Retry budget for transient transport failures (0 = no retries)."""

    request_timeout: float = 30.0
    """Per-request wall-clock timeout (seconds)."""


class RemoteVMSandboxClient(
    BaseSandboxClient[RemoteVMSandboxClientOptions],
    abc.ABC,
):
    """Intermediate base for every hosted-bridge client.

    Provides shared HTTP construction + retry policy + error mapping
    via the module-level ``build_httpx_client``, ``retry_async_call``,
    ``map_http_error_to_sandbox_error``, and ``parse_exec_result``
    helpers. Subclasses override create / resume / delete to call
    the provider's specific API.
    """

    backend_id = "remote_vm"

    def __init__(self) -> None:
        _require_httpx()

    @abc.abstractmethod
    @override
    async def create(
        self,
        *,
        snapshot: SnapshotSpec | None = None,
        snapshot_store: Any | None = None,
        manifest: Manifest | None = None,
        options: RemoteVMSandboxClientOptions,
    ) -> BaseSandboxSession: ...

    @override
    async def delete(self, session: BaseSandboxSession) -> BaseSandboxSession:
        await session.aclose()
        return session

    @abc.abstractmethod
    @override
    async def resume(self, state: SandboxSessionState) -> BaseSandboxSession: ...

    @override
    def deserialize_session_state(self, payload: dict[str, Any]) -> SandboxSessionState:
        return SandboxSessionState.model_validate(payload)
