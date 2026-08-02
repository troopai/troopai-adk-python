"""``RemoteVMSandboxSession`` — HTTP-backed live session (TRV.5+TRV.6).

A concrete-but-overridable session implementation. Hosted bridges
(E2B, Vercel, Modal, ...) instantiate this directly or subclass
to swap specific endpoints. The session holds the httpx client
+ session-endpoint URL and routes ABC method calls to HTTP requests
with retry / error-mapping shared by every hosted bridge.
"""

from __future__ import annotations

import logging
from io import BytesIO, IOBase
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, override

from troopai.adk.exceptions.exceptions import (
    ExecTimeoutError,
    ExecTransportError,
    SandboxConfigurationError,
)
from troopai.adk.sandbox.clients.hosted.remote_vm.remote_vm_client import (
    map_http_error_to_sandbox_error,
    parse_exec_result,
    retry_async_call,
)
from troopai.adk.sandbox.clients.session import (
    BaseSandboxSession,
    FileEntry,
    MaterializationResult,
)
from troopai.adk.types.sandbox.exec_result import (
    ExecResult,
    ExposedPortEndpoint,
    PtyHandle,
)

if TYPE_CHECKING:
    from troopai.adk.types.sandbox.permissions import User

logger = logging.getLogger(__name__)

__all__ = ["RemoteVMSandboxSession"]


def _user_to_str(user: str | User) -> str:
    """Project a ``user`` argument into a wire string.

    Callers ensure ``user is not None`` before invoking. ``str`` users
    pass through; ``User`` instances surface their ``.name``. Using a
    helper centralises the narrowing so the call sites stay
    ``# type: ignore``-free.
    """
    if isinstance(user, str):
        return user
    return user.name


class RemoteVMSandboxSession(BaseSandboxSession):
    """HTTP-backed sandbox session.

    Hosted bridges instantiate this with their httpx client + an
    abstract endpoint contract:
    - ``POST {prefix}/exec``  body: {command, timeout, shell, user}
      response: ``ExecResult``-shaped JSON
    - ``GET  {prefix}/fs/read?path=...``
    - ``POST {prefix}/fs/write``  body: {path, data_b64}
    - ``GET  {prefix}/fs/ls?path=...``
    - ``POST {prefix}/fs/rm``  body: {path, recursive}
    - ``POST {prefix}/fs/mkdir``  body: {path, parents}
    - ``POST {prefix}/fs/extract``  body: {path, archive_b64,
      compression_scheme}
    - ``POST {prefix}/snapshot/persist``  body: {} → tar bytes
    - ``POST {prefix}/snapshot/hydrate``  body: {archive_b64}
    - ``POST {prefix}/manifest/apply``  body: {only_ephemeral}
    - ``POST {prefix}/patch``  body: {patch, user}
    - ``GET  {prefix}/status``
    - ``GET  {prefix}/ports/{port}``

    Subclasses with different shapes override the corresponding
    method.
    """

    def __init__(
        self,
        *,
        http_client: Any,
        session_endpoint: str,
        backend_id: str = "remote_vm",
        session_id: str | None = None,
        max_retries: int = 0,
    ) -> None:
        self._http = http_client
        self._endpoint = session_endpoint.rstrip("/")
        self._backend_id = backend_id
        self._session_id = session_id
        self._max_retries = max_retries
        self._started = False
        self._stopped = False

    @property
    @override
    def session_id(self) -> str | None:
        return self._session_id

    # --- HTTP plumbing ---------------------------------------------------

    async def _post(self, path: str, json: dict[str, Any] | None = None) -> Any:
        """POST to ``{endpoint}{path}`` with retry + error mapping."""
        import httpx

        async def _do() -> Any:
            response = await self._http.post(f"{self._endpoint}{path}", json=json or {})
            if response.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"{response.status_code} {response.text[:200]}",
                    request=response.request,
                    response=response,
                )
            return response

        try:
            response = await retry_async_call(_do, max_retries=self._max_retries)
        except httpx.HTTPStatusError as exc:
            raise map_http_error_to_sandbox_error(
                exc.response.status_code,
                exc.response.text,
                backend_id=self._backend_id,
            ) from exc
        return response

    async def _get(self, path: str, params: dict[str, str] | None = None) -> Any:
        """GET ``{endpoint}{path}`` with retry + error mapping."""
        import httpx

        async def _do() -> Any:
            response = await self._http.get(
                f"{self._endpoint}{path}",
                params=params or {},
            )
            if response.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"{response.status_code} {response.text[:200]}",
                    request=response.request,
                    response=response,
                )
            return response

        try:
            response = await retry_async_call(_do, max_retries=self._max_retries)
        except httpx.HTTPStatusError as exc:
            raise map_http_error_to_sandbox_error(
                exc.response.status_code,
                exc.response.text,
                backend_id=self._backend_id,
            ) from exc
        return response

    # --- Lifecycle -------------------------------------------------------

    @override
    async def start(self) -> None:
        # Many hosted providers start the session at create-time. The
        # default impl is a no-op + status check; subclasses override
        # if their provider has an explicit start endpoint.
        self._started = True

    @override
    async def stop(self) -> None:
        # Default: no-op. Subclasses persist snapshots before exit
        # when applicable.
        self._stopped = True

    @override
    async def shutdown(self) -> None:
        # Close the HTTP client so the connection pool is released.
        try:
            await self._http.aclose()
        except Exception:
            logger.warning("RemoteVMSandboxSession.shutdown: aclose failed", exc_info=True)

    @override
    async def aclose(self) -> None:
        if not self._stopped:
            await self.stop()
        await self.shutdown()

    # --- Command execution ----------------------------------------------

    @override
    async def run(
        self,
        *command: str | Path,
        timeout: float | None = None,
        shell: bool | list[str] = True,
        user: str | User | None = None,
    ) -> ExecResult:
        # ``shell`` mirrors the ABC contract: ``True`` wraps the command in the
        # provider's shell (single joined string), ``False`` invokes argv
        # directly, and a list is the shell command + flags (e.g.
        # ``["bash", "-lc"]``) — forwarded verbatim so the developer's explicit
        # shell selection is not silently collapsed to the provider default.
        if isinstance(shell, list):
            body: dict[str, Any] = {
                "command": [str(c) for c in command],
                "shell": [str(part) for part in shell],
            }
        elif shell:
            body = {
                "command": " ".join(str(c) for c in command),
                "shell": True,
            }
        else:
            body = {
                "command": [str(c) for c in command],
                "shell": False,
            }
        if timeout is not None:
            body["timeout"] = timeout
        if user is not None:
            body["user"] = _user_to_str(user)
        response = await self._post("/exec", json=body)
        return parse_exec_result(response.json())

    @override
    async def pty_start(
        self,
        *command: str | Path,
        user: str | User | None = None,
    ) -> PtyHandle:
        # PTY streams typically use WebSocket; concrete bridges override.
        # The default implementation returns an opaque handle pointing
        # at the command for tracing — interaction must go through the
        # provider's own WS surface.
        return PtyHandle(
            session_id=self._session_id or "remote_vm",
            command=" ".join(str(c) for c in command),
            backend_payload={"impl": "remote_vm_default", "user": user},
        )

    @override
    async def pty_write_stdin(self, handle: PtyHandle, data: bytes) -> None:
        # No-op in the default impl. Subclasses with WS support override.
        del handle, data

    @override
    async def pty_terminate_all(self) -> None:
        await self._post("/pty/terminate_all")

    # --- File operations -------------------------------------------------

    @override
    async def read(
        self,
        path: Path | str,
        *,
        user: str | User | None = None,
    ) -> IOBase:
        import base64

        params = {"path": str(path)}
        if user is not None:
            params["user"] = _user_to_str(user)
        response = await self._get("/fs/read", params=params)
        payload = response.json()
        # A present-but-empty value is a legitimately empty file; only a
        # response carrying NEITHER key is malformed and must not silently
        # decode to empty bytes.
        encoded = payload.get("data_b64")
        if encoded is None:
            encoded = payload.get("data")
        if encoded is None:
            raise ExecTransportError(
                f"RemoteVM /fs/read returned neither 'data_b64' nor 'data' for path {str(path)!r}; "
                "the response is malformed.",
            )
        return BytesIO(base64.b64decode(encoded))

    @override
    async def write(
        self,
        path: Path | str,
        data: IOBase,
        *,
        user: str | User | None = None,
    ) -> None:
        import base64

        payload = data.read()
        body: dict[str, Any] = {
            "path": str(path),
            "data_b64": base64.b64encode(payload).decode("ascii"),
        }
        if user is not None:
            body["user"] = _user_to_str(user)
        await self._post("/fs/write", json=body)

    @override
    async def ls(
        self,
        path: Path | str,
        *,
        user: str | User | None = None,
    ) -> list[FileEntry]:
        params = {"path": str(path)}
        if user is not None:
            params["user"] = _user_to_str(user)
        response = await self._get("/fs/ls", params=params)
        entries: list[FileEntry] = []
        for item in response.json().get("entries", []):
            entries.append(
                FileEntry(
                    name=str(item.get("name", "")),
                    is_directory=bool(item.get("is_directory", False)),
                    size_bytes=int(item.get("size_bytes", -1)),
                ),
            )
        return entries

    @override
    async def rm(
        self,
        path: Path | str,
        *,
        recursive: bool = False,
        user: str | User | None = None,
    ) -> None:
        body: dict[str, Any] = {"path": str(path), "recursive": recursive}
        if user is not None:
            body["user"] = _user_to_str(user)
        await self._post("/fs/rm", json=body)

    @override
    async def mkdir(
        self,
        path: Path | str,
        *,
        parents: bool = False,
        user: str | User | None = None,
    ) -> None:
        body: dict[str, Any] = {"path": str(path), "parents": parents}
        if user is not None:
            body["user"] = _user_to_str(user)
        await self._post("/fs/mkdir", json=body)

    @override
    async def extract(
        self,
        path: Path | str,
        data: IOBase,
        *,
        compression_scheme: Literal["tar", "zip"] | None = None,
    ) -> None:
        import base64

        payload = data.read()
        body: dict[str, Any] = {
            "path": str(path),
            "archive_b64": base64.b64encode(payload).decode("ascii"),
        }
        if compression_scheme is not None:
            body["compression_scheme"] = compression_scheme
        await self._post("/fs/extract", json=body)

    # --- Workspace persistence -------------------------------------------

    @override
    async def persist_workspace(self) -> IOBase:
        import base64

        response = await self._post("/snapshot/persist")
        encoded = response.json().get("archive_b64")
        # A workspace archive always carries tar headers; a missing key or an
        # empty body means the persist failed. Raise rather than write a
        # 0-byte snapshot the store would silently accept as durable.
        if encoded is None or len(encoded) == 0:
            raise ExecTransportError(
                "RemoteVM /snapshot/persist returned no 'archive_b64' payload; "
                "refusing to persist an empty workspace snapshot.",
            )
        return BytesIO(base64.b64decode(encoded))

    @override
    async def hydrate_workspace(self, data: IOBase) -> None:
        import base64

        payload = data.read()
        await self._post(
            "/snapshot/hydrate",
            json={"archive_b64": base64.b64encode(payload).decode("ascii")},
        )

    @override
    async def apply_manifest(
        self,
        *,
        only_ephemeral: bool = False,
    ) -> MaterializationResult:
        await self._post("/manifest/apply", json={"only_ephemeral": only_ephemeral})
        # The provider may return a materialization summary; default
        # to empty MaterializationResult — callers needing the full
        # summary should override.
        return MaterializationResult()

    @override
    async def apply_patch(
        self,
        patch: str,
        *,
        user: str | User | None = None,
    ) -> str:
        body: dict[str, Any] = {"patch": patch}
        if user is not None:
            body["user"] = _user_to_str(user)
        response = await self._post("/patch", json=body)
        return str(response.json().get("summary", ""))

    # --- Utilities -------------------------------------------------------

    @override
    async def running(self) -> bool:
        if not self._started or self._stopped:
            return False
        try:
            response = await self._get("/status")
            return bool(response.json().get("running", False))
        except SandboxConfigurationError:
            # Auth / 404: the caller MUST learn about this — don't
            # silently report "not running" when really "can't talk".
            raise
        except (ExecTransportError, ExecTimeoutError) as exc:
            # Transient connectivity blip. Treat as "we can't confirm
            # right now" and surface to caller via logs.
            logger.warning("RemoteVM running() transport blip: %s", exc)
            return False

    @override
    async def resolve_exposed_port(self, port: int) -> ExposedPortEndpoint:
        response = await self._get(f"/ports/{port}")
        data = response.json()
        return ExposedPortEndpoint(
            host=str(data.get("host", "localhost")),
            port=int(data.get("port", port)),
            tls=bool(data.get("tls", False)),
            query=str(data.get("query", "")),
        )
