"""K8sPodSandboxSession — real lifecycle backed by the kubernetes SDK.

The kubernetes Python client is synchronous; this session wraps every
call in ``asyncio.to_thread`` to keep the agent loop responsive. File
operations use tar streamed over the pod's exec channel — the
equivalent of ``kubectl cp``.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import tarfile
import time
from io import BytesIO, IOBase
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, ClassVar, Literal, override

from troopai.adk.exceptions.exceptions import (
    ExecTimeoutError,
    SandboxStartFailed,
    SandboxStopFailed,
    WorkspaceReadNotFoundError,
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
    from troopai.adk.types.sandbox.manifest import Manifest
    from troopai.adk.types.sandbox.network import NetworkPolicy
    from troopai.adk.types.sandbox.permissions import User
    from troopai.adk.types.sandbox.resource_limits import SandboxResourceLimits

logger = logging.getLogger(__name__)

__all__ = ["K8sPodSandboxSession"]

# Kubernetes exec WebSocket channel index for stdin (mirrors
# ``kubernetes.stream.ws_client.STDIN_CHANNEL``). Used to signal stdin EOF
# without closing the whole socket.
_STDIN_CHANNEL = 0


class K8sPodSandboxSession(BaseSandboxSession):
    """Live sandbox session backed by a Kubernetes pod.

    ``core_v1`` is the ``CoreV1Api`` client. ``pod_name`` + ``namespace``
    identify the running Pod. The Pod's first container is the exec
    target; multi-container Pods MUST set ``container_name`` explicitly.
    """

    def __init__(
        self,
        *,
        core_v1: Any,
        pod_name: str,
        namespace: str = "default",
        container_name: str | None = None,
        working_directory: str = "/workspace",
        network_policy: NetworkPolicy | None = None,
        networking_v1: Any = None,
        resource_limits: SandboxResourceLimits | None = None,
        environment: dict[str, str] | None = None,
        stream_api: Any = None,
        manifest: Manifest | None = None,
    ) -> None:
        self._core_v1 = core_v1
        self._pod_name = pod_name
        self._namespace = namespace
        self._container_name = container_name
        self._working_directory = working_directory
        self._network_policy = network_policy
        # ``NetworkingV1Api`` used to delete the per-pod NetworkPolicy CR on
        # teardown so it does not outlive its pod. ``None`` when no policy was
        # configured (no CR to clean up).
        self._networking_v1 = networking_v1
        self._resource_limits = resource_limits
        self._environment = dict(environment or {})
        self._stream_api = stream_api
        self._manifest = manifest
        self._started = False
        self._stopped = False
        self._session_id = f"k8s-{namespace}-{pod_name}"
        self._pty_handles: dict[str, dict[str, Any]] = {}

    def get_manifest(self) -> Manifest | None:
        """Return the manifest this session was created with, or ``None``."""
        return self._manifest

    @property
    @override
    def session_id(self) -> str | None:
        return self._session_id

    @override
    def supports_docker_volume_mounts(self) -> bool:
        return False

    @override
    def supports_pty(self) -> bool:
        return True

    # --- lifecycle -----------------------------------------------------

    @override
    async def start(self) -> None:
        if self._started:
            return
        try:
            deadline = time.monotonic() + 60.0
            while time.monotonic() < deadline:
                pod = await asyncio.to_thread(
                    self._core_v1.read_namespaced_pod_status,
                    name=self._pod_name,
                    namespace=self._namespace,
                )
                phase = getattr(pod.status, "phase", None) if pod else None
                if phase == "Running":
                    break
                if phase in {"Failed", "Succeeded"}:
                    raise SandboxStartFailed(
                        backend_id="k8s_pod",
                        reason=f"pod entered terminal phase {phase!r}",
                    )
                await asyncio.sleep(0.5)
            else:
                raise SandboxStartFailed(
                    backend_id="k8s_pod",
                    reason="pod did not reach Running phase within 60s",
                )
        except SandboxStartFailed:
            raise
        except Exception as exc:
            raise SandboxStartFailed(
                backend_id="k8s_pod",
                reason=f"K8s start failed: {exc}",
            ) from exc
        # Pod is Running — flip the flag BEFORE the post-start mkdir
        # so a retry won't re-poll the already-running pod and the
        # mkdir failure (if any) bubbles as its own SandboxStartFailed
        # without leaving `_started` desynced.
        self._started = True
        try:
            await self.mkdir(self._working_directory, parents=True)
        except Exception as exc:
            raise SandboxStartFailed(
                backend_id="k8s_pod",
                reason=f"mkdir {self._working_directory!r} failed: {exc}",
            ) from exc
        # Workspace helpers must exist before any capability uses
        # them; a failed install fails the start (attributed to the
        # backend) rather than surfacing later in a dependent
        # capability. Function-local import: clients/ keeps no
        # top-level edge into the heavy session package (same
        # rationale as the lazy kubernetes import in this file).
        from troopai.adk.sandbox.session.runtime_helpers import (
            install_runtime_helpers,
        )

        await install_runtime_helpers(self, backend_id="k8s_pod")

    @override
    async def stop(self) -> None:
        if self._stopped:
            return
        try:
            await asyncio.to_thread(
                self._core_v1.delete_namespaced_pod,
                name=self._pod_name,
                namespace=self._namespace,
                grace_period_seconds=10,
            )
        except Exception as exc:
            logger.warning("K8sPodSandboxSession.stop failed: %s", exc)
            raise SandboxStopFailed(
                f"K8sPodSandboxSession.stop failed: {exc}",
            ) from exc
        finally:
            # The per-pod NetworkPolicy CR has no ownerReference to the pod,
            # so it must be reclaimed even when pod deletion fails — otherwise
            # a failed teardown orphans the policy object. This helper is
            # itself best-effort (never raises), so it cannot mask the pod
            # delete failure re-raised above.
            await self._delete_network_policy_best_effort()
        self._stopped = True

    async def _delete_network_policy_best_effort(self) -> None:
        """Delete the per-pod NetworkPolicy CR so it does not outlive its pod.

        The CR has no ``ownerReference`` to the pod, so Kubernetes GC does not
        reclaim it on pod deletion; without this every policy-configured
        session would orphan one NetworkPolicy object. Best-effort: a failure
        is logged, never raised, so it cannot mask a successful pod teardown.
        """
        if self._network_policy is None or self._networking_v1 is None:
            return
        netpol_name = f"{self._pod_name}-egress-policy"
        try:
            await asyncio.to_thread(
                self._networking_v1.delete_namespaced_network_policy,
                name=netpol_name,
                namespace=self._namespace,
            )
        except Exception:
            logger.warning(
                "K8sPodSandboxSession: NetworkPolicy cleanup failed for %s",
                netpol_name,
                exc_info=True,
            )

    @override
    async def shutdown(self) -> None:
        if not self._stopped:
            try:
                await self.stop()
            except SandboxStopFailed:
                logger.warning(
                    "K8sPodSandboxSession.shutdown: stop raised, treating as fire-and-forget",
                )

    @override
    async def aclose(self) -> None:
        # Close any open PTY websockets first so their sockets + pod exec
        # streams do not leak until GC.
        await self.pty_terminate_all()
        await self.shutdown()

    # --- exec / run ----------------------------------------------------

    def _resolve_stream_callable(self) -> Any:
        """Return the ``kubernetes.stream.stream`` function (lazily)."""
        if self._stream_api is not None:
            return self._stream_api
        try:
            # kubernetes lives in the [sandbox-k8s] extra; not in core deps.
            from kubernetes import (  # pyright: ignore[reportMissingImports]
                stream as k8s_stream,
            )

            self._stream_api = k8s_stream.stream
        except ImportError as exc:
            raise SandboxStartFailed(
                backend_id="k8s_pod",
                reason=f"kubernetes.stream not importable: {exc}",
            ) from exc
        return self._stream_api

    def _build_exec_argv(
        self,
        command: tuple[str | Path, ...],
        shell: bool | list[str],
    ) -> list[str]:
        if shell is True:
            joined = " ".join(str(p) for p in command)
            return ["sh", "-c", joined]
        if isinstance(shell, list):
            return [*shell, *[str(p) for p in command]]
        return [str(p) for p in command]

    _EXEC_SYNC_MAX_SECONDS: ClassVar[float] = 300.0
    """Wall-clock cap for any single ``_exec_sync`` invocation."""

    def _exec_sync(
        self,
        argv: list[str],
        *,
        stdin_payload: bytes | None = None,
        capture_stderr_separately: bool = True,
        max_seconds: float | None = None,
    ) -> tuple[int, bytes, bytes]:
        """Run argv via WS exec; return ``(exit_code, stdout, stderr)``.

        Loop is bounded by ``max_seconds`` (default
        ``_EXEC_SYNC_MAX_SECONDS``); raising ``ExecTimeoutError`` on
        breach so a misbehaving exec channel can never hang the
        worker thread indefinitely.
        """
        deadline_s = max_seconds if max_seconds is not None else self._EXEC_SYNC_MAX_SECONDS
        deadline = time.monotonic() + deadline_s
        stream_fn = self._resolve_stream_callable()
        kwargs: dict[str, Any] = {
            "name": self._pod_name,
            "namespace": self._namespace,
            "command": argv,
            "stderr": True,
            "stdin": stdin_payload is not None,
            "stdout": True,
            "tty": False,
            "_preload_content": False,
            # File ops stream a tar archive (arbitrary binary) over the exec
            # channel; without binary the client utf-8-replace-decodes every
            # frame and silently corrupts non-UTF-8 bytes.
            "binary": True,
        }
        if self._container_name is not None:
            kwargs["container"] = self._container_name
        ws = stream_fn(
            self._core_v1.connect_get_namespaced_pod_exec,
            **kwargs,
        )
        stdout_chunks: list[bytes] = []
        stderr_chunks: list[bytes] = []
        if stdin_payload is not None:
            ws.write_stdin(stdin_payload)
            # Signal stdin EOF so the remote command (e.g. ``tar xf -``) sees
            # end-of-input and terminates. Closing the whole socket here would
            # abort the drain loop before stdout/stderr/exit-status arrive, so
            # close only the stdin channel (v5 protocol). On the older v4
            # protocol ``close_channel`` is a no-op; commands that read until
            # EOF require v5, which the client negotiates first.
            close_channel = getattr(ws, "close_channel", None)
            if callable(close_channel):
                close_channel(_STDIN_CHANNEL)
        while ws.is_open():
            if time.monotonic() > deadline:
                try:
                    ws.close()
                except Exception:
                    logger.debug("K8s exec ws.close after deadline failed", exc_info=True)
                raise ExecTimeoutError(
                    f"K8s exec exceeded {deadline_s:.1f}s wall-clock deadline",
                )
            ws.update(timeout=1)
            if ws.peek_stdout():
                out = ws.read_stdout()
                stdout_chunks.append(out.encode("utf-8") if isinstance(out, str) else out)
            if capture_stderr_separately and ws.peek_stderr():
                err = ws.read_stderr()
                stderr_chunks.append(err.encode("utf-8") if isinstance(err, str) else err)
        return (
            self._exec_returncode(ws),
            b"".join(stdout_chunks),
            b"".join(stderr_chunks),
        )

    @staticmethod
    def _exec_returncode(ws: Any) -> int:
        """Read the exec exit status, defaulting to 0 when unavailable.

        The kubernetes client exposes ``returncode`` as a property that, on a
        closed socket with no status frame on the error channel, parses an
        empty payload and raises ``TypeError``/``KeyError`` rather than
        returning ``None``. Treat any such read failure as a clean exit so a
        missing status frame never crashes the caller.
        """
        try:
            returncode = getattr(ws, "returncode", None)
        except (TypeError, KeyError, ValueError):
            return 0
        if returncode is None:
            return 0
        return int(returncode)

    @override
    async def run(
        self,
        *command: str | Path,
        timeout: float | None = None,
        shell: bool | list[str] = True,
        user: str | User | None = None,
    ) -> ExecResult:
        del user
        argv = self._build_exec_argv(command, shell)
        start_time = time.monotonic()

        def _run() -> tuple[int, bytes, bytes]:
            # Forward the asyncio-level timeout as the thread-level deadline so
            # that a timed-out asyncio task does not leave the worker thread
            # spinning in the WebSocket recv loop for up to the class-level
            # 300 s cap, which would exhaust the thread pool under concurrent
            # short-timeout workloads.
            return self._exec_sync(argv, max_seconds=timeout)

        try:
            if timeout is None:
                exit_code, stdout, stderr = await asyncio.to_thread(_run)
            else:
                exit_code, stdout, stderr = await asyncio.wait_for(
                    asyncio.to_thread(_run),
                    timeout=timeout,
                )
        except TimeoutError as exc:
            elapsed = time.monotonic() - start_time
            raise ExecTimeoutError(
                f"K8s command timed out after {elapsed:.2f}s (limit={timeout}s)",
            ) from exc
        duration_ms = int((time.monotonic() - start_time) * 1000)
        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_ms=duration_ms,
        )

    # --- PTY via kubernetes.stream with tty=True ----------------------

    @override
    async def pty_start(
        self,
        *command: str | Path,
        user: str | User | None = None,
    ) -> PtyHandle:
        """Open an interactive exec over the pod's WebSocket exec channel.

        ``command`` parts are passed directly to the K8s exec channel
        as an argv list — no ``sh -c`` interpolation. Callers wanting
        shell expansion MUST pass ``"sh"``, ``"-c"``, ``"<line>"``
        explicitly so the interpolation site is auditable.
        """
        del user
        stream_fn = self._resolve_stream_callable()
        argv = [str(p) for p in command]

        def _start() -> Any:
            kwargs: dict[str, Any] = {
                "name": self._pod_name,
                "namespace": self._namespace,
                "command": argv,
                "stderr": True,
                "stdin": True,
                "stdout": True,
                "tty": True,
                "_preload_content": False,
            }
            if self._container_name is not None:
                kwargs["container"] = self._container_name
            return stream_fn(self._core_v1.connect_get_namespaced_pod_exec, **kwargs)

        try:
            ws = await asyncio.to_thread(_start)
        except Exception as exc:
            raise SandboxStopFailed(
                f"K8sPodSandboxSession.pty_start failed: {exc}",
            ) from exc
        pty_id = f"pty-{id(ws)}"
        self._pty_handles[pty_id] = {"ws": ws}
        return PtyHandle(
            session_id=self._session_id,
            command=" ".join(str(c) for c in command),
            backend_payload={"impl": "k8s_exec_ws", "pty_id": pty_id},
        )

    @override
    async def pty_write_stdin(self, handle: PtyHandle, data: bytes) -> None:
        """Write ``data`` to the PTY WebSocket backing ``handle``."""
        payload = handle.backend_payload
        if not isinstance(payload, dict):
            raise SandboxStopFailed("K8sPodSandboxSession.pty_write_stdin: invalid handle")
        pty_id = payload.get("pty_id")
        if not isinstance(pty_id, str):
            raise SandboxStopFailed("K8sPodSandboxSession.pty_write_stdin: invalid pty_id")
        entry = self._pty_handles.get(pty_id)
        if entry is None:
            raise SandboxStopFailed(
                f"K8sPodSandboxSession.pty_write_stdin: unknown PTY {pty_id!r}",
            )
        ws = entry["ws"]

        def _write() -> None:
            # Deliver bytes verbatim — ``write_channel`` sends a binary frame
            # for ``bytes`` input, so the PTY receives exactly what was passed.
            # A utf-8 decode here would lossily mangle non-UTF-8 stdin.
            ws.write_stdin(data)

        await asyncio.to_thread(_write)

    @override
    async def pty_terminate_all(self) -> None:
        """Close every open PTY WebSocket; best-effort."""
        handles = list(self._pty_handles.items())
        self._pty_handles.clear()
        for pty_id, entry in handles:
            ws = entry.get("ws")
            try:
                close_fn = getattr(ws, "close", None)
                if callable(close_fn):
                    await asyncio.to_thread(close_fn)
            except Exception:
                logger.debug(
                    "K8sPodSandboxSession.pty_terminate_all: close failed for %s",
                    pty_id,
                    exc_info=True,
                )

    # --- file ops via tar-over-exec (kubectl cp equivalent) -----------

    def _resolve_path(self, path: Path | str) -> str:
        p = PurePosixPath(str(path))
        if p.is_absolute():
            return str(p)
        return str(PurePosixPath(self._working_directory) / p)

    @override
    async def read(
        self,
        path: Path | str,
        *,
        user: str | User | None = None,
    ) -> IOBase:
        del user
        target = self._resolve_path(path)
        parent = str(PurePosixPath(target).parent)
        basename = PurePosixPath(target).name

        def _read() -> bytes:
            exit_code, stdout, stderr = self._exec_sync(
                ["tar", "cf", "-", "-C", parent, basename],
            )
            if exit_code != 0:
                raise WorkspaceReadNotFoundError(
                    f"Workspace path not found: {target} (stderr={stderr.decode(errors='replace')[:200]})",
                )
            return stdout

        tar_bytes = await asyncio.to_thread(_read)
        # A single-file read must yield exactly one regular-file entry.
        # ``tar cf`` on a directory streams the directory entry plus its
        # contents (or, when empty, just the directory entry), so anything
        # other than one regular file means ``target`` is a directory or
        # special file — raise instead of silently returning the first file
        # found inside.
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tar:
            members = tar.getmembers()
            if len(members) != 1 or not members[0].isfile():
                raise WorkspaceReadNotFoundError(
                    f"Workspace path is not a regular file: {target}",
                )
            extracted = tar.extractfile(members[0])
            if extracted is None:
                raise WorkspaceReadNotFoundError(
                    f"Workspace path not readable: {target}",
                )
            return BytesIO(extracted.read())

    @override
    async def write(
        self,
        path: Path | str,
        data: IOBase,
        *,
        user: str | User | None = None,
    ) -> None:
        del user
        target = self._resolve_path(path)
        parent = str(PurePosixPath(target).parent)
        basename = PurePosixPath(target).name
        payload = data.read()

        await self.mkdir(parent, parents=True)
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w") as tar:
            info = tarfile.TarInfo(name=basename)
            info.size = len(payload)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(payload))

        def _write() -> None:
            exit_code, _stdout, stderr = self._exec_sync(
                ["tar", "xf", "-", "-C", parent],
                stdin_payload=tar_buf.getvalue(),
            )
            if exit_code != 0:
                raise SandboxStopFailed(
                    f"K8s write failed for {target}: {stderr.decode(errors='replace')[:200]}",
                )

        await asyncio.to_thread(_write)

    @override
    async def ls(
        self,
        path: Path | str,
        *,
        user: str | User | None = None,
    ) -> list[FileEntry]:
        del user
        target = self._resolve_path(path)
        result = await self.run("ls", "-1A", target, shell=False)
        if result.exit_code != 0:
            raise WorkspaceReadNotFoundError(
                f"Workspace path not found or not a directory: {target}",
            )
        entries: list[FileEntry] = []
        for line in result.stdout.decode("utf-8", errors="replace").splitlines():
            name = line.strip()
            if len(name) == 0:
                continue
            stat_result = await self.run(
                "test",
                "-d",
                f"{target}/{name}",
                shell=False,
            )
            is_dir = stat_result.exit_code == 0
            entries.append(FileEntry(name=name, is_directory=is_dir, size_bytes=-1))
        return entries

    @override
    async def rm(
        self,
        path: Path | str,
        *,
        recursive: bool = False,
        user: str | User | None = None,
    ) -> None:
        del user
        target = self._resolve_path(path)
        if recursive:
            await self.run("rm", "-rf", target, shell=False)
        else:
            await self.run("rm", target, shell=False)

    @override
    async def mkdir(
        self,
        path: Path | str,
        *,
        parents: bool = False,
        user: str | User | None = None,
    ) -> None:
        del user
        target = self._resolve_path(path)
        if parents:
            await self.run("mkdir", "-p", target, shell=False)
        else:
            await self.run("mkdir", target, shell=False)

    @override
    async def extract(
        self,
        path: Path | str,
        data: IOBase,
        *,
        compression_scheme: Literal["tar", "zip"] | None = None,
    ) -> None:
        del compression_scheme
        target = self._resolve_path(path)
        await self.mkdir(target, parents=True)
        payload = data.read()

        def _extract() -> None:
            exit_code, _stdout, stderr = self._exec_sync(
                ["tar", "xf", "-", "-C", target],
                stdin_payload=payload,
            )
            if exit_code != 0:
                raise SandboxStopFailed(
                    f"K8s extract failed for {target}: {stderr.decode(errors='replace')[:200]}",
                )

        await asyncio.to_thread(_extract)

    # --- persist / hydrate ---------------------------------------------

    @override
    async def persist_workspace(self) -> IOBase:
        parent = str(PurePosixPath(self._working_directory).parent)
        basename = PurePosixPath(self._working_directory).name

        def _archive() -> bytes:
            exit_code, stdout, stderr = self._exec_sync(
                ["tar", "cf", "-", "-C", parent, basename],
            )
            if exit_code != 0:
                raise SandboxStopFailed(
                    f"K8s persist_workspace failed: {stderr.decode(errors='replace')[:200]}",
                )
            return stdout

        return BytesIO(await asyncio.to_thread(_archive))

    @override
    async def hydrate_workspace(self, data: IOBase) -> None:
        await self.run("rm", "-rf", self._working_directory, shell=False)
        await self.mkdir(self._working_directory, parents=True)
        parent = str(PurePosixPath(self._working_directory).parent)
        payload = data.read()

        def _restore() -> None:
            exit_code, _stdout, stderr = self._exec_sync(
                ["tar", "xf", "-", "-C", parent],
                stdin_payload=payload,
            )
            if exit_code != 0:
                raise SandboxStopFailed(
                    f"K8s hydrate_workspace failed: {stderr.decode(errors='replace')[:200]}",
                )

        await asyncio.to_thread(_restore)

    @override
    async def apply_manifest(
        self,
        *,
        only_ephemeral: bool = False,
    ) -> MaterializationResult:
        manifest = self.get_manifest()
        if manifest is None:
            return MaterializationResult()
        from troopai.adk.sandbox.session.materialization import materialize_manifest

        return await materialize_manifest(self, manifest, only_ephemeral=only_ephemeral)

    @override
    async def apply_patch(
        self,
        patch: str,
        *,
        user: str | User | None = None,
    ) -> str:
        del user
        await self.write(".troopai_patch.diff", BytesIO(patch.encode("utf-8")))
        try:
            result = await self.run(
                "patch",
                "-p1",
                "-i",
                ".troopai_patch.diff",
                shell=False,
            )
        finally:
            with contextlib.suppress(Exception):
                await self.rm(".troopai_patch.diff")
        if result.exit_code != 0:
            return f"apply_patch failed: exit={result.exit_code}; stderr={result.stderr.decode(errors='replace')[:300]}"
        return f"apply_patch: OK; patch_size={len(patch)} bytes"

    # --- utilities -----------------------------------------------------

    @override
    async def running(self) -> bool:
        if not self._started or self._stopped:
            return False
        try:
            pod = await asyncio.to_thread(
                self._core_v1.read_namespaced_pod_status,
                name=self._pod_name,
                namespace=self._namespace,
            )
            phase = getattr(pod.status, "phase", None) if pod else None
            return phase == "Running"
        except Exception:
            return False

    @override
    async def resolve_exposed_port(self, port: int) -> ExposedPortEndpoint:
        try:
            pod = await asyncio.to_thread(
                self._core_v1.read_namespaced_pod,
                name=self._pod_name,
                namespace=self._namespace,
            )
        except Exception:
            return ExposedPortEndpoint(host="127.0.0.1", port=port)
        ip = getattr(pod.status, "pod_ip", None) if pod else None
        host = ip if isinstance(ip, str) and len(ip) > 0 else "127.0.0.1"
        return ExposedPortEndpoint(host=host, port=port)
