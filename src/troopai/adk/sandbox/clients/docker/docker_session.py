"""DockerSandboxSession — real lifecycle backed by the docker SDK (TDK.1-TDK.11).

The docker SDK is synchronous; this session wraps every call in
``asyncio.to_thread`` so it doesn't block the agent loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import math
import tarfile
import time
from io import BytesIO, IOBase
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal, override

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

# Exit status the in-container 'timeout' wrapper reports when it killed
# the command (coreutils and busybox both use 124 for the default TERM).
_IN_CONTAINER_TIMEOUT_EXIT = 124
# Extra host-side budget on top of the in-container deadline, covering
# daemon round-trip and wrapper startup before the backstop fires.
_TIMEOUT_BACKSTOP_GRACE_SECONDS = 5.0

__all__ = ["DockerSandboxSession"]


class DockerSandboxSession(BaseSandboxSession):
    """Live sandbox session backed by a Docker container."""

    def __init__(
        self,
        *,
        container: Any,
        working_directory: str = "/workspace",
        network_policy: NetworkPolicy | None = None,
        resource_limits: SandboxResourceLimits | None = None,
        environment: dict[str, str] | None = None,
        manifest: Manifest | None = None,
    ) -> None:
        self._container = container
        self._working_directory = working_directory
        self._network_policy = network_policy
        self._resource_limits = resource_limits
        self._environment = dict(environment or {})
        self._manifest = manifest
        # Lazily probed: whether the container image ships a 'timeout'
        # binary for in-container deadline enforcement (see run()).
        self._timeout_bin: bool | None = None
        self._started = False
        self._stopped = False
        self._session_id = f"docker-{container.id[:12]}"
        self._pty_handles: dict[str, dict[str, Any]] = {}

    def get_manifest(self) -> Manifest | None:
        """Return the manifest this session was created with, or ``None``."""
        return self._manifest

    async def _timeout_bin_available(self) -> bool:
        """Probe (once) whether the container image ships a 'timeout' binary.

        Used by :meth:`run` to enforce deadlines in-container. A probe
        failure (no shell, exec error) records the binary as unavailable
        so timed runs fall back to host-side enforcement.
        """
        if self._timeout_bin is None:

            def _probe() -> Any:
                return self._container.exec_run(["sh", "-c", "command -v timeout"], demux=True)

            try:
                result = await asyncio.to_thread(_probe)
                self._timeout_bin = int(getattr(result, "exit_code", 1)) == 0
            except Exception:
                self._timeout_bin = False
            if not self._timeout_bin:
                logger.info(
                    "Container image lacks the 'timeout' binary; timed runs fall back to "
                    "host-side enforcement only (a timed-out command keeps running in the "
                    "container until session teardown)"
                )
        return self._timeout_bin

    @property
    @override
    def session_id(self) -> str | None:
        return self._session_id

    @override
    def supports_docker_volume_mounts(self) -> bool:
        return True

    @override
    def supports_pty(self) -> bool:
        return True

    # --- TDK.3: start --------------------------------------------------

    @override
    async def start(self) -> None:
        if self._started:
            return
        try:
            await asyncio.to_thread(self._container.start)
            # Wait for the container to be running. Poll because
            # docker.wait blocks until the container exits, which is
            # wrong for our long-lived sandbox use case.
            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                await asyncio.to_thread(self._container.reload)
                if self._container.status == "running":
                    break
                await asyncio.sleep(0.1)
            else:
                raise SandboxStartFailed(
                    backend_id="docker",
                    reason=f"container did not reach running state within 30s (status={self._container.status})",
                )
        except Exception as exc:
            if isinstance(exc, SandboxStartFailed):
                raise
            raise SandboxStartFailed(
                backend_id="docker",
                reason=f"start failed: {exc}",
            ) from exc
        # Container reached Running — flip the flag BEFORE the post-
        # start mkdir so a retry won't double-call container.start.
        # The mkdir failure (if any) will then bubble as its own
        # SandboxStartFailed without leaving `_started` desynced.
        self._started = True
        try:
            await self.mkdir(self._working_directory, parents=True)
        except Exception as exc:
            raise SandboxStartFailed(
                backend_id="docker",
                reason=f"mkdir {self._working_directory!r} failed: {exc}",
            ) from exc
        # Workspace helpers must exist before any capability uses
        # them; a failed install fails the start (attributed to the
        # backend) rather than surfacing later in a dependent
        # capability. Function-local import: clients/ keeps no
        # top-level edge into the heavy session package (same
        # rationale as the lazy kubernetes import elsewhere).
        from troopai.adk.sandbox.session.runtime_helpers import (
            install_runtime_helpers,
        )

        await install_runtime_helpers(self, backend_id="docker")

    # --- TDK.4: stop / shutdown / aclose -------------------------------

    @override
    async def stop(self) -> None:
        if self._stopped:
            return
        try:
            await asyncio.to_thread(self._container.stop, timeout=10)
        except Exception as exc:
            logger.warning("DockerSandboxSession.stop failed: %s", exc)
            raise SandboxStopFailed(
                f"DockerSandboxSession.stop failed: {exc}",
            ) from exc
        self._stopped = True

    @override
    async def shutdown(self) -> None:
        try:
            await asyncio.to_thread(self._container.remove, force=True)
        except Exception:
            logger.warning("DockerSandboxSession.shutdown remove failed", exc_info=True)

    @override
    async def aclose(self) -> None:
        # Close any open PTY sockets first, while the container is still up, so
        # their fds + container exec sessions do not leak until GC.
        await self.pty_terminate_all()
        if not self._stopped and self._started:
            try:
                await self.stop()
            except SandboxStopFailed:
                logger.warning("DockerSandboxSession.aclose: stop raised, continuing to shutdown")
        await self.shutdown()

    # --- TDK.5: run ----------------------------------------------------

    @override
    async def run(
        self,
        *command: str | Path,
        timeout: float | None = None,
        shell: bool | list[str] = True,
        user: str | User | None = None,
    ) -> ExecResult:
        argv: list[str]
        if shell is True:
            joined = " ".join(str(p) for p in command)
            argv = ["sh", "-c", joined]
        elif isinstance(shell, list):
            argv = [*shell, *[str(p) for p in command]]
        else:
            argv = [str(p) for p in command]
        # The synchronous Docker SDK offers no way to cancel a running
        # exec, so a host-side timeout alone strands the command in the
        # container (and its carrier thread on the host). When the image
        # ships a 'timeout' binary, enforce the deadline in-container —
        # the wrapper TERMs the command at expiry and the exec returns
        # promptly with exit 124; the host wait_for stays as a padded
        # backstop. Images without 'timeout' keep host-only enforcement.
        wrapped_deadline: int | None = None
        if timeout is not None and await self._timeout_bin_available():
            # busybox 'timeout' only accepts whole seconds, so sub-second
            # deadlines round up to 1s for the in-container kill; the
            # exact host-side deadline is not enforced on this path.
            wrapped_deadline = max(1, math.ceil(timeout))
            argv = ["timeout", str(wrapped_deadline), *argv]
        user_str: str | None
        if user is None:
            user_str = None
        elif isinstance(user, str):
            user_str = user
        else:
            user_str = getattr(user, "name", None)
        start_time = time.monotonic()

        def _run_sync() -> Any:
            return self._container.exec_run(
                argv,
                workdir=self._working_directory,
                environment=self._environment if len(self._environment) > 0 else None,
                user=user_str or "",
                demux=True,
            )

        if timeout is None:
            try:
                exec_result = await asyncio.to_thread(_run_sync)
            except Exception as exc:
                raise SandboxStopFailed(
                    f"DockerSandboxSession.run failed: {exc}",
                ) from exc
        else:
            host_budget = (
                timeout if wrapped_deadline is None else float(wrapped_deadline) + _TIMEOUT_BACKSTOP_GRACE_SECONDS
            )
            try:
                exec_result = await asyncio.wait_for(
                    asyncio.to_thread(_run_sync),
                    timeout=host_budget,
                )
            except TimeoutError as exc:
                elapsed = time.monotonic() - start_time
                logger.warning(
                    "DockerSandboxSession.run timed out after %.2fs (limit=%.2fs) — "
                    "asyncio task cancelled but the underlying OS thread running "
                    "exec_run may still be running (no clean cancellation API for "
                    "synchronous Docker SDK calls)",
                    elapsed,
                    timeout,
                )
                raise ExecTimeoutError(
                    f"Docker command timed out after {elapsed:.2f}s (limit={timeout}s)",
                ) from exc
            if (
                wrapped_deadline is not None
                and int(getattr(exec_result, "exit_code", -1)) == _IN_CONTAINER_TIMEOUT_EXIT
            ):
                elapsed = time.monotonic() - start_time
                raise ExecTimeoutError(
                    f"Docker command killed by in-container timeout after {elapsed:.2f}s (limit={timeout}s)",
                )
        duration_ms = int((time.monotonic() - start_time) * 1000)
        exit_code = int(getattr(exec_result, "exit_code", -1))
        output = getattr(exec_result, "output", (b"", b""))
        if isinstance(output, tuple) and len(output) == 2:
            stdout = output[0] or b""
            stderr = output[1] or b""
        else:
            stdout = output if isinstance(output, bytes) else b""
            stderr = b""
        return ExecResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_ms=duration_ms,
        )

    # --- TDK.6: PTY via exec_run(socket=True) --------------------------

    @override
    async def pty_start(
        self,
        *command: str | Path,
        user: str | User | None = None,
    ) -> PtyHandle:
        """Open a PTY-backed exec via the Docker socket interface.

        The docker SDK's ``exec_create`` + ``exec_start(socket=True)``
        returns a bidirectional socket the framework drives directly.
        The framework stores ``(exec_id, sock)`` in the backend payload
        keyed by a synthetic PTY id so subsequent ``pty_write_stdin``
        calls find the right socket.
        """
        user_str: str | None
        if user is None:
            user_str = None
        elif isinstance(user, str):
            user_str = user
        else:
            user_str = getattr(user, "name", None)
        # Direct argv — no ``sh -c`` interpolation. Callers wanting
        # shell expansion MUST pass "sh", "-c", "<line>" explicitly.
        argv = [str(p) for p in command]

        def _start() -> tuple[str, Any]:
            api_client = self._container.client.api  # low-level APIClient
            exec_handle = api_client.exec_create(
                self._container.id,
                cmd=argv,
                stdin=True,
                stdout=True,
                stderr=True,
                tty=True,
                workdir=self._working_directory,
                environment=self._environment if len(self._environment) > 0 else None,
                user=user_str or "",
            )
            exec_id = exec_handle["Id"] if isinstance(exec_handle, dict) else exec_handle
            sock = api_client.exec_start(exec_id, detach=False, tty=True, stream=False, socket=True)
            return exec_id, sock

        try:
            exec_id, sock = await asyncio.to_thread(_start)
        except Exception as exc:
            raise SandboxStopFailed(
                f"DockerSandboxSession.pty_start failed: {exc}",
            ) from exc
        pty_id = f"pty-{exec_id[:12]}"
        self._pty_handles[pty_id] = {"exec_id": exec_id, "sock": sock}
        return PtyHandle(
            session_id=self._session_id,
            command=" ".join(str(c) for c in command),
            backend_payload={"impl": "docker_exec_socket", "pty_id": pty_id},
        )

    @override
    async def pty_write_stdin(self, handle: PtyHandle, data: bytes) -> None:
        """Write ``data`` to the PTY backing ``handle``."""
        payload = handle.backend_payload
        if not isinstance(payload, dict):
            raise SandboxStopFailed("DockerSandboxSession.pty_write_stdin: invalid handle")
        pty_id = payload.get("pty_id")
        if not isinstance(pty_id, str):
            raise SandboxStopFailed("DockerSandboxSession.pty_write_stdin: invalid pty_id")
        entry = self._pty_handles.get(pty_id)
        if entry is None:
            raise SandboxStopFailed(
                f"DockerSandboxSession.pty_write_stdin: unknown PTY {pty_id!r}",
            )
        sock = entry["sock"]

        def _write() -> None:
            # docker-py wraps the socket; the underlying _sock supports send().
            underlying = getattr(sock, "_sock", sock)
            underlying.send(data)

        await asyncio.to_thread(_write)

    @override
    async def pty_terminate_all(self) -> None:
        """Close every open PTY socket; best-effort."""
        handles = list(self._pty_handles.items())
        self._pty_handles.clear()
        for pty_id, entry in handles:
            sock = entry.get("sock")
            try:
                close_fn = getattr(sock, "close", None)
                if callable(close_fn):
                    await asyncio.to_thread(close_fn)
            except Exception:
                logger.debug("pty_terminate_all: close failed for %s", pty_id, exc_info=True)

    # --- TDK.7-TDK.9: File ops via tar archives -----------------------

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

        def _get_archive() -> bytes:
            try:
                bits, _stat = self._container.get_archive(target)
            except Exception as exc:
                raise WorkspaceReadNotFoundError(
                    f"Workspace path not found: {target} ({exc})",
                ) from exc
            buf = io.BytesIO()
            for chunk in bits:
                buf.write(chunk)
            return buf.getvalue()

        tar_bytes = await asyncio.to_thread(_get_archive)
        # A single-file read must yield exactly one regular-file entry.
        # A directory archives as its own entry plus its contents (or, when
        # empty, just the directory entry), so anything other than one
        # regular file means `target` is a directory or special file —
        # raise instead of silently returning the first file found inside.
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
        filename = PurePosixPath(target).name
        payload = data.read()
        # Build a tar archive containing the single file.
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w") as tar:
            info = tarfile.TarInfo(name=filename)
            info.size = len(payload)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(payload))
        tar_buf.seek(0)
        # Ensure parent exists.
        await self.mkdir(parent, parents=True)
        await asyncio.to_thread(
            self._container.put_archive,
            parent,
            tar_buf.getvalue(),
        )

    @override
    async def ls(
        self,
        path: Path | str,
        *,
        user: str | User | None = None,
    ) -> list[FileEntry]:
        del user
        target = self._resolve_path(path)
        # Use `ls -1a --file-type` via exec for portability.
        result = await self.run(
            "ls",
            "-1A",
            target,
            shell=False,
        )
        if result.exit_code != 0:
            raise WorkspaceReadNotFoundError(
                f"Workspace path not found or not a directory: {target}",
            )
        entries: list[FileEntry] = []
        for line in result.stdout.decode("utf-8", errors="replace").splitlines():
            name = line.strip()
            if len(name) == 0:
                continue
            # Cheap is-directory check via stat -c.
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
        await asyncio.to_thread(
            self._container.put_archive,
            target,
            payload,
        )

    # --- TDK.10: Persist / hydrate -------------------------------------

    @override
    async def persist_workspace(self) -> IOBase:
        def _get_archive() -> bytes:
            bits, _stat = self._container.get_archive(self._working_directory)
            buf = io.BytesIO()
            for chunk in bits:
                buf.write(chunk)
            return buf.getvalue()

        tar_bytes = await asyncio.to_thread(_get_archive)
        return BytesIO(tar_bytes)

    @override
    async def hydrate_workspace(self, data: IOBase) -> None:
        # Wipe + recreate the workspace dir.
        await self.run(
            "rm",
            "-rf",
            self._working_directory,
            shell=False,
        )
        await self.mkdir(self._working_directory, parents=True)
        parent = str(PurePosixPath(self._working_directory).parent)
        payload = data.read()
        await asyncio.to_thread(
            self._container.put_archive,
            parent,
            payload,
        )

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
        # Write patch to a temp file inside the container + apply via patch.
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

    # --- TDK.11: Utilities --------------------------------------------

    @override
    async def running(self) -> bool:
        if not self._started or self._stopped:
            return False
        try:
            await asyncio.to_thread(self._container.reload)
        except Exception:
            return False
        return bool(self._container.status == "running")

    @override
    async def resolve_exposed_port(self, port: int) -> ExposedPortEndpoint:
        await asyncio.to_thread(self._container.reload)
        ports = self._container.attrs.get("NetworkSettings", {}).get("Ports") or {}
        for spec, mappings in ports.items():
            if not spec.startswith(f"{port}/"):
                continue
            if mappings is None or len(mappings) == 0:
                continue
            mapping = mappings[0]
            host = mapping.get("HostIp") or "127.0.0.1"
            host_port = int(mapping.get("HostPort", port))
            return ExposedPortEndpoint(host=host, port=host_port)
        # Fall back to the container IP if no host mapping.
        net = self._container.attrs.get("NetworkSettings", {})
        ip = net.get("IPAddress") or "127.0.0.1"
        return ExposedPortEndpoint(host=ip, port=port)
