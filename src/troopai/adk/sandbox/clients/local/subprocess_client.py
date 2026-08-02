"""LocalSubprocessSandboxClient — dev-only sandbox backed by host subprocesses.

WARNING — this backend provides NO isolation. Commands run as
children of the host Python process under a temporary working
directory. Use ONLY for development and example code. Production
deployments MUST use DockerSandboxClient, K8sPodSandboxClient,
or a hosted-bridge subclass of RemoteVMSandboxClient.

The backend logs a WARNING banner on construction to make the
risk explicit in tracing + audit.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import tarfile
import tempfile
import time
from io import BytesIO, IOBase
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, override

from troopai.adk.exceptions.exceptions import (
    ExecTimeoutError,
    SandboxNetworkPolicyViolation,
    SandboxStartFailed,
    WorkspaceReadNotFoundError,
)
from troopai.adk.sandbox.clients.base import (
    BaseSandboxClient,
    BaseSandboxClientOptions,
    reject_unsupported_snapshot_store,
    warn_discarded_snapshot,
)
from troopai.adk.sandbox.clients.session import (
    BaseSandboxSession,
    FileEntry,
    MaterializationResult,
)
from troopai.adk.sandbox.session import (
    SandboxArchiveLimits,
    validate_tar_archive_for_extraction,
)
from troopai.adk.types.sandbox.cost import SandboxBackendCapabilities, SandboxCostDescriptor
from troopai.adk.types.sandbox.exec_result import (
    ExecResult,
    ExposedPortEndpoint,
    PtyHandle,
)
from troopai.adk.types.sandbox.session_state import SandboxSessionState

if TYPE_CHECKING:
    from troopai.adk.types.sandbox.manifest import Manifest
    from troopai.adk.types.sandbox.network import NetworkPolicy
    from troopai.adk.types.sandbox.permissions import User
    from troopai.adk.types.sandbox.snapshot import SnapshotSpec

logger = logging.getLogger(__name__)

__all__ = [
    "LocalSandboxClientOptions",
    "LocalSandboxSession",
    "LocalSubprocessSandboxClient",
]


_DEV_ONLY_BANNER = (
    "LocalSubprocessSandboxClient provides NO isolation. "
    "Use ONLY for development; production deployments MUST use "
    "DockerSandboxClient, K8sPodSandboxClient, or a hosted bridge."
)


class LocalSandboxClientOptions(BaseSandboxClientOptions):
    """Options for the local subprocess backend.

    Attributes:
        working_directory: Optional host filesystem path to use as the
            workspace root. ``None`` creates a temporary directory that
            the framework creates and cleans up on session close.
        default_env: Environment variables added to every subprocess
            invocation in addition to the manifest's environment.
        archive_limits: Resource bounds enforced when validating
            archives before ``extract()`` / ``hydrate_workspace()``
            (member count, summed extracted bytes). ``None`` applies
            the deny-by-default bounds; pass
            ``SandboxArchiveLimits.unbounded()`` to disable.
    """

    working_directory: str | None = None
    default_env: dict[str, str] = {}
    archive_limits: SandboxArchiveLimits | None = None


class LocalSandboxSession(BaseSandboxSession):
    """Concrete session for the local-subprocess backend."""

    def __init__(
        self,
        *,
        working_directory: Path,
        cleanup_on_shutdown: bool,
        default_env: dict[str, str],
        network_policy: NetworkPolicy | None = None,
        manifest: Manifest | None = None,
        archive_limits: SandboxArchiveLimits | None = None,
    ) -> None:
        self._working_directory = working_directory
        self._cleanup_on_shutdown = cleanup_on_shutdown
        self._default_env = dict(default_env)
        self._network_policy = network_policy
        self._manifest = manifest
        self._archive_limits = archive_limits
        self._started = False
        self._stopped = False
        self._session_id = f"local-{id(self):x}"

    def get_manifest(self) -> Manifest | None:
        """Return the manifest this session was created with, or ``None``."""
        return self._manifest

    @property
    @override
    def session_id(self) -> str | None:
        return self._session_id

    @override
    async def start(self) -> None:
        if self._started:
            return
        if not self._working_directory.exists():
            try:
                self._working_directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise SandboxStartFailed(
                    backend_id="unix_local",
                    reason=f"could not create workspace directory: {exc}",
                ) from exc
        # Network policy: refuse deny_default=True since we cannot
        # enforce filesystem-level network isolation inside a
        # subprocess of the host process.
        if self._network_policy is not None and self._network_policy.deny_default:
            raise SandboxNetworkPolicyViolation(
                "LocalSubprocessSandboxClient cannot enforce a deny-default "
                "NetworkPolicy inside a subprocess of the host process; use "
                "DockerSandboxClient or K8sPodSandboxClient for enforced policy."
            )
        self._started = True

    @override
    async def stop(self) -> None:
        # No snapshot persistence in the local backend by default;
        # callers needing snapshots wire a SnapshotStore via the
        # SandboxRunConfig and stop returns after the store persists.
        self._stopped = True

    @override
    async def shutdown(self) -> None:
        if self._cleanup_on_shutdown and self._working_directory.exists():
            shutil.rmtree(self._working_directory, ignore_errors=True)

    @override
    async def aclose(self) -> None:
        if not self._stopped:
            await self.stop()
        await self.shutdown()

    def _resolve_inside_workspace(self, path: Path | str) -> Path:
        """Resolve ``path`` inside the workspace, rejecting any escape attempt.

        Absolute paths and relative traversal components (``../../``) that
        would place the resolved path outside the workspace root raise
        ``PermissionError``.  ``os.path.realpath`` is used so that symlinks
        are followed before the containment check — a symlink pointing out of
        the workspace is indistinguishable from a plain traversal from the
        host's perspective.
        """
        workspace = Path(os.path.realpath(self._working_directory))
        candidate = Path(os.path.realpath(workspace / path))
        try:
            candidate.relative_to(workspace)
        except ValueError:
            raise PermissionError(f"Path {path!r} escapes the sandbox workspace") from None
        return candidate

    @override
    async def run(
        self,
        *command: str | Path,
        timeout: float | None = None,
        shell: bool | list[str] = True,
        user: str | User | None = None,
    ) -> ExecResult:
        del user  # local backend ignores per-call user identity
        argv = [str(p) for p in command]
        env = {**os.environ, **self._default_env}
        cwd = str(self._working_directory)
        start_time = time.monotonic()
        if shell is True:
            joined = argv[0] if len(argv) == 1 else " ".join(argv)
            proc = await asyncio.create_subprocess_shell(
                joined,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        elif isinstance(shell, list):
            proc = await asyncio.create_subprocess_exec(
                *shell,
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError as exc:
            proc.kill()
            await proc.wait()
            elapsed = time.monotonic() - start_time
            raise ExecTimeoutError(f"Command timed out after {elapsed:.2f}s (limit={timeout}s)") from exc
        finally:
            # CancelledError path: the process was not yet reaped (returncode
            # is still None) — kill it and wait so no orphan runs with open
            # PIPE file descriptors that can fill the kernel buffer and block.
            if proc.returncode is None:
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
        duration_ms = int((time.monotonic() - start_time) * 1000)
        return ExecResult(
            stdout=stdout or b"",
            stderr=stderr or b"",
            exit_code=proc.returncode if proc.returncode is not None else -1,
            duration_ms=duration_ms,
        )

    @override
    async def pty_start(
        self,
        *command: str | Path,
        user: str | User | None = None,
    ) -> PtyHandle:
        del user
        # The local backend does not yet implement a streaming PTY
        # surface; concrete impl arrives with the PTY-streaming
        # phase if needed. For now we synchronously run the command
        # and return a handle pointing at the captured stdout.
        joined = " ".join(str(c) for c in command)
        return PtyHandle(
            session_id=self._session_id,
            command=joined,
            backend_payload={"impl": "local_subprocess_passthrough"},
        )

    @override
    async def pty_write_stdin(self, handle: PtyHandle, data: bytes) -> None:
        del handle, data
        # No-op for the local passthrough impl.

    @override
    async def pty_terminate_all(self) -> None:
        # No-op for the local passthrough impl.
        pass

    @override
    def supports_pty(self) -> bool:
        return False

    @override
    async def read(self, path: Path | str, *, user: str | User | None = None) -> IOBase:
        del user
        full = self._resolve_inside_workspace(path)
        try:
            return open(full, "rb")
        except FileNotFoundError as exc:
            raise WorkspaceReadNotFoundError(f"Workspace path not found: {full}") from exc

    @override
    async def write(
        self,
        path: Path | str,
        data: IOBase,
        *,
        user: str | User | None = None,
    ) -> None:
        del user
        full = self._resolve_inside_workspace(path)
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(data.read())

    @override
    async def ls(
        self,
        path: Path | str,
        *,
        user: str | User | None = None,
    ) -> list[FileEntry]:
        del user
        full = self._resolve_inside_workspace(path)
        if not full.exists():
            raise WorkspaceReadNotFoundError(f"Workspace path not found: {full}")
        result: list[FileEntry] = []
        for child in sorted(full.iterdir()):
            is_dir = child.is_dir()
            # lstat() does not follow symlinks, so a dangling symlink (which
            # sandboxed commands can create via `ln -s`) reports the link's
            # own size instead of raising FileNotFoundError on the missing
            # target. For regular files lstat().st_size == stat().st_size.
            size = -1 if is_dir else child.lstat().st_size
            result.append(FileEntry(name=child.name, is_directory=is_dir, size_bytes=size))
        return result

    @override
    async def rm(
        self,
        path: Path | str,
        *,
        recursive: bool = False,
        user: str | User | None = None,
    ) -> None:
        del user
        full = self._resolve_inside_workspace(path)
        if not full.exists():
            return
        if full.is_dir():
            if not recursive:
                raise IsADirectoryError(f"rm: {full} is a directory; pass recursive=True to remove")
            shutil.rmtree(full)
        else:
            full.unlink()

    @override
    async def mkdir(
        self,
        path: Path | str,
        *,
        parents: bool = False,
        user: str | User | None = None,
    ) -> None:
        del user
        full = self._resolve_inside_workspace(path)
        full.mkdir(parents=parents, exist_ok=True)

    @override
    async def extract(
        self,
        path: Path | str,
        data: IOBase,
        *,
        compression_scheme: Literal["tar", "zip"] | None = None,
    ) -> None:
        del compression_scheme
        full = self._resolve_inside_workspace(path)
        full.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=data, mode="r:*") as tar:
            # extractall's filter="data" rejects traversal and link
            # members but enforces no resource bounds; validate first so
            # a tar bomb is refused before any member lands on disk.
            validate_tar_archive_for_extraction(tar, archive_limits=self._archive_limits)
            tar.extractall(full, filter="data")

    @override
    async def persist_workspace(self) -> IOBase:
        buf = BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.add(self._working_directory, arcname=".")
        buf.seek(0)
        return buf

    @override
    async def hydrate_workspace(self, data: IOBase) -> None:
        # Validate BEFORE the destructive wipe so a rejected archive leaves
        # the existing workspace intact instead of emptying it. Symlinks are
        # permitted here (dev workspaces ship them — e.g. venv) so a snapshot
        # the framework itself produced round-trips; filter="data" still
        # enforces symlink-target safety at extraction time.
        with tarfile.open(fileobj=data, mode="r") as tar:
            validate_tar_archive_for_extraction(
                tar,
                archive_limits=self._archive_limits,
                allow_symlinks=True,
            )
            shutil.rmtree(self._working_directory, ignore_errors=True)
            self._working_directory.mkdir(parents=True, exist_ok=True)
            tar.extractall(self._working_directory, filter="data")

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
    async def apply_patch(self, patch: str, *, user: str | User | None = None) -> str:
        del user
        # Minimal patch handler: invoke `patch` if present. ``run`` does not
        # pipe stdin, so ``patch -i -`` would never receive the diff — the
        # patch would silently apply nothing (or hang on an interactive
        # stdin). Write the diff to a temp file inside the working dir and
        # feed it via ``-i <file>``. Production-grade patch handling lives in
        # the Filesystem capability's apply_patch tool.
        fd, patch_path = tempfile.mkstemp(suffix=".diff", dir=str(self._working_directory))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(patch)
            # shell=False routes through create_subprocess_exec so each arg —
            # including the temp-file path, which may contain spaces when the
            # caller's working_directory does — is passed verbatim without
            # shell word-splitting.
            result = await self.run("patch", "-p1", "-i", patch_path, shell=False, timeout=30.0)
        finally:
            Path(patch_path).unlink(missing_ok=True)
        if result.exit_code != 0:
            return f"apply_patch: exit_code={result.exit_code}; stderr={result.stderr!r}"
        return f"apply_patch: OK; patch_len={len(patch)}"

    @override
    async def running(self) -> bool:
        return self._started and not self._stopped

    @override
    async def resolve_exposed_port(self, port: int) -> ExposedPortEndpoint:
        # Local backend has no port-mapping layer; ports are reachable
        # as-is on localhost.
        return ExposedPortEndpoint(host="127.0.0.1", port=port)


class LocalSubprocessSandboxClient(BaseSandboxClient[LocalSandboxClientOptions]):
    """Dev-only sandbox client that runs commands as host subprocesses."""

    backend_id = "unix_local"
    # Self-hosted: no per-minute compute charge.
    cost = SandboxCostDescriptor(free=True)
    capabilities = SandboxBackendCapabilities(network=True, persistent=False)

    def __init__(
        self,
        *,
        network_policy: NetworkPolicy | None = None,
        warn_banner: bool = True,
    ) -> None:
        self._network_policy = network_policy
        if warn_banner:
            logger.warning(_DEV_ONLY_BANNER)

    @override
    async def create(
        self,
        *,
        snapshot: SnapshotSpec | None = None,
        snapshot_store: Any | None = None,
        manifest: Manifest | None = None,
        options: LocalSandboxClientOptions | None = None,
    ) -> LocalSandboxSession:
        reject_unsupported_snapshot_store(snapshot_store, self.backend_id)
        warn_discarded_snapshot(snapshot, self.backend_id, logger)
        del snapshot
        opts = options or LocalSandboxClientOptions()
        if opts.working_directory is not None:
            workdir = Path(opts.working_directory)
            cleanup = False
        else:
            workdir = Path(tempfile.mkdtemp(prefix="troopai-sandbox-"))
            cleanup = True
        return LocalSandboxSession(
            working_directory=workdir,
            cleanup_on_shutdown=cleanup,
            default_env=opts.default_env,
            network_policy=self._network_policy,
            manifest=manifest,
            archive_limits=opts.archive_limits,
        )

    @override
    async def delete(self, session: BaseSandboxSession) -> BaseSandboxSession:
        await session.aclose()
        return session

    @override
    async def resume(self, state: SandboxSessionState) -> BaseSandboxSession:
        if state.backend_id != self.backend_id:
            raise ValueError(f"LocalSubprocessSandboxClient cannot resume backend_id={state.backend_id!r}")
        # Local backend has no live resource to reconnect to. Make a
        # fresh session and let hydrate_workspace seed it from the
        # snapshot if the caller does that explicitly.
        return await self.create()

    @override
    def deserialize_session_state(self, payload: dict[str, Any]) -> SandboxSessionState:
        return SandboxSessionState.model_validate(payload)
