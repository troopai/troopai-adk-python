"""``BaseSandboxSession`` — abstract live-session contract.

Async context manager. Every concrete backend implements lifecycle
(start / stop / shutdown / aclose), command execution (the
run-a-command primitive + PTY family), file operations (read /
write / ls / rm / mkdir / extract), workspace persistence
(persist_workspace / hydrate_workspace / apply_manifest /
apply_patch), and utility introspection (running / resolve_exposed_port
/ normalize_path / supports_docker_volume_mounts / supports_pty).

The class sets ``_TROOPAI_SANDBOX_SESSION_MARKER`` so
``SandboxCapability._clone_value`` can detect and reset session
references on per-run capability clones.
"""

from __future__ import annotations

import abc
from collections.abc import Iterable, Iterator
from io import IOBase
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal, Self

if TYPE_CHECKING:
    from troopai.adk.types.sandbox.entries import MaterializedFile
    from troopai.adk.types.sandbox.exec_result import (
        ExecResult,
        ExposedPortEndpoint,
        PtyHandle,
    )
    from troopai.adk.types.sandbox.permissions import User

__all__ = ["BaseSandboxSession", "FileEntry", "MaterializationResult"]


class FileEntry:
    """Lightweight record returned by ``BaseSandboxSession.ls``.

    Concrete sessions construct this directly. Stays a plain class
    (not a Pydantic model) so backends with thousands of files don't
    pay validation cost on listing.

    Attributes:
        name: File or directory name.
        is_directory: True when the entry is a directory.
        size_bytes: Byte size; ``-1`` when unknown or irrelevant
            (directories on most backends).
    """

    __slots__ = ("is_directory", "name", "size_bytes")

    def __init__(self, *, name: str, is_directory: bool, size_bytes: int = -1) -> None:
        self.name = name
        self.is_directory = is_directory
        self.size_bytes = size_bytes


class MaterializationResult:
    """Record returned by ``BaseSandboxSession.apply_manifest``.

    Attributes:
        files: Materialized files (one per non-Mount manifest entry).
        skipped_mounts: Mount entries deferred to backend-native
            attach instead of file-copy materialization.
    """

    __slots__ = ("files", "skipped_mounts")

    def __init__(
        self,
        *,
        files: Iterable[MaterializedFile] = (),
        skipped_mounts: Iterable[str] = (),
    ) -> None:
        self.files = list(files)
        self.skipped_mounts = list(skipped_mounts)

    def __iter__(self) -> Iterator[MaterializedFile]:
        return iter(self.files)


class BaseSandboxSession(abc.ABC):
    """Abstract live sandbox session — async context manager.

    Use the context-manager form for the common path::

        async with await client.create(...) as session:
            result = await session.run("ls", "/tmp")

    Or call lifecycle methods directly when you need explicit
    control::

        session = await client.create(...)
        try:
            await session.start()
            ...
        finally:
            await session.aclose()

    ``aclose`` is the full cleanup path: it runs pre-stop hooks,
    calls ``stop``, then ``shutdown``, then releases session-scoped
    dependencies. ``stop`` alone only persists snapshot-backed state.
    """

    # Sentinel marker the capability clone helper checks to skip
    # deep-copying session references.
    _TROOPAI_SANDBOX_SESSION_MARKER: ClassVar[bool] = True

    # --- Async context-manager protocol -----------------------------------

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        await self.aclose()

    # --- Lifecycle --------------------------------------------------------

    @abc.abstractmethod
    async def start(self) -> None:
        """Initialize backend resources and materialize the manifest."""

    @abc.abstractmethod
    async def stop(self) -> None:
        """Persist snapshot-backed workspace contents.

        Does NOT release backend resources — call ``aclose`` for full
        teardown. Safe to call multiple times; later calls re-persist
        the snapshot if the workspace mutated since the last stop.
        """

    @abc.abstractmethod
    async def shutdown(self) -> None:
        """Best-effort backend-resource release.

        Called inside ``aclose`` after ``stop``. Implementations MUST
        not raise on missing resources (already-deleted containers,
        revoked sessions, etc.).
        """

    @abc.abstractmethod
    async def aclose(self) -> None:
        """Full session cleanup: pre-stop hooks then stop then shutdown."""

    # --- Command execution ------------------------------------------------

    @abc.abstractmethod
    async def run(
        self,
        *command: str | Path,
        timeout: float | None = None,
        shell: bool | list[str] = True,
        user: str | User | None = None,
    ) -> ExecResult:
        """Run ``command`` to completion and return an ``ExecResult``.

        Method name is ``run`` (not the shell builtin) to avoid
        shadowing Python's exec primitive in user code while staying
        unambiguous in tracing.

        Args:
            command: Argv parts (or a single shell-string when
                ``shell=True``).
            timeout: Wall-clock cap in seconds, or ``None`` for the
                backend default.
            shell: ``True`` (default) wraps the command in a shell;
                ``False`` invokes argv directly; a list is the shell
                command + flags (e.g. ``["bash", "-lc"]``).
            user: Optional override for the per-call user identity.
        """

    @abc.abstractmethod
    async def pty_start(
        self,
        *command: str | Path,
        user: str | User | None = None,
    ) -> PtyHandle:
        """Start a PTY-streamed command; returns an opaque handle.

        The backend streams output via its own channel (events,
        callback, ...); the handle is passed back to ``pty_write_stdin``
        and ``pty_terminate_all``.
        """

    @abc.abstractmethod
    async def pty_write_stdin(self, handle: PtyHandle, data: bytes) -> None:
        """Send ``data`` to the PTY's stdin."""

    @abc.abstractmethod
    async def pty_terminate_all(self) -> None:
        """Terminate every active PTY session in this sandbox."""

    # --- File operations --------------------------------------------------

    @abc.abstractmethod
    async def read(self, path: Path | str, *, user: str | User | None = None) -> IOBase:
        """Open ``path`` for reading and return a binary stream."""

    @abc.abstractmethod
    async def write(
        self,
        path: Path | str,
        data: IOBase,
        *,
        user: str | User | None = None,
    ) -> None:
        """Write ``data`` to ``path`` (binary, atomic when possible)."""

    @abc.abstractmethod
    async def ls(
        self,
        path: Path | str,
        *,
        user: str | User | None = None,
    ) -> list[FileEntry]:
        """List entries in ``path``."""

    @abc.abstractmethod
    async def rm(
        self,
        path: Path | str,
        *,
        recursive: bool = False,
        user: str | User | None = None,
    ) -> None:
        """Remove ``path``; recursive when explicitly requested."""

    @abc.abstractmethod
    async def mkdir(
        self,
        path: Path | str,
        *,
        parents: bool = False,
        user: str | User | None = None,
    ) -> None:
        """Create ``path``; create intermediate directories when ``parents``."""

    @abc.abstractmethod
    async def extract(
        self,
        path: Path | str,
        data: IOBase,
        *,
        compression_scheme: Literal["tar", "zip"] | None = None,
    ) -> None:
        """Extract an archive ``data`` into ``path``.

        ``compression_scheme=None`` lets the backend sniff the
        archive header.
        """

    # --- Workspace persistence -------------------------------------------

    @abc.abstractmethod
    async def persist_workspace(self) -> IOBase:
        """Serialize the workspace tree into a tar stream."""

    @abc.abstractmethod
    async def hydrate_workspace(self, data: IOBase) -> None:
        """Restore the workspace from a previously persisted stream."""

    @abc.abstractmethod
    async def apply_manifest(
        self,
        *,
        only_ephemeral: bool = False,
    ) -> MaterializationResult:
        """Materialize the (already-validated) manifest into the workspace.

        When ``only_ephemeral`` is True, the backend re-materializes
        only ephemeral entries (mounts re-attach, transient files
        re-write); durable entries assumed already-present.
        """

    @abc.abstractmethod
    async def apply_patch(self, patch: str, *, user: str | User | None = None) -> str:
        """Apply a unified-diff patch to the workspace; return summary."""

    # --- Utilities --------------------------------------------------------

    @abc.abstractmethod
    async def running(self) -> bool:
        """Return True iff the backend reports the session alive."""

    @abc.abstractmethod
    async def resolve_exposed_port(self, port: int) -> ExposedPortEndpoint:
        """Return the host- or tunnel-routable endpoint for sandbox ``port``."""

    def normalize_path(self, path: Path | str, *, for_write: bool = False) -> Path:
        """Normalize ``path`` to the sandbox's canonical form.

        Default returns the input as ``Path`` unchanged. Backends with
        case-insensitive filesystems or path-translation requirements
        override this. ``for_write`` lets backends rewrite paths
        differently for read vs write surfaces.
        """
        _ = for_write
        return Path(path)

    def supports_docker_volume_mounts(self) -> bool:
        """Default False; ``DockerSandboxSession`` overrides to True."""
        return False

    def supports_pty(self) -> bool:
        """Default False; backends that proxy PTY override to True."""
        return False

    @property
    def session_id(self) -> str | None:
        """Backend-assigned session ID, or ``None`` when not yet started.

        Default returns ``None``. Backends override to expose their
        own identifier so callers can correlate logs / spans / audit.
        """
        return None
