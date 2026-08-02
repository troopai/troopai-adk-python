"""DockerSandboxClient — production container backend (TDK.2 + TDK.12).

Spawns a long-lived container per session with ``sleep infinity`` as
the PID-1 command so ``exec_run`` calls can drive workloads inside.
NetworkPolicy + ResourceLimits flow through the
``apply_network_policy_to_docker`` + ``apply_resource_limits_to_docker``
helpers.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
from typing import TYPE_CHECKING, Any, override

from pydantic import Field

from troopai.adk.exceptions.exceptions import (
    SandboxStartFailed,
    UnsupportedSandboxClientError,
)
from troopai.adk.sandbox.clients.base import (
    BaseSandboxClient,
    BaseSandboxClientOptions,
    reject_unsupported_snapshot_store,
    warn_discarded_snapshot,
)
from troopai.adk.sandbox.clients.docker.docker_session import DockerSandboxSession
from troopai.adk.sandbox.clients.session import BaseSandboxSession
from troopai.adk.types.sandbox.cost import SandboxBackendCapabilities, SandboxCostDescriptor
from troopai.adk.types.sandbox.network import NetworkPolicy
from troopai.adk.types.sandbox.resource_limits import SandboxResourceLimits
from troopai.adk.types.sandbox.session_state import SandboxSessionState

if TYPE_CHECKING:
    from troopai.adk.types.sandbox.manifest import Manifest
    from troopai.adk.types.sandbox.snapshot import SnapshotSpec

logger = logging.getLogger(__name__)

__all__ = ["DockerSandboxClient", "DockerSandboxClientOptions"]


def _require_docker() -> None:
    try:
        # docker package lives in the [sandbox-docker] extra; bare import
        # for the ImportError signal only. mypy/pyright ignore optional dep.
        import docker as _docker  # type: ignore[import-untyped]  # noqa: F401  # pyright: ignore[reportMissingModuleSource, reportUnusedImport]
    except ImportError as exc:
        raise UnsupportedSandboxClientError(
            "DockerSandboxClient requires the optional 'docker' package: pip install 'troopai-adk-python[sandbox-docker]'",
        ) from exc


def _volume_name(target: str, driver: str, driver_options: dict[str, str]) -> str:
    """Deterministic, collision-safe driver-volume name.

    Same (target, driver, options) ⇒ same name (so a re-create is the
    daemon's idempotent no-op); any differing spec ⇒ a different name
    (sha256 of the triple). Slugged to Docker's allowed charset.
    """
    opts = ",".join(f"{k}={v}" for k, v in sorted(driver_options.items()))
    digest = hashlib.sha256(f"{target}|{driver}|{opts}".encode()).hexdigest()[:12]
    slug = target.strip("/").replace("/", "-").replace(".", "-")[-24:]
    return f"troopai-{slug}-{digest}"


def _require_spec_keys(spec: dict[str, Any], keys: tuple[str, ...]) -> None:
    """Fail loud (typed) if a mount spec is missing a key its branch needs.

    Without this, a malformed spec would raise a bare ``KeyError`` deep
    in materialization; ``create``'s ``except Exception`` would then
    re-wrap it as an opaque ``provisioning failed: 'driver'`` message
    indistinguishable from a Docker daemon fault. Validating up front
    yields an explicit ``malformed mount spec`` cause instead. Two
    call sites; isolates the typed-raise-on-missing-key concern.
    """
    missing = [k for k in keys if k not in spec]
    if len(missing) > 0:
        raise SandboxStartFailed(
            backend_id="docker",
            reason=f"malformed mount spec (strategy={spec.get('strategy')!r}): missing keys {missing!r}",
        )


def _spec_to_docker_mount(docker_client: Any, docker_types: Any, spec: dict[str, Any]) -> Any:
    """Convert ONE neutral mount-spec dict to a ``docker.types.Mount``.

    * ``in_container`` → an anonymous volume (``source=None`` is
      docker-py's idiom for a fresh writable mountpoint dir). The
      in-container FUSE tool does the real mount and enforces
      read-only itself, so the Docker mountpoint stays writable
      regardless of the mount's declared ``read_only``.
    * ``docker_volume`` → pre-create the driver-backed named volume
      (idempotent: the daemon returns the existing volume on a name
      clash), then reference it by name. ``DriverConfig`` is NOT
      passed — the daemon ignores it for a pre-existing named source.

    ANY malformed spec fails loud as ``SandboxStartFailed`` — a
    missing required key (validated per branch via
    ``_require_spec_keys``) and an unknown / missing ``strategy`` (the
    exhaustive final raise) both produce an explicit typed cause, never
    a bare ``KeyError`` and never a silent wrong-spec routed to a
    volume. Symmetric with the policy layer's exhaustive dispatch.

    Extracted from ``_materialize_docker_mounts`` so the per-spec
    conversion (the exhaustive strategy dispatch) is independently
    testable and keeps the materializer within the function-length
    bound; it is the single conversion site, deliberately so.
    """
    strategy = spec.get("strategy")
    if strategy == "in_container":
        _require_spec_keys(spec, ("target", "read_only"))
        if spec["read_only"]:
            logger.debug(
                "in-container mount %s: Docker mountpoint is writable; "
                "read-only is enforced by the in-container tool, not the bind",
                spec["target"],
            )
        return docker_types.Mount(target=spec["target"], source=None, type="volume", read_only=False)
    if strategy == "docker_volume":
        _require_spec_keys(spec, ("target", "read_only", "driver", "driver_options"))
        name = _volume_name(spec["target"], spec["driver"], spec["driver_options"])
        docker_client.volumes.create(
            name=name,
            driver=spec["driver"],
            driver_opts=spec["driver_options"] if len(spec["driver_options"]) > 0 else None,
        )
        logger.warning(
            "pre-created operator-managed driver volume %r (driver=%r); NOT "
            "auto-removed on session close or partial-create failure — "
            "`docker volume prune` owns its lifecycle",
            name,
            spec["driver"],
        )
        return docker_types.Mount(target=spec["target"], source=name, type="volume", read_only=spec["read_only"])
    raise SandboxStartFailed(
        backend_id="docker",
        reason=f"mount-spec materialization: unrecognized strategy {strategy!r}",
    )


def _materialize_docker_mounts(docker_client: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Convert the policy layer's neutral mount-spec dicts to docker.types.Mount.

    ``apply_mounts_to_docker`` leaves ``kwargs["mounts"]`` as
    provider-agnostic ``InContainerMountSpec`` / ``DockerVolumeMountSpec``
    dicts — the policy layer must not import the docker SDK. This is
    the consuming client's conversion step; the spec dicts are popped
    so they never reach ``containers.run``. Per-spec conversion and
    the exhaustive strategy dispatch live in ``_spec_to_docker_mount``.

    Lifecycle: pre-created driver volumes are NOT auto-removed —
    neither on session close (driver-backed volumes may be shared /
    externally managed; teardown has no volume-tracking surface) NOR
    on a partial-creation failure (if a later spec raises, or
    ``containers.run`` fails after volumes were created, the
    already-created volumes stay registered). Mitigation: volume
    names are deterministic (``_volume_name``), so retrying the same
    session-create reuses them idempotently instead of leaking a
    fresh one per attempt; and ``_spec_to_docker_mount`` emits a
    ``logger.warning`` naming each pre-created volume so an orphan is
    findable in logs, not only in this docstring. ``docker volume
    prune`` owns final lifecycle. Deliberately operator-managed and
    runtime-surfaced — not a silent leak.
    """
    specs: list[Any] = kwargs.pop("mounts", [])
    if len(specs) == 0:
        return kwargs
    # docker is the optional [sandbox-docker] extra; lazy-import here
    # mirrors __init__'s discipline (never module-level).
    import docker.types  # type: ignore[import-untyped]  # pyright: ignore[reportMissingModuleSource]

    kwargs["mounts"] = [_spec_to_docker_mount(docker_client, docker.types, spec) for spec in specs]
    return kwargs


def _build_run_kwargs(options: DockerSandboxClientOptions, manifest: Manifest | None) -> dict[str, Any]:
    """Assemble docker ``containers.run`` kwargs from framework options.

    Translates framework-agnostic NetworkPolicy / ResourceLimits /
    Mount entries to docker SDK kwargs. Imports are deferred to break
    the policy↔client import cycle. ``resource_limits`` overlays the
    direct cpu / memory / pid kwargs: the limits helper runs last, so a
    set ``resource_limits`` field wins over the matching direct kwarg
    (and its cpu form replaces the direct ``nano_cpus``). Mount entries are left as
    neutral spec dicts in ``kwargs["mounts"]`` —
    ``_materialize_docker_mounts`` converts them inside the create
    try-block so a volume-driver failure is a typed create-failure.

    Extracted from ``create`` so the framework-options → docker-kwargs
    translation is one named concern and ``create`` stays within the
    function-length bound; single call site, deliberately so.
    """
    from troopai.adk.sandbox.policy import (
        apply_mounts_to_docker,
        apply_network_policy_to_docker,
        apply_resource_limits_to_docker,
    )
    from troopai.adk.types.sandbox.mounts import Mount

    kwargs: dict[str, Any] = {
        "image": options.image,
        "command": ["sleep", "infinity"],
        "detach": True,
        "environment": options.environment,
        "working_dir": options.working_directory,
    }
    if options.cpu_count is not None:
        # A set resource_limits.cpu_cores (applied below) overlays this:
        # the helper drops nano_cpus and writes cpu_period/cpu_quota instead.
        kwargs["nano_cpus"] = int(options.cpu_count * 1_000_000_000)
    if options.memory_mb is not None:
        kwargs["mem_limit"] = f"{options.memory_mb}m"
    if options.pid_limit is not None:
        kwargs["pids_limit"] = options.pid_limit
    kwargs = apply_network_policy_to_docker(options.network_policy, kwargs)
    kwargs = apply_resource_limits_to_docker(options.resource_limits, kwargs)
    mounts = [m for m in (manifest.entries.values() if manifest is not None else []) if isinstance(m, Mount)]
    # workspace_root MUST be the container's actual WORKDIR so mount
    # targets resolve under it (else a mount lands at /workspace/x
    # while the agent looks under options.working_directory).
    return apply_mounts_to_docker(mounts, kwargs, workspace_root=options.working_directory)


class DockerSandboxClientOptions(BaseSandboxClientOptions):
    """Options for the Docker backend.

    Attributes:
        image: Container image (e.g. ``python:3.12-slim``). Required; no
            default.
        environment: Environment variables injected into the container at
            run time.
        cpu_count: Fractional CPU allotment; translated to ``nano_cpus``
            (1.0 = 1 core). ``None`` leaves Docker's default.
        memory_mb: Memory limit in MiB; translated to Docker's
            ``mem_limit`` string. ``None`` leaves Docker's default.
        pid_limit: Per-container PID cap; translated to Docker's
            ``pids_limit``. ``None`` leaves Docker's default.
        working_directory: Container ``WORKDIR`` and the session's default
            for relative paths.
        network_policy: Framework ``NetworkPolicy`` translated to Docker
            network kwargs on session create.
        resource_limits: Framework resource limits that overlay the direct
            cpu/memory/pid kwargs above.
    """

    image: str
    """Container image (e.g. ``python:3.12-slim``). Required; no default."""

    environment: dict[str, str] = Field(default_factory=dict)
    """Environment variables injected into the container at run time."""

    cpu_count: float | None = None
    """Fractional CPU allotment; translated to ``nano_cpus`` (1.0 = 1 core)."""

    memory_mb: int | None = None
    """Memory limit in MiB; translated to Docker's ``mem_limit`` string."""

    pid_limit: int | None = None
    """Per-container PID cap; translated to Docker's ``pids_limit``."""

    working_directory: str = "/workspace"
    """Container ``WORKDIR`` and the session's default for relative paths."""

    network_policy: NetworkPolicy | None = None
    """Framework NetworkPolicy translated to docker network kwargs on create."""

    resource_limits: SandboxResourceLimits | None = None
    """Framework resource limits; overlays the direct cpu/memory/pid kwargs."""


class DockerSandboxClient(BaseSandboxClient[DockerSandboxClientOptions]):
    """Production Docker-backed sandbox client."""

    backend_id = "docker"
    # Self-hosted: no per-minute compute charge.
    cost = SandboxCostDescriptor(free=True)
    capabilities = SandboxBackendCapabilities(network=True, persistent=True)

    def __init__(self, *, docker_client: Any = None) -> None:
        if docker_client is not None:
            self._docker = docker_client
            return
        _require_docker()
        # Lazy-imported because docker is an optional dependency. No
        # [import-untyped] is needed on this bare import: _require_docker()
        # above already imported `docker` with that suppression, which
        # registers the module as known-untyped for this whole file, so
        # mypy no longer flags it here. pyright still needs the
        # missing-module-source suppression for the stub-less extra.
        import docker  # pyright: ignore[reportMissingModuleSource]

        self._docker = docker.from_env()

    @override
    async def create(
        self,
        *,
        snapshot: SnapshotSpec | None = None,
        snapshot_store: Any | None = None,
        manifest: Manifest | None = None,
        options: DockerSandboxClientOptions,
    ) -> BaseSandboxSession:
        """Start a fresh container session from ``options`` + ``manifest``.

        Snapshot persistence is NOT implemented for the Docker
        backend. ``snapshot_store`` raises
        ``UnsupportedSnapshotFeatureError`` — a configured store
        silently dropped would be a data-durability lie. ``snapshot``
        is accepted for ``BaseSandboxClient`` ABC conformance and
        explicitly discarded (``del snapshot``), not silently
        honoured. Use ``resume`` to rebind an existing container by
        id. A caller needing snapshot-based restore must select a
        backend that implements it.
        """
        reject_unsupported_snapshot_store(snapshot_store, self.backend_id)
        warn_discarded_snapshot(snapshot, self.backend_id, logger)
        del snapshot
        kwargs = _build_run_kwargs(options, manifest)
        container = None
        try:
            # Convert neutral spec dicts to docker.types.Mount + pre-create
            # driver-backed volumes (blocking daemon calls → to_thread).
            # Inside the try so a volume-driver failure surfaces as
            # SandboxStartFailed like containers.run — not a raw SDK error
            # that bypasses the create-failure contract.
            kwargs = await asyncio.to_thread(_materialize_docker_mounts, self._docker, kwargs)
            container = await asyncio.to_thread(
                self._docker.containers.run,
                **kwargs,
            )
        except SandboxStartFailed:
            # Already the typed create-failure (e.g. an unrecognized mount
            # strategy raised by _materialize_docker_mounts) — re-raise
            # without re-wrapping so the precise reason is preserved.
            raise
        except asyncio.CancelledError:
            # The asyncio task was cancelled while the Docker daemon thread
            # was running. The thread may have completed (container created)
            # but CancelledError bypasses the normal except-Exception handler.
            # Remove the orphaned container before re-raising so no runaway
            # container consumes host resources indefinitely.
            if container is not None:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(container.remove, force=True)
            raise
        except Exception as exc:
            raise SandboxStartFailed(
                backend_id="docker",
                reason=f"container/volume provisioning failed: {exc}",
                details={"image": options.image},
            ) from exc
        return DockerSandboxSession(
            container=container,
            working_directory=options.working_directory,
            network_policy=options.network_policy,
            resource_limits=options.resource_limits,
            environment=options.environment,
            manifest=manifest,
        )

    @override
    async def delete(self, session: BaseSandboxSession) -> BaseSandboxSession:
        await session.aclose()
        return session

    @override
    async def resume(self, state: SandboxSessionState) -> BaseSandboxSession:
        """Reconnect to a previously-saved container.

        Looks for ``provider_payload["container_id"]``. If the
        container still exists in Docker, rebinds. Otherwise raises
        ``SandboxStartFailed``.
        """
        container_id = state.provider_payload.get("container_id")
        if not isinstance(container_id, str) or len(container_id) == 0:
            raise SandboxStartFailed(
                backend_id="docker",
                reason="resume requires provider_payload['container_id']",
            )
        try:
            container = await asyncio.to_thread(
                self._docker.containers.get,
                container_id,
            )
        except Exception as exc:
            raise SandboxStartFailed(
                backend_id="docker",
                reason=f"container {container_id!r} not found: {exc}",
            ) from exc
        return DockerSandboxSession(container=container)

    @override
    def deserialize_session_state(
        self,
        payload: dict[str, Any],
    ) -> SandboxSessionState:
        return SandboxSessionState.model_validate(payload)
