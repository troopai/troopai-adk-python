"""Mount → backend translation helpers.

Framework Mount types (S3Mount, GCSMount, R2Mount, AzureBlobMount,
BoxMount, S3FilesMount) describe cloud storage to materialize inside
a sandbox workspace. Each Mount carries a ``mount_strategy``:

- ``InContainerMountStrategy`` — an in-sandbox FUSE tool
  (rclone / mount-s3 / blobfuse2 / mount.s3files, selected by the
  strategy's ``pattern``) performs the mount inside the workspace.
- ``DockerVolumeMountStrategy`` — a Docker volume-driver-backed
  named volume (Docker backend only).

Translation is strategy-dispatched; each backend realizes it
differently:

- Docker (``apply_mounts_to_docker``): emits provider-agnostic
  ``InContainerMountSpec`` / ``DockerVolumeMountSpec`` dicts (plus
  ``cap_add=SYS_ADMIN`` when an in-container mount is present). The
  Docker client converts those to ``docker.types.Mount`` /
  ``DriverConfig`` at session-create — the provider-SDK boundary
  stays inside ``clients/docker/``, never in this policy layer.
- K8s (``apply_mounts_to_k8s_pod``): CSI volume + volumeMount
  entries on the pod spec. K8s has no Docker daemon, so a
  ``DockerVolumeMountStrategy`` raises ``UnsupportedMountStrategyError``
  rather than being silently materialized as a CSI mount.
- LocalSubprocess (``describe_mount_for_local``): shell argv for
  ``rclone`` / ``gcsfuse`` / ``blobfuse2``; likewise rejects
  non-in-container strategies (no Docker daemon).
- Hosted bridges (``apply_mounts_to_hosted_bridge``): each
  provider's cloud-bucket attribute on the create-body JSON.
  Deliberately strategy-agnostic — the managed provider owns mount
  realization, so ``mount_strategy`` is advisory there.

``mount.mount_path`` is the workspace-relative destination; when
None the target falls back to ``mount-<mount.type>`` (the stable
wire discriminator, NOT the Python class name) so a class rename
cannot silently relocate a sandbox's mountpoint.
"""

from __future__ import annotations

from typing import Any

from troopai.adk.exceptions import (
    SandboxConfigurationError,
    UnsupportedMountPatternError,
    UnsupportedMountStrategyError,
)
from troopai.adk.types.sandbox.mounts import (
    AzureBlobMount,
    BoxMount,
    DockerVolumeMountSpec,
    DockerVolumeMountStrategy,
    FuseMountPattern,
    GCSMount,
    InContainerMountSpec,
    InContainerMountStrategy,
    Mount,
    MountpointMountPattern,
    R2Mount,
    RcloneMountPattern,
    S3FilesMount,
    S3FilesMountPattern,
    S3Mount,
)

__all__ = [
    "apply_mounts_to_docker",
    "apply_mounts_to_hosted_bridge",
    "apply_mounts_to_k8s_pod",
    "build_docker_volume_mount_spec",
    "build_in_container_mount_spec",
    "describe_mount_for_local",
]


def _bucket_uri(mount: Mount) -> str:
    """Return a ``<scheme>://bucket/prefix`` URI for the mount."""
    if isinstance(mount, S3Mount):
        prefix = mount.prefix if mount.prefix is not None else ""
        return f"s3://{mount.bucket}/{prefix}"
    if isinstance(mount, GCSMount):
        prefix = mount.prefix if mount.prefix is not None else ""
        return f"gs://{mount.bucket}/{prefix}"
    if isinstance(mount, R2Mount):
        prefix = mount.prefix if mount.prefix is not None else ""
        return f"r2://{mount.account_id}/{mount.bucket}/{prefix}"
    if isinstance(mount, AzureBlobMount):
        prefix = mount.prefix if mount.prefix is not None else ""
        return f"azure://{mount.account}/{mount.container}/{prefix}"
    if isinstance(mount, BoxMount):
        return f"box://{mount.folder_id}"
    if isinstance(mount, S3FilesMount):
        return f"s3files://{mount.mount_target_id}"
    return "mount://unknown"


def _mount_target(mount: Mount) -> str:
    """Return the workspace-relative target path for the mount.

    When ``mount_path`` is unset the fallback is derived from
    ``mount.type`` — the stable wire discriminator (e.g.
    ``s3_mount``) — NOT ``type(mount).__name__``. The Python class
    name is an implementation detail: renaming the class would
    silently relocate every unpathed sandbox's mount directory. The
    discriminator is the serialized contract, so the generated path
    stays stable across refactors and matches how every other
    dispatch in this module keys off ``.type``.
    """
    if mount.mount_path is None:
        return f"mount-{mount.type}"
    return mount.mount_path


# In-container mount-tool compatibility. A pattern names a specific
# CLI tool; only some mount backings can be served by each. rclone
# speaks every backing; mount-s3 is S3-protocol only; blobfuse2 is
# Azure Blob only; mount.s3files targets an S3 Files mount target.
_PATTERN_SUPPORTED_MOUNTS: dict[str, tuple[type[Mount], ...]] = {
    "rclone": (S3Mount, GCSMount, R2Mount, AzureBlobMount, BoxMount, S3FilesMount),
    "mountpoint": (S3Mount, R2Mount),
    "fuse": (AzureBlobMount,),
    "s3files": (S3FilesMount,),
}

_PatternConfig = dict[str, str | int | bool | list[str]]

# Zero-width / formatting code points (Unicode category Cf) that
# render visually blank but are NOT ``str.isspace()`` (so ``strip()``
# leaves them). Held as auditable hex ordinals — NEVER a string
# literal of the characters themselves, which would be invisible and
# un-reviewable in source and diffs. U+200B ZWSP, U+200C ZWNJ,
# U+200D ZWJ, U+2060 WORD JOINER, U+FEFF ZW NO-BREAK SPACE / BOM.
_BLANKISH_ZERO_WIDTH: frozenset[int] = frozenset({0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF})


def _is_blank_driver(driver: str) -> bool:
    """True if ``driver`` is empty, or only whitespace / zero-width.

    A driver name carrying any real (visible, non-format) character
    is accepted (host-registration is the backend's concern). One
    composed solely of whitespace and/or zero-width format code
    points is effectively empty and must fail loud at translation
    time rather than reach the Docker daemon as an opaque error.
    ``all(...)`` over an empty string is ``True`` — an empty driver
    is correctly blank.
    """
    return all(ch.isspace() or ord(ch) in _BLANKISH_ZERO_WIDTH for ch in driver)


def _resolve_target(mount: Mount, workspace_root: str) -> str:
    """Absolute in-container destination: ``workspace_root`` / target.

    ``mount.mount_path`` is validated workspace-relative (no absolute,
    no ``..``) by ``Mount``, so a plain join is safe.
    """
    rel = _mount_target(mount)
    return f"{workspace_root.rstrip('/')}/{rel.lstrip('/')}"


def _rclone_pattern_config(pattern: RcloneMountPattern) -> _PatternConfig:
    cfg: _PatternConfig = {
        "mode": pattern.mode,
        "remote_name": pattern.remote_name,
        "extra_args": list(pattern.extra_args),
        "nfs_mount_options": list(pattern.nfs_mount_options),
    }
    if pattern.nfs_addr is not None:
        cfg["nfs_addr"] = pattern.nfs_addr
    if pattern.config_file_path is not None:
        cfg["config_file_path"] = pattern.config_file_path
    return cfg


def _mountpoint_pattern_config(pattern: MountpointMountPattern) -> _PatternConfig:
    cfg: _PatternConfig = {
        "bucket": pattern.bucket,
        "extra_args": list(pattern.extra_args),
    }
    if pattern.prefix is not None:
        cfg["prefix"] = pattern.prefix
    if pattern.endpoint_url is not None:
        cfg["endpoint_url"] = pattern.endpoint_url
    return cfg


def _fuse_pattern_config(pattern: FuseMountPattern) -> _PatternConfig:
    cfg: _PatternConfig = {
        "container": pattern.container,
        "allow_other": pattern.allow_other,
        "extra_args": list(pattern.extra_args),
    }
    if pattern.cache_path is not None:
        cfg["cache_path"] = pattern.cache_path
    if pattern.cache_size_mb is not None:
        cfg["cache_size_mb"] = pattern.cache_size_mb
    return cfg


def _s3files_pattern_config(pattern: S3FilesMountPattern) -> _PatternConfig:
    return {
        "mount_target_id": pattern.mount_target_id,
        "extra_args": list(pattern.extra_args),
    }


def _require_strategy_matches_mount(
    mount: Mount,
    strategy: InContainerMountStrategy | DockerVolumeMountStrategy,
) -> None:
    """Reject a ``strategy`` that is not the mount's own strategy.

    Building a spec from a strategy unrelated to ``mount`` would
    silently materialize the wrong tool's config (e.g. an rclone
    mount served as mount-s3 against a different bucket) with no
    error. Identity OR value equality passes (a value-equal copy is
    fine); anything else fails loud at translation time. Shared by
    the in-container and docker-volume builders, so it accepts
    either strategy kind and the message names both discriminators.
    """
    if strategy is not mount.mount_strategy and strategy != mount.mount_strategy:
        raise SandboxConfigurationError(
            f"mount-spec translation: supplied strategy does not match "
            f"{type(mount).__name__}.mount_strategy — supplied "
            f"{type(strategy).__name__} (type={strategy.type!r}) vs "
            f"mount's {type(mount.mount_strategy).__name__} "
            f"(type={mount.mount_strategy.type!r}); pass the mount's "
            f"own mount strategy"
        )


def build_in_container_mount_spec(
    mount: Mount,
    strategy: InContainerMountStrategy,
    workspace_root: str,
) -> InContainerMountSpec:
    """Translate a Mount + ``InContainerMountStrategy`` to a neutral spec.

    The strategy's ``pattern`` names the in-container CLI tool and
    carries its configuration; the Mount subclass selects which
    patterns are valid and supplies ``read_only`` + the destination.

    ``pattern_config`` follows the spec's documented contract: keys
    whose source pattern field is ``None`` are OMITTED (a missing key
    means "unspecified"); ``tuple`` fields are emitted as ``list``.

    Raises:
        SandboxConfigurationError: ``strategy`` is not ``mount``'s own
            mount strategy (a mismatch would silently build the wrong
            tool's config — rejected at translation time instead).
        UnsupportedMountPatternError: the mount subclass cannot be
            served by the strategy's pattern, OR an unrecognized /
            future ``MountPattern`` subclass reached the dispatch
            (silent drop forbidden — the misconfiguration surfaces
            here, at translation time, never as a mistyped spec).
    """
    _require_strategy_matches_mount(mount, strategy)
    pattern = strategy.pattern
    supported = _PATTERN_SUPPORTED_MOUNTS.get(pattern.type, ())
    if not isinstance(mount, supported):
        raise UnsupportedMountPatternError(
            mount_type=type(mount).__name__,
            pattern_type=pattern.type,
        )
    if isinstance(pattern, RcloneMountPattern):
        pattern_config = _rclone_pattern_config(pattern)
    elif isinstance(pattern, MountpointMountPattern):
        pattern_config = _mountpoint_pattern_config(pattern)
    elif isinstance(pattern, FuseMountPattern):
        pattern_config = _fuse_pattern_config(pattern)
    elif isinstance(pattern, S3FilesMountPattern):
        pattern_config = _s3files_pattern_config(pattern)
    else:
        # Unreachable while InContainerMountStrategy.pattern is closed
        # over the four patterns above. If a fifth is added and reaches
        # here, fail loud at translation time rather than silently
        # emit a mistyped config under the wrong pattern_type.
        raise UnsupportedMountPatternError(
            mount_type=type(mount).__name__,
            pattern_type=pattern.type,
        )
    return InContainerMountSpec(
        type="bind",
        target=_resolve_target(mount, workspace_root),
        read_only=mount.read_only,
        strategy="in_container",
        pattern_type=pattern.type,
        pattern_config=pattern_config,
    )


def build_docker_volume_mount_spec(
    mount: Mount,
    strategy: DockerVolumeMountStrategy,
    workspace_root: str,
) -> DockerVolumeMountSpec:
    """Translate a Mount + ``DockerVolumeMountStrategy`` to a neutral spec.

    Docker attaches a volume-driver-backed volume before the
    container starts. The strategy supplies the developer-registered
    ``driver`` + ``driver_options`` verbatim (no mount-subclass
    branching — the driver config is entirely developer-supplied);
    the Mount supplies only ``read_only`` and the destination.
    ``driver_options`` is copied so the spec is independent of the
    frozen strategy model.

    Raises:
        SandboxConfigurationError: ``strategy`` is not ``mount``'s own
            mount strategy (a mismatch would silently build a volume
            with the wrong driver), OR ``strategy.driver`` is
            empty/blank. A blank driver is statically invalid on
            *every* Docker host (no host registers an empty-named
            driver), so it is rejected here at translation time
            rather than deferred to an opaque daemon error layers
            later. (An unregistered non-blank driver name IS host-
            dependent and correctly surfaces at the backend.)
    """
    _require_strategy_matches_mount(mount, strategy)
    if _is_blank_driver(strategy.driver):
        raise SandboxConfigurationError(
            f"build_docker_volume_mount_spec: {type(mount).__name__}"
            f".mount_strategy has an empty/blank volume driver "
            f"({strategy.driver!r}); a Docker volume driver name is "
            f"required (e.g. 'local', 'rclone')"
        )
    return DockerVolumeMountSpec(
        type="volume",
        target=_resolve_target(mount, workspace_root),
        read_only=mount.read_only,
        strategy="docker_volume",
        driver=strategy.driver,
        driver_options=dict(strategy.driver_options),
    )


def apply_mounts_to_docker(
    mounts: list[Mount],
    container_kwargs: dict[str, Any],
    *,
    workspace_root: str = "/workspace",
) -> dict[str, Any]:
    """Translate framework mounts to docker ``containers.run`` kwargs.

    Each mount is dispatched on its ``mount_strategy`` (the actual
    runtime kind) to the matching neutral-spec builder —
    ``in_container`` → ``build_in_container_mount_spec``,
    ``docker_volume`` → ``build_docker_volume_mount_spec`` — and the
    specs accumulate into ``container_kwargs["mounts"]`` as
    provider-agnostic ``InContainerMountSpec`` / ``DockerVolumeMountSpec``
    dicts. They are deliberately NOT ``docker.types.Mount`` objects:
    converting each spec to ``docker.types.Mount`` / ``DriverConfig``
    before ``containers.run`` is the consuming Docker backend
    client's responsibility (the provider-SDK boundary stays inside
    ``clients/docker/``, never in this policy layer). When any
    in-container mount is present,
    ``SYS_ADMIN`` is added ONCE to ``container_kwargs["cap_add"]``
    (the list is extended, never replaced) so the in-container FUSE
    tool can mount. Dispatching on the runtime strategy kind means
    each builder only ever receives its matching strategy — a
    wrong-kind strategy cannot reach a builder via this path.

    Returns the (possibly-mutated) kwargs dict.

    Raises:
        UnsupportedMountStrategyError: a mount carries a strategy
            kind the Docker backend cannot dispatch. The typed
            union is exhaustively handled; an out-of-union kind
            fails loud here rather than being silently skipped.
    """
    if len(mounts) == 0:
        return container_kwargs
    specs: list[InContainerMountSpec | DockerVolumeMountSpec] = list(container_kwargs.get("mounts", []))
    any_in_container = False
    for mount in mounts:
        strategy = mount.mount_strategy
        if isinstance(strategy, InContainerMountStrategy):
            specs.append(build_in_container_mount_spec(mount, strategy, workspace_root))
            any_in_container = True
        elif isinstance(strategy, DockerVolumeMountStrategy):
            specs.append(build_docker_volume_mount_spec(mount, strategy, workspace_root))
        else:
            raise UnsupportedMountStrategyError(
                mount_type=type(mount).__name__,
                strategy_type=strategy.type,
                backend="docker",
            )
    container_kwargs["mounts"] = specs
    if any_in_container:
        cap_add = list(container_kwargs.get("cap_add", []))
        if "SYS_ADMIN" not in cap_add:
            cap_add.append("SYS_ADMIN")
        container_kwargs["cap_add"] = cap_add
    return container_kwargs


def _require_in_container_strategy(mount: Mount, backend: str) -> None:
    """Fail loud if a mount's strategy cannot be honored off-Docker.

    The k8s backend (CSI driver) and the local backend (rclone /
    gcsfuse / blobfuse2 subprocess) realize a cloud mount via their
    own in-namespace mechanism and have NO Docker daemon, so a
    ``DockerVolumeMountStrategy`` (which pre-creates a Docker
    volume-driver volume) cannot be attached there. Surfacing it
    eagerly as ``UnsupportedMountStrategyError`` at translation time
    beats silently materializing an in-container-style mount that
    ignores the configured strategy. Symmetric with
    ``apply_mounts_to_docker``'s exhaustive dispatch; two call sites
    (k8s + local), isolating the typed-raise-on-wrong-strategy
    concern.
    """
    strategy = mount.mount_strategy
    if not isinstance(strategy, InContainerMountStrategy):
        raise UnsupportedMountStrategyError(
            mount_type=type(mount).__name__,
            strategy_type=strategy.type,
            backend=backend,
        )


def _k8s_volume_for_mount(mount: Mount, volume_name: str) -> dict[str, Any]:
    """Build the K8s pod ``volumes[]`` entry for one cloud mount.

    S3 / GCS / Azure map to their CSI driver; R2 / Box / S3Files fall
    back to an ``emptyDir`` a sidecar populates via rclone (documented
    limitation). Extracted from ``apply_mounts_to_k8s_pod`` so the
    per-mount CSI mapping is one named concern and the caller stays
    within the function-length bound; single conversion site,
    deliberately so.
    """
    if isinstance(mount, S3Mount):
        prefix = mount.prefix if mount.prefix is not None else ""
        return {
            "name": volume_name,
            "csi": {
                "driver": "s3.csi.aws.com",
                "readOnly": mount.read_only,
                "volumeAttributes": {
                    "bucketName": mount.bucket,
                    "mountOptions": f"allow-other,prefix={prefix}",
                },
            },
        }
    if isinstance(mount, GCSMount):
        prefix = mount.prefix if mount.prefix is not None else ""
        return {
            "name": volume_name,
            "csi": {
                "driver": "gcs.csi.ofek.dev",
                "readOnly": mount.read_only,
                "volumeAttributes": {
                    "bucketName": mount.bucket,
                    "mountOptions": f"implicit-dirs,only-dir={prefix}",
                },
            },
        }
    if isinstance(mount, AzureBlobMount):
        return {
            "name": volume_name,
            "csi": {
                "driver": "blob.csi.azure.com",
                "readOnly": mount.read_only,
                "volumeAttributes": {
                    "containerName": mount.container,
                    "mountOptions": "allow_other",
                },
            },
        }
    # R2 / Box / S3Files: emptyDir + sidecar provisioning.
    return {"name": volume_name, "emptyDir": {}}


def apply_mounts_to_k8s_pod(
    mounts: list[Mount],
    pod_spec: dict[str, Any],
    *,
    workspace_root: str = "/workspace",
) -> dict[str, Any]:
    """Translate framework mounts to K8s pod-spec volume + volumeMount entries.

    Each cloud mount becomes a CSI volume entry (driver chosen per
    backing scheme: ``s3.csi.aws.com`` for S3, ``gcs.csi.ofek.dev``
    for GCS, ``blob.csi.azure.com`` for Azure) via
    ``_k8s_volume_for_mount``; a matching volumeMount is appended to
    every container in the pod. The ``mountPath`` is resolved against
    ``workspace_root`` (the container ``workingDir``) — Kubernetes
    rejects a relative ``mountPath``, so the workspace-relative target
    is joined to an absolute destination here, mirroring the Docker
    backend's ``/workspace``-rooted targets.

    R2 / Box / S3Files fall back to emptyDir; a sidecar populates via
    rclone — documented limitation.

    Returns the (possibly-mutated) pod_spec.

    Raises:
        UnsupportedMountStrategyError: a mount carries a
            ``DockerVolumeMountStrategy`` (or any non-in-container
            strategy). K8s has no Docker daemon, so it is rejected
            eagerly rather than silently materialized as an
            in-container CSI mount that ignores the configured
            strategy.
    """
    if len(mounts) == 0:
        return pod_spec
    # All-or-nothing: validate EVERY strategy before mutating
    # pod_spec. The per-container ``volumeMounts`` write below is
    # in-place and not rolled back, so a bad strategy on a later
    # mount would otherwise leave the caller's spec with volumeMounts
    # referencing volumes that were never appended. Pre-validating
    # makes the mutation loop raise-free (mirrors
    # apply_mounts_to_docker's validate-then-commit discipline).
    for mount in mounts:
        _require_in_container_strategy(mount, "k8s")
    volumes: list[dict[str, Any]] = list(pod_spec.get("volumes", []))
    for idx, mount in enumerate(mounts):
        volume_name = f"troopai-mount-{idx}"
        volumes.append(_k8s_volume_for_mount(mount, volume_name))
        for container in pod_spec.get("containers", []):
            mount_list = list(container.get("volumeMounts", []))
            mount_list.append(
                {
                    "name": volume_name,
                    "mountPath": _resolve_target(mount, workspace_root),
                    "readOnly": mount.read_only,
                },
            )
            container["volumeMounts"] = mount_list
    pod_spec["volumes"] = volumes
    return pod_spec


def describe_mount_for_local(mount: Mount) -> dict[str, Any]:
    """Return a dict describing how a local subprocess should mount.

    Local backends don't have a kernel namespace to enforce mount
    isolation, so they shell out to ``rclone mount`` / ``gcsfuse`` /
    ``blobfuse2`` per the mount's scheme. The helper hands the caller
    the argv list ready for ``asyncio.create_subprocess_exec``.

    Raises:
        UnsupportedMountStrategyError: the mount carries a
            ``DockerVolumeMountStrategy`` (or any non-in-container
            strategy). The local backend has no Docker daemon, so it
            is rejected eagerly rather than silently mounted via
            rclone as if the strategy had been in-container.
    """
    _require_in_container_strategy(mount, "local")
    target = _mount_target(mount)
    if isinstance(mount, S3Mount):
        prefix = mount.prefix if mount.prefix is not None else ""
        return {
            "tool": "rclone",
            "argv": ["rclone", "mount", f"s3:{mount.bucket}/{prefix}", target, "--allow-other"],
        }
    if isinstance(mount, GCSMount):
        prefix = mount.prefix if mount.prefix is not None else ""
        argv: list[str] = ["gcsfuse"]
        if len(prefix) > 0:
            argv.extend(["--only-dir", prefix])
        argv.extend([mount.bucket, target])
        return {"tool": "gcsfuse", "argv": argv}
    if isinstance(mount, AzureBlobMount):
        return {
            "tool": "blobfuse2",
            "argv": ["blobfuse2", "mount", target, "--container-name", mount.container],
        }
    return {"tool": "rclone", "argv": ["rclone", "mount", _bucket_uri(mount), target]}


def apply_mounts_to_hosted_bridge(
    mounts: list[Mount],
    provider_id: str,
    request_body: dict[str, Any],
) -> dict[str, Any]:
    """Translate framework mounts to a hosted-bridge create-body field.

    Most provider APIs accept a mounts array on create. The exact
    shape varies (Modal: ``cloud_bucket_mounts``; Daytona: ``buckets``;
    Cloudflare: ``r2_bindings``; Blaxel: ``cloud_buckets``). This
    helper centralizes the per-provider naming and emits a uniform
    list of ``{type, uri, target_path, read_only}`` dicts.

    Unlike ``apply_mounts_to_k8s_pod`` / ``describe_mount_for_local``,
    this helper deliberately does NOT dispatch on or reject
    ``mount.mount_strategy``. A hosted bridge is a remote *managed*
    sandbox: the provider's infrastructure owns the actual mount
    realization (it consumes the declarative bucket spec via its own
    cloud-bucket field). The framework builds neither a CSI volume nor
    a FUSE subprocess here, so ``mount_strategy`` is advisory for
    hosted providers and is intentionally not consulted — passing a
    ``DockerVolumeMountStrategy`` is a no-op overlay, NOT a
    silently-dropped mount (the bucket is still mounted by the
    provider). The asymmetry with the k8s / local strategy guards is
    by design — there the framework constructs the mount itself, so an
    unsupportable strategy must fail loud; here it does not.
    """
    if len(mounts) == 0:
        return request_body
    items: list[dict[str, Any]] = []
    for mount in mounts:
        items.append(
            {
                "type": mount.type,
                "uri": _bucket_uri(mount),
                "target_path": _mount_target(mount),
                "read_only": mount.read_only,
            },
        )
    if provider_id == "modal":
        field_name = "cloud_bucket_mounts"
    elif provider_id == "daytona":
        field_name = "buckets"
    elif provider_id == "cloudflare":
        field_name = "r2_bindings"
    elif provider_id == "blaxel":
        field_name = "cloud_buckets"
    else:
        field_name = "mounts"
    request_body[field_name] = items
    return request_body
