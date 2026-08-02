"""K8sPodSandboxClient — production Kubernetes pod backend.

Spawns an ephemeral Pod per session. NetworkPolicy + ResourceQuota +
PodSecurity admission labels flow from framework types via the
``apply_*_to_k8s_pod`` helpers in ``troopai.adk.sandbox.policy``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Literal, override

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
from troopai.adk.sandbox.clients.k8s.k8s_session import K8sPodSandboxSession
from troopai.adk.sandbox.clients.session import BaseSandboxSession
from troopai.adk.types.sandbox.cost import SandboxBackendCapabilities, SandboxCostDescriptor
from troopai.adk.types.sandbox.network import NetworkPolicy
from troopai.adk.types.sandbox.resource_limits import SandboxResourceLimits
from troopai.adk.types.sandbox.session_state import SandboxSessionState

if TYPE_CHECKING:
    from troopai.adk.types.sandbox.manifest import Manifest
    from troopai.adk.types.sandbox.snapshot import SnapshotSpec

logger = logging.getLogger(__name__)

__all__ = ["K8sPodSandboxClient", "K8sSandboxClientOptions"]


def _require_k8s() -> None:
    try:
        # kubernetes lives in the [sandbox-k8s] extra; not in core deps.
        # Bare import is intentional — we only need the ImportError signal.
        import kubernetes  # noqa: F401  # pyright: ignore[reportMissingImports, reportUnusedImport]
    except ImportError as exc:
        raise UnsupportedSandboxClientError(
            "K8sPodSandboxClient requires the optional 'kubernetes' package: pip install 'troopai-adk-python[sandbox-k8s]'",
        ) from exc


class K8sSandboxClientOptions(BaseSandboxClientOptions):
    """Options for the Kubernetes Pod backend.

    Attributes:
        image: Container image for the pod's sandbox container (e.g.
            ``python:3.12-slim``). Required; no default.
        namespace: Kubernetes namespace the pod and NetworkPolicy CR are
            created in.
        service_account: Pod-level ``serviceAccountName``. ``None`` uses
            the namespace default.
        pod_security_standard: PodSecurity admission profile.
            ``"restricted"`` enforces non-root + drop-all capabilities.
        container_name: Name of the sandbox container within the pod;
            used as the target for ``exec`` calls.
        working_directory: Container ``workingDir``; created by the session
            on first start.
        environment: Environment variables injected into the sandbox
            container.
        network_policy: Framework ``NetworkPolicy`` translated to a
            Kubernetes ``NetworkPolicy`` CR on session create.
        resource_limits: Framework resource limits translated to the
            container's ``resources.limits`` (and ``resources.requests``).
        node_selector: ``spec.nodeSelector`` key-value pairs to pin the pod
            to specific nodes. ``None`` means no selector.
        tolerations: ``spec.tolerations`` entries for tainted nodes.
            ``None`` means no tolerations.
        labels: Extra labels merged onto the pod's metadata. Framework
            labels (``app.kubernetes.io/managed-by``, etc.) are always
            present regardless.
    """

    image: str
    """Container image (e.g. ``python:3.12-slim``) for the pod's sandbox container."""

    namespace: str = "default"
    """Kubernetes namespace the pod and NetworkPolicy CR are created in."""

    service_account: str | None = None
    """Pod-level ``serviceAccountName``. None uses the namespace default."""

    pod_security_standard: Literal["baseline", "restricted"] = "restricted"
    """PodSecurity admission profile. ``restricted`` enforces non-root + drop-all caps."""

    container_name: str = "sandbox"
    """Name of the sandbox container within the pod; used for ``exec`` targeting."""

    working_directory: str = "/workspace"
    """Container ``workingDir``; created by the session on first start."""

    environment: dict[str, str] = Field(default_factory=dict)
    """Environment variables injected into the sandbox container."""

    network_policy: NetworkPolicy | None = None
    """Framework NetworkPolicy translated to a Kubernetes NetworkPolicy CR on create."""

    resource_limits: SandboxResourceLimits | None = None
    """Framework resource limits translated to container ``resources.limits``."""

    node_selector: dict[str, str] | None = None
    """``spec.nodeSelector`` to pin the pod to specific nodes."""

    tolerations: list[dict[str, Any]] | None = None
    """``spec.tolerations`` for tainted nodes."""

    labels: dict[str, str] | None = None
    """Extra labels merged onto the pod's metadata. Framework labels always present."""


def _build_pod_manifest(options: K8sSandboxClientOptions, pod_name: str) -> dict[str, Any]:
    """Render the V1Pod object as a plain dict, ready for create_namespaced_pod."""
    labels: dict[str, str] = {
        "app.kubernetes.io/managed-by": "troopai-adk-python",
        "troopai.adk.io/role": "sandbox",
    }
    if options.labels is not None:
        labels.update(options.labels)
    # PodSecurity admission labels go on the *namespace*, not the pod. The
    # client validates the namespace's profile externally; here we tag the
    # pod for observability.
    labels["troopai.adk.io/pss"] = options.pod_security_standard
    # The NetworkPolicy CR's podSelector matches this label; without it the
    # CR selects zero pods and egress isolation is silently not enforced.
    labels["troopai.sandbox/pod"] = pod_name
    annotations: dict[str, str] = {}
    env_vars = [{"name": k, "value": v} for k, v in options.environment.items()]
    container: dict[str, Any] = {
        "name": options.container_name,
        "image": options.image,
        "command": ["sleep", "infinity"],
        "workingDir": options.working_directory,
        "env": env_vars,
        "imagePullPolicy": "IfNotPresent",
    }
    # restricted PSS minimum: non-root, no privilege escalation, seccomp default.
    if options.pod_security_standard == "restricted":
        container["securityContext"] = {
            "allowPrivilegeEscalation": False,
            "runAsNonRoot": True,
            "runAsUser": 1000,
            "capabilities": {"drop": ["ALL"]},
            "seccompProfile": {"type": "RuntimeDefault"},
        }
    if options.resource_limits is not None:
        from troopai.adk.sandbox.policy import (
            apply_resource_limits_to_k8s_pod,
        )

        container = apply_resource_limits_to_k8s_pod(
            options.resource_limits,
            container,
        )
    pod_spec: dict[str, Any] = {
        "containers": [container],
        "restartPolicy": "Never",
        "terminationGracePeriodSeconds": 10,
    }
    if options.service_account is not None:
        pod_spec["serviceAccountName"] = options.service_account
    if options.node_selector is not None:
        pod_spec["nodeSelector"] = options.node_selector
    if options.tolerations is not None:
        pod_spec["tolerations"] = options.tolerations
    if options.pod_security_standard == "restricted":
        pod_spec["securityContext"] = {
            "runAsNonRoot": True,
            "runAsUser": 1000,
            "seccompProfile": {"type": "RuntimeDefault"},
        }
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": pod_name,
            "namespace": options.namespace,
            "labels": labels,
            "annotations": annotations,
        },
        "spec": pod_spec,
    }


def _build_network_policy_cr(
    options: K8sSandboxClientOptions,
    pod_name: str,
) -> dict[str, Any] | None:
    """Translate the framework NetworkPolicy to a Kubernetes NetworkPolicy CR."""
    if options.network_policy is None:
        return None
    from troopai.adk.sandbox.policy import (
        apply_network_policy_to_k8s_pod,
    )

    return apply_network_policy_to_k8s_pod(
        options.network_policy,
        namespace=options.namespace,
        pod_name=pod_name,
    )


class K8sPodSandboxClient(BaseSandboxClient[K8sSandboxClientOptions]):
    """Production K8s-backed sandbox client."""

    backend_id = "k8s_pod"
    # Self-hosted: no per-minute compute charge.
    cost = SandboxCostDescriptor(free=True)
    capabilities = SandboxBackendCapabilities(network=True, persistent=True)

    def __init__(
        self,
        *,
        core_v1: Any = None,
        networking_v1: Any = None,
    ) -> None:
        """Construct the client.

        ``core_v1`` and ``networking_v1`` accept pre-built kubernetes
        clients (primary use case: tests). When both are ``None`` the
        client lazily loads in-cluster config or kube-config.
        """
        if core_v1 is not None:
            self._core_v1 = core_v1
            self._networking_v1 = networking_v1
            return
        _require_k8s()
        # Lazy-imported because kubernetes is an optional dependency
        # (the [sandbox-k8s] extra); _require_k8s above raised if absent.
        from kubernetes import (  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]
            client as k8s_client,
            config as k8s_config,
        )

        try:
            k8s_config.load_incluster_config()
        except Exception:
            k8s_config.load_kube_config()
        self._core_v1 = k8s_client.CoreV1Api()
        self._networking_v1 = networking_v1 if networking_v1 is not None else k8s_client.NetworkingV1Api()

    async def _delete_pod_best_effort(self, pod_name: str, namespace: str) -> None:
        """Delete a pod, swallowing errors — teardown on a failed create.

        Used to undo a pod whose NetworkPolicy could not be applied, so the
        caller never receives a session that violates the isolation contract.
        """
        try:
            await asyncio.to_thread(
                self._core_v1.delete_namespaced_pod,
                name=pod_name,
                namespace=namespace,
                grace_period_seconds=0,
            )
        except Exception:
            logger.warning(
                "K8sPodSandboxClient: best-effort pod cleanup failed",
                exc_info=True,
            )

    @override
    async def create(
        self,
        *,
        snapshot: SnapshotSpec | None = None,
        snapshot_store: Any | None = None,
        manifest: Manifest | None = None,
        options: K8sSandboxClientOptions,
    ) -> BaseSandboxSession:
        reject_unsupported_snapshot_store(snapshot_store, self.backend_id)
        warn_discarded_snapshot(snapshot, self.backend_id, logger)
        del snapshot
        import uuid

        pod_name = f"troopai-sandbox-{uuid.uuid4().hex[:12]}"
        pod_body = _build_pod_manifest(options, pod_name=pod_name)
        if manifest is not None:
            from troopai.adk.sandbox.policy import apply_mounts_to_k8s_pod
            from troopai.adk.types.sandbox.mounts import Mount

            mounts = [m for m in manifest.entries.values() if isinstance(m, Mount)]
            pod_body["spec"] = apply_mounts_to_k8s_pod(
                mounts, pod_body["spec"], workspace_root=options.working_directory
            )
        try:
            await asyncio.to_thread(
                self._core_v1.create_namespaced_pod,
                namespace=options.namespace,
                body=pod_body,
            )
        except asyncio.CancelledError:
            # The asyncio task was cancelled while create_namespaced_pod was
            # executing in a thread. The thread cannot be interrupted; the pod
            # may have been created by the time CancelledError is raised here.
            # Best-effort cleanup before re-raising to avoid an orphaned pod.
            await self._delete_pod_best_effort(pod_name, options.namespace)
            raise
        except Exception as exc:
            raise SandboxStartFailed(
                backend_id="k8s_pod",
                reason=f"create_namespaced_pod failed: {exc}",
                details={"image": options.image, "namespace": options.namespace},
            ) from exc
        netpol = _build_network_policy_cr(options, pod_name=pod_name)
        if netpol is not None:
            if self._networking_v1 is None:
                # A NetworkPolicy was configured but this client has no
                # NetworkingV1Api (core_v1 was injected without networking_v1).
                # Silently skipping it would hand back a session running with
                # UNRESTRICTED network despite the requested isolation. Tear
                # the pod down + raise instead.
                await self._delete_pod_best_effort(pod_name, options.namespace)
                raise SandboxStartFailed(
                    backend_id="k8s_pod",
                    reason="NetworkPolicy configured but no NetworkingV1Api available (pod torn down)",
                    details={"namespace": options.namespace, "pod_name": pod_name},
                )
            try:
                await asyncio.to_thread(
                    self._networking_v1.create_namespaced_network_policy,
                    namespace=options.namespace,
                    body=netpol,
                )
            except asyncio.CancelledError:
                # Task cancelled during NetworkPolicy creation. The pod is
                # already running without its isolation policy, which violates
                # the declared isolation contract. Tear the pod down before
                # re-raising so the caller never inherits a policy-violating
                # session.
                await self._delete_pod_best_effort(pod_name, options.namespace)
                raise
            except Exception as exc:
                # Restrictive network policy could not be applied; the
                # pod would run un-restricted. Tear it down + raise so
                # the caller never gets a session that violates the
                # policy contract.
                await self._delete_pod_best_effort(pod_name, options.namespace)
                raise SandboxStartFailed(
                    backend_id="k8s_pod",
                    reason=f"NetworkPolicy CR create failed (pod torn down): {exc}",
                    details={"namespace": options.namespace, "pod_name": pod_name},
                ) from exc
            else:
                logger.debug(
                    "K8sPodSandboxClient.create: NetworkPolicy CR %s applied",
                    netpol.get("metadata", {}).get("name", "<unknown>"),
                )
        return K8sPodSandboxSession(
            core_v1=self._core_v1,
            pod_name=pod_name,
            namespace=options.namespace,
            container_name=options.container_name,
            working_directory=options.working_directory,
            network_policy=options.network_policy,
            # Hand the session the networking client so it can delete the
            # per-pod NetworkPolicy CR on teardown (the CR has no
            # ownerReference, so K8s GC will not reclaim it otherwise).
            networking_v1=self._networking_v1,
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
        """Reconnect to a previously-saved pod.

        ``provider_payload`` keys: ``pod_name``, ``namespace``,
        ``container_name`` (optional), ``working_directory`` (optional).
        """
        payload = state.provider_payload
        pod_name = payload.get("pod_name")
        namespace = payload.get("namespace", "default")
        if not isinstance(pod_name, str) or len(pod_name) == 0:
            raise SandboxStartFailed(
                backend_id="k8s_pod",
                reason="resume requires provider_payload['pod_name']",
            )
        try:
            await asyncio.to_thread(
                self._core_v1.read_namespaced_pod_status,
                name=pod_name,
                namespace=namespace,
            )
        except Exception as exc:
            raise SandboxStartFailed(
                backend_id="k8s_pod",
                reason=f"pod {pod_name!r} not found in namespace {namespace!r}: {exc}",
            ) from exc
        container_name = payload.get("container_name")
        working_directory = payload.get("working_directory", "/workspace")
        return K8sPodSandboxSession(
            core_v1=self._core_v1,
            pod_name=pod_name,
            namespace=namespace if isinstance(namespace, str) else "default",
            container_name=container_name if isinstance(container_name, str) else None,
            working_directory=working_directory if isinstance(working_directory, str) else "/workspace",
        )

    @override
    def deserialize_session_state(
        self,
        payload: dict[str, Any],
    ) -> SandboxSessionState:
        return SandboxSessionState.model_validate(payload)
