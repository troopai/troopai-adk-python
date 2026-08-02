"""Sandbox policy translation helpers.

``NetworkPolicy``, ``SandboxResourceLimits`` and ``Mount`` strategies
are backend-agnostic declarations; these helpers translate them to
each backend's wire format. The ``apply_*`` helpers are pure
functions backends call at session-create time to enrich their
per-backend options; the ``build_*`` helpers produce the
provider-agnostic per-mount neutral specs the Docker applier
accumulates.
"""

from __future__ import annotations

from troopai.adk.sandbox.policy.mounts import (
    apply_mounts_to_docker,
    apply_mounts_to_hosted_bridge,
    apply_mounts_to_k8s_pod,
    build_docker_volume_mount_spec,
    build_in_container_mount_spec,
    describe_mount_for_local,
)
from troopai.adk.sandbox.policy.network_policy import (
    apply_network_policy_to_docker,
    apply_network_policy_to_k8s_pod,
    apply_network_policy_to_local,
)
from troopai.adk.sandbox.policy.resource_limits import (
    apply_resource_limits_to_docker,
    apply_resource_limits_to_k8s_pod,
)

__all__ = [
    "apply_mounts_to_docker",
    "apply_mounts_to_hosted_bridge",
    "apply_mounts_to_k8s_pod",
    "apply_network_policy_to_docker",
    "apply_network_policy_to_k8s_pod",
    "apply_network_policy_to_local",
    "apply_resource_limits_to_docker",
    "apply_resource_limits_to_k8s_pod",
    "build_docker_volume_mount_spec",
    "build_in_container_mount_spec",
    "describe_mount_for_local",
]
