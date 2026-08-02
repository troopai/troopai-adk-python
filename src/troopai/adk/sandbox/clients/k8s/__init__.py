"""K8sPodSandboxClient — production Kubernetes pod backend.

Requires the ``[sandbox-k8s]`` extra. Spawns an ephemeral Pod
per session with NetworkPolicy + ResourceQuota + restricted
PodSecurity profile.
"""

from __future__ import annotations

from troopai.adk.sandbox.clients.k8s.k8s_client import (
    K8sPodSandboxClient,
    K8sSandboxClientOptions,
)
from troopai.adk.sandbox.clients.k8s.k8s_session import K8sPodSandboxSession

__all__ = [
    "K8sPodSandboxClient",
    "K8sPodSandboxSession",
    "K8sSandboxClientOptions",
]
