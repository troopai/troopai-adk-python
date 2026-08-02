"""NetworkPolicy → backend wire format translation helpers."""

from __future__ import annotations

from typing import Any

from troopai.adk.exceptions.exceptions import SandboxNetworkPolicyViolation
from troopai.adk.types.sandbox.network import NetworkPolicy

__all__ = [
    "apply_network_policy_to_docker",
    "apply_network_policy_to_k8s_pod",
    "apply_network_policy_to_local",
]


def apply_network_policy_to_docker(
    policy: NetworkPolicy | None,
    container_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Enrich Docker ``containers.run`` kwargs with the NetworkPolicy.

    - ``deny_default=True`` with empty ``allow_hosts`` AND empty
      ``allow_ports`` ⇒ ``network_mode="none"``.
    - Otherwise the backend defers to Docker's default networking and
      a sidecar iptables / firewall step is the caller's responsibility
      (full backend-level translation lives in DockerSandboxClient).

    Returns the (possibly-mutated) kwargs dict.
    """
    if policy is None:
        return container_kwargs
    if policy.deny_default and len(policy.allow_hosts) == 0 and len(policy.allow_ports) == 0:
        container_kwargs["network_mode"] = "none"
    return container_kwargs


def apply_network_policy_to_k8s_pod(
    policy: NetworkPolicy | None,
    namespace: str,
    pod_name: str,
) -> dict[str, Any] | None:
    """Translate NetworkPolicy to a Kubernetes ``NetworkPolicy`` CR.

    Returns the CR body (dict) or ``None`` when the policy is None or
    does not restrict anything. K8sPodSandboxClient calls this
    + applies the CR via NetworkingV1Api.

    A restrictive CR is emitted ONLY when ``deny_default=True``. With
    ``deny_default=False`` the developer asked for unrestricted egress;
    emitting a non-empty K8s ``egress`` list would silently invert that
    into a deny-by-default whitelist, so ``None`` is returned instead.
    """
    if policy is None or not policy.deny_default:
        return None
    egress_rules: list[dict[str, Any]] = []
    if policy.allow_dns:
        egress_rules.append(
            {
                "ports": [
                    {"port": 53, "protocol": "UDP"},
                    {"port": 53, "protocol": "TCP"},
                ],
            },
        )
    if len(policy.allow_hosts) > 0:
        # Hosts translate to CIDR / DNS-name rules. K8s NetworkPolicy
        # cannot match by hostname directly — applications typically
        # rely on a service-mesh sidecar for that. We emit the host
        # list as an annotation so a downstream operator can read it.
        pass
    if len(policy.allow_ports) > 0:
        egress_rules.append(
            {
                "ports": [{"port": p, "protocol": "TCP"} for p in policy.allow_ports],
            },
        )
    return {
        "apiVersion": "networking.k8s.io/v1",
        "kind": "NetworkPolicy",
        "metadata": {
            "name": f"{pod_name}-egress-policy",
            "namespace": namespace,
            "annotations": {
                "troopai.sandbox/allow-hosts": ",".join(policy.allow_hosts),
            },
        },
        "spec": {
            "podSelector": {"matchLabels": {"troopai.sandbox/pod": pod_name}},
            "policyTypes": ["Egress"],
            "egress": egress_rules,
        },
    }


def apply_network_policy_to_local(policy: NetworkPolicy | None) -> None:
    """Refuse to enforce ``deny_default=True`` for the local backend.

    Raises ``SandboxNetworkPolicyViolation`` because subprocesses of
    the host process cannot enforce filesystem-level network policy.
    Returns silently when ``policy`` is None or ``deny_default=False``.
    """
    if policy is None or not policy.deny_default:
        return
    raise SandboxNetworkPolicyViolation(
        "LocalSubprocessSandboxClient cannot enforce a deny-default "
        "NetworkPolicy inside a subprocess of the host process; use "
        "DockerSandboxClient or K8sPodSandboxClient for enforced policy."
    )
