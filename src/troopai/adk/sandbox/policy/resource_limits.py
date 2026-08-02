"""SandboxResourceLimits → backend wire format translation helpers."""

from __future__ import annotations

from typing import Any

from troopai.adk.types.sandbox.resource_limits import SandboxResourceLimits

__all__ = [
    "apply_resource_limits_to_docker",
    "apply_resource_limits_to_k8s_pod",
]


def apply_resource_limits_to_docker(
    limits: SandboxResourceLimits | None,
    container_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Enrich Docker ``containers.run`` kwargs with resource limits.

    Translates the agnostic fields to Docker SDK options:
    - cpu_cores → cpu_period + cpu_quota (or cpus= when float).
    - memory_mb → mem_limit (string form: "{n}m").
    - max_processes → pids_limit.
    - disk_mb → not directly supported by Docker; emit via tmpfs
      configuration the caller may add.

    Returns the (possibly-mutated) kwargs dict.
    """
    if limits is None:
        return container_kwargs
    if limits.cpu_cores is not None:
        # Docker rejects NanoCPUs alongside CpuQuota ("Conflicting options:
        # Nano CPUs and CPU Quota cannot both be set"). cpu_cores expresses the
        # cap as period/quota, so drop any nano_cpus a direct cpu_count set.
        container_kwargs.pop("nano_cpus", None)
        container_kwargs["cpu_period"] = 100_000
        container_kwargs["cpu_quota"] = int(limits.cpu_cores * 100_000)
    if limits.memory_mb is not None:
        container_kwargs["mem_limit"] = f"{limits.memory_mb}m"
    if limits.max_processes is not None:
        container_kwargs["pids_limit"] = limits.max_processes
    return container_kwargs


def apply_resource_limits_to_k8s_pod(
    limits: SandboxResourceLimits | None,
    container_spec: dict[str, Any],
) -> dict[str, Any]:
    """Enrich a K8s container spec with resource limits.

    Sets ``resources.limits`` (and ``resources.requests`` to the same
    values when set) on the container dict. Returns the spec.
    """
    if limits is None:
        return container_spec
    resources = container_spec.setdefault("resources", {})
    limits_dict = resources.setdefault("limits", {})
    requests_dict = resources.setdefault("requests", {})
    if limits.cpu_cores is not None:
        cpu_str = f"{int(limits.cpu_cores * 1000)}m"
        limits_dict["cpu"] = cpu_str
        requests_dict.setdefault("cpu", cpu_str)
    if limits.memory_mb is not None:
        mem_str = f"{limits.memory_mb}Mi"
        limits_dict["memory"] = mem_str
        requests_dict.setdefault("memory", mem_str)
    if limits.disk_mb is not None:
        limits_dict["ephemeral-storage"] = f"{limits.disk_mb}Mi"
    return container_spec
