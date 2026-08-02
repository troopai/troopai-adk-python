"""Regression tests for DockerSandboxClient run-kwargs assembly.

Covers ``_build_run_kwargs`` — the framework-options → docker
``containers.run`` kwargs translation. These exercise the resource-limit
overlay contract without an actual Docker daemon (no docker SDK import is
reached: ``_build_run_kwargs`` only calls the provider-agnostic policy
helpers, never ``_materialize_docker_mounts``).
"""

from __future__ import annotations

from troopai.adk.sandbox.clients.docker.docker_client import (
    DockerSandboxClientOptions,
    _build_run_kwargs,
)
from troopai.adk.types.sandbox.resource_limits import SandboxResourceLimits


class TestCpuKwargMutualExclusion:
    """Docker rejects NanoCPUs alongside CpuQuota.

    Setting both ``cpu_count`` (→ nano_cpus) and ``resource_limits.cpu_cores``
    (→ cpu_period/cpu_quota) must NOT emit both keys, or ``containers.run``
    fails with "Conflicting options: Nano CPUs and CPU Quota cannot both be
    set" and create() re-wraps it as SandboxStartFailed.
    """

    def test_cpu_cores_drops_nano_cpus(self) -> None:
        opts = DockerSandboxClientOptions(
            image="python:3.12-slim",
            cpu_count=2.0,
            resource_limits=SandboxResourceLimits(cpu_cores=1.0),
        )
        kwargs = _build_run_kwargs(opts, None)
        # resource_limits.cpu_cores overlays the direct cpu_count: only the
        # period/quota form survives, never both.
        assert "nano_cpus" not in kwargs
        assert kwargs["cpu_period"] == 100_000
        assert kwargs["cpu_quota"] == 100_000

    def test_cpu_count_alone_keeps_nano_cpus(self) -> None:
        opts = DockerSandboxClientOptions(image="python:3.12-slim", cpu_count=2.0)
        kwargs = _build_run_kwargs(opts, None)
        # No resource_limits.cpu_cores: the direct cpu_count form stands and
        # no conflicting quota is emitted.
        assert kwargs["nano_cpus"] == 2_000_000_000
        assert "cpu_quota" not in kwargs


class TestResourceLimitsOverlayPrecedence:
    """resource_limits overlays the direct memory / pid kwargs (it wins)."""

    def test_resource_limits_memory_overlays_direct(self) -> None:
        opts = DockerSandboxClientOptions(
            image="python:3.12-slim",
            memory_mb=512,
            resource_limits=SandboxResourceLimits(memory_mb=256),
        )
        kwargs = _build_run_kwargs(opts, None)
        assert kwargs["mem_limit"] == "256m"

    def test_resource_limits_pids_overlays_direct(self) -> None:
        opts = DockerSandboxClientOptions(
            image="python:3.12-slim",
            pid_limit=128,
            resource_limits=SandboxResourceLimits(max_processes=64),
        )
        kwargs = _build_run_kwargs(opts, None)
        assert kwargs["pids_limit"] == 64

    def test_direct_kwargs_stand_without_resource_limits(self) -> None:
        opts = DockerSandboxClientOptions(
            image="python:3.12-slim",
            memory_mb=512,
            pid_limit=128,
        )
        kwargs = _build_run_kwargs(opts, None)
        assert kwargs["mem_limit"] == "512m"
        assert kwargs["pids_limit"] == 128
