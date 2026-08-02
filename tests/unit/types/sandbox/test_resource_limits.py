"""Tests for ``troopai.adk.types.sandbox.resource_limits``."""

from __future__ import annotations

import pytest

from troopai.adk.types.sandbox.resource_limits import SandboxResourceLimits


class TestSandboxResourceLimitsDefaults:
    def test_all_none_is_unbounded(self) -> None:
        limits = SandboxResourceLimits()
        assert limits.is_unbounded() is True
        assert limits.cpu_cores is None
        assert limits.memory_mb is None

    def test_one_set_is_not_unbounded(self) -> None:
        limits = SandboxResourceLimits(memory_mb=512)
        assert limits.is_unbounded() is False


class TestSandboxResourceLimitsValidation:
    def test_negative_cpu_rejected(self) -> None:
        with pytest.raises(ValueError, match="cpu_cores"):
            SandboxResourceLimits(cpu_cores=-1.0)

    def test_zero_memory_rejected(self) -> None:
        with pytest.raises(ValueError, match="memory_mb"):
            SandboxResourceLimits(memory_mb=0)

    def test_negative_exec_timeout_rejected(self) -> None:
        with pytest.raises(ValueError, match="exec_timeout"):
            SandboxResourceLimits(exec_timeout=-0.5)

    def test_fractional_cpu_allowed(self) -> None:
        limits = SandboxResourceLimits(cpu_cores=0.5)
        assert limits.cpu_cores == 0.5

    def test_full_config(self) -> None:
        limits = SandboxResourceLimits(
            cpu_cores=2.0,
            memory_mb=1024,
            disk_mb=2048,
            exec_timeout=30.0,
            session_timeout=600.0,
            max_processes=128,
            max_egress_bytes=10 * 1024 * 1024,
        )
        assert limits.is_unbounded() is False
        assert limits.cpu_cores == 2.0
        assert limits.memory_mb == 1024
