"""Tests for policy backend-translation helpers (P36 + P37)."""

from __future__ import annotations

import pytest

from troopai.adk.exceptions.exceptions import SandboxNetworkPolicyViolation
from troopai.adk.sandbox.policy import (
    apply_network_policy_to_docker,
    apply_network_policy_to_k8s_pod,
    apply_network_policy_to_local,
    apply_resource_limits_to_docker,
    apply_resource_limits_to_k8s_pod,
)
from troopai.adk.types.sandbox.network import NetworkPolicy
from troopai.adk.types.sandbox.resource_limits import SandboxResourceLimits


class TestNetworkPolicyDocker:
    def test_none_policy_no_op(self) -> None:
        kwargs: dict = {"image": "x"}
        result = apply_network_policy_to_docker(None, kwargs)
        assert result == {"image": "x"}

    def test_deny_default_with_no_allows_sets_network_none(self) -> None:
        kwargs: dict = {"image": "x"}
        result = apply_network_policy_to_docker(NetworkPolicy(), kwargs)
        assert result["network_mode"] == "none"

    def test_allow_lists_skip_network_mode(self) -> None:
        kwargs: dict = {"image": "x"}
        policy = NetworkPolicy(allow_hosts=("api.example.com",))
        result = apply_network_policy_to_docker(policy, kwargs)
        assert "network_mode" not in result


class TestNetworkPolicyK8s:
    def test_none_returns_none(self) -> None:
        assert apply_network_policy_to_k8s_pod(None, "default", "pod") is None

    def test_emits_cr_for_deny_default(self) -> None:
        cr = apply_network_policy_to_k8s_pod(NetworkPolicy(), "default", "pod-1")
        assert cr is not None
        assert cr["kind"] == "NetworkPolicy"
        assert cr["metadata"]["namespace"] == "default"
        # DNS allowed by default.
        egress = cr["spec"]["egress"]
        assert any(53 in (p["port"] for p in r["ports"]) for r in egress)

    def test_allow_hosts_emitted_as_annotation(self) -> None:
        policy = NetworkPolicy(allow_hosts=("api.example.com", "*.foo.com"))
        cr = apply_network_policy_to_k8s_pod(policy, "default", "pod-1")
        assert cr is not None
        annotation = cr["metadata"]["annotations"]["troopai.sandbox/allow-hosts"]
        assert "api.example.com" in annotation


class TestNetworkPolicyLocal:
    def test_none_no_op(self) -> None:
        apply_network_policy_to_local(None)  # no raise

    def test_deny_default_raises(self) -> None:
        with pytest.raises(SandboxNetworkPolicyViolation):
            apply_network_policy_to_local(NetworkPolicy())

    def test_deny_default_false_no_op(self) -> None:
        apply_network_policy_to_local(NetworkPolicy(deny_default=False))  # no raise


class TestResourceLimitsDocker:
    def test_none_no_op(self) -> None:
        kwargs: dict = {"image": "x"}
        result = apply_resource_limits_to_docker(None, kwargs)
        assert result == {"image": "x"}

    def test_translates_fields(self) -> None:
        kwargs: dict = {"image": "x"}
        limits = SandboxResourceLimits(
            cpu_cores=1.5,
            memory_mb=512,
            max_processes=64,
        )
        result = apply_resource_limits_to_docker(limits, kwargs)
        assert result["cpu_period"] == 100_000
        assert result["cpu_quota"] == 150_000
        assert result["mem_limit"] == "512m"
        assert result["pids_limit"] == 64


class TestResourceLimitsK8s:
    def test_none_no_op(self) -> None:
        spec: dict = {"name": "x"}
        result = apply_resource_limits_to_k8s_pod(None, spec)
        assert result == {"name": "x"}

    def test_translates_fields(self) -> None:
        spec: dict = {"name": "x"}
        limits = SandboxResourceLimits(
            cpu_cores=0.5,
            memory_mb=256,
            disk_mb=1024,
        )
        result = apply_resource_limits_to_k8s_pod(limits, spec)
        r = result["resources"]
        assert r["limits"]["cpu"] == "500m"
        assert r["limits"]["memory"] == "256Mi"
        assert r["limits"]["ephemeral-storage"] == "1024Mi"
        # Requests filled from limits when not pre-set.
        assert r["requests"]["cpu"] == "500m"
