"""Tests for NetworkPolicy → backend wire-format translation helpers."""

from __future__ import annotations

import pytest

from troopai.adk.exceptions.exceptions import SandboxNetworkPolicyViolation
from troopai.adk.sandbox.policy import (
    apply_network_policy_to_docker,
    apply_network_policy_to_k8s_pod,
    apply_network_policy_to_local,
)
from troopai.adk.types.sandbox.network import NetworkPolicy


class TestK8sDenyDefaultGuard:
    """``deny_default=False`` must NEVER emit a restrictive K8s CR.

    In Kubernetes, a NetworkPolicy that lists ``Egress`` in
    ``policyTypes`` with a non-empty ``egress`` list whitelists only the
    listed rules and drops all other egress. So emitting such a CR for an
    open policy (``deny_default=False``) would silently invert the
    declared intent into a deny-by-default lockdown.
    """

    def test_open_policy_with_allow_ports_emits_no_cr(self) -> None:
        # Regression: deny_default=False with a non-empty allow list used to
        # fall through and build a restrictive Egress CR.
        policy = NetworkPolicy(deny_default=False, allow_ports=(8080,))
        assert apply_network_policy_to_k8s_pod(policy, "default", "pod-1") is None

    def test_open_policy_with_allow_hosts_emits_no_cr(self) -> None:
        policy = NetworkPolicy(deny_default=False, allow_hosts=("api.example.com",))
        assert apply_network_policy_to_k8s_pod(policy, "default", "pod-1") is None

    def test_open_policy_factory_emits_no_cr(self) -> None:
        cr = apply_network_policy_to_k8s_pod(NetworkPolicy.open(), "default", "pod-1")
        assert cr is None

    def test_open_policy_with_both_allows_emits_no_cr(self) -> None:
        policy = NetworkPolicy(
            deny_default=False,
            allow_hosts=("api.example.com",),
            allow_ports=(8080, 443),
        )
        assert apply_network_policy_to_k8s_pod(policy, "default", "pod-1") is None

    def test_none_policy_returns_none(self) -> None:
        assert apply_network_policy_to_k8s_pod(None, "default", "pod") is None


class TestK8sDenyDefaultEmitsCr:
    """``deny_default=True`` still emits the restrictive CR (unchanged)."""

    def test_default_policy_emits_egress_cr(self) -> None:
        cr = apply_network_policy_to_k8s_pod(NetworkPolicy(), "default", "pod-1")
        assert cr is not None
        assert cr["kind"] == "NetworkPolicy"
        assert cr["spec"]["policyTypes"] == ["Egress"]
        # DNS allowed by default.
        egress = cr["spec"]["egress"]
        assert any(53 in (p["port"] for p in r["ports"]) for r in egress)

    def test_deny_default_with_allow_ports_emits_port_rule(self) -> None:
        policy = NetworkPolicy(deny_default=True, allow_ports=(8080,))
        cr = apply_network_policy_to_k8s_pod(policy, "default", "pod-1")
        assert cr is not None
        egress = cr["spec"]["egress"]
        assert any(8080 in (p["port"] for p in r["ports"]) for r in egress)

    def test_deny_default_with_allow_hosts_emits_annotation(self) -> None:
        policy = NetworkPolicy(deny_default=True, allow_hosts=("api.example.com",))
        cr = apply_network_policy_to_k8s_pod(policy, "default", "pod-1")
        assert cr is not None
        annotation = cr["metadata"]["annotations"]["troopai.sandbox/allow-hosts"]
        assert "api.example.com" in annotation


class TestDockerOpenPolicyNoLockdown:
    """The Docker helper must not lock down an open policy either."""

    def test_open_policy_with_allows_leaves_network_mode_unset(self) -> None:
        policy = NetworkPolicy(deny_default=False, allow_ports=(8080,))
        result = apply_network_policy_to_docker(policy, {"image": "x"})
        assert "network_mode" not in result

    def test_deny_default_no_allows_sets_network_none(self) -> None:
        result = apply_network_policy_to_docker(NetworkPolicy(), {"image": "x"})
        assert result["network_mode"] == "none"


class TestLocalOpenPolicyNoRaise:
    def test_open_policy_no_raise(self) -> None:
        apply_network_policy_to_local(NetworkPolicy(deny_default=False))

    def test_deny_default_raises(self) -> None:
        with pytest.raises(SandboxNetworkPolicyViolation):
            apply_network_policy_to_local(NetworkPolicy())
