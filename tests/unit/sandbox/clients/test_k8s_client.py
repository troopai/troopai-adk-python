"""Tests for K8sPodSandboxClient manifest + NetworkPolicy wiring."""

from __future__ import annotations

from troopai.adk.sandbox.clients.k8s import K8sSandboxClientOptions
from troopai.adk.sandbox.clients.k8s.k8s_client import (
    _build_network_policy_cr,
    _build_pod_manifest,
)
from troopai.adk.types.sandbox.network import NetworkPolicy


class TestPodNetworkPolicySelectorMatch:
    """The pod must carry the label the NetworkPolicy CR's podSelector matches.

    Without it the CR selects zero pods and egress isolation is silently
    not enforced — a sandbox runs with unrestricted egress while the
    framework believes the deny-default policy is in effect.
    """

    def test_pod_carries_network_policy_selector_label(self) -> None:
        options = K8sSandboxClientOptions(image="python:3.12-slim")
        pod_name = "troopai-sandbox-deadbeef0001"

        manifest = _build_pod_manifest(options, pod_name=pod_name)

        pod_labels = manifest["metadata"]["labels"]
        assert pod_labels.get("troopai.sandbox/pod") == pod_name

    def test_network_policy_selector_matches_pod_labels(self) -> None:
        options = K8sSandboxClientOptions(
            image="python:3.12-slim",
            network_policy=NetworkPolicy(deny_default=True, allow_ports=[443]),
        )
        pod_name = "troopai-sandbox-deadbeef0002"

        manifest = _build_pod_manifest(options, pod_name=pod_name)
        netpol = _build_network_policy_cr(options, pod_name=pod_name)
        assert netpol is not None

        pod_labels = manifest["metadata"]["labels"]
        selector = netpol["spec"]["podSelector"]["matchLabels"]
        # Every selector key/value must be present on the pod, else the CR
        # selects zero pods and the egress restriction never applies.
        for key, value in selector.items():
            assert pod_labels.get(key) == value
