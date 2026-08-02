"""Tests for K8sPodSandboxClient stub (P20)."""

from __future__ import annotations

from troopai.adk.sandbox.clients.k8s import K8sSandboxClientOptions


class TestK8sClientImports:
    def test_options_class_imports(self) -> None:
        opts = K8sSandboxClientOptions(image="python:3.12-slim")
        assert opts.image == "python:3.12-slim"
        assert opts.namespace == "default"
        assert opts.pod_security_standard == "restricted"
