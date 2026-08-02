"""Tests for DockerSandboxClient stub (P19)."""

from __future__ import annotations


class TestDockerClientImports:
    def test_options_class_imports(self) -> None:
        from troopai.adk.sandbox.clients.docker import DockerSandboxClientOptions

        opts = DockerSandboxClientOptions(image="python:3.12-slim")
        assert opts.image == "python:3.12-slim"

    def test_client_class_imports(self) -> None:
        from troopai.adk.sandbox.clients.docker import DockerSandboxClient

        # Just verify the class exists; instantiation requires
        # the optional ``docker`` extra.
        assert DockerSandboxClient.backend_id == "docker"
