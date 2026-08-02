"""Tests for hosted-bridge stubs (P22-P26)."""

from __future__ import annotations

import pytest

HOSTED_PROVIDERS = [
    ("e2b", "E2bSandboxClient", "E2bSandboxClientOptions"),
    ("vercel", "VercelSandboxClient", "VercelSandboxClientOptions"),
    ("modal", "ModalSandboxClient", "ModalSandboxClientOptions"),
    ("daytona", "DaytonaSandboxClient", "DaytonaSandboxClientOptions"),
    ("cloudflare", "CloudflareSandboxClient", "CloudflareSandboxClientOptions"),
    ("blaxel", "BlaxelSandboxClient", "BlaxelSandboxClientOptions"),
    ("runloop", "RunloopSandboxClient", "RunloopSandboxClientOptions"),
]


@pytest.mark.parametrize("provider,client_name,options_name", HOSTED_PROVIDERS)
class TestHostedStub:
    def test_options_import(self, provider: str, client_name: str, options_name: str) -> None:
        import importlib

        module = importlib.import_module(
            f"troopai.adk.sandbox.clients.hosted.{provider}",
        )
        opts_cls = getattr(module, options_name)
        opts = opts_cls()
        assert opts.api_key is None

    def test_client_class_attr(self, provider: str, client_name: str, options_name: str) -> None:
        import importlib

        module = importlib.import_module(
            f"troopai.adk.sandbox.clients.hosted.{provider}",
        )
        client_cls = getattr(module, client_name)
        assert client_cls.backend_id == provider
