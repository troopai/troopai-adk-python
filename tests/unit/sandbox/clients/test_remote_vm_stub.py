"""Tests for RemoteVMSandboxClient base (P21)."""

from __future__ import annotations

from troopai.adk.sandbox.clients.hosted.remote_vm import RemoteVMSandboxClientOptions


class TestRemoteVMOptions:
    def test_defaults(self) -> None:
        opts = RemoteVMSandboxClientOptions()
        assert opts.base_url is None
        # Cost-conservative default: no retries unless the developer opts in
        # (each retry is a real billable provider call).
        assert opts.max_retries == 0
        assert opts.request_timeout == 30.0
