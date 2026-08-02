"""E2bSandboxClient — hosted bridge.

Requires the [sandbox-e2b] extra. Extends
RemoteVMSandboxClient for shared HTTP / retry / port-forward
machinery; overrides only provider-specific create / resume / auth.
"""

from __future__ import annotations

from troopai.adk.sandbox.clients.hosted.e2b.e2b_client import (
    E2bSandboxClient,
    E2bSandboxClientOptions,
)

__all__ = ["E2bSandboxClient", "E2bSandboxClientOptions"]
