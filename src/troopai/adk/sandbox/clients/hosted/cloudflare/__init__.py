"""CloudflareSandboxClient — hosted bridge.

Requires the [sandbox-cloudflare] extra. Extends
RemoteVMSandboxClient for shared HTTP / retry / port-forward
machinery; overrides only provider-specific create / resume / auth.
"""

from __future__ import annotations

from troopai.adk.sandbox.clients.hosted.cloudflare.cloudflare_client import (
    CloudflareSandboxClient,
    CloudflareSandboxClientOptions,
)

__all__ = ["CloudflareSandboxClient", "CloudflareSandboxClientOptions"]
