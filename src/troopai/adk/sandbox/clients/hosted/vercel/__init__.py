"""VercelSandboxClient — hosted bridge.

Requires the [sandbox-vercel] extra. Extends
RemoteVMSandboxClient for shared HTTP / retry / port-forward
machinery; overrides only provider-specific create / resume / auth.
"""

from __future__ import annotations

from troopai.adk.sandbox.clients.hosted.vercel.vercel_client import (
    VercelSandboxClient,
    VercelSandboxClientOptions,
)

__all__ = ["VercelSandboxClient", "VercelSandboxClientOptions"]
