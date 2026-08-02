"""Local-subprocess sandbox client (dev-only).

The local backend runs the agent's commands as subprocesses of the
host Python process inside a temporary working directory. NO
isolation. NO network policy enforcement. NO resource caps beyond
basic timeout. Use ONLY for development and example code.
"""

from __future__ import annotations

from troopai.adk.sandbox.clients.local.subprocess_client import (
    LocalSandboxClientOptions,
    LocalSandboxSession,
    LocalSubprocessSandboxClient,
)

__all__ = [
    "LocalSandboxClientOptions",
    "LocalSandboxSession",
    "LocalSubprocessSandboxClient",
]
