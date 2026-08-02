"""Sandbox backend clients — abstract contract + concrete implementations.

The package root exposes the ABCs (``BaseSandboxClient``,
``BaseSandboxSession``, ``BaseSandboxClientOptions``) plus the
client-side records (``FileEntry``, ``MaterializationResult``).
Concrete backends live in subpackages: ``local/``,
``docker/``, ``k8s/``, and ``hosted/``.
"""

from __future__ import annotations

from troopai.adk.sandbox.clients.base import (
    BaseSandboxClient,
    BaseSandboxClientOptions,
    ClientOptionsT,
)
from troopai.adk.sandbox.clients.session import (
    BaseSandboxSession,
    FileEntry,
    MaterializationResult,
)

__all__ = [
    "BaseSandboxClient",
    "BaseSandboxClientOptions",
    "BaseSandboxSession",
    "ClientOptionsT",
    "FileEntry",
    "MaterializationResult",
]
