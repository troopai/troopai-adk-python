"""Sandbox agents — isolated execution environments for agent runs.

A SandboxAgent pairs an Agent with a workspace-backed sandbox session
(local subprocess, Docker container, K8s pod, or hosted compute
provider) so the model can manipulate files and run commands inside a
controlled boundary.

Public re-exports live here; concrete implementations are organized
under sub-packages: ``types/`` for Layer-1 data types, ``capabilities/``
for composable capability extensions, ``clients/`` for backend
implementations, ``guardrails/`` for policy, ``observability/`` for
audit + tracing, ``snapshot/`` for workspace persistence,
``runner_integration/`` for lifecycle glue.
"""

from __future__ import annotations

from troopai.adk.sandbox.config import SandboxRunConfig

__all__: list[str] = [
    "SandboxRunConfig",
]
