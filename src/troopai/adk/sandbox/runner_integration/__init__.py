"""Runner-side glue that brackets the agent loop with a sandbox lifecycle.

The runner integration layer is loaded by ``Runner.arun`` when a
``SandboxAgent`` (or non-None ``RunConfig.sandbox``) is detected.
This package houses:

- ``concurrency_guard.py`` — per-agent exclusivity primitive.
- ``capability_lifecycle.py`` — clone → bind → process → collect.
- ``lifecycle.py`` — async context manager wrapping the agent loop.
- ``instructions_composer.py`` — sandbox-aware system prompt builder.
- ``iac_runner.py`` — Terraform / Pulumi apply + destroy.

The per-run entry points are re-exported here as the package's
public surface so callers import from the package rather than
reaching into submodules.
"""

from __future__ import annotations

from troopai.adk.sandbox.runner_integration.concurrency_guard import (
    SandboxConcurrencyGuard,
)
from troopai.adk.sandbox.runner_integration.iac_runner import (
    apply_iac,
    destroy_iac,
)
from troopai.adk.sandbox.runner_integration.instructions_composer import (
    compose_sandbox_prompt,
)
from troopai.adk.sandbox.runner_integration.lifecycle import (
    SandboxLifecycleHandle,
    sandbox_run_context,
)

__all__ = [
    "SandboxConcurrencyGuard",
    "SandboxLifecycleHandle",
    "apply_iac",
    "compose_sandbox_prompt",
    "destroy_iac",
    "sandbox_run_context",
]
