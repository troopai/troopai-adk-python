"""TroopAI Workflows — durable execution engine abstraction.

Provides the :class:`DurableEngine` Protocol and configuration dataclasses
shared by all durable execution backends.  Concrete backends live in
sub-packages:

- ``temporal/`` — Temporal.io-backed durable execution.
- ``restate/`` — Restate-backed durable execution.

Install the matching optional extra to activate a backend::

    pip install "troopai-adk-python[temporal]"

The core module is dependency-free; only the sub-packages carry their
respective SDK dependencies.
"""

from __future__ import annotations

from troopai.adk.workflows.engine import (
    DurableEngine,
    ModelActivityConfig,
    ToolActivityConfig,
)

__all__ = [
    "DurableEngine",
    "ModelActivityConfig",
    "ToolActivityConfig",
]
