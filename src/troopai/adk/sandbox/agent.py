"""``SandboxAgent`` — an Agent paired with a sandbox session.

A SandboxAgent IS an Agent. Same instructions / tools / handoffs /
mcp_servers / model_settings / output_type / guardrails / hooks.
The only additions are sandbox-specific defaults: ``default_manifest``,
``capabilities``, ``run_as``.

A SandboxAgent stays config-only; orchestration belongs to the Runner.
The Runner detects ``isinstance(agent, SandboxAgent)`` and brackets
the agent loop with ``sandbox_run_context``.
"""

from __future__ import annotations

import dataclasses
import threading
from typing import TYPE_CHECKING

from troopai.adk.agents.agent import Agent
from troopai.adk.sandbox.capabilities.capabilities import Capabilities

if TYPE_CHECKING:
    from troopai.adk.sandbox.capabilities.base import SandboxCapability
    from troopai.adk.sandbox.runner_integration.concurrency_guard import (
        SandboxConcurrencyGuard,
    )
    from troopai.adk.types.sandbox.manifest import Manifest
    from troopai.adk.types.sandbox.permissions import User

__all__ = ["SandboxAgent"]


# Sentinel system_prompt the framework recognises and replaces at
# turn-time with the resolved sandbox prompt + agent instructions.
# The canonical constant lives in runner_integration.instructions_composer.
from troopai.adk.sandbox.runner_integration.instructions_composer import (
    SANDBOX_PLACEHOLDER_SYSTEM_PROMPT,
)


@dataclasses.dataclass
class SandboxAgent(Agent):
    """Agent subclass carrying sandbox-specific defaults.

    Attributes:
        default_manifest: Fresh-session workspace contract applied
            when the Runner creates a new sandbox session.
        capabilities: Composable extensions (Shell, Filesystem,
            Skills, Memory, Compaction). Defaults to
            ``Capabilities.default()`` — cost-conservative
            ``[CompactionCapability()]``.
        run_as: Model-facing user identity. Shell + Filesystem +
            Skills tools run as this user when the backend supports
            account provisioning.
    """

    default_manifest: Manifest | None = None
    """Fresh-session workspace contract."""

    capabilities: list[SandboxCapability] = dataclasses.field(
        default_factory=Capabilities.default,
    )
    """Composable extensions for sandbox-native behaviour."""

    run_as: User | str | None = None
    """Model-facing user identity (User object or username string)."""

    def __post_init__(self) -> None:
        # When no system_prompt is provided, inject a sentinel so
        # Agent.__post_init__'s "instructions cannot be empty" check
        # passes. The Runner-side instructions composer replaces the
        # sentinel with the resolved sandbox prompt at turn-time.
        # Callers MAY set ``system_prompt`` explicitly to override
        # the entire prompt.
        if self.system_prompt is None:
            self.system_prompt = SANDBOX_PLACEHOLDER_SYSTEM_PROMPT
        super().__post_init__()
        self._validate_run_as()
        # Concurrency guard is lazily allocated by ``get_concurrency_guard``
        # so newly-constructed agents don't pay the asyncio.Lock cost
        # until they actually enter a Runner.arun.
        object.__setattr__(self, "_concurrency_guard", None)
        object.__setattr__(self, "_guard_init_lock", threading.Lock())

    def _validate_run_as(self) -> None:
        if self.run_as is None:
            return
        if isinstance(self.run_as, str):
            if len(self.run_as) == 0:
                raise ValueError("SandboxAgent.run_as must be a non-empty string or User")
            return
        # Otherwise must be a User instance; allow ducktyping but
        # surface a clear error if it's neither str nor User.
        if not hasattr(self.run_as, "name"):
            raise TypeError(
                f"SandboxAgent.run_as must be a User or non-empty str, got {type(self.run_as).__name__}",
            )

    def get_concurrency_guard(self) -> SandboxConcurrencyGuard:
        """Return the per-agent concurrency guard, allocating lazily.

        The private ``_concurrency_guard`` attribute is encapsulated
        behind this accessor; callers MUST NEVER touch the underscore
        attribute from outside this class.
        """
        existing = object.__getattribute__(self, "_concurrency_guard")
        if existing is not None:
            return existing  # type: ignore[no-any-return]
        # Late import to avoid a load-time cycle between sandbox/
        # agent.py and sandbox/runner_integration/.
        from troopai.adk.sandbox.runner_integration.concurrency_guard import (
            SandboxConcurrencyGuard,
        )

        lock = object.__getattribute__(self, "_guard_init_lock")
        with lock:
            existing = object.__getattribute__(self, "_concurrency_guard")
            if existing is not None:
                return existing  # type: ignore[no-any-return]
            guard = SandboxConcurrencyGuard()
            object.__setattr__(self, "_concurrency_guard", guard)
            return guard
