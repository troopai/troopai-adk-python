"""``Capabilities`` — the default capability set for SandboxAgents.

A plain (non-Pydantic) container class with one classmethod
``default()``. Returns a fresh ``list[SandboxCapability]`` callers
can extend with plain list concatenation:

    capabilities = Capabilities.default() + [ShellCapability(), Filesystem()]

**Cost-conservative default**: OpenAI's ``Capabilities.default()``
returns ``[Filesystem(), Shell(), Compaction()]`` — every fresh
``SandboxAgent`` gets shell-exec out of the box. We return ONLY
``[CompactionCapability()]`` so developers never pay for shell or
filesystem capabilities they didn't explicitly opt into. Shell
and Filesystem are opt-in.

Compaction is included by default because (1) it is a no-cost
hook until a long-running session actually approaches the context
window, (2) running without compaction in a long session silently
truncates history at the LLM provider, and (3) the policy
auto-selects per-model context windows so the default is sensible
across providers.
"""

from __future__ import annotations

from troopai.adk.sandbox.capabilities.base import SandboxCapability

__all__ = ["Capabilities"]


class Capabilities:
    """Capability-set factory + sentinel placeholder for the default list."""

    @classmethod
    def default(cls) -> list[SandboxCapability]:
        """Return the cost-conservative default capability list.

        Returns a FRESH list every call so callers can mutate
        without affecting other agents.
        """
        # Local import keeps capabilities/__init__.py free of an
        # eager dependency on the concrete compaction module.
        from troopai.adk.sandbox.capabilities.compaction import CompactionCapability

        return [CompactionCapability()]
