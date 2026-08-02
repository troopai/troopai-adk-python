"""Sandbox system-prompt composer (extracted from run/loop.py).

The composer resolves the SandboxAgent placeholder marker, appends
each capability's instruction fragment, and renders the manifest's
filesystem tree. Lives in runner_integration/ so the Runner-side
inline composition can delegate here instead of carrying the logic
in run/loop.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from troopai.adk.sandbox.capabilities.base import SandboxCapability
    from troopai.adk.types.sandbox.manifest import Manifest

__all__ = [
    "DEFAULT_SANDBOX_PROMPT",
    "SANDBOX_PLACEHOLDER_SYSTEM_PROMPT",
    "compose_sandbox_prompt",
]

SANDBOX_PLACEHOLDER_SYSTEM_PROMPT = "__troopai_sandbox_placeholder_system_prompt__"
"""Sentinel set on ``Agent.system_prompt`` when only ``base_instructions``
is configured. The Runner-side composer rewrites it at turn time."""

DEFAULT_SANDBOX_PROMPT = (
    "You are an agent operating inside an isolated sandbox workspace. "
    "You have access to tools that let you read files, run shell "
    "commands, and inspect the workspace filesystem. Use the tools "
    "to ground every claim in what is actually in the workspace; "
    "never fabricate file contents."
)


async def compose_sandbox_prompt(
    prompt_str: str,
    *,
    capabilities: list[SandboxCapability],
    manifest: Manifest | None,
    base_instructions: Any = None,
) -> str:
    """Build the final per-turn system prompt for a sandbox run.

    Layering (in order):
      1. ``base_instructions`` if provided, otherwise the placeholder
         is resolved to ``DEFAULT_SANDBOX_PROMPT``; an explicit
         ``prompt_str`` short-circuits both.
      2. The composed capability instruction fragments (separated by
         blank lines).
      3. The rendered manifest filesystem tree, if any entries exist.

    Returns the assembled prompt string.
    """
    # Resolve placeholder.
    if prompt_str == SANDBOX_PLACEHOLDER_SYSTEM_PROMPT:
        prompt_str = (
            str(base_instructions)
            if base_instructions is not None and len(str(base_instructions)) > 0
            else DEFAULT_SANDBOX_PROMPT
        )
    # Capability instruction fragments are manifest-independent — a
    # ShellCapability surfaces its run_command primer whether or not a
    # workspace manifest was configured, so collect them unconditionally.
    from troopai.adk.sandbox.runner_integration.capability_lifecycle import (
        collect_capability_instructions,
    )

    fragments = await collect_capability_instructions(capabilities, manifest)
    if len(fragments) > 0:
        prompt_str = prompt_str + "\n\n" + "\n\n".join(fragments)
    # Only the workspace-tree rendering needs a manifest with entries.
    if manifest is not None and len(manifest.entries) > 0:
        tree_lines = ["Workspace layout:"]
        for path, _entry in manifest.iter_entries():
            tree_lines.append(f"  /{path}")
        prompt_str = prompt_str + "\n\n" + "\n".join(tree_lines)
    return prompt_str
