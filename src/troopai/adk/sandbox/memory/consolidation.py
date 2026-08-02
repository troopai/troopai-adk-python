"""Consolidation step — consolidate raw memories into MEMORY.md + memory_summary.md.

This step is a tool-using LLM call: the model reads
``raw_memories.md`` and the prior ``MEMORY.md`` /
``memory_summary.md``, then rewrites them in place. We do NOT
extract structured output from the response — the model's tool
calls land the new contents on disk; the response itself is
discarded once the call returns.

The runner is injected so tests can drive the consolidation step against a fake
agent without spinning up a real provider.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from troopai.adk.sandbox.memory.prompts import render_memory_consolidation_prompt
from troopai.adk.sandbox.memory.storage import ConsolidationInputSelection

logger = logging.getLogger(__name__)

__all__ = [
    "CONSOLIDATION_INSTRUCTION_HEADER",
    "MemoryConsolidationRunner",
    "run_consolidation",
]

MemoryConsolidationRunner = Callable[[str, str], Awaitable[None]]
"""Callback signature: ``async (system_prompt, user_prompt) -> None``.

The runner is expected to drive a tool-using agent that edits the
memory files in-place; the return value is intentionally
discarded because the result lives on the workspace, not in the
LLM's text response.
"""

CONSOLIDATION_INSTRUCTION_HEADER: str = (
    "Read the files described in the system prompt, then rewrite the consolidated memory."
)


async def run_consolidation(
    *,
    memory_root: str,
    selection: ConsolidationInputSelection,
    runner: MemoryConsolidationRunner,
    extra_prompt: str | None = None,
) -> None:
    """Drive the consolidation-step LLM consolidation pass.

    Args:
        memory_root: Workspace-relative path to the memories
            directory (e.g. ``"memories"``). The system prompt
            references this path so the model knows where to find
            ``MEMORY.md`` / ``raw_memories.md``.
        selection: The current the current input selection — the model
            sees the diff against the prior consolidation so it
            can drop or summarize evicted entries.
        runner: Async callable that drives the tool-using agent.
        extra_prompt: Optional per-deployment guidance appended to
            the system prompt.
    """
    system_prompt = render_memory_consolidation_prompt(
        memory_root=memory_root,
        selection=selection,
        extra_prompt=extra_prompt,
    )
    user_prompt = CONSOLIDATION_INSTRUCTION_HEADER
    logger.debug(
        "consolidate: invoking runner with %d retained, %d removed",
        len(selection.retained_rollout_ids),
        len(selection.removed),
    )
    await runner(system_prompt, user_prompt)
