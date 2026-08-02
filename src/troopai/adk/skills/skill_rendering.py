"""Skill instruction rendering for system prompt enrichment.

Renders activated skill instructions as a structured section
appended to the agent's system prompt.  Used by the Runner
during EAGER and LAZY skill activation.
"""

from __future__ import annotations

import logging
from inspect import isawaitable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from troopai.adk.run.context import RunContext
    from troopai.adk.skills.skill import Skill

logger = logging.getLogger(__name__)


async def check_skill_enabled(
    skill: Skill,
    context: RunContext[Any] | None,
) -> bool:
    """Evaluate whether a skill is enabled.

    Handles both static ``bool`` and dynamic callables (sync/async).

    Args:
        skill: The skill to check.
        context: Run context passed to dynamic callables.

    Returns:
        ``True`` if the skill is enabled.
    """
    if isinstance(skill.enabled, bool):
        return skill.enabled
    if context is None:
        # A callable ``enabled`` is a developer-supplied gate that needs the
        # run context to evaluate. Without it we cannot honor the condition,
        # so fail closed rather than injecting a skill whose gate we could
        # not check.
        logger.debug(
            "Skill %r has a callable 'enabled' but no run context; treating as disabled",
            skill.name,
        )
        return False
    result = skill.enabled(context)
    if isawaitable(result):
        result = await result
    return result


async def render_skill_instructions(
    skills: list[Skill],
    context: RunContext[Any] | None = None,
    *,
    exclude: set[str] | None = None,
) -> str | None:
    """Render activated skill instructions as a system prompt section.

    Only includes enabled skills that have non-empty instructions.
    Returns ``None`` if no skills have instructions to inject.

    Args:
        skills: The skills to render instructions for.
        context: Run context for dynamic ``enabled`` checks.
        exclude: Optional set of skill names to exclude
            (used by LAZY activation to exclude not-yet-activated skills).

    Returns:
        A formatted string with skill instructions, or ``None``.
    """
    sections: list[str] = []
    excluded = exclude or set()

    for skill in skills:
        if skill.name in excluded:
            continue
        if skill.instructions is None or len(skill.instructions) == 0:
            continue
        if not await check_skill_enabled(skill, context):
            continue
        sections.append(f"### {skill.name}\n\n{skill.instructions}")

    if len(sections) == 0:
        return None

    header = "## Available Skills\n\n"
    rendered = header + "\n\n".join(sections)

    logger.debug(
        "Rendered skill instructions for %d skill(s), ~%d chars",
        len(sections),
        len(rendered),
    )
    return rendered


def render_single_skill_instruction(skill: Skill) -> str | None:
    """Render a single skill's instructions for LAZY injection.

    Args:
        skill: The skill to render.

    Returns:
        A formatted string, or ``None`` if no instructions.
    """
    if skill.instructions is None or len(skill.instructions) == 0:
        return None
    return f"\n\n### {skill.name}\n\n{skill.instructions}"
