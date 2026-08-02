"""System prompt prefix for skill-enabled agents.

Provides context to the LLM about how to use skill tools
(list_skills, load_skill, load_skill_resource) effectively.

Opt-in — never auto-injected.  Use ``prompt_with_skill_instructions()``
to prepend to an agent's system prompt.

Follows the same pattern as ``handoffs/handoff_prompt.py``.
"""

RECOMMENDED_SKILL_INSTRUCTIONS = (
    "# Skill system context\n\n"
    "You have access to specialized **skills** that extend your capabilities "
    "for domain-specific tasks.  You MUST use the skill tools to interact "
    "with these skills.\n\n"
    "Each skill bundles detailed instructions, tools, and optional resources "
    "(reference files, guides, scripts).  Skills may be loaded from directories "
    "containing:\n"
    "- **SKILL.md** (required): The main instruction file with metadata and "
    "detailed markdown instructions.\n"
    "- **references/** (optional): Additional documentation or examples.\n"
    "- **assets/** (optional): Templates, data files, or configuration.\n"
    "- **scripts/** (optional): Executable scripts (Python, Bash).\n\n"
    "## Rules (very important)\n\n"
    "1. Use `list_skills` to discover all available skills with their names "
    "and descriptions.\n"
    "2. If a skill seems relevant to the user's request, you MUST use "
    '`load_skill` with `name="<SKILL_NAME>"` to read its full instructions '
    "and see what tools and resources it provides BEFORE acting.\n"
    "3. Once you have read the instructions, follow them exactly as documented.  "
    "If the instructions list multiple steps, complete all of them in order.\n"
    "4. Use `load_skill_resource` to read reference files or guides from a "
    "skill's resources.  Do NOT attempt to access these files through other means.\n"
    "5. When a tool search returns no results, you MUST check if the answer "
    "is in a skill's resources — use `load_skill` to see available resources, "
    "then `load_skill_resource` to read them.\n"
)
"""Recommended prompt prefix for agents with SkillDiscoveryToolset.

Teaches the LLM the discovery workflow: list → load → act.
Only useful when the agent has discovery tools (list_skills,
load_skill, etc.) in its tool list.

Inspired by Google ADK's ``_DEFAULT_SKILL_SYSTEM_INSTRUCTION``.
"""


def prompt_with_skill_instructions(prompt: str) -> str:
    """Prepend recommended skill instructions to a prompt string.

    Args:
        prompt: The original system prompt string.

    Returns:
        The prompt with skill instructions prepended.
    """
    return f"{RECOMMENDED_SKILL_INSTRUCTIONS}\n{prompt}"
