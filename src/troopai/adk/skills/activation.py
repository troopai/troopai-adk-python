"""Skill activation strategies.

Controls when skill instructions are injected into the agent's
system prompt during execution.
"""

from __future__ import annotations

from enum import StrEnum


class SkillActivation(StrEnum):
    """Controls when skill instructions enter the system prompt.

    Attributes:
        EAGER: Instructions injected at run start — always in context.
            Best for agents with few, always-relevant skills.
        LAZY: Instructions injected only when one of the skill's
            tools is first called.  Zero-overhead discovery — tool
            descriptions are already in the tool list, and
            instructions load only when needed.  Best for agents
            with many skills where only a few are used per run.
    """

    EAGER = "eager"
    """Instructions injected at run start (always in context)."""

    LAZY = "lazy"
    """Instructions injected on first skill tool call (zero overhead)."""
