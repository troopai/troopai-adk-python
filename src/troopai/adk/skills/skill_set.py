"""Skill set — named collection of skills.

A convenience container for organizing related skills and
reusing them across multiple agents.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from troopai.adk.skills.skill import Skill

logger = logging.getLogger(__name__)


@dataclass
class SkillSet:
    """A named collection of skills that can be assigned to agents.

    Useful for organizing related skills and reusing them across
    multiple agents without duplicating skill lists.

    Attributes:
        name: Name for this skill collection.
        skills: The skills in this set.
    """

    name: str
    """Name for this skill collection."""

    skills: list[Skill] = field(default_factory=list)
    """The skills in this set."""

    def find(self, name: str) -> Skill | None:
        """Find a skill by name.

        Args:
            name: The skill name to search for.

        Returns:
            The matching ``Skill``, or ``None`` if not found.
        """
        for skill in self.skills:
            if skill.name == name:
                return skill
        return None

    def filter_by_tag(self, tag: str) -> list[Skill]:
        """Return skills matching a tag.

        Args:
            tag: The tag to filter by.

        Returns:
            A list of skills whose metadata contains the given tag.
        """
        return [skill for skill in self.skills if skill.metadata is not None and tag in skill.metadata.tags]
