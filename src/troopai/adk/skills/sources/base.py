"""Abstract base class for skill sources.

Skill sources provide a uniform interface for loading skills
from different locations (filesystem, remote URLs, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.skills.skill import Skill


class SkillSource(ABC):
    """Abstract base for skill content sources.

    Subclasses implement ``load()`` to fetch and parse skill content
    from their specific source (filesystem, URL, etc.).
    """

    @abstractmethod
    async def load(self) -> Skill:
        """Load the full skill content from this source.

        Returns:
            A fully populated ``Skill`` instance.
        """
        ...

    @abstractmethod
    async def load_metadata(self) -> tuple[str, str]:
        """Load only name and description from the skill source.

        Used for discovery without loading the full skill instructions.

        Returns:
            A ``(name, description)`` tuple.
        """
        ...
