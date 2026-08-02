"""Remote URL-based skill source.

Loads a skill by fetching a ``SKILL.md`` file from a remote URL
(HTTP GET).  Supports optional authentication headers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

from troopai.adk.skills.sources.base import SkillSource

if TYPE_CHECKING:
    from troopai.adk.skills.skill import Skill

logger = logging.getLogger(__name__)


@dataclass
class RemoteSkillSource(SkillSource):
    """Load a skill from a remote URL.

    Fetches the content at ``url`` via HTTP GET and parses it as
    a ``SKILL.md`` file (YAML frontmatter + markdown body).

    Attributes:
        url: URL to the SKILL.md file.
        headers: Optional HTTP headers (e.g. for authentication).
    """

    url: str
    """URL to the SKILL.md file."""

    headers: dict[str, str] | None = field(default=None)
    """Optional HTTP headers (e.g. for authentication)."""

    @override
    async def load(self) -> Skill:
        """Fetch and parse the skill from the remote URL.

        Returns:
            A ``Skill`` with metadata and instructions.

        Raises:
            httpx.HTTPStatusError: If the HTTP request fails.
            ValueError: If the response is not a valid SKILL.md.
        """
        import httpx

        from troopai.adk.skills.skill import Skill, SkillMetadata
        from troopai.adk.skills.sources.directory import parse_skill_md

        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.url,
                headers=self.headers,
                follow_redirects=True,
            )
            response.raise_for_status()

        content = response.text
        frontmatter, body = parse_skill_md(content)

        name = frontmatter.get("name")
        if name is None or len(name) == 0:
            raise ValueError(f"Remote SKILL.md at {self.url} is missing required 'name' field")

        description = frontmatter.get("description")
        if description is None or len(description) == 0:
            raise ValueError(f"Remote SKILL.md at {self.url} is missing required 'description' field")

        tags_str = frontmatter.get("tags", "")
        tags = tuple(t.strip() for t in tags_str.split(",") if t.strip()) if len(tags_str) > 0 else ()

        metadata = SkillMetadata(
            version=frontmatter.get("version"),
            author=frontmatter.get("author"),
            tags=tags,
            license=frontmatter.get("license"),
        )

        logger.info(
            "Loaded skill '%s' from %s (%d chars instructions)",
            name,
            self.url,
            len(body),
        )

        return Skill(
            name=name,
            description=description,
            instructions=body if len(body) > 0 else None,
            metadata=metadata,
        )

    @override
    async def load_metadata(self) -> tuple[str, str]:
        """Fetch and extract only name and description.

        Note: This still fetches the full file since HTTP doesn't
        support partial content easily.  For performance-sensitive
        discovery, cache the metadata locally.

        Returns:
            A ``(name, description)`` tuple.
        """
        import httpx

        from troopai.adk.skills.sources.directory import parse_skill_md

        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.url,
                headers=self.headers,
                follow_redirects=True,
            )
            response.raise_for_status()

        frontmatter, _ = parse_skill_md(response.text)
        return frontmatter.get("name", ""), frontmatter.get("description", "")
