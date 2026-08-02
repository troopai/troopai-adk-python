"""Directory-based skill source.

Loads skills from a filesystem directory containing a ``SKILL.md``
file with YAML frontmatter and markdown instructions.  Compatible
with LangChain/CrewAI/Google ADK skill format.

Expected directory structure::

    my-skill/
        SKILL.md          # Required: YAML frontmatter + markdown body
        scripts/           # Optional: executable scripts
        references/        # Optional: reference documents
        assets/            # Optional: static files (configs, data)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override

from troopai.adk.skills.sources.base import SkillSource

if TYPE_CHECKING:
    from troopai.adk.skills.skill import Skill

logger = logging.getLogger(__name__)

# Regex to extract YAML frontmatter between --- delimiters. The body
# group and its leading newline are optional so a frontmatter-only file
# saved without a trailing newline still parses.
_FRONTMATTER_RE = re.compile(
    r"\A\s*---\s*\n(.*?)\n---\s*(?:\n(.*))?\Z",
    re.DOTALL,
)


def parse_skill_md(content: str) -> tuple[dict[str, str], str]:
    """Parse a SKILL.md file into frontmatter dict and body.

    Args:
        content: The full text of SKILL.md.

    Returns:
        A ``(frontmatter_dict, body)`` tuple.

    Raises:
        ValueError: If the file has no valid frontmatter.
    """
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        raise ValueError("SKILL.md must start with YAML frontmatter delimited by '---' lines")

    frontmatter_text = match.group(1)
    body = (match.group(2) or "").strip()

    # Simple YAML-like parser for flat key: value pairs
    # Avoids requiring pyyaml as a dependency for skill loading
    frontmatter: dict[str, str] = {}
    for line in frontmatter_text.splitlines():
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        if ":" in line:
            key, _, value = line.partition(":")
            v = value.strip()
            # Strip one matching pair of surrounding YAML quotes so that
            # quoted scalars (e.g. ``name: "my-skill"``, used when a value
            # contains a comma or colon) normalize to the bare string.
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                v = v[1:-1]
            frontmatter[key.strip()] = v

    return frontmatter, body


def collect_resources(skill_dir: Path) -> dict[str, str] | None:
    """Collect resources from skill subdirectories.

    Scans ``scripts/``, ``references/``, and ``assets/`` for files
    and returns a mapping of ``"subdir/filename"`` → file path.

    Symlinks and any files whose resolved path falls outside
    ``skill_dir`` are skipped and logged as warnings. This blocks the
    symlink-escape attack where a malicious skill archive contains
    ``scripts/x -> /etc/passwd`` (or similar) and turns every consumer
    of ``skill.resources`` into a read/execute primitive for arbitrary
    paths on the host.

    Args:
        skill_dir: The skill root directory.

    Returns:
        A resource dict, or ``None`` if no resources found.
    """
    try:
        skill_root_real = skill_dir.resolve(strict=True)
    except (OSError, RuntimeError):
        logger.warning(
            "Skill directory %s could not be resolved; skipping resources",
            skill_dir,
        )
        return None

    resources: dict[str, str] = {}
    for subdir_name in ("scripts", "references", "assets"):
        subdir = skill_dir / subdir_name
        if not subdir.is_dir():
            continue
        for file_path in subdir.rglob("*"):
            if file_path.is_symlink():
                logger.warning(
                    "Skipping symlinked skill resource %s (symlinks not permitted to prevent directory escape)",
                    file_path,
                )
                continue
            if not file_path.is_file():
                continue
            try:
                resolved = file_path.resolve(strict=True)
                resolved.relative_to(skill_root_real)
            except (OSError, RuntimeError, ValueError):
                logger.warning(
                    "Skipping skill resource %s: resolves outside skill directory %s",
                    file_path,
                    skill_dir,
                )
                continue
            rel = file_path.relative_to(skill_dir)
            resources[str(rel)] = str(resolved)
    return resources if len(resources) > 0 else None


@dataclass
class DirectorySkillSource(SkillSource):
    """Load a skill from a filesystem directory with ``SKILL.md``.

    The directory must contain a ``SKILL.md`` file with YAML
    frontmatter (name, description) and a markdown body
    (instructions).

    Supported frontmatter fields:
        - ``name`` (required): Skill identifier
        - ``description`` (required): Short description
        - ``version``: Semantic version
        - ``author``: Skill author
        - ``tags``: Comma-separated tags
        - ``license``: License identifier

    Attributes:
        path: Path to the skill directory.
    """

    path: Path
    """Path to the skill directory."""

    @override
    async def load(self) -> Skill:
        """Load the full skill from the directory.

        Returns:
            A ``Skill`` with metadata, instructions, and resources.

        Raises:
            FileNotFoundError: If SKILL.md doesn't exist.
            ValueError: If SKILL.md is malformed.
        """
        return self.load_sync()

    def load_sync(self) -> Skill:
        """Synchronous version of ``load()``.

        Returns:
            A ``Skill`` with metadata, instructions, and resources.
        """
        from troopai.adk.skills.skill import Skill, SkillMetadata

        skill_md = self.path / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"SKILL.md not found in {self.path}")

        content = skill_md.read_text(encoding="utf-8")
        frontmatter, body = parse_skill_md(content)

        name = frontmatter.get("name")
        if name is None or len(name) == 0:
            raise ValueError(f"SKILL.md in {self.path} is missing required 'name' field")

        description = frontmatter.get("description")
        if description is None or len(description) == 0:
            raise ValueError(f"SKILL.md in {self.path} is missing required 'description' field")

        # Parse optional metadata fields
        tags_str = frontmatter.get("tags", "")
        tags = tuple(t.strip() for t in tags_str.split(",") if t.strip()) if len(tags_str) > 0 else ()

        metadata = SkillMetadata(
            version=frontmatter.get("version"),
            author=frontmatter.get("author"),
            tags=tags,
            license=frontmatter.get("license"),
        )

        resources = collect_resources(self.path)

        # Canonical, fully-resolved skill root. Populated even when no
        # resources were collected, so callers that later hand-extend
        # ``resources`` still get execute-time containment enforced.
        try:
            resource_root = self.path.resolve(strict=True)
        except (OSError, RuntimeError):
            resource_root = None

        logger.info(
            "Loaded skill '%s' from %s (%d chars instructions, %d resources)",
            name,
            self.path,
            len(body),
            len(resources) if resources is not None else 0,
        )

        return Skill(
            name=name,
            description=description,
            instructions=body if len(body) > 0 else None,
            metadata=metadata,
            resources=resources,
            resource_root=resource_root,
        )

    @override
    async def load_metadata(self) -> tuple[str, str]:
        """Load only name and description from frontmatter.

        Returns:
            A ``(name, description)`` tuple.
        """
        skill_md = self.path / "SKILL.md"
        if not skill_md.exists():
            raise FileNotFoundError(f"SKILL.md not found in {self.path}")

        content = skill_md.read_text(encoding="utf-8")
        frontmatter, _ = parse_skill_md(content)

        name = frontmatter.get("name", "")
        description = frontmatter.get("description", "")
        return name, description
