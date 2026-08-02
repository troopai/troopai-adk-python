"""Core skill data model.

A Skill is a reusable, self-contained capability package that bundles
instructions, tools, resources, and governance into a composable unit.
Skills occupy the design space between a FunctionTool (single function)
and an Agent (full autonomous entity).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from troopai.adk.utils.typedef import MaybeAwaitable

if TYPE_CHECKING:
    from troopai.adk.run.context import RunContext
    from troopai.adk.tools import Tool
    from troopai.adk.tools.tool_guardrails import ToolGuardrails

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillMetadata:
    """Immutable metadata about a skill.

    Used for discovery, cataloging, and filtering skills.

    Attributes:
        version: Semantic version string (e.g. ``"1.0.0"``).
        author: Skill author or organization.
        tags: Searchable tags for discovery and filtering.
        license: License identifier (e.g. ``"MIT"``).
    """

    version: str | None = None
    """Semantic version string (e.g. ``"1.0.0"``)."""

    author: str | None = None
    """Skill author or organization."""

    tags: tuple[str, ...] = ()
    """Searchable tags for discovery and filtering."""

    license: str | None = None
    """License identifier (e.g. ``"MIT"``)."""


@dataclass
class SkillGovernance:
    """Resource governance applied to all tools within a skill.

    Values here act as defaults — a tool's own governance settings
    take precedence over skill-level governance.

    Attributes:
        timeout: Per-tool execution timeout in seconds.
        max_result_tokens: Max tokens for each tool's result.
        max_retries: LLM retry budget for skill tools.
    """

    timeout: float | None = None
    """Per-tool execution timeout in seconds.

    Applied as default to tools that do not set their own timeout.
    """

    max_result_tokens: int | None = None
    """Max tokens for each tool's result.

    Applied as default to tools that do not set their own
    ``max_result_tokens``.
    """

    max_retries: int | None = None
    """LLM retry budget for skill tools.

    Applied as default to tools that do not set their own
    ``max_retries``.
    """


@dataclass
class Skill:
    """A reusable capability package that augments agent behavior.

    Skills combine domain instructions with optional tools, resources,
    and governance into a single composable unit.  They occupy the
    middle ground between a FunctionTool (single function) and a full
    Agent (autonomous entity).

    Skills are attached to agents via ``Agent(skills=[...])``.  The
    Runner handles activation based on the agent's ``skill_activation``
    strategy (EAGER or LAZY).

    Attributes:
        name: Unique identifier for this skill.
        description: Short description for discovery and matching.
        instructions: Detailed markdown instructions injected into
            the agent's system prompt when the skill is activated.
        tools: Tools this skill provides to the agent.
        guardrails: ``ToolGuardrails`` config holding ``input`` and
            ``output`` phase-typed lists. Each entry is prepended to
            every contained tool's own guardrails when the skill
            activates, so skill-level guardrails run first.
        enabled: Whether this skill is active (static or dynamic).
        metadata: Additional metadata (version, author, tags, license).
        governance: Resource governance applied to all skill tools.
        resources: Named resources available to skill tools.
        resource_root: Absolute, fully-resolved directory that bounds
            ``resources``. When set, any attempt to read or execute a
            resource whose resolved path escapes this root is refused.
            ``None`` means the caller's paths are trusted unconditionally,
            which is appropriate for hand-built ``Skill`` objects with
            inline resources.
    """

    name: str
    """Unique identifier for this skill."""

    description: str
    """Short description of what this skill provides.

    Used for discovery and matching.  Stored in the ``SKILL.md``
    frontmatter for directory-sourced skills and loaded during
    discovery without reading the full instructions.  Should be
    concise (under 1024 characters).
    """

    instructions: str | None = None
    """Detailed markdown instructions injected into the agent's
    system prompt when the skill is activated.

    Only loaded when the skill is activated: at run start for EAGER
    activation, or on the first call to one of the skill's tools
    for LAZY activation.
    """

    tools: list[Tool] = field(default_factory=list)
    """Tools this skill provides to the agent.

    Merged into the agent's tool list when the skill is activated.
    Skill-level governance and guardrails are applied as defaults
    to each tool.
    """

    guardrails: ToolGuardrails | None = None
    """Per-phase tool-level guardrails applied to every tool in this skill.

    A single :class:`~troopai.adk.tools.tool_guardrails.ToolGuardrails`
    config object holds ``input`` and ``output`` phase-typed lists.
    ``None`` (default) means the skill adds no guardrails — each
    contained tool's own ``guardrails`` still applies.

    When set, skill-level entries are prepended to each tool's own
    guardrails so skill-level guardrails run first.
    """

    enabled: bool | Callable[[RunContext[Any]], MaybeAwaitable[bool]] = True
    """Whether this skill is enabled.

    When ``False``, the skill's tools and instructions are excluded
    from the agent.  Can be a callable that receives ``RunContext``
    and returns a bool (sync or async) for dynamic enablement.
    """

    metadata: SkillMetadata | None = None
    """Additional metadata (version, author, tags, license).

    Used for discovery, cataloging, and filtering.
    """

    governance: SkillGovernance | None = None
    """Resource governance applied to all tools in this skill.

    Governance values act as defaults — a tool's own settings take
    precedence.  For example, ``SkillGovernance(timeout=30.0)``
    sets a 30s timeout on all skill tools that don't have their
    own ``timeout``.
    """

    resources: dict[str, str] | None = None
    """Named resources available to skill tools.

    Keys are resource identifiers, values are file paths or content
    strings.  Accessible to discovery tools via
    ``SkillDiscoveryToolset.load_skill_resource()``.
    """

    resource_root: Path | None = None
    """Absolute, fully-resolved directory that bounds ``resources``.

    Populated by ``DirectorySkillSource`` to the real path of the
    skill directory it loaded from. When set, ``run_skill_script``
    refuses to execute any script whose resolved path escapes this
    root — a defence against both the hand-crafted
    ``resources={"scripts/x.sh": "/etc/passwd"}`` case and a
    post-load swap that relocates the file outside the skill tree.

    ``None`` means "trust the caller's paths unconditionally" and is
    the right default for hand-built ``Skill`` objects with inline
    resources (e.g. unit tests).  Prod deployments loading from
    disk or from a remote archive should always land here with
    ``resource_root`` set.
    """

    def __post_init__(self) -> None:
        """Validate skill configuration after initialization."""
        if len(self.name) == 0:
            raise ValueError("Skill name cannot be empty")
        if len(self.description) == 0:
            raise ValueError("Skill description cannot be empty")

    @classmethod
    def from_directory(cls, path: str | Path) -> Skill:
        """Load a skill from a directory containing ``SKILL.md``.

        The directory must contain a ``SKILL.md`` file with YAML
        frontmatter (name, description) and a markdown body
        (instructions).  Optional subdirectories ``scripts/``,
        ``references/``, and ``assets/`` are loaded as resources.

        Compatible with LangChain/CrewAI/Google ADK skill format.

        Args:
            path: Path to the skill directory.

        Returns:
            A ``Skill`` loaded from the directory.

        Raises:
            FileNotFoundError: If the directory or SKILL.md doesn't exist.
            ValueError: If SKILL.md is malformed.
        """
        from troopai.adk.skills.sources.directory import DirectorySkillSource

        source = DirectorySkillSource(path=Path(path))
        return source.load_sync()

    @classmethod
    async def from_url(
        cls,
        url: str,
        *,
        headers: dict[str, str] | None = None,
    ) -> Skill:
        """Load a skill from a remote URL.

        Fetches a ``SKILL.md`` file from the given URL and parses
        it the same way as ``from_directory()``.

        Args:
            url: URL to the SKILL.md file.
            headers: Optional HTTP headers (e.g. for authentication).

        Returns:
            A ``Skill`` loaded from the remote source.

        Raises:
            httpx.HTTPError: If the HTTP request fails.
            ValueError: If the response is not a valid SKILL.md.
        """
        from troopai.adk.skills.sources.remote import RemoteSkillSource

        source = RemoteSkillSource(url=url, headers=headers)
        return await source.load()
