"""``SkillsCapability`` — skill discovery + materialization.

Three skill-source modes (mutually exclusive):
- ``skills=[Skill(...)]``: pre-defined eager skills materialized at
  manifest-processing time.
- ``from_=<entry>``: single ``BaseEntry`` whose children are skills.
- ``lazy_from=LocalDirLazySkillSource | GitRepoLazySkillSource``:
  skills loaded on demand via the ``load_skill`` tool.

Progressive disclosure: with `lazy_from`, the system prompt lists
the skill index and the LLM calls ``load_skill(name)`` to bring a
single skill's files into the workspace.
"""

from __future__ import annotations

import atexit
import contextlib
import shutil
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, override

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from troopai.adk.exceptions.exceptions import SkillsConfigError
from troopai.adk.sandbox.capabilities.base import SandboxCapability
from troopai.adk.types.sandbox.entries import BaseEntry

if TYPE_CHECKING:
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.types.sandbox.manifest import Manifest

__all__ = [
    "GitRepoLazySkillSource",
    "LocalDirLazySkillSource",
    "Skill",
    "SkillMetadata",
    "SkillsCapability",
]


# ---------------------------------------------------------------------------
# Skill + SkillMetadata
# ---------------------------------------------------------------------------


class SkillMetadata(BaseModel):
    """Lightweight skill descriptor used by the skill index."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    path: str


def _validate_skill_path_component(name: str, label: str) -> None:
    """Reject absolute paths, ``..`` traversals, slashes, null bytes."""
    if len(name) == 0:
        raise SkillsConfigError(f"Skill {label} must be non-empty")
    if name.startswith("/") or name.startswith("\\"):
        raise SkillsConfigError(
            f"Skill {label} must be workspace-relative, got {name!r}",
        )
    if "/" in name or "\\" in name or ".." in name or "\x00" in name:
        raise SkillsConfigError(
            f"Skill {label} must be a simple name (no separators, no traversal, no null bytes); got {name!r}",
        )


class Skill(BaseModel):
    """Eagerly-loaded skill bundle.

    Attributes:
        name: Workspace-safe skill name (no separators, no ``..``).
        description: Short human-readable description.
        content: Skill body (SKILL.md content); accepts str or bytes.
        scripts: Optional ``filename -> content`` map placed under
            ``{name}/scripts/``.
        references: Optional reference docs under ``{name}/references/``.
        assets: Optional binary assets under ``{name}/assets/``.
        deferred: When True, do NOT materialize at process_manifest
            time — the runtime loads on demand.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    content: str | bytes
    scripts: dict[str, str | bytes] = Field(default_factory=dict)
    references: dict[str, str | bytes] = Field(default_factory=dict)
    assets: dict[str, str | bytes] = Field(default_factory=dict)
    deferred: bool = False

    @override
    def model_post_init(self, context: Any) -> None:
        del context
        _validate_skill_path_component(self.name, "name")
        for label, mapping in (
            ("scripts key", self.scripts),
            ("references key", self.references),
            ("assets key", self.assets),
        ):
            for key in mapping:
                _validate_skill_path_component(key, label)


# ---------------------------------------------------------------------------
# Lazy sources
# ---------------------------------------------------------------------------


class LocalDirLazySkillSource(BaseModel):
    """Lazy source loading skills from a host filesystem directory.

    Each immediate subdirectory containing a ``SKILL.md`` is a skill.
    The directory's name is the skill name; ``SKILL.md`` is its body
    (the description); any sibling ``scripts/`` / ``references/`` /
    ``assets/`` subdirectories are loaded under matching names.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_path: str
    """Host filesystem directory containing skill subdirectories."""

    def list_skill_metadata(self) -> list[SkillMetadata]:
        """Scan ``source_path`` for skill subdirectories."""
        root = Path(self.source_path)
        if not root.exists() or not root.is_dir():
            return []
        result: list[SkillMetadata] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue
            description = ""
            with contextlib.suppress(OSError):
                description = skill_md.read_text(encoding="utf-8")[:200]
            result.append(
                SkillMetadata(
                    name=child.name,
                    description=description,
                    path=str(child),
                ),
            )
        return result

    def load_skill_payload(self, skill_name: str) -> dict[str, bytes] | None:
        """Load every file in ``source_path/skill_name`` as a flat dict.

        Returns ``None`` when the skill is not found. The returned
        dict maps workspace-relative-to-skill path → file bytes.
        """
        root = Path(self.source_path) / skill_name
        if not root.exists() or not root.is_dir():
            return None
        payload: dict[str, bytes] = {}
        for file in root.rglob("*"):
            if not file.is_file():
                continue
            rel = file.relative_to(root).as_posix()
            payload[rel] = file.read_bytes()
        return payload


class GitRepoLazySkillSource(BaseModel):
    """Lazy source backed by a Git repository.

    Clones the repo to a temp directory on first access + scans for
    SKILL.md. Subsequent ``load_skill_payload`` calls reuse the clone.

    Attributes:
        repo: ``owner/name`` form.
        host: Git host (default: github.com).
        ref: Branch / tag / SHA to check out.
        subpath: Optional repo-relative path where skills live.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo: str
    host: str = "github.com"
    ref: str = "main"
    subpath: str | None = None

    _clone_path: Path | None = PrivateAttr(default=None)

    @override
    def model_post_init(self, context: Any) -> None:
        del context
        if "/" not in self.repo or self.repo.count("/") != 1:
            raise SkillsConfigError(
                f"GitRepoLazySkillSource.repo must be 'owner/name', got {self.repo!r}",
            )

    _CLONE_TIMEOUT_SECONDS: ClassVar[float] = 120.0
    """Wall-clock cap on ``git clone`` so a hung network never blocks materialisation."""

    def _ensure_clone(self) -> Path:
        if self._clone_path is not None and self._clone_path.exists():
            return self._clone_path
        clone_dir = Path(tempfile.mkdtemp(prefix="troopai-skills-"))
        url = f"https://{self.host}/{self.repo}.git"
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    "--branch",
                    self.ref,
                    url,
                    str(clone_dir),
                ],
                check=True,
                capture_output=True,
                timeout=self._CLONE_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            shutil.rmtree(clone_dir, ignore_errors=True)
            from troopai.adk.exceptions.exceptions import SkillsConfigError

            raise SkillsConfigError(
                f"git clone {url} timed out after {self._CLONE_TIMEOUT_SECONDS:.0f}s",
            ) from exc
        except subprocess.CalledProcessError as exc:
            shutil.rmtree(clone_dir, ignore_errors=True)
            from troopai.adk.exceptions.exceptions import SkillsConfigError

            raise SkillsConfigError(
                f"git clone {url} failed: {exc.stderr.decode(errors='replace')[:300]}",
            ) from exc
        self._clone_path = clone_dir
        atexit.register(shutil.rmtree, str(clone_dir), True)
        return clone_dir

    def _effective_root(self) -> Path:
        clone = self._ensure_clone()
        return clone / self.subpath if self.subpath else clone

    def list_skill_metadata(self) -> list[SkillMetadata]:
        try:
            root = self._effective_root()
        except SkillsConfigError:
            return []
        if not root.exists() or not root.is_dir():
            return []
        result: list[SkillMetadata] = []
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue
            description = ""
            with contextlib.suppress(OSError):
                description = skill_md.read_text(encoding="utf-8")[:200]
            result.append(
                SkillMetadata(name=child.name, description=description, path=str(child)),
            )
        return result

    def load_skill_payload(self, skill_name: str) -> dict[str, bytes] | None:
        try:
            root = self._effective_root() / skill_name
        except SkillsConfigError:
            return None
        if not root.exists() or not root.is_dir():
            return None
        payload: dict[str, bytes] = {}
        for file in root.rglob("*"):
            if not file.is_file():
                continue
            rel = file.relative_to(root).as_posix()
            payload[rel] = file.read_bytes()
        return payload


# ---------------------------------------------------------------------------
# SkillsCapability
# ---------------------------------------------------------------------------


class SkillsCapability(SandboxCapability):
    """Capability that materializes skills into the sandbox workspace.

    Exactly one of ``skills``, ``from_``, or ``lazy_from`` must be
    set (validated at construction).

    Attributes:
        type: Discriminator. Always ``"skills"``.
        skills: Eagerly-loaded skill bundles materialized into the
            workspace at manifest-processing time. Mutually exclusive
            with ``from_`` and ``lazy_from``.
        from_: A single ``BaseEntry`` whose children are treated as
            skills. Mutually exclusive with ``skills`` and ``lazy_from``.
        lazy_from: A lazy skill source (local directory or Git repo).
            Skills are loaded on demand via the ``load_skill`` tool.
            Mutually exclusive with ``skills`` and ``from_``.
        skills_path: Workspace-relative root directory under which all
            skill subdirectories are placed (default ``".agents"``).
    """

    # extra="forbid" so an unknown key fails loudly instead of silently
    # producing an empty capability — e.g. the wire alias ``{"from": ...}``
    # (the field is the Python attribute ``from_``) or a misspelled kwarg.
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    type: Literal["skills"] = "skills"

    skills: list[Skill] = Field(default_factory=list)
    from_: BaseEntry | None = None
    lazy_from: LocalDirLazySkillSource | GitRepoLazySkillSource | None = None
    skills_path: str = ".agents"

    _metadata_cache: list[SkillMetadata] | None = PrivateAttr(default=None)
    _metadata_cache_key: tuple[Any, ...] | None = PrivateAttr(default=None)

    @override
    def model_post_init(self, context: Any) -> None:
        del context
        sources = [
            ("skills", len(self.skills) > 0),
            ("from_", self.from_ is not None),
            ("lazy_from", self.lazy_from is not None),
        ]
        active = [name for name, present in sources if present]
        if len(active) == 0:
            raise SkillsConfigError("SkillsCapability requires exactly one of skills / from_ / lazy_from; got none.")
        if len(active) > 1:
            raise SkillsConfigError(
                f"SkillsCapability accepts exactly one of skills / from / lazy_from, got: {active}",
            )
        if len(self.skills_path) == 0:
            raise SkillsConfigError("SkillsCapability.skills_path must be non-empty")
        if self.skills_path.startswith("/"):
            raise SkillsConfigError(
                f"SkillsCapability.skills_path must be workspace-relative, got absolute: {self.skills_path!r}",
            )

    @override
    def bind(self, session: Any) -> None:
        super().bind(session)
        # Invalidate cache on rebind so per-run sessions see fresh metadata.
        self._metadata_cache = None
        self._metadata_cache_key = None

    @override
    def process_manifest(self, manifest: Manifest) -> Manifest:
        """Materialize skill directories into the manifest.

        For inline ``skills``: every Skill becomes a Dir entry under
        ``{skills_path}/{skill.name}`` with SKILL.md + scripts/ +
        references/ + assets/ children.

        For ``from_``: the source ``Dir``'s children are spliced in as
        the skills root's children (each child is treated as a skill).

        For ``lazy_from``: reserves the ``skills_path`` namespace as
        an empty Dir entry (LoadSkillTool populates it on demand).

        Raises ``SkillsConfigError`` if the ``skills_path`` overlaps
        an existing manifest entry, or if ``from_`` is not a ``Dir``.
        """
        from troopai.adk.types.sandbox.entries import Dir, File
        from troopai.adk.types.sandbox.manifest import Manifest as _Manifest

        if self.skills_path in manifest.entries:
            raise SkillsConfigError(
                f"SkillsCapability.skills_path {self.skills_path!r} overlaps an existing manifest entry",
            )
        # Build the skills root Dir.
        children: dict[str, BaseEntry] = {}
        if len(self.skills) > 0:
            for skill in self.skills:
                if skill.deferred:
                    continue
                skill_children: dict[str, BaseEntry] = {}
                content_bytes = skill.content.encode("utf-8") if isinstance(skill.content, str) else skill.content
                skill_children["SKILL.md"] = File(content=content_bytes)
                for sub, mapping in (
                    ("scripts", skill.scripts),
                    ("references", skill.references),
                    ("assets", skill.assets),
                ):
                    if len(mapping) == 0:
                        continue
                    sub_children: dict[str, BaseEntry] = {}
                    for filename, body in mapping.items():
                        body_bytes = body.encode("utf-8") if isinstance(body, str) else body
                        sub_children[filename] = File(content=body_bytes)
                    skill_children[sub] = Dir(children=sub_children)
                children[skill.name] = Dir(children=skill_children)
        elif self.from_ is not None:
            # The from_ entry's children ARE the skills; splice them into the
            # skills root. A non-Dir source has no children to contribute, so
            # it is a misconfiguration rather than a silently-empty directory.
            if not isinstance(self.from_, Dir):
                raise SkillsConfigError(
                    f"SkillsCapability.from_ must be a Dir whose children are skills, got {type(self.from_).__name__}",
                )
            children = dict(self.from_.children)
        elif self.lazy_from is not None:
            # Reserve namespace; LoadSkillTool fills it on demand.
            pass
        new_entries: dict[str, BaseEntry] = dict(manifest.entries)
        new_entries[self.skills_path] = Dir(children=children)
        return _Manifest(
            root=manifest.root,
            entries=new_entries,
            environment=manifest.environment,
            users=manifest.users,
            groups=manifest.groups,
            extra_path_grants=manifest.extra_path_grants,
            remote_mount_command_allowlist=manifest.remote_mount_command_allowlist,
        )

    @override
    def tools(self) -> list[FunctionTool]:
        if self.lazy_from is None or self.session is None:
            return []
        from troopai.adk.sandbox.tools.load_skill_tool import make_load_skill_tool

        return [make_load_skill_tool(loader=self)]

    async def load_skill(self, skill_name: str) -> dict[str, str]:
        """Materialize one lazy-source skill into the session workspace.

        Returns ``{"status": "loaded", "skill_name": ..., "path": ...,
        "files_written": N}`` on success; raises ``SkillsConfigError``
        if not configured for lazy loading or the skill is not found.
        """
        if self.lazy_from is None:
            raise SkillsConfigError("Skills.load_skill requires lazy_from to be set")
        if self.session is None:
            raise SkillsConfigError("Skills.load_skill requires a bound session")
        # ``skill_name`` is model-supplied; reject separators / ``..`` / absolute
        # paths / null bytes before any join so a crafted name cannot read host
        # files outside the lazy source or write outside ``skills_path``.
        _validate_skill_path_component(skill_name, "skill_name")
        payload = self.lazy_from.load_skill_payload(skill_name)
        if payload is None:
            raise SkillsConfigError(f"Skill {skill_name!r} not found in lazy source")
        target_root = f"{self.skills_path}/{skill_name}"
        files_written = 0
        for rel_path, data in payload.items():
            full_path = f"{target_root}/{rel_path}"
            await self.session.write(full_path, BytesIO(data))
            files_written += 1
        return {
            "status": "loaded",
            "skill_name": skill_name,
            "path": target_root,
            "files_written": str(files_written),
        }

    def _resolve_runtime_metadata(self) -> list[SkillMetadata]:
        """Return cached or freshly-resolved skill metadata."""
        cache_key: tuple[Any, ...] = (
            id(self.lazy_from),
            len(self.skills),
        )
        if self._metadata_cache is not None and self._metadata_cache_key == cache_key:
            return self._metadata_cache
        if self.lazy_from is not None:
            metadata = self.lazy_from.list_skill_metadata()
        elif len(self.skills) > 0:
            metadata = [SkillMetadata(name=s.name, description=s.description, path=s.name) for s in self.skills]
        else:
            metadata = []
        self._metadata_cache = metadata
        self._metadata_cache_key = cache_key
        return metadata

    @override
    async def instructions(self, manifest: Manifest | None) -> str | None:
        _ = manifest
        metadata = self._resolve_runtime_metadata()
        if len(metadata) == 0:
            return None
        lines = [f"Available skills (materialized under `{self.skills_path}/`):"]
        for md in metadata:
            lines.append(f"- {md.name}: {md.description.strip()[:120]}")
        if self.lazy_from is not None:
            lines.append("")
            lines.append(
                "Use the `load_skill(skill_name=...)` tool to load a specific "
                "skill's files into the workspace before reading them.",
            )
        return "\n".join(lines)
