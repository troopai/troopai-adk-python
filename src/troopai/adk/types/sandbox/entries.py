"""Manifest entries — declarative workspace artifacts.

Each entry describes a file, directory, or repository that should
appear inside a sandbox workspace when a fresh session starts. Entries
are workspace-relative; absolute paths and ``..`` traversals are
rejected at validation time.

Concrete subclasses are registered automatically via Pydantic's
``__pydantic_init_subclass__`` so that ``BaseEntry.parse({...})``
dispatches by ``type`` discriminator.

The materialization (actual copy, clone, write to disk) is performed
by the backend session inside ``apply_manifest()`` — entries here are
pure data.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path, PurePosixPath
from typing import Any, Literal, override

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import PydanticUndefined

from troopai.adk.types.sandbox.permissions import Group, Permissions, User

__all__ = [
    "BaseEntry",
    "Dir",
    "File",
    "GitRepo",
    "LocalDir",
    "LocalFile",
    "MaterializedFile",
]


_ENTRY_REGISTRY: dict[str, type[BaseEntry]] = {}


def _validate_relative_workspace_path(name: str, value: str | Path) -> str:
    """Reject absolute paths, Windows drive prefixes, and ``..`` traversals.

    Workspace entries must be portable across local, container, and
    hosted backends, so paths are always relative to the manifest root.
    The path is interpreted as POSIX regardless of host OS so that a
    manifest authored on Windows materializes identically inside a
    Linux container.
    """
    raw = str(value)
    if len(raw) == 0:
        raise ValueError(f"{name} must be a non-empty path")
    # Windows drive prefixes ("C:foo", "C:\\foo") are POSIX-absolute
    # only when fully qualified; `pathlib.PurePosixPath.is_absolute()`
    # treats them as relative. Reject them explicitly so a Windows
    # manifest cannot smuggle a host-rooted path into the sandbox.
    if len(raw) >= 2 and raw[1] == ":" and raw[0].isalpha():
        raise ValueError(f"{name} must be workspace-relative POSIX; got Windows drive path: {raw!r}")
    # Interpret as POSIX regardless of host OS: backslashes are separators
    # (a Windows-authored manifest), not literal filename characters. Without
    # this normalization, ``Path`` on a POSIX host treats ``a\\..\\b`` as a
    # single part, so the ``..`` traversal guard below never fires.
    p = PurePosixPath(raw.replace("\\", "/"))
    if p.is_absolute() or raw.startswith("/") or raw.startswith("\\"):
        raise ValueError(f"{name} must be workspace-relative, got absolute path: {raw!r}")
    parts = p.parts
    if ".." in parts:
        raise ValueError(f"{name} must not contain '..': {raw!r}")
    return p.as_posix()


@dataclasses.dataclass(frozen=True)
class MaterializedFile:
    """Result record describing one file materialized into a sandbox.

    Returned by backend ``apply_manifest`` calls so callers can audit
    exactly which files landed where, with which permissions.

    Attributes:
        path: Workspace-relative POSIX path of the materialized file.
        size_bytes: Final byte size of the file on the sandbox filesystem.
        permissions: Effective POSIX permission triplet applied.
        is_directory: True when the materialized entry is a directory.
    """

    path: str
    """Workspace-relative POSIX path of the materialized file."""

    size_bytes: int
    """Final byte size of the file on the sandbox filesystem."""

    permissions: Permissions
    """Effective POSIX permission triplet applied."""

    is_directory: bool = False
    """True when the materialized entry is a directory."""


class BaseEntry(BaseModel):
    """Abstract base for every manifest entry.

    Subclasses MUST set ``type`` to a unique ``Literal[...]`` string;
    that string becomes the discriminator used by ``parse`` /
    ``parse_many`` to reconstruct the correct subclass from JSON.

    The framework reserves ``type`` strings prefixed with ``"sandbox_"``;
    third-party entries should use a vendor-qualified prefix.

    Attributes:
        type: Discriminator string for subclass dispatch.
        description: Optional human-readable description; surfaced in
            tracing spans and audit events.
        ephemeral: When True, this entry is treated as transient — it
            is materialized into a fresh session but never copied into
            snapshots that persist workspace state.
        group: Optional owning user or group for POSIX-style filesystem
            permissions on backends that honor them.
        permissions: POSIX permission triplet applied at materialization.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str
    """Discriminator string for subclass dispatch."""

    description: str | None = None
    """Optional human-readable description surfaced in tracing + audit."""

    ephemeral: bool = False
    """When True, never copied into durable workspace snapshots."""

    group: User | Group | None = None
    """Owning user or group for POSIX-style permissions when supported."""

    permissions: Permissions = Field(default_factory=Permissions)
    """POSIX permission triplet applied at materialization.

    Defaults to ``Permissions()`` (``rw-r--r--``): developers opt INTO
    executable / world-traversable bits on directory-bearing entries
    (``Dir``, ``LocalDir``, ``GitRepo``) when needed, rather than
    opting out from a permissive default.
    """

    @classmethod
    @override
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        type_field = cls.model_fields.get("type")
        if type_field is None:
            raise TypeError(f"BaseEntry subclass {cls.__name__!r} must declare a 'type' field")
        default = type_field.default
        # An intermediate abstract subclass (e.g. ``Mount``) inherits
        # ``type: str`` without overriding it. Pydantic stores
        # ``PydanticUndefined`` as the field default in that case;
        # skip registration silently — concrete subclasses below it
        # will register themselves.
        if default is PydanticUndefined:
            return
        if default is None or default == "":
            raise TypeError(
                f"BaseEntry subclass {cls.__name__!r} must set a non-empty "
                f"'type' default (e.g. type: Literal['my_entry'] = 'my_entry')"
            )
        if not isinstance(default, str):
            raise TypeError(
                f"BaseEntry subclass {cls.__name__!r} 'type' default must be a "
                f"str literal, got {type(default).__name__}"
            )
        if default in _ENTRY_REGISTRY and _ENTRY_REGISTRY[default] is not cls:
            raise TypeError(
                f"BaseEntry discriminator {default!r} already registered for {_ENTRY_REGISTRY[default].__name__!r}"
            )
        _ENTRY_REGISTRY[default] = cls

    @classmethod
    def parse(cls, payload: dict[str, Any]) -> BaseEntry:
        """Reconstruct a subclass instance from a JSON-shaped dict.

        Uses the ``type`` discriminator to look up the registered
        subclass; raises ``ValueError`` if the discriminator is missing
        or unknown.
        """
        if "type" not in payload:
            raise ValueError("BaseEntry payload missing required 'type' discriminator")
        discriminator = payload["type"]
        impl = _ENTRY_REGISTRY.get(discriminator)
        if impl is None:
            raise ValueError(f"Unknown BaseEntry discriminator: {discriminator!r}")
        return impl.model_validate(payload)

    @classmethod
    def known_types(cls) -> tuple[str, ...]:
        """Return every registered discriminator string (test/diagnostics)."""
        return tuple(sorted(_ENTRY_REGISTRY.keys()))

    def is_dir(self) -> bool:
        """Return True when this entry represents a directory.

        Default is False; ``Dir`` / ``LocalDir`` override.
        """
        return False


class File(BaseEntry):
    """Synthetic file written into the workspace at materialization.

    Use for small inline content (config files, fixtures, helper
    snippets). For host files, use ``LocalFile`` instead.

    Attributes:
        content: Raw bytes written to the file.
    """

    type: Literal["file"] = "file"
    """Discriminator. Always ``"file"``."""

    content: bytes = b""
    """Raw bytes written to the file at materialization."""

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_content(cls, value: object) -> bytes:
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        if isinstance(value, bytearray):
            return bytes(value)
        raise TypeError(f"File.content must be bytes or str, got {type(value).__name__}")


class Dir(BaseEntry):
    """Synthetic directory; optionally pre-populated with child entries.

    Use for output directories the agent should write into, or for
    nested synthetic file trees. ``children`` keys are workspace-relative
    names of the immediate children (no ``/`` and no ``..``).

    Attributes:
        children: Mapping of child name → child entry.
    """

    type: Literal["dir"] = "dir"
    """Discriminator. Always ``"dir"``."""

    children: dict[str, BaseEntry] = Field(default_factory=dict)
    """Mapping of child name → child entry. Keys are direct child names."""

    @field_validator("children", mode="before")
    @classmethod
    def _parse_children(cls, value: object) -> dict[str, BaseEntry]:
        if not isinstance(value, dict):
            raise TypeError("Dir.children must be a dict")
        parsed: dict[str, BaseEntry] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if len(key) == 0:
                raise ValueError("Dir.children keys must be non-empty")
            # Reject every form that could break out of a single directory
            # name: path separators (both flavors), parent/current traversal,
            # and embedded null bytes (classic path-injection surface).
            if "/" in key or "\\" in key or "\x00" in key or key in {".", ".."}:
                raise ValueError(f"Dir.children keys must be simple directory names, got {key!r}")
            if isinstance(raw_value, BaseEntry):
                parsed[key] = raw_value
            elif isinstance(raw_value, dict):
                parsed[key] = BaseEntry.parse(raw_value)
            else:
                raise TypeError(f"Dir.children values must be BaseEntry or dict, got {type(raw_value).__name__}")
        return parsed

    @override
    def is_dir(self) -> bool:
        return True


class LocalFile(BaseEntry):
    """File copied from the host filesystem into the workspace.

    ``src`` is resolved against the host process working directory and
    MUST stay under that base unless covered by a
    ``SandboxPathGrant`` declared on the manifest.

    Attributes:
        src: Host filesystem path to the source file.
    """

    type: Literal["local_file"] = "local_file"
    """Discriminator. Always ``"local_file"``."""

    src: Path
    """Host filesystem path to the source file."""

    @field_validator("src", mode="before")
    @classmethod
    def _coerce_src(cls, value: object) -> Path:
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            if len(value) == 0:
                raise ValueError("LocalFile.src must be non-empty")
            return Path(value)
        raise TypeError(f"LocalFile.src must be str or Path, got {type(value).__name__}")


class LocalDir(BaseEntry):
    """Directory copied from the host filesystem into the workspace.

    Recursive copy; symlink behavior is backend-defined (Unix-local
    follows by default; container backends typically resolve at copy
    time). Source MUST stay under the host process working directory
    unless covered by a ``SandboxPathGrant``.

    Attributes:
        src: Host filesystem path to the source directory.
        follow_symlinks: When False, symlinks are not followed during copy.
    """

    type: Literal["local_dir"] = "local_dir"
    """Discriminator. Always ``"local_dir"``."""

    src: Path | None = None
    """Host filesystem path to the source directory (None ⇒ create empty)."""

    follow_symlinks: bool = True
    """When False, symlinks are not followed during the recursive copy."""

    @field_validator("src", mode="before")
    @classmethod
    def _coerce_src(cls, value: object) -> Path | None:
        if value is None:
            return None
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            if len(value) == 0:
                return None
            return Path(value)
        raise TypeError(f"LocalDir.src must be str, Path, or None, got {type(value).__name__}")

    @override
    def is_dir(self) -> bool:
        return True


class GitRepo(BaseEntry):
    """Git repository cloned into the workspace at materialization.

    Backends are expected to use the git CLI inside the sandbox image;
    fetch depth is shallow by default to minimize materialization cost.

    Attributes:
        host: Git host hostname.
        repo: Repository identifier in ``owner/name`` form.
        ref: Branch, tag, or commit SHA to check out.
        subpath: Optional relative path inside the repo to materialize.
        depth: Clone depth. ``1`` for a shallow clone (cost-conservative
            default); set to ``None`` for a full clone when history is
            needed.
    """

    type: Literal["git_repo"] = "git_repo"
    """Discriminator. Always ``"git_repo"``."""

    host: str = "github.com"
    """Git host hostname."""

    repo: str
    """Repository identifier in ``owner/name`` form."""

    ref: str = "main"
    """Branch, tag, or commit SHA to check out."""

    subpath: str | None = None
    """Optional relative path inside the repo to materialize."""

    depth: int | None = 1
    """Clone depth. ``1`` = shallow (default); ``None`` = full history."""

    @field_validator("host")
    @classmethod
    def _validate_host(cls, value: str) -> str:
        if len(value) == 0:
            raise ValueError("GitRepo.host must be non-empty")
        if value.startswith(("http://", "https://", "git://", "ssh://")):
            raise ValueError(f"GitRepo.host is a hostname only; protocol prefixes must be stripped (got {value!r})")
        if "/" in value:
            raise ValueError(f"GitRepo.host may not contain '/' (got {value!r}); use repo='owner/name'")
        return value

    @field_validator("repo")
    @classmethod
    def _validate_repo(cls, value: str) -> str:
        if "/" not in value or value.count("/") != 1:
            raise ValueError(f"GitRepo.repo must be 'owner/name', got {value!r}")
        owner, name = value.split("/", 1)
        if len(owner) == 0 or len(name) == 0:
            raise ValueError(f"GitRepo.repo owner and name must both be non-empty, got {value!r}")
        return value

    @field_validator("subpath")
    @classmethod
    def _validate_subpath(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_relative_workspace_path("GitRepo.subpath", value)

    @field_validator("depth")
    @classmethod
    def _validate_depth(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError(f"GitRepo.depth must be positive or None, got {value}")
        return value

    @override
    def is_dir(self) -> bool:
        return True
