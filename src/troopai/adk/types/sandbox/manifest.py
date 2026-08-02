"""Manifest — declarative workspace contract for a fresh sandbox.

A ``Manifest`` describes what a fresh sandbox workspace should contain
when a new session starts: files and directories, mounted storage,
environment variables, users and groups, and any absolute path grants
outside the workspace root.

Manifests are workspace-relative-by-default: entry keys are paths
under ``root``, and absolute paths / ``..`` traversals are rejected at
validation time. The contract is portable across local, Docker, and
hosted backends.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any, Literal, override

from pydantic import BaseModel, ConfigDict, Field, field_validator

from troopai.adk.types.sandbox.entries import BaseEntry, Dir
from troopai.adk.types.sandbox.permissions import Group, User
from troopai.adk.types.sandbox.workspace_paths import SandboxPathGrant

__all__ = [
    "EnvEntry",
    "Environment",
    "Manifest",
    "StrEnvValue",
]


_DEFAULT_REMOTE_MOUNT_COMMAND_ALLOWLIST: tuple[str, ...] = (
    "ls",
    "cat",
    "head",
    "tail",
    "find",
    "stat",
    "file",
    "rg",
    "grep",
    "wc",
    "du",
    "df",
)


class EnvEntry(BaseModel):
    """Abstract base for environment-value entries that resolve at runtime.

    Concrete subclasses (e.g. ``StrEnvValue``) implement ``async resolve()``
    so vault / secret-store integrations can resolve to a real value
    just before the sandbox starts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str
    """Discriminator string for subclass dispatch."""

    async def resolve(self) -> str:
        """Resolve this entry to a literal string value.

        Concrete subclasses (e.g. :class:`StrEnvValue`, or a vault /
        secret-store integration) override this. The base raises so an entry
        type with no resolution logic fails loudly at sandbox start rather
        than silently injecting nothing.
        """
        raise NotImplementedError(f"EnvEntry of type {self.type!r} does not implement resolve()")


class StrEnvValue(EnvEntry):
    """Plain-string environment variable.

    Attributes:
        value: The literal value injected into the sandbox env.
    """

    type: Literal["str"] = "str"
    """Discriminator. Always ``"str"``."""

    value: str
    """The literal value injected into the sandbox env."""

    @override
    async def resolve(self) -> str:
        return self.value


class Environment(BaseModel):
    """Environment-variable bundle injected into a fresh sandbox.

    Values can be plain strings (for trivial cases) or ``EnvEntry``
    instances (for vault / secret-store integrations that resolve at
    sandbox-start time).

    Attributes:
        variables: Mapping of env-var name → value.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    variables: dict[str, str | EnvEntry] = Field(default_factory=dict)
    """Mapping of env-var name → value."""

    @field_validator("variables", mode="before")
    @classmethod
    def _coerce_variables(cls, value: object) -> dict[str, str | EnvEntry]:
        if not isinstance(value, dict):
            raise TypeError("Environment.variables must be a dict")
        out: dict[str, str | EnvEntry] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if len(key) == 0:
                raise ValueError("Environment.variables keys must be non-empty")
            if isinstance(raw_value, (str, EnvEntry)):
                out[key] = raw_value
            elif isinstance(raw_value, dict):
                # Strict dispatch: only known concrete EnvEntry types are
                # accepted. Validating an unknown discriminator against the
                # abstract `EnvEntry` would silently produce a degraded
                # instance with no `resolve()` implementation, surfacing
                # the bug only later when the sandbox starts. Fail loud here.
                entry_type = raw_value.get("type")
                if entry_type == "str":
                    out[key] = StrEnvValue.model_validate(raw_value)
                else:
                    raise ValueError(
                        f"Environment.variables[{key!r}] has unknown EnvEntry type {entry_type!r}; expected 'str'"
                    )
            else:
                raise TypeError(
                    f"Environment.variables[{key!r}] must be str or EnvEntry, got {type(raw_value).__name__}"
                )
        return out

    def normalized(self) -> dict[str, str]:
        """Return only the literal-string env vars; entries are skipped.

        Use ``resolve()`` when you need the full set including resolved
        ``EnvEntry`` values.
        """
        return {k: v for k, v in self.variables.items() if isinstance(v, str)}

    async def resolve(self) -> dict[str, str]:
        """Resolve every entry and return a flat ``str → str`` dict.

        Plain-string values pass through; every ``EnvEntry`` is resolved via
        its own ``resolve()`` so custom secret-store subclasses are honored.
        An ``EnvEntry`` subclass that does not implement ``resolve()`` raises
        (loud, not silent).
        """
        out: dict[str, str] = {}
        for k, v in self.variables.items():
            if isinstance(v, str):
                out[k] = v
            else:
                out[k] = await v.resolve()
        return out


class Manifest(BaseModel):
    """Declarative workspace contract for a fresh sandbox session.

    A manifest captures everything a backend needs to materialize a
    workspace before the agent starts: files, directories, repos,
    mounts, environment, users, groups, and absolute-path grants.

    Manifests describe the FRESH-SESSION contract only. Reused or
    resumed sessions keep their existing workspace state; the manifest
    is consulted only when the backend creates a new workspace.

    Attributes:
        root: Workspace root path inside the sandbox.
        entries: Workspace-relative entries to materialize.
        environment: Environment variables for the sandbox process.
        users: Sandbox-local user identities.
        groups: Sandbox-local group identities.
        extra_path_grants: Absolute paths outside ``root`` the sandbox
            may access.
        remote_mount_command_allowlist: Commands allowed against
            remote-mount paths (cost-conservative read-only defaults).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: str = "/workspace"
    """Workspace root path inside the sandbox."""

    entries: dict[str, BaseEntry] = Field(default_factory=dict)
    """Workspace-relative entries to materialize."""

    environment: Environment = Field(default_factory=Environment)
    """Environment variables for the sandbox process."""

    users: list[User] = Field(default_factory=list)
    """Sandbox-local user identities."""

    groups: list[Group] = Field(default_factory=list)
    """Sandbox-local group identities."""

    extra_path_grants: tuple[SandboxPathGrant, ...] = ()
    """Absolute paths outside ``root`` the sandbox may access."""

    remote_mount_command_allowlist: tuple[str, ...] = _DEFAULT_REMOTE_MOUNT_COMMAND_ALLOWLIST
    """Commands allowed against remote-mount paths."""

    @field_validator("root")
    @classmethod
    def _validate_root(cls, value: str) -> str:
        if len(value) == 0:
            raise ValueError("Manifest.root must be non-empty")
        if not value.startswith("/"):
            raise ValueError(f"Manifest.root must be an absolute POSIX path, got {value!r}")
        return value

    @field_validator("entries", mode="before")
    @classmethod
    def _parse_entries(cls, value: object) -> dict[str, BaseEntry]:
        if not isinstance(value, dict):
            raise TypeError("Manifest.entries must be a dict")
        out: dict[str, BaseEntry] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if len(key) == 0:
                raise ValueError("Manifest.entries keys must be non-empty")
            # Windows drive prefixes ("C:foo", "C:\\foo") are POSIX-absolute
            # only when fully qualified; `PurePosixPath.is_absolute()` treats
            # them as relative. Reject them explicitly so a Windows manifest
            # cannot smuggle a host-rooted path into the sandbox.
            if len(key) >= 2 and key[1] == ":" and key[0].isalpha():
                raise ValueError(
                    f"Manifest.entries keys must be workspace-relative POSIX; got Windows drive path: {key!r}"
                )
            # Interpret as POSIX regardless of host OS: backslashes are
            # separators (a Windows-authored manifest), not literal filename
            # characters. Without this normalization, `Path` on a POSIX host
            # treats `..\\..\\etc` as a single part, so the `..` traversal
            # guard below never fires and the key is stored verbatim.
            p = PurePosixPath(key.replace("\\", "/"))
            if p.is_absolute() or key.startswith("/") or key.startswith("\\"):
                raise ValueError(f"Manifest.entries keys must be workspace-relative, got absolute {key!r}")
            if ".." in p.parts:
                raise ValueError(f"Manifest.entries keys must not contain '..': {key!r}")
            normalized = p.as_posix()
            if isinstance(raw_value, BaseEntry):
                out[normalized] = raw_value
            elif isinstance(raw_value, dict):
                out[normalized] = BaseEntry.parse(raw_value)
            else:
                raise TypeError(f"Manifest.entries values must be BaseEntry or dict, got {type(raw_value).__name__}")
        return out

    def iter_entries(self) -> list[tuple[str, BaseEntry]]:
        """Depth-first traversal yielding ``(workspace_path, entry)`` pairs.

        Yields every DECLARED entry. ``Dir`` children (synthetic
        directories) are descended into recursively. ``LocalDir``,
        ``GitRepo`` and ``Mount`` entries are yielded as opaque
        directory entries — their contents are materialized by the
        backend at session start and are not introspectable here.
        """
        result: list[tuple[str, BaseEntry]] = []

        def _walk(prefix: str, entry: BaseEntry) -> None:
            result.append((prefix, entry))
            if isinstance(entry, Dir):
                for child_name, child in entry.children.items():
                    child_path = f"{prefix}/{child_name}" if prefix else child_name
                    _walk(child_path, child)

        for path, entry in self.entries.items():
            _walk(path, entry)
        return result

    @override
    def model_post_init(self, context: Any, /) -> None:
        # Pydantic post-init signature requires `context`; unused here.
        # Deduplicate user / group names so backends can rely on the
        # invariant when provisioning OS-level accounts.
        del context
        user_names = [u.name for u in self.users]
        if len(set(user_names)) != len(user_names):
            raise ValueError(f"Manifest.users contains duplicate names: {user_names!r}")
        group_names = [g.name for g in self.groups]
        if len(set(group_names)) != len(group_names):
            raise ValueError(f"Manifest.groups contains duplicate names: {group_names!r}")
