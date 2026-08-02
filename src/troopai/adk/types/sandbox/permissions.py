"""POSIX-style permissions, users, and groups for sandbox manifests.

These types model filesystem-level identity and access semantics that
materialization applies inside the sandbox. They are NOT sandbox-side
RBAC, approval policy, or API credential models — those belong in
guardrails and ``run_config`` respectively.

``Permissions`` carries owner/group/other triplets; ``FileMode`` is the
classic POSIX bitmask. ``User`` and ``Group`` are hashable identity
records so a manifest can declare them once and reference them from
each entry.
"""

from __future__ import annotations

from enum import IntFlag
from typing import ClassVar, override

from pydantic import BaseModel, field_validator

__all__ = [
    "FileMode",
    "Group",
    "Permissions",
    "User",
]


class FileMode(IntFlag):
    """POSIX read/write/execute bits.

    Values mirror the classic ``r=4``, ``w=2``, ``x=1`` encoding so
    ``FileMode.READ | FileMode.WRITE`` arithmetic works as expected.
    """

    NONE = 0
    """No permissions."""

    EXEC = 1
    """Execute / list."""

    WRITE = 2
    """Write."""

    READ = 4
    """Read."""

    ALL = 7
    """Read + write + execute."""


_MODE_POSITION_EXPECTED: tuple[tuple[str, FileMode], ...] = (
    ("r", FileMode.READ),
    ("w", FileMode.WRITE),
    ("x", FileMode.EXEC),
)


class User(BaseModel):
    """Sandbox-local user identity.

    A manifest declares users so that entries can reference them via
    ``group=`` and so the agent can run as them via
    ``SandboxAgent.run_as``. Backends that support OS-level account
    provisioning (Docker, K8s, hosted) honor the declaration; backends
    that don't (local subprocess) ignore it.

    Attributes:
        name: Username. Must be a non-empty alphanumeric (plus ``-``
            and ``_``) string. Case-sensitive.
    """

    name: str
    """Sandbox-local username."""

    model_config: ClassVar = {"frozen": True}

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if len(value) == 0:
            raise ValueError("User.name must be non-empty")
        for ch in value:
            if not (ch.isalnum() or ch in {"-", "_"}):
                raise ValueError(f"User.name may contain only alphanumerics, '-', '_'; got {value!r}")
        return value

    @override
    def __hash__(self) -> int:
        return hash(("user", self.name))


class Group(BaseModel):
    """Sandbox-local group identity.

    Attributes:
        name: Group name (same validation as ``User.name``).
        users: Member users referenced by name.
    """

    name: str
    """Group name."""

    users: tuple[User, ...] = ()
    """Member users referenced by name."""

    model_config: ClassVar = {"frozen": True}

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if len(value) == 0:
            raise ValueError("Group.name must be non-empty")
        for ch in value:
            if not (ch.isalnum() or ch in {"-", "_"}):
                raise ValueError(f"Group.name may contain only alphanumerics, '-', '_'; got {value!r}")
        return value

    @override
    def __hash__(self) -> int:
        return hash(("group", self.name, tuple(u.name for u in self.users)))


class Permissions(BaseModel):
    """POSIX permission triplet (owner / group / other).

    Defaults to ``rw-r--r--`` for files (``directory=False``); set
    ``directory=True`` for entries representing directories so the
    executable bit on owner/group is interpreted as traverse.

    Attributes:
        owner: Bits applied to the owner.
        group: Bits applied to members of the owning group.
        other: Bits applied to everyone else.
        directory: True when the entry is a directory.
    """

    owner: FileMode = FileMode.READ | FileMode.WRITE
    """Bits applied to the owner."""

    group: FileMode = FileMode.READ
    """Bits applied to members of the owning group."""

    other: FileMode = FileMode.READ
    """Bits applied to everyone else."""

    directory: bool = False
    """True when the entry is a directory."""

    @field_validator("owner", "group", "other", mode="before")
    @classmethod
    def _coerce_mode(cls, value: object) -> FileMode:
        if isinstance(value, FileMode):
            return value
        # `bool` is a subclass of `int`; reject explicitly so a stray
        # `Permissions(owner=True)` does not silently become `EXEC`.
        if isinstance(value, bool):
            raise TypeError(f"Permissions bits must not be bool, got {value!r}")
        if isinstance(value, int):
            if value < 0 or value > 7:
                raise ValueError(f"Permissions bits must be in 0..7, got {value}")
            return FileMode(value)
        raise TypeError(f"Permissions bits must be FileMode or int 0..7, got {type(value).__name__}")

    def to_mode(self) -> int:
        """Return the classic POSIX numeric mode (e.g. ``0o755``)."""
        return (int(self.owner) << 6) | (int(self.group) << 3) | int(self.other)

    def owner_can(self, mode: FileMode) -> bool:
        """Return True iff the owner has all the bits in ``mode``."""
        return (int(self.owner) & int(mode)) == int(mode)

    def group_can(self, mode: FileMode) -> bool:
        """Return True iff group members have all the bits in ``mode``."""
        return (int(self.group) & int(mode)) == int(mode)

    def other_can(self, mode: FileMode) -> bool:
        """Return True iff others have all the bits in ``mode``."""
        return (int(self.other) & int(mode)) == int(mode)

    @classmethod
    def from_mode(cls, mode: int, *, directory: bool = False) -> Permissions:
        """Construct from a classic POSIX numeric mode (e.g. ``0o755``).

        Accepts only the low 9 bits; setuid/setgid/sticky (bits above
        ``0o777``) are rejected — those are backend-specific and must
        be set explicitly via a typed surface rather than silently
        truncated here.
        """
        if mode < 0:
            raise ValueError(f"Permissions.from_mode requires non-negative, got {mode}")
        if mode > 0o777:
            raise ValueError(f"Permissions.from_mode does not support setuid/setgid/sticky bits; got 0o{mode:o}")
        owner = FileMode((mode >> 6) & 0o7)
        group = FileMode((mode >> 3) & 0o7)
        other = FileMode(mode & 0o7)
        return cls(owner=owner, group=group, other=other, directory=directory)

    @classmethod
    def from_str(cls, text: str, *, directory: bool = False) -> Permissions:
        """Parse a strict 9-char ``ls -l`` triplet such as ``"rwxr-xr--"``.

        Each triplet is positional: char 0 ∈ {``r``,``-``}, char 1 ∈
        {``w``,``-``}, char 2 ∈ {``x``,``-``}. Non-positional strings
        (e.g. ``"rwwr--r--"``) are rejected rather than silently
        OR-accumulated.
        """
        if len(text) != 9:
            raise ValueError(f"Permissions.from_str expects 9 chars, got {len(text)}: {text!r}")
        triplets: list[FileMode] = []
        for offset in (0, 3, 6):
            chunk = text[offset : offset + 3]
            bits = FileMode.NONE
            for index, ch in enumerate(chunk):
                expected_char, expected_bit = _MODE_POSITION_EXPECTED[index]
                if ch == "-":
                    continue
                if ch != expected_char:
                    raise ValueError(
                        f"Permissions.from_str invalid char {ch!r} at position "
                        f"{offset + index} in {text!r}; expected {expected_char!r} "
                        f"or '-'"
                    )
                bits |= expected_bit
            triplets.append(bits)
        return cls(
            owner=triplets[0],
            group=triplets[1],
            other=triplets[2],
            directory=directory,
        )
