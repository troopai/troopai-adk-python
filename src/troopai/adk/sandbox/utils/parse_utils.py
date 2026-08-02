"""Parse ``ls -la`` output into structured entries.

Sandboxes that drive ``ls`` via a shell command (Docker exec, K8s
exec stream, remote VM REST) receive raw stdout; this module turns
that text back into typed records the higher layers can act on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from troopai.adk.types.sandbox.permissions import Permissions

logger = logging.getLogger(__name__)

__all__ = ["EntryKind", "LsLaEntry", "parse_ls_la"]


class EntryKind(StrEnum):
    """High-level type of an ``ls -la`` row."""

    DIRECTORY = "directory"
    """Directory entry (``d`` prefix in the permissions column)."""

    FILE = "file"
    """Regular file (``-`` prefix)."""

    SYMLINK = "symlink"
    """Symbolic link (``l`` prefix)."""

    OTHER = "other"
    """Any other type (``c``, ``b``, ``p``, ``s``)."""


@dataclass(frozen=True, kw_only=True)
class LsLaEntry:
    """One parsed row of ``ls -la`` output.

    Attributes:
        path: Absolute or workspace-relative path. The parser
            joins the ``base`` argument with the bare filename
            unless the filename is already absolute.
        permissions: Owner/group/other RWX triplet.
        owner: Owner username.
        group: Owning group name.
        size: Reported byte size (note: the value ``ls`` reports for
            symlinks is the target-name length, not the link-target
            file size).
        kind: Discriminator — directory / file / symlink / other.
    """

    path: str
    """Absolute or workspace-relative path."""

    permissions: Permissions
    """Owner/group/other RWX triplet."""

    owner: str
    """Owner username."""

    group: str
    """Owning group name."""

    size: int
    """Reported byte size."""

    kind: EntryKind = EntryKind.FILE
    """Discriminator — directory / file / symlink / other."""

    @property
    def is_directory(self) -> bool:
        """True iff ``kind == EntryKind.DIRECTORY``."""
        return self.kind == EntryKind.DIRECTORY


_KIND_MAP: dict[str, EntryKind] = {
    "d": EntryKind.DIRECTORY,
    "-": EntryKind.FILE,
    "l": EntryKind.SYMLINK,
}


def parse_ls_la(output: str, *, base: str) -> list[LsLaEntry]:
    """Parse coreutils ``ls -la`` output into typed entries.

    Skips: the ``total`` header line, the ``.`` / ``..`` entries,
    and lines that don't have at least 9 whitespace-delimited
    columns or whose 5th column isn't an integer.

    Args:
        output: Raw stdout from ``ls -la <path>``.
        base: Directory the listing came from; joined with each row's
            filename to produce the entry path. Pass ``/`` for the
            workspace root.

    Returns:
        List of ``LsLaEntry`` records in the same order as the input.
    """
    entries: list[LsLaEntry] = []
    skipped = 0
    for raw_line in output.splitlines():
        line = raw_line.strip("\n")
        if len(line) == 0 or line.startswith("total"):
            continue

        parts = line.split(maxsplit=8)
        if len(parts) < 9:
            logger.debug("parse_ls_la: skip line with <9 columns: %r", line)
            skipped += 1
            continue

        permissions_str = parts[0]
        owner = parts[2]
        group = parts[3]
        try:
            size = int(parts[4])
        except ValueError:
            logger.debug("parse_ls_la: skip line with non-integer size: %r", line)
            skipped += 1
            continue

        kind = _KIND_MAP.get(permissions_str[:1], EntryKind.OTHER)

        # `Permissions.from_str` requires a 9-char triplet without
        # the leading type marker (which our model carries via the
        # `directory` flag, not the RWX bits).
        triplet = permissions_str[1:10]
        if len(triplet) != 9:
            logger.debug("parse_ls_la: skip line with malformed permissions %r: %r", permissions_str, line)
            skipped += 1
            continue
        try:
            permissions = Permissions.from_str(triplet, directory=(kind == EntryKind.DIRECTORY))
        except ValueError as exc:
            logger.debug("parse_ls_la: skip unparseable permissions %r: %s", permissions_str, exc)
            skipped += 1
            continue

        name = parts[8]
        if kind == EntryKind.SYMLINK and " -> " in name:
            name = name.split(" -> ", 1)[0]

        if name in {".", ".."}:
            continue

        if name.startswith("/"):
            entry_path = name
        elif base == "/":
            entry_path = f"/{name}"
        else:
            entry_path = f"{base.rstrip('/')}/{name}"

        entries.append(
            LsLaEntry(
                path=entry_path,
                permissions=permissions,
                owner=owner,
                group=group,
                size=size,
                kind=kind,
            )
        )

    return entries
