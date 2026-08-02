"""Regression tests for P01 review-gate fail-loud fixes.

Each test pins one fix surfaced by the §6c review pass: silent failures
the audit identified were converted to typed exceptions, and the
corresponding tests live here to prevent regression.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from troopai.adk.types.sandbox.entries import BaseEntry, Dir, File, GitRepo
from troopai.adk.types.sandbox.manifest import Environment
from troopai.adk.types.sandbox.mounts import (
    InContainerMountStrategy,
    RcloneMountPattern,
    S3Mount,
)
from troopai.adk.types.sandbox.permissions import FileMode, Permissions


class TestBaseEntryFailLoudSubclassRegistration:
    """The audit found silent-skip branches in __pydantic_init_subclass__."""

    def test_subclass_with_empty_type_default_raises(self) -> None:
        with pytest.raises(TypeError, match="non-empty"):

            class _BadEntry(BaseEntry):
                type: str = ""

    def test_subclass_with_non_str_type_default_raises(self) -> None:
        with pytest.raises(TypeError, match="str literal"):

            class _BadEntry(BaseEntry):
                type: int = 1  # type: ignore[assignment]


class TestPermissionsFailLoud:
    """Audit findings: bool accepted as int; setuid silently truncated."""

    def test_bool_rejected_for_bits(self) -> None:
        with pytest.raises(TypeError, match="must not be bool"):
            Permissions(owner=True)  # type: ignore[arg-type]

    def test_from_mode_rejects_setuid(self) -> None:
        with pytest.raises(ValueError, match="setuid/setgid/sticky"):
            Permissions.from_mode(0o4755)

    def test_from_str_rejects_non_positional(self) -> None:
        # "rwwr--r--" has 'w' in slot 0 of owner triplet, which is the
        # READ slot — should reject, not silently OR-accumulate.
        with pytest.raises(ValueError, match="invalid char"):
            Permissions.from_str("rwwr--r--")

    def test_from_str_accepts_valid_positional(self) -> None:
        p = Permissions.from_str("rwxr-xr--")
        assert p.owner == FileMode.ALL
        assert p.group == FileMode.READ | FileMode.EXEC
        assert p.other == FileMode.READ


class TestEntryDefaultIsConservative:
    """The audit flagged BaseEntry's permissions default as too permissive."""

    def test_file_default_is_rw_for_owner_r_for_others(self) -> None:
        f = File()
        # rw-r--r-- (0o644) — no executable bit anywhere by default.
        assert f.permissions.to_mode() == 0o644

    def test_executable_bit_must_be_opted_in(self) -> None:
        f = File(
            permissions=Permissions(
                owner=FileMode.READ | FileMode.WRITE | FileMode.EXEC,
                group=FileMode.READ,
                other=FileMode.READ,
            )
        )
        assert (f.permissions.owner & FileMode.EXEC) == FileMode.EXEC


class TestMountPathRejectsAbsolute:
    """The audit found `Mount._validate_mount_path` skipped `..` on absolute paths."""

    def test_absolute_mount_path_rejected(self) -> None:
        strategy = InContainerMountStrategy(
            pattern=RcloneMountPattern(remote_name="r"),
        )
        with pytest.raises(ValueError, match="workspace-relative"):
            S3Mount(
                bucket="b",
                mount_path="/etc",
                mount_strategy=strategy,
            )

    def test_absolute_mount_path_with_dot_dot_rejected_at_absolute_check(self) -> None:
        strategy = InContainerMountStrategy(
            pattern=RcloneMountPattern(remote_name="r"),
        )
        with pytest.raises(ValueError, match="workspace-relative"):
            S3Mount(
                bucket="b",
                mount_path="/workspace/../etc",
                mount_strategy=strategy,
            )


class TestGitRepoHostValidation:
    """The audit flagged missing validator on GitRepo.host."""

    def test_empty_host_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            GitRepo(repo="o/r", host="")

    def test_url_prefix_rejected(self) -> None:
        with pytest.raises(ValueError, match="protocol prefixes"):
            GitRepo(repo="o/r", host="https://github.com")

    def test_slash_in_host_rejected(self) -> None:
        with pytest.raises(ValueError, match="may not contain '/'"):
            GitRepo(repo="o/r", host="github.com/extra")


class TestDirRejectsDangerousKeys:
    """The audit flagged missing rejection of \\0 and \\\\."""

    def test_null_byte_rejected(self) -> None:
        with pytest.raises(ValueError, match="simple directory names"):
            Dir(children={"file\x00name": File()})

    def test_backslash_rejected(self) -> None:
        with pytest.raises(ValueError, match="simple directory names"):
            Dir(children={"sub\\dir": File()})

    def test_empty_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Dir(children={"": File()})


class TestEnvironmentRejectsUnknownEntryType:
    """The audit found the dict path silently accepted unknown EnvEntry types."""

    def test_unknown_entry_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown EnvEntry type"):
            Environment(
                variables={"X": {"type": "vault_secret", "secret_id": "abc"}},  # type: ignore[dict-item]
            )

    def test_str_entry_type_accepted(self) -> None:
        e = Environment(variables={"X": {"type": "str", "value": "y"}})  # type: ignore[dict-item]
        assert e.variables["X"].type == "str"  # type: ignore[union-attr]

    def test_unknown_non_dict_type_rejected(self) -> None:
        # A BaseModel that is NOT an EnvEntry: should be rejected.
        class _NotAnEntry(BaseModel):
            x: int

        with pytest.raises(TypeError, match="must be str or EnvEntry"):
            Environment(variables={"X": _NotAnEntry(x=1)})  # type: ignore[dict-item]
