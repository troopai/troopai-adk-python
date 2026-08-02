"""Tests for ``troopai.adk.types.sandbox.permissions``."""

from __future__ import annotations

import pytest

from troopai.adk.types.sandbox.permissions import (
    FileMode,
    Group,
    Permissions,
    User,
)


class TestFileMode:
    def test_bitwise_or_combines_flags(self) -> None:
        rwx = FileMode.READ | FileMode.WRITE | FileMode.EXEC
        assert int(rwx) == 7
        assert rwx == FileMode.ALL

    def test_none_is_zero(self) -> None:
        assert int(FileMode.NONE) == 0

    def test_individual_bits(self) -> None:
        assert int(FileMode.READ) == 4
        assert int(FileMode.WRITE) == 2
        assert int(FileMode.EXEC) == 1


class TestPermissions:
    def test_default_is_rw_for_owner_r_for_others(self) -> None:
        p = Permissions()
        assert p.owner == FileMode.READ | FileMode.WRITE
        assert p.group == FileMode.READ
        assert p.other == FileMode.READ
        assert p.directory is False

    def test_to_mode_round_trip(self) -> None:
        p = Permissions(
            owner=FileMode.READ | FileMode.WRITE | FileMode.EXEC,
            group=FileMode.READ | FileMode.EXEC,
            other=FileMode.READ,
        )
        assert p.to_mode() == 0o754

    def test_from_mode_handles_0o755(self) -> None:
        p = Permissions.from_mode(0o755)
        assert p.owner == FileMode.ALL
        assert p.group == FileMode.READ | FileMode.EXEC
        assert p.other == FileMode.READ | FileMode.EXEC

    def test_from_str_rwxr_xr__(self) -> None:
        p = Permissions.from_str("rwxr-xr--")
        assert p.owner == FileMode.ALL
        assert p.group == FileMode.READ | FileMode.EXEC
        assert p.other == FileMode.READ

    def test_from_str_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="9 chars"):
            Permissions.from_str("rwx")

    def test_from_str_rejects_bad_char(self) -> None:
        with pytest.raises(ValueError, match="invalid char"):
            Permissions.from_str("rwzr-xr--")

    def test_owner_can_checks_subset(self) -> None:
        p = Permissions(owner=FileMode.READ | FileMode.WRITE)
        assert p.owner_can(FileMode.READ) is True
        assert p.owner_can(FileMode.WRITE) is True
        assert p.owner_can(FileMode.EXEC) is False

    def test_from_mode_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            Permissions.from_mode(-1)

    def test_int_coercion_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="0..7"):
            Permissions(owner=8)  # type: ignore[arg-type]


class TestUser:
    def test_name_validation(self) -> None:
        u = User(name="analyst")
        assert u.name == "analyst"

    def test_name_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            User(name="")

    def test_name_rejects_special_chars(self) -> None:
        with pytest.raises(ValueError, match="alphanumerics"):
            User(name="user/name")

    def test_user_is_hashable(self) -> None:
        u1 = User(name="alice")
        u2 = User(name="alice")
        u3 = User(name="bob")
        assert hash(u1) == hash(u2)
        assert hash(u1) != hash(u3)
        # Set membership works.
        assert len({u1, u2, u3}) == 2


class TestGroup:
    def test_group_with_users(self) -> None:
        alice = User(name="alice")
        bob = User(name="bob")
        g = Group(name="admins", users=[alice, bob])
        assert len(g.users) == 2

    def test_group_is_hashable(self) -> None:
        alice = User(name="alice")
        g1 = Group(name="admins", users=[alice])
        g2 = Group(name="admins", users=[alice])
        g3 = Group(name="admins", users=[])
        assert hash(g1) == hash(g2)
        assert hash(g1) != hash(g3)

    def test_users_is_immutable_tuple(self) -> None:
        # A list literal is coerced to an immutable tuple so the frozen
        # model has no in-place mutation escape hatch.
        g = Group(name="dev", users=[User(name="alice")])
        assert isinstance(g.users, tuple)
        with pytest.raises(AttributeError):
            g.users.append(User(name="bob"))  # type: ignore[attr-defined]

    def test_hash_stable_against_member_mutation(self) -> None:
        # The hash must remain stable for the lifetime of the object so a
        # Group placed in a set / used as a dict key stays findable.
        g = Group(name="dev", users=[User(name="alice")])
        bucket: set[Group] = {g}
        before = hash(g)
        # Any attempt to append a member raises rather than silently
        # changing the hash out from under the bucket.
        with pytest.raises(AttributeError):
            g.users.append(User(name="bob"))  # type: ignore[attr-defined]
        assert hash(g) == before
        assert g in bucket
