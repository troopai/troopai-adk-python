"""Tests for ``troopai.adk.types.sandbox.manifest``."""

from __future__ import annotations

import pytest

from troopai.adk.types.sandbox.entries import Dir, File
from troopai.adk.types.sandbox.manifest import (
    Environment,
    Manifest,
    StrEnvValue,
)
from troopai.adk.types.sandbox.permissions import Group, User
from troopai.adk.types.sandbox.workspace_paths import SandboxPathGrant


class TestManifestDefaults:
    def test_empty_manifest(self) -> None:
        m = Manifest()
        assert m.root == "/workspace"
        assert m.entries == {}
        assert m.users == []
        assert m.groups == []
        assert m.extra_path_grants == ()
        # Cost-conservative default: only read-only inspection commands.
        assert "rm" not in m.remote_mount_command_allowlist
        assert "ls" in m.remote_mount_command_allowlist

    def test_root_must_be_absolute(self) -> None:
        with pytest.raises(ValueError, match="absolute POSIX path"):
            Manifest(root="workspace")

    def test_root_must_be_non_empty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Manifest(root="")


class TestManifestEntries:
    def test_typed_entry_passed_through(self) -> None:
        m = Manifest(entries={"hello.txt": File(content=b"hi")})
        assert isinstance(m.entries["hello.txt"], File)

    def test_dict_entry_parsed_by_discriminator(self) -> None:
        m = Manifest(
            entries={"hello.txt": {"type": "file", "content": b"hi"}},  # type: ignore[dict-item]
        )
        assert isinstance(m.entries["hello.txt"], File)
        assert m.entries["hello.txt"].content == b"hi"

    def test_absolute_entry_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="workspace-relative"):
            Manifest(entries={"/etc/hosts": File()})

    def test_dot_dot_entry_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="must not contain '..'"):
            Manifest(entries={"../escape": File()})

    def test_empty_entry_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Manifest(entries={"": File()})

    @pytest.mark.parametrize(
        "key",
        [
            "..\\..\\etc\\passwd",
            "a\\..\\b",
            "..\\escape",
        ],
    )
    def test_backslash_dot_dot_entry_key_rejected(self, key: str) -> None:
        # Backslashes are POSIX-path separators for a Windows-authored
        # manifest; without normalization the host's `Path` treats the
        # whole key as one component and the `..` traversal guard never
        # fires, storing the escaping key verbatim.
        with pytest.raises(ValueError, match="must not contain '..'"):
            Manifest(entries={key: File()})

    @pytest.mark.parametrize("key", ["C:\\windows\\system32", "C:foo", "Z:/data"])
    def test_windows_drive_entry_key_rejected(self, key: str) -> None:
        with pytest.raises(ValueError, match="workspace-relative"):
            Manifest(entries={key: File()})

    @pytest.mark.parametrize("key", ["\\unc\\path", "\\escape"])
    def test_leading_backslash_entry_key_rejected(self, key: str) -> None:
        with pytest.raises(ValueError, match="workspace-relative"):
            Manifest(entries={key: File()})

    def test_backslash_separators_normalized_to_posix(self) -> None:
        # A valid Windows-authored key normalizes to a portable POSIX key
        # so it materializes identically across backends.
        m = Manifest(entries={"a\\b\\c.txt": File(content=b"hi")})
        assert "a/b/c.txt" in m.entries
        assert "a\\b\\c.txt" not in m.entries


class TestManifestDeduplication:
    def test_duplicate_user_names_rejected(self) -> None:
        alice = User(name="alice")
        alice2 = User(name="alice")
        with pytest.raises(ValueError, match="duplicate"):
            Manifest(users=[alice, alice2])

    def test_duplicate_group_names_rejected(self) -> None:
        admins1 = Group(name="admins")
        admins2 = Group(name="admins", users=[User(name="bob")])
        with pytest.raises(ValueError, match="duplicate"):
            Manifest(groups=[admins1, admins2])


class TestManifestIterEntries:
    def test_flat_entries(self) -> None:
        m = Manifest(entries={"a.txt": File(), "b.txt": File()})
        result = m.iter_entries()
        assert len(result) == 2
        paths = {p for p, _ in result}
        assert paths == {"a.txt", "b.txt"}

    def test_nested_dir_entries_yielded(self) -> None:
        m = Manifest(
            entries={
                "out": Dir(
                    children={
                        "report.md": File(content=b"# Report"),
                        "data": Dir(children={"x.csv": File()}),
                    }
                ),
            }
        )
        result = m.iter_entries()
        paths = {p for p, _ in result}
        assert "out" in paths
        assert "out/report.md" in paths
        assert "out/data" in paths
        assert "out/data/x.csv" in paths


class TestEnvironment:
    def test_default_empty(self) -> None:
        e = Environment()
        assert e.variables == {}

    def test_string_values_kept(self) -> None:
        e = Environment(variables={"FOO": "bar"})
        assert e.variables["FOO"] == "bar"

    def test_str_env_value_kept(self) -> None:
        e = Environment(variables={"FOO": StrEnvValue(value="bar")})
        v = e.variables["FOO"]
        assert isinstance(v, StrEnvValue)
        assert v.value == "bar"

    def test_normalized_returns_str_only(self) -> None:
        e = Environment(
            variables={
                "PLAIN": "x",
                "ENTRY": StrEnvValue(value="y"),
            },
        )
        n = e.normalized()
        assert n == {"PLAIN": "x"}

    async def test_resolve_yields_both(self) -> None:
        e = Environment(
            variables={
                "PLAIN": "x",
                "ENTRY": StrEnvValue(value="y"),
            },
        )
        resolved = await e.resolve()
        assert resolved == {"PLAIN": "x", "ENTRY": "y"}


class TestManifestExtraPathGrants:
    def test_grants_stored_as_tuple(self) -> None:
        g = SandboxPathGrant(path="/tmp")
        m = Manifest(extra_path_grants=(g,))
        assert len(m.extra_path_grants) == 1
        assert m.extra_path_grants[0].path == "/tmp"


class TestEnvironmentResolveDispatch:
    """``Environment.resolve`` dispatches to each ``EnvEntry.resolve``."""

    async def test_custom_env_entry_resolve_is_called(self) -> None:
        from typing import Literal, override

        from troopai.adk.types.sandbox.manifest import EnvEntry

        class SecretEnvValue(EnvEntry):
            type: Literal["secret"] = "secret"
            secret: str

            @override
            async def resolve(self) -> str:
                return f"resolved:{self.secret}"

        env = Environment(
            variables={
                "API": SecretEnvValue(secret="xyz"),
                "PLAIN": "p",
                "ENTRY": StrEnvValue(value="y"),
            }
        )
        resolved = await env.resolve()
        assert resolved == {"API": "resolved:xyz", "PLAIN": "p", "ENTRY": "y"}

    async def test_unimplemented_env_entry_raises(self) -> None:
        from typing import Literal

        from troopai.adk.types.sandbox.manifest import EnvEntry

        class NoResolve(EnvEntry):
            type: Literal["noresolve"] = "noresolve"

        env = Environment(variables={"A": NoResolve()})
        with pytest.raises(NotImplementedError):
            await env.resolve()
