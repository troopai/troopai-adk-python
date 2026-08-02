"""Tests for ``troopai.adk.types.sandbox.entries``."""

from __future__ import annotations

from pathlib import Path

import pytest

from troopai.adk.types.sandbox.entries import (
    BaseEntry,
    Dir,
    File,
    GitRepo,
    LocalDir,
    LocalFile,
)


class TestBaseEntryRegistry:
    def test_known_types_includes_concrete_subclasses(self) -> None:
        known = BaseEntry.known_types()
        assert "file" in known
        assert "dir" in known
        assert "local_file" in known
        assert "local_dir" in known
        assert "git_repo" in known

    def test_parse_dispatches_by_type(self) -> None:
        entry = BaseEntry.parse({"type": "file", "content": b"hi"})
        assert isinstance(entry, File)
        assert entry.content == b"hi"

    def test_parse_rejects_unknown_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown BaseEntry discriminator"):
            BaseEntry.parse({"type": "nope"})

    def test_parse_rejects_missing_type(self) -> None:
        with pytest.raises(ValueError, match="missing required 'type'"):
            BaseEntry.parse({"content": b"hi"})


class TestFile:
    def test_default_is_empty_bytes(self) -> None:
        f = File()
        assert f.content == b""

    def test_str_coerced_to_bytes(self) -> None:
        f = File(content="hi")  # type: ignore[arg-type]
        assert f.content == b"hi"

    def test_is_dir_is_false(self) -> None:
        assert File().is_dir() is False


class TestDir:
    def test_empty_children(self) -> None:
        d = Dir()
        assert d.children == {}
        assert d.is_dir() is True

    def test_with_typed_child(self) -> None:
        d = Dir(children={"hello.txt": File(content=b"hi")})
        assert isinstance(d.children["hello.txt"], File)

    def test_dict_child_parsed_via_discriminator(self) -> None:
        d = Dir(children={"x": {"type": "file", "content": b"y"}})  # type: ignore[arg-type]
        assert isinstance(d.children["x"], File)
        assert d.children["x"].content == b"y"

    def test_child_key_rejects_slash(self) -> None:
        with pytest.raises(ValueError, match="simple directory names"):
            Dir(children={"sub/dir": File()})

    def test_child_key_rejects_dot_dot(self) -> None:
        with pytest.raises(ValueError, match="simple directory names"):
            Dir(children={"..": File()})


class TestLocalFile:
    def test_src_coerced_from_str(self) -> None:
        lf = LocalFile(src="/etc/hosts")  # type: ignore[arg-type]
        assert lf.src == Path("/etc/hosts")

    def test_src_rejects_empty_str(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            LocalFile(src="")  # type: ignore[arg-type]

    def test_is_dir_is_false(self) -> None:
        assert LocalFile(src=Path("/x")).is_dir() is False


class TestLocalDir:
    def test_src_optional(self) -> None:
        ld = LocalDir()
        assert ld.src is None
        assert ld.is_dir() is True

    def test_follow_symlinks_default_true(self) -> None:
        assert LocalDir().follow_symlinks is True


class TestGitRepo:
    def test_repo_owner_name_ok(self) -> None:
        gr = GitRepo(repo="anthropics/claude-code")
        assert gr.repo == "anthropics/claude-code"
        assert gr.depth == 1  # shallow by default

    def test_repo_rejects_no_slash(self) -> None:
        with pytest.raises(ValueError, match="owner/name"):
            GitRepo(repo="claude-code")

    def test_repo_rejects_multiple_slashes(self) -> None:
        with pytest.raises(ValueError, match="owner/name"):
            GitRepo(repo="a/b/c")

    def test_repo_rejects_empty_owner_or_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            GitRepo(repo="/name")
        with pytest.raises(ValueError, match="non-empty"):
            GitRepo(repo="owner/")

    def test_subpath_rejects_dot_dot(self) -> None:
        with pytest.raises(ValueError, match="must not contain '..'"):
            GitRepo(repo="o/r", subpath="../outside")

    def test_subpath_rejects_absolute(self) -> None:
        with pytest.raises(ValueError, match="workspace-relative"):
            GitRepo(repo="o/r", subpath="/etc")

    def test_subpath_rejects_backslash_dot_dot_traversal(self) -> None:
        # A Windows-authored manifest uses backslash separators. The path is
        # interpreted as POSIX regardless of host OS, so the ``..`` traversal
        # must be rejected even when separated by backslashes.
        with pytest.raises(ValueError, match="must not contain '..'"):
            GitRepo(repo="o/r", subpath="src\\..\\..\\..\\etc\\passwd")

    def test_subpath_rejects_backslash_absolute(self) -> None:
        with pytest.raises(ValueError, match="workspace-relative"):
            GitRepo(repo="o/r", subpath="\\etc\\passwd")

    def test_subpath_normalizes_backslash_separators(self) -> None:
        # A clean backslash-separated relative path normalizes to POSIX.
        gr = GitRepo(repo="o/r", subpath="src\\pkg\\mod")
        assert gr.subpath == "src/pkg/mod"

    def test_depth_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError, match="positive or None"):
            GitRepo(repo="o/r", depth=0)

    def test_depth_none_allowed(self) -> None:
        gr = GitRepo(repo="o/r", depth=None)
        assert gr.depth is None

    def test_is_dir_is_true(self) -> None:
        assert GitRepo(repo="o/r").is_dir() is True
