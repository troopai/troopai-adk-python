"""Tests for ``troopai.adk.sandbox.session.tar_workspace``."""

from __future__ import annotations

from pathlib import Path

from troopai.adk.sandbox.session.tar_workspace import shell_tar_exclude_args


def test_empty_input_returns_empty_list() -> None:
    assert shell_tar_exclude_args([]) == []


def test_each_path_produces_two_excludes() -> None:
    out = shell_tar_exclude_args([Path(".sandbox"), Path("memories")])
    assert "--exclude=.sandbox" in out
    assert "--exclude=./.sandbox" in out
    assert "--exclude=memories" in out
    assert "--exclude=./memories" in out


def test_root_and_dot_paths_skipped() -> None:
    assert shell_tar_exclude_args([Path("/"), Path(".")]) == []


def test_paths_are_shell_quoted() -> None:
    out = shell_tar_exclude_args([Path("a b c")])
    assert "--exclude='a b c'" in out
    assert "--exclude='./a b c'" in out


def test_paths_are_sorted_for_determinism() -> None:
    out = shell_tar_exclude_args([Path("zeta"), Path("alpha")])
    # First two entries belong to alpha (alphabetically first).
    assert out[0] == "--exclude=alpha"
    assert out[1] == "--exclude=./alpha"
    assert out[2] == "--exclude=zeta"
    assert out[3] == "--exclude=./zeta"


def test_leading_slash_stripped() -> None:
    out = shell_tar_exclude_args([Path("/srv/data")])
    assert "--exclude=srv/data" in out
    assert "--exclude=./srv/data" in out
