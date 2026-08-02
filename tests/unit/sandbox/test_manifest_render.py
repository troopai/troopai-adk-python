"""Tests for ``troopai.adk.sandbox.manifest_render``."""

from __future__ import annotations

from pathlib import Path

import pytest

from troopai.adk.sandbox.manifest_render import (
    MAX_MANIFEST_DESCRIPTION_CHARS,
    coerce_rel_path,
    render_manifest_description,
)
from troopai.adk.types.sandbox.entries import Dir, File
from troopai.adk.types.sandbox.permissions import Permissions


def _coerce(value: str | Path) -> Path:
    return coerce_rel_path(value)


class TestRenderBasic:
    def test_empty_manifest_renders_root_only(self) -> None:
        out = render_manifest_description(
            root="/workspace",
            entries={},
            coerce_rel_path=_coerce,
        )
        assert out.startswith("/workspace")
        assert out.endswith("\n")

    def test_single_file_entry(self) -> None:
        entries: dict[str | Path, object] = {
            "src/app.py": File(content=b"# code", description="entry point"),
        }
        out = render_manifest_description(
            root="/workspace",
            entries=entries,  # type: ignore[arg-type]
            coerce_rel_path=_coerce,
        )
        assert "/workspace" in out
        assert "src" in out
        assert "└──" in out or "├──" in out

    def test_nested_dir_with_depth_1_renders_only_first_level(self) -> None:
        nested = Dir(
            permissions=Permissions(directory=True),
            children={"inner.py": File(content=b"x")},  # type: ignore[dict-item]
        )
        entries: dict[str | Path, object] = {"outer": nested}
        out = render_manifest_description(
            root="/work",
            entries=entries,  # type: ignore[arg-type]
            coerce_rel_path=_coerce,
            depth=1,
        )
        assert "outer" in out
        assert "inner.py" not in out

    def test_nested_dir_with_depth_2_renders_children(self) -> None:
        nested = Dir(
            permissions=Permissions(directory=True),
            children={"inner.py": File(content=b"x", description="leaf")},
        )
        entries: dict[str | Path, object] = {"outer": nested}
        out = render_manifest_description(
            root="/work",
            entries=entries,  # type: ignore[arg-type]
            coerce_rel_path=_coerce,
            depth=2,
        )
        assert "inner.py" in out
        assert "leaf" in out


class TestArgValidation:
    def test_rejects_zero_depth(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            render_manifest_description(
                root="/x",
                entries={},
                coerce_rel_path=_coerce,
                depth=0,
            )

    def test_rejects_negative_max_chars(self) -> None:
        with pytest.raises(ValueError, match="max_chars"):
            render_manifest_description(
                root="/x",
                entries={},
                coerce_rel_path=_coerce,
                max_chars=-1,
            )

    def test_unbounded_depth_accepted(self) -> None:
        render_manifest_description(
            root="/x",
            entries={},
            coerce_rel_path=_coerce,
            depth=None,
        )

    def test_unbounded_max_chars_accepted(self) -> None:
        render_manifest_description(
            root="/x",
            entries={},
            coerce_rel_path=_coerce,
            max_chars=None,
        )


class TestTruncation:
    def test_truncates_when_exceeding_limit(self) -> None:
        from troopai.adk.types.sandbox.entries import BaseEntry

        children: dict[str, BaseEntry] = {f"f{i}.py": File(content=b"x", description=f"file {i}") for i in range(200)}
        nested = Dir(permissions=Permissions(directory=True), children=children)
        entries: dict[str | Path, object] = {"big": nested}
        out = render_manifest_description(
            root="/work",
            entries=entries,  # type: ignore[arg-type]
            coerce_rel_path=_coerce,
            depth=2,
            max_chars=400,
        )
        assert "truncated" in out
        assert len(out) <= 400 + 5  # tolerance for the marker rounding loop

    def test_default_max_chars_constant(self) -> None:
        assert MAX_MANIFEST_DESCRIPTION_CHARS == 5000


def test_coerce_rel_path_handles_absolute_input() -> None:
    out = coerce_rel_path("/x/y/z")
    assert out == Path("x/y/z")


def test_coerce_rel_path_passthrough() -> None:
    out = coerce_rel_path("rel/path")
    assert out == Path("rel/path")


def test_coerce_rel_path_strips_leading_parent_refs() -> None:
    # The docstring guarantees the result is always workspace-relative;
    # leading ".." traversals must not survive.
    out = coerce_rel_path("../../etc/passwd")
    assert out == Path("etc/passwd")
    assert ".." not in out.parts


def test_coerce_rel_path_strips_interior_escaping_parent_refs() -> None:
    # A relative path that climbs above its anchor ("a/../../b" -> "../b")
    # must still resolve to a workspace-relative path.
    out = coerce_rel_path("a/../../b")
    assert out == Path("b")
    assert ".." not in out.parts


def test_coerce_rel_path_strips_windows_drive_anchor() -> None:
    out = coerce_rel_path("C:/foo")
    assert out == Path("foo")
    assert "C:" not in out.parts


def test_coerce_rel_path_strips_windows_drive_anchor_with_backslashes() -> None:
    out = coerce_rel_path("C:\\foo\\bar")
    assert out == Path("foo/bar")
    assert "C:" not in out.parts


def test_coerce_rel_path_bare_parent_ref_yields_dot() -> None:
    assert coerce_rel_path("..") == Path(".")
