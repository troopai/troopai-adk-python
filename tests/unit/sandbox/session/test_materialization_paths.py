"""Tests for materialization.paths — workspace-key normalize + overlap."""

from __future__ import annotations

import pytest

from troopai.adk.sandbox.session.materialization.paths import (
    normalize_workspace_key,
    paths_overlap,
)


class TestNormalizeWorkspaceKey:
    def test_passthrough(self) -> None:
        assert normalize_workspace_key("src/foo.py") == "src/foo.py"

    def test_collapses_dot_segments(self) -> None:
        assert normalize_workspace_key("a/./b.txt") == "a/b.txt"

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            normalize_workspace_key("")

    def test_absolute_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be relative"):
            normalize_workspace_key("/etc/passwd")

    def test_parent_traversal_rejected(self) -> None:
        with pytest.raises(ValueError, match=r"must not contain '\.\.'"):
            normalize_workspace_key("a/../../etc")


class TestPathsOverlap:
    def test_equal_overlaps(self) -> None:
        assert paths_overlap("a/b", "a/b") is True

    def test_ancestor_overlaps(self) -> None:
        assert paths_overlap("a", "a/b/c") is True

    def test_descendant_overlaps(self) -> None:
        assert paths_overlap("a/b/c", "a") is True

    def test_siblings_do_not_overlap(self) -> None:
        assert paths_overlap("a/b", "a/c") is False

    def test_unrelated_do_not_overlap(self) -> None:
        assert paths_overlap("x/y", "p/q") is False
