"""Tests for :mod:`troopai.adk.rag.loaders.github`.

PyGithub is stubbed via ``sys.modules`` so the lazy ``from github import ...``
inside the worker thread resolves to in-test fakes; no network is touched.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from dataclasses import dataclass

import pytest

from troopai.adk.rag.loaders.github import GithubLoader


class _GithubException(Exception):
    """Stand-in for PyGithub's ``GithubException``."""


@dataclass
class _FakeTreeElement:
    type: str
    path: str


@dataclass
class _FakeTree:
    tree: list[_FakeTreeElement]


@dataclass
class _FakeBlob:
    decoded_content: bytes


class _FakeRepo:
    """Records every ``get_contents`` call to assert the fetch cap."""

    def __init__(self, *, elements: list[_FakeTreeElement], contents: dict[str, bytes]) -> None:
        self.default_branch = "main"
        self._elements = elements
        self._contents = contents
        self.get_contents_calls: list[str] = []

    def get_git_tree(self, _branch: str, recursive: bool) -> _FakeTree:
        assert recursive is True
        return _FakeTree(tree=self._elements)

    def get_contents(self, path: str, ref: str) -> _FakeBlob:
        assert ref == self.default_branch
        self.get_contents_calls.append(path)
        return _FakeBlob(decoded_content=self._contents[path])


class _FakeGithubClient:
    def __init__(self, repo: _FakeRepo) -> None:
        self._repo = repo

    def get_repo(self, _full_name: str) -> _FakeRepo:
        return self._repo


def _install_fake_github(monkeypatch: pytest.MonkeyPatch, repo: _FakeRepo) -> None:
    module = types.ModuleType("github")
    # A real spec lets ``find_spec`` succeed in ``ensure_dependencies``.
    module.__spec__ = importlib.machinery.ModuleSpec("github", loader=None)
    module.Github = lambda *args, **kwargs: _FakeGithubClient(repo)  # type: ignore[attr-defined]
    module.GithubException = _GithubException  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "github", module)


async def test_max_files_caps_blob_fetches_not_nonempty_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    """``max_files`` must bound ``get_contents`` calls, not appended documents.

    With many empty/whitespace-only matching blobs preceding content-bearing
    ones, the cap must still fire after exactly ``max_files`` fetches — the
    documented contract is a per-repo blob-fetch bound.
    """
    elements = [_FakeTreeElement(type="blob", path=f"f{i}.py") for i in range(10)]
    # First 5 matching blobs are whitespace-only; the rest carry content.
    contents = {f"f{i}.py": (b"   " if i < 5 else b"real content") for i in range(10)}
    repo = _FakeRepo(elements=elements, contents=contents)
    _install_fake_github(monkeypatch, repo)

    loader = GithubLoader(max_files=3)
    documents = await loader.load("https://github.com/owner/repo")

    # Before the fix, get_contents was called for every blob until 3 non-empty
    # docs accumulated (8 calls); after the fix it stops at exactly max_files.
    assert len(repo.get_contents_calls) == 3
    # Those first 3 blobs are whitespace-only, so no documents survive.
    assert len(documents) == 0


async def test_max_files_allows_content_within_the_fetch_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """Within the fetch budget, content-bearing blobs still become documents."""
    elements = [_FakeTreeElement(type="blob", path=f"f{i}.py") for i in range(5)]
    contents = {f"f{i}.py": b"real content" for i in range(5)}
    repo = _FakeRepo(elements=elements, contents=contents)
    _install_fake_github(monkeypatch, repo)

    loader = GithubLoader(max_files=2)
    documents = await loader.load("https://github.com/owner/repo")

    assert len(repo.get_contents_calls) == 2
    assert len(documents) == 2
    assert documents[0].metadata["path"] == "f0.py"
    assert documents[0].metadata["repo"] == "owner/repo"
