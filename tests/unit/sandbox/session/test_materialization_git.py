"""Tests for materialization.git — in-sandbox git clone orchestration.

Hermetic: a programmable fake session records every ``run`` argv and
returns canned exit codes (no real git, no network). The valuable
surface is command construction + control flow, exercised here.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

from troopai.adk.exceptions.exceptions import GitArtifactError
from troopai.adk.sandbox.session.materialization.git import materialize_git_repo
from troopai.adk.types.sandbox.entries import GitRepo


def _git_session(responder: Callable[[tuple[str, ...]], int] | None = None) -> Any:
    """Fake session: records run/mkdir; ``responder`` maps argv→exit code (default 0)."""

    class _Rec:
        def __init__(self) -> None:
            self.runs: list[tuple[str, ...]] = []
            self.mkdirs: list[str] = []

        async def run(
            self, *command: object, timeout: float | None = None, shell: bool = True, user: object = None
        ) -> Any:
            argv = tuple(str(c) for c in command)
            self.runs.append(argv)
            code = 0 if responder is None else responder(argv)
            return SimpleNamespace(exit_code=code, stdout=b"", stderr=b"boom")

        async def mkdir(self, path: object, *, parents: bool = False, user: object = None) -> None:
            self.mkdirs.append(str(path))

    return _Rec()


def _clone_argv(session: Any) -> tuple[str, ...]:
    return next(a for a in session.runs if "clone" in a)


class TestMaterializeGitRepo:
    async def test_git_missing_raises(self) -> None:
        session = _git_session(lambda argv: 1 if "--version" in argv else 0)
        with pytest.raises(GitArtifactError, match="git is not available"):
            await materialize_git_repo(session, "dst", GitRepo(repo="acme/widgets"))

    async def test_ref_starting_with_dash_rejected(self) -> None:
        # ref has no entry validator; a leading '-' is a git
        # argument-injection vector — rejected before any git runs.
        session = _git_session()
        with pytest.raises(GitArtifactError, match="argument-injection"):
            await materialize_git_repo(session, "dst", GitRepo(repo="o/n", ref="--upload-pack=evil"))
        assert session.runs == []

    async def test_named_ref_shallow_clone_command(self) -> None:
        session = _git_session()
        result = await materialize_git_repo(session, "dst", GitRepo(repo="acme/widgets"))
        clone = _clone_argv(session)
        assert clone[:3] == ("env", "GIT_TERMINAL_PROMPT=0", "GIT_ASKPASS=true")
        assert clone[3:9] == ("git", "clone", "--depth", "1", "--no-tags", "--single-branch")
        assert clone[9:12] == ("--branch", "main", "https://github.com/acme/widgets.git")
        assert clone[12].startswith("/tmp/troopai-git-")
        assert "dst" in session.mkdirs
        assert any(a[:3] == ("cp", "-R", "--") and a[3].endswith("/.") for a in session.runs)
        assert any(a[0] == "rm" and a[1] == "-rf" for a in session.runs)  # finally cleanup
        assert result.is_directory is True
        assert result.path == "dst"

    async def test_depth_none_omits_depth_flag(self) -> None:
        session = _git_session()
        await materialize_git_repo(session, "dst", GitRepo(repo="o/n", depth=None))
        clone = _clone_argv(session)
        assert "--depth" not in clone

    async def test_subpath_copied_from_subdir(self) -> None:
        session = _git_session()
        await materialize_git_repo(session, "dst", GitRepo(repo="o/n", subpath="pkg/inner"))
        cp = next(a for a in session.runs if a[:3] == ("cp", "-R", "--"))
        assert cp[3].endswith("/pkg/inner/.")

    async def test_custom_host_url(self) -> None:
        session = _git_session()
        await materialize_git_repo(session, "dst", GitRepo(host="gitlab.example.com", repo="grp/proj"))
        assert "https://gitlab.example.com/grp/proj.git" in _clone_argv(session)

    async def test_commit_sha_ref_uses_fetch_strategy(self) -> None:
        session = _git_session()
        await materialize_git_repo(session, "dst", GitRepo(repo="o/n", ref="a1b2c3d4e5f6"))
        joined = [" ".join(a) for a in session.runs]
        assert any("git init" in j for j in joined)
        assert any("fetch" in j and "a1b2c3d4e5f6" in j for j in joined)
        assert any("checkout --detach FETCH_HEAD" in j for j in joined)
        assert not any("clone" in a for a in session.runs)

    async def test_commit_fetch_failure_falls_back_to_named_clone(self) -> None:
        # A hex-looking BRANCH: commit-fetch fails, tmp is reset, named
        # clone is attempted (mirrors upstream fallback).
        def responder(argv: tuple[str, ...]) -> int:
            return 1 if "fetch" in argv else 0

        session = _git_session(responder)
        await materialize_git_repo(session, "dst", GitRepo(repo="o/n", ref="abcdef1234"))
        rm_runs = [a for a in session.runs if a[0] == "rm" and a[1] == "-rf"]
        assert len(rm_runs) >= 2  # intermediate tmp reset BEFORE fallback + finally cleanup
        assert any("clone" in a for a in session.runs)  # named-ref fallback ran

    async def test_clone_failure_raises_and_still_cleans_up(self) -> None:
        def responder(argv: tuple[str, ...]) -> int:
            return 1 if "clone" in argv else 0

        session = _git_session(responder)
        with pytest.raises(GitArtifactError, match="git clone"):
            await materialize_git_repo(session, "dst", GitRepo(repo="o/n"))
        # finally still attempted the tmp rm despite the clone failure.
        assert any(a[0] == "rm" and a[1] == "-rf" for a in session.runs)

    async def test_cleanup_failure_does_not_mask_success(self) -> None:
        # rm cleanup non-zero must NOT raise / mask the successful clone.
        def responder(argv: tuple[str, ...]) -> int:
            return 1 if argv[0] == "rm" else 0

        session = _git_session(responder)
        result = await materialize_git_repo(session, "dst", GitRepo(repo="o/n"))
        assert result.is_directory is True
