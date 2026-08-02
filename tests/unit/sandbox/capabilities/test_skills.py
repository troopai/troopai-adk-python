"""Regression tests for ``adk/sandbox/capabilities/skills.py`` defects.

Covers two confirmed bugs:

- The model-supplied ``skill_name`` passed to ``SkillsCapability.load_skill``
  was joined into host + workspace paths with no sanitization, so a crafted
  name (``..`` traversal or an absolute path) could read host files outside
  the lazy source and write outside ``skills_path``.
- ``GitRepoLazySkillSource`` caught ``subprocess.CalledProcessError`` in its
  fallback handlers, but ``_ensure_clone`` re-raises clone failures as
  ``SkillsConfigError`` — so the handlers were dead code and a transient
  clone failure aborted system-prompt construction instead of degrading to
  an empty skill index.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.exceptions.exceptions import SkillsConfigError
from troopai.adk.sandbox.capabilities.skills import (
    GitRepoLazySkillSource,
    LocalDirLazySkillSource,
    SkillsCapability,
)


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    skill = tmp_path / "linter"
    skill.mkdir()
    (skill / "SKILL.md").write_text("A linter skill")
    return tmp_path


@pytest.fixture
def secret_file(tmp_path: Path) -> Path:
    """A host file OUTSIDE the lazy source dir, simulating a host secret."""
    secret = tmp_path / "secret.txt"
    secret.write_text("super-secret")
    return secret


class TestLoadSkillPathTraversal:
    """``load_skill`` must reject crafted ``skill_name`` before any path join."""

    @pytest.mark.asyncio
    async def test_dotdot_traversal_rejected_before_read(self, tmp_path: Path, secret_file: Path) -> None:
        # Lazy source rooted at a subdir; ``..`` would escape to the secret.
        source_root = tmp_path / "skills"
        source_root.mkdir()
        session = MagicMock()
        session.write = AsyncMock()
        cap = SkillsCapability(lazy_from=LocalDirLazySkillSource(source_path=str(source_root)))
        cap.bind(session)

        with pytest.raises(SkillsConfigError, match="simple name"):
            await cap.load_skill("..")

        # No file was exfiltrated into the workspace.
        assert session.write.await_count == 0

    @pytest.mark.asyncio
    async def test_absolute_skill_name_rejected(self, tmp_path: Path) -> None:
        source_root = tmp_path / "skills"
        source_root.mkdir()
        session = MagicMock()
        session.write = AsyncMock()
        cap = SkillsCapability(lazy_from=LocalDirLazySkillSource(source_path=str(source_root)))
        cap.bind(session)

        # Absolute right operand would discard source_path and read host root.
        with pytest.raises(SkillsConfigError, match="workspace-relative"):
            await cap.load_skill("/etc")
        assert session.write.await_count == 0

    @pytest.mark.asyncio
    async def test_separator_in_skill_name_rejected(self, skill_dir: Path) -> None:
        session = MagicMock()
        session.write = AsyncMock()
        cap = SkillsCapability(lazy_from=LocalDirLazySkillSource(source_path=str(skill_dir)))
        cap.bind(session)

        with pytest.raises(SkillsConfigError, match="simple name"):
            await cap.load_skill("../../etc/passwd")
        assert session.write.await_count == 0

    @pytest.mark.asyncio
    async def test_valid_skill_name_still_loads(self, skill_dir: Path) -> None:
        """Sanitization must not break the legitimate single-name case."""
        session = MagicMock()
        session.write = AsyncMock()
        cap = SkillsCapability(lazy_from=LocalDirLazySkillSource(source_path=str(skill_dir)))
        cap.bind(session)

        result = await cap.load_skill("linter")
        assert result["status"] == "loaded"
        assert result["skill_name"] == "linter"
        assert session.write.await_count == 1


class TestGitCloneFailureDegradesGracefully:
    """A failed/timed-out clone (``SkillsConfigError``) must be caught, not raised."""

    def test_list_skill_metadata_returns_empty_on_clone_failure(self) -> None:
        source = GitRepoLazySkillSource(repo="owner/name")

        def _boom() -> Path:
            raise SkillsConfigError("git clone failed")

        source._ensure_clone = _boom  # type: ignore[method-assign]

        # Before the fix this propagated SkillsConfigError and aborted the run.
        assert source.list_skill_metadata() == []

    def test_load_skill_payload_returns_none_on_clone_failure(self) -> None:
        source = GitRepoLazySkillSource(repo="owner/name")

        def _boom() -> Path:
            raise SkillsConfigError("git clone timed out")

        source._ensure_clone = _boom  # type: ignore[method-assign]

        assert source.load_skill_payload("anything") is None

    @pytest.mark.asyncio
    async def test_instructions_survive_clone_failure(self) -> None:
        """System-prompt construction must not crash when the clone fails."""
        from troopai.adk.types.sandbox.manifest import Manifest

        source = GitRepoLazySkillSource(repo="owner/name")

        def _boom() -> Path:
            raise SkillsConfigError("git clone failed")

        source._ensure_clone = _boom  # type: ignore[method-assign]
        cap = SkillsCapability(lazy_from=source)

        # Empty index -> instructions return None instead of raising.
        assert await cap.instructions(Manifest()) is None
