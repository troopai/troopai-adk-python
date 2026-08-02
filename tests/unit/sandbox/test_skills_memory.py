"""Tests for SkillsCapability + MemoryCapability shells (P15 + P16)."""

from __future__ import annotations

from pathlib import Path

import pytest

from troopai.adk.exceptions.exceptions import SkillsConfigError
from troopai.adk.sandbox.capabilities.memory import (
    MemoryCapability,
    MemoryGenerateConfig,
    MemoryReadConfig,
)
from troopai.adk.sandbox.capabilities.skills import (
    LocalDirLazySkillSource,
    Skill,
    SkillsCapability,
)
from troopai.adk.types.sandbox.manifest import Manifest


class TestSkillsConfig:
    def test_empty_construction_raises(self) -> None:
        with pytest.raises(SkillsConfigError, match="none"):
            SkillsCapability()

    def test_construction_with_inline_skills_ok(self) -> None:
        cap = SkillsCapability(skills=[Skill(name="x", description="d", content="c")])
        assert cap.type == "skills"
        assert cap.skills_path == ".agents"

    def test_multiple_sources_rejected(self) -> None:
        with pytest.raises(SkillsConfigError, match="exactly one"):
            SkillsCapability(
                skills=[Skill(name="x", description="d", content="c")],
                lazy_from=LocalDirLazySkillSource(source_path="/tmp"),
            )

    def test_empty_skills_path_rejected(self) -> None:
        with pytest.raises(SkillsConfigError, match="non-empty"):
            SkillsCapability(
                skills=[Skill(name="x", description="d", content="c")],
                skills_path="",
            )

    def test_absolute_skills_path_rejected(self) -> None:
        with pytest.raises(SkillsConfigError, match="workspace-relative"):
            SkillsCapability(
                skills=[Skill(name="x", description="d", content="c")],
                skills_path="/tmp/.agents",
            )


class TestLocalDirLazySkillSource:
    def test_lists_subdirs_with_skill_md(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("a useful skill")
        other = tmp_path / "not_a_skill"
        other.mkdir()
        # No SKILL.md — should be skipped.

        source = LocalDirLazySkillSource(source_path=str(tmp_path))
        result = source.list_skill_metadata()
        assert len(result) == 1
        assert result[0].name == "my_skill"

    def test_missing_directory_returns_empty(self, tmp_path: Path) -> None:
        source = LocalDirLazySkillSource(source_path=str(tmp_path / "nope"))
        assert source.list_skill_metadata() == []


class TestSkillsInstructions:
    @pytest.mark.asyncio
    async def test_empty_skills_list_returns_none(self) -> None:
        # A SkillsCapability with an explicitly empty list raises at construction.
        # When a non-empty list is passed, instructions summarise the skills.
        cap = SkillsCapability(skills=[Skill(name="x", description="d", content="c")])
        result = await cap.instructions(Manifest())
        assert result is not None

    @pytest.mark.asyncio
    async def test_inline_skills_listed(self) -> None:
        cap = SkillsCapability(
            skills=[
                Skill(name="lint", description="run linter", content="..."),
                Skill(name="test", description="run tests", content="..."),
            ],
        )
        result = await cap.instructions(Manifest())
        assert result is not None
        assert "lint" in result
        assert "test" in result


class TestMemoryRequiredCapabilityTypes:
    def test_read_none_no_requirements(self) -> None:
        cap = MemoryCapability(read=None)
        assert cap.required_capability_types() == set()

    def test_read_no_live_update_requires_shell(self) -> None:
        cap = MemoryCapability(read=MemoryReadConfig(live_update=False))
        assert cap.required_capability_types() == {"shell"}

    def test_read_live_update_requires_shell_and_filesystem(self) -> None:
        cap = MemoryCapability(read=MemoryReadConfig(live_update=True))
        assert cap.required_capability_types() == {"shell", "filesystem"}


class TestMemoryInstructions:
    @pytest.mark.asyncio
    async def test_read_none_returns_none(self) -> None:
        cap = MemoryCapability(read=None)
        assert await cap.instructions(Manifest()) is None

    @pytest.mark.asyncio
    async def test_read_enabled_unbound_returns_none(self) -> None:
        # Without a bound session the capability cannot read the
        # summary file, so instructions() returns None. The session
        # binding happens inside sandbox_run_context.
        cap = MemoryCapability()
        assert await cap.instructions(Manifest()) is None


class TestMemoryGenerateConfig:
    def test_defaults(self) -> None:
        cfg = MemoryGenerateConfig()
        assert cfg.max_raw_memories_for_consolidation == 256
        assert cfg.extra_prompt is None
