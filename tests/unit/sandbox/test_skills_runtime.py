"""Tests for Skills runtime (TS.1-TS.7)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.exceptions.exceptions import SkillsConfigError
from troopai.adk.sandbox.capabilities.skills import (
    LocalDirLazySkillSource,
    Skill,
    SkillsCapability,
)
from troopai.adk.types.sandbox.manifest import Manifest


@pytest.fixture
def skill_dir(tmp_path: Path) -> Path:
    skill = tmp_path / "linter"
    skill.mkdir()
    (skill / "SKILL.md").write_text("A linter skill")
    (skill / "run.sh").write_text("#!/bin/sh\nlinter --check")
    (skill / "config.json").write_text("{}")
    other = tmp_path / "not_a_skill"
    other.mkdir()
    return tmp_path


class TestSkillPathValidation:
    def test_skill_with_slash_name_rejected(self) -> None:
        with pytest.raises(SkillsConfigError, match="simple name"):
            Skill(name="bad/name", description="x", content="x")

    def test_skill_with_dotdot_rejected(self) -> None:
        with pytest.raises(SkillsConfigError, match="simple name"):
            Skill(name="..", description="x", content="x")

    def test_skill_with_null_byte_in_scripts_rejected(self) -> None:
        with pytest.raises(SkillsConfigError):
            Skill(
                name="ok",
                description="x",
                content="x",
                scripts={"bad\x00name": "..."},
            )


class TestLocalDirLoadSkillPayload:
    def test_load_existing_skill(self, skill_dir: Path) -> None:
        source = LocalDirLazySkillSource(source_path=str(skill_dir))
        payload = source.load_skill_payload("linter")
        assert payload is not None
        assert "SKILL.md" in payload
        assert "run.sh" in payload
        assert "config.json" in payload

    def test_load_missing_returns_none(self, skill_dir: Path) -> None:
        source = LocalDirLazySkillSource(source_path=str(skill_dir))
        assert source.load_skill_payload("nope") is None


class TestSkillsProcessManifest:
    def test_inline_skills_materialize_into_manifest(self) -> None:
        cap = SkillsCapability(
            skills=[
                Skill(name="lint", description="run lint", content="lint body"),
                Skill(
                    name="fmt",
                    description="format",
                    content="fmt body",
                    scripts={"run.sh": "#!/bin/sh"},
                ),
            ],
        )
        result = cap.process_manifest(Manifest())
        assert ".agents" in result.entries
        skills_dir = result.entries[".agents"]
        from troopai.adk.types.sandbox.entries import Dir

        assert isinstance(skills_dir, Dir)
        assert "lint" in skills_dir.children
        assert "fmt" in skills_dir.children
        # fmt has scripts/
        fmt = skills_dir.children["fmt"]
        assert isinstance(fmt, Dir)
        assert "SKILL.md" in fmt.children
        assert "scripts" in fmt.children

    def test_lazy_skills_reserve_namespace(self, skill_dir: Path) -> None:
        cap = SkillsCapability(
            lazy_from=LocalDirLazySkillSource(source_path=str(skill_dir)),
        )
        result = cap.process_manifest(Manifest())
        from troopai.adk.types.sandbox.entries import Dir

        assert ".agents" in result.entries
        # Lazy: empty Dir reservation.
        assert isinstance(result.entries[".agents"], Dir)
        assert len(result.entries[".agents"].children) == 0

    def test_from_source_children_materialize(self) -> None:
        """``from_=Dir(...)`` splices its children into the skills root.

        Regression: ``process_manifest`` had branches for ``skills`` and
        ``lazy_from`` but none for ``from_``, so a validated, documented
        ``SkillsCapability(from_=entry)`` silently wrote an EMPTY ``.agents``
        directory instead of the source's skills.
        """
        from troopai.adk.types.sandbox.entries import Dir, File

        source = Dir(
            children={
                "alpha": Dir(children={"SKILL.md": File(content=b"a")}),
                "beta": Dir(children={"SKILL.md": File(content=b"b")}),
            },
        )
        cap = SkillsCapability(from_=source)
        result = cap.process_manifest(Manifest())

        assert ".agents" in result.entries
        skills_dir = result.entries[".agents"]
        assert isinstance(skills_dir, Dir)
        # The from_ children were spliced in, not dropped.
        assert set(skills_dir.children.keys()) == {"alpha", "beta"}

    def test_from_source_non_dir_raises(self) -> None:
        """A non-Dir ``from_`` cannot supply skill children — raise, don't no-op."""
        from troopai.adk.types.sandbox.entries import File

        cap = SkillsCapability(from_=File(content=b"not a dir"))
        with pytest.raises(SkillsConfigError, match="must be a Dir"):
            cap.process_manifest(Manifest())

    def test_unknown_key_raises_not_silent(self) -> None:
        """An unknown construction key (e.g. the wire alias ``from``, or a typo)
        must raise, not silently produce an empty capability.

        The field is the Python attribute ``from_``; ``extra="forbid"`` makes a
        ``model_validate({"from": ...})`` (or a misspelled kwarg) fail loudly
        instead of leaving ``from_`` unset and writing an empty skills dir.
        """
        from pydantic import ValidationError

        from troopai.adk.types.sandbox.entries import Dir

        with pytest.raises(ValidationError):
            SkillsCapability.model_validate({"from": Dir(children={})})

    def test_overlap_raises(self) -> None:
        from troopai.adk.types.sandbox.entries import Dir

        cap = SkillsCapability(
            skills=[Skill(name="lint", description="x", content="x")],
        )
        manifest = Manifest(entries={".agents": Dir(children={})})
        with pytest.raises(SkillsConfigError, match="overlaps"):
            cap.process_manifest(manifest)

    def test_zero_sources_raises_at_construction(self) -> None:
        """Regression: SkillsCapability() with no skills/from_/lazy_from was
        silently instantiable and wrote an empty .agents/ directory at manifest
        time. The fix adds a zero-source guard in model_post_init."""
        with pytest.raises(SkillsConfigError, match="none"):
            SkillsCapability()


class TestLoadSkill:
    @pytest.mark.asyncio
    async def test_load_skill_writes_files(self, skill_dir: Path) -> None:
        session = MagicMock()
        session.write = AsyncMock()
        cap = SkillsCapability(
            lazy_from=LocalDirLazySkillSource(source_path=str(skill_dir)),
        )
        cap.bind(session)
        result = await cap.load_skill("linter")
        assert result["status"] == "loaded"
        assert result["skill_name"] == "linter"
        # Three files in the linter skill — SKILL.md, run.sh, config.json.
        assert int(result["files_written"]) == 3
        # session.write was called 3 times.
        assert session.write.await_count == 3

    @pytest.mark.asyncio
    async def test_load_skill_without_lazy_raises(self) -> None:
        cap = SkillsCapability(
            skills=[Skill(name="x", description="d", content="c")],
        )
        cap.bind(MagicMock())
        with pytest.raises(SkillsConfigError, match="requires lazy_from"):
            await cap.load_skill("x")

    @pytest.mark.asyncio
    async def test_load_skill_without_session_raises(self, skill_dir: Path) -> None:
        cap = SkillsCapability(
            lazy_from=LocalDirLazySkillSource(source_path=str(skill_dir)),
        )
        with pytest.raises(SkillsConfigError, match="bound session"):
            await cap.load_skill("linter")

    @pytest.mark.asyncio
    async def test_load_missing_skill_raises(self, skill_dir: Path) -> None:
        cap = SkillsCapability(
            lazy_from=LocalDirLazySkillSource(source_path=str(skill_dir)),
        )
        cap.bind(MagicMock())
        with pytest.raises(SkillsConfigError, match="not found"):
            await cap.load_skill("missing")


class TestSkillsTools:
    def test_unbound_returns_empty(self) -> None:
        cap = SkillsCapability(skills=[Skill(name="x", description="x", content="x")])
        assert cap.tools() == []

    def test_lazy_bound_exposes_load_skill(self, skill_dir: Path) -> None:
        cap = SkillsCapability(
            lazy_from=LocalDirLazySkillSource(source_path=str(skill_dir)),
        )
        cap.bind(MagicMock())
        tools = cap.tools()
        assert len(tools) == 1
        assert tools[0].name == "load_skill"

    def test_inline_skills_no_tool(self) -> None:
        cap = SkillsCapability(skills=[Skill(name="x", description="x", content="x")])
        cap.bind(MagicMock())
        # Inline skills materialize at process_manifest time; no tool needed.
        assert cap.tools() == []


class TestSkillsInstructionsProgressive:
    @pytest.mark.asyncio
    async def test_lazy_instructions_include_load_skill_hint(self, skill_dir: Path) -> None:
        cap = SkillsCapability(
            lazy_from=LocalDirLazySkillSource(source_path=str(skill_dir)),
        )
        primer = await cap.instructions(Manifest())
        assert primer is not None
        assert "linter" in primer
        assert "load_skill" in primer

    @pytest.mark.asyncio
    async def test_inline_instructions_no_load_skill_hint(self) -> None:
        cap = SkillsCapability(
            skills=[Skill(name="lint", description="run lint", content="x")],
        )
        primer = await cap.instructions(Manifest())
        assert primer is not None
        assert "lint" in primer
        assert "load_skill" not in primer


class TestMetadataCacheInvalidation:
    def test_bind_invalidates_cache(self, skill_dir: Path) -> None:
        cap = SkillsCapability(
            lazy_from=LocalDirLazySkillSource(source_path=str(skill_dir)),
        )
        _ = cap._resolve_runtime_metadata()
        assert cap._metadata_cache is not None
        cap.bind(MagicMock())
        assert cap._metadata_cache is None
