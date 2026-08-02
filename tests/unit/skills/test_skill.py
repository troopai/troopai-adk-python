"""Tests for Skill, SkillMetadata, SkillGovernance dataclasses."""

import pytest

from troopai.adk.skills import (
    Skill,
    SkillActivation,
    SkillGovernance,
    SkillMetadata,
    SkillSet,
)
from troopai.adk.tools import FunctionTool


class TestSkill:
    """Test Skill dataclass construction and validation."""

    def test_basic_construction(self) -> None:
        skill = Skill(name="review", description="Code review")
        assert skill.name == "review"
        assert skill.description == "Code review"
        assert skill.instructions is None
        assert skill.tools == []
        assert skill.enabled is True
        assert skill.metadata is None
        assert skill.governance is None
        assert skill.resources is None

    def test_full_construction(self) -> None:
        tool = FunctionTool(
            name="analyze",
            schema={"type": "object", "properties": {}},
            description="Analyze code",
        )
        skill = Skill(
            name="review",
            description="Code review",
            instructions="Review carefully",
            tools=[tool],
            metadata=SkillMetadata(version="1.0", tags=("python",)),
            governance=SkillGovernance(timeout=30.0),
            resources={"guide": "/path/to/guide.md"},
        )
        assert skill.instructions == "Review carefully"
        assert len(skill.tools) == 1
        assert skill.metadata is not None
        assert skill.metadata.version == "1.0"
        assert skill.governance is not None
        assert skill.governance.timeout == 30.0
        assert skill.resources is not None
        assert "guide" in skill.resources

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="name cannot be empty"):
            Skill(name="", description="test")

    def test_empty_description_raises(self) -> None:
        with pytest.raises(ValueError, match="description cannot be empty"):
            Skill(name="test", description="")

    def test_enabled_default_true(self) -> None:
        skill = Skill(name="test", description="test")
        assert skill.enabled is True

    def test_enabled_static_false(self) -> None:
        skill = Skill(name="test", description="test", enabled=False)
        assert skill.enabled is False

    def test_enabled_callable(self) -> None:
        skill = Skill(
            name="test",
            description="test",
            enabled=lambda _ctx: True,
        )
        assert callable(skill.enabled)


class TestSkillMetadata:
    """Test SkillMetadata frozen dataclass."""

    def test_basic(self) -> None:
        meta = SkillMetadata(version="1.0.0", author="test")
        assert meta.version == "1.0.0"
        assert meta.author == "test"
        assert meta.tags == ()
        assert meta.license is None

    def test_frozen(self) -> None:
        meta = SkillMetadata(version="1.0")
        with pytest.raises(AttributeError):
            meta.version = "2.0"  # type: ignore[misc]

    def test_tags(self) -> None:
        meta = SkillMetadata(tags=("python", "security"))
        assert "python" in meta.tags
        assert "security" in meta.tags


class TestSkillGovernance:
    """Test SkillGovernance dataclass."""

    def test_defaults(self) -> None:
        gov = SkillGovernance()
        assert gov.timeout is None
        assert gov.max_result_tokens is None
        assert gov.max_retries is None

    def test_full(self) -> None:
        gov = SkillGovernance(
            timeout=30.0,
            max_result_tokens=500,
            max_retries=2,
        )
        assert gov.timeout == 30.0
        assert gov.max_result_tokens == 500
        assert gov.max_retries == 2


class TestSkillSet:
    """Test SkillSet container."""

    def test_find(self) -> None:
        s1 = Skill(name="review", description="Review")
        s2 = Skill(name="debug", description="Debug")
        ss = SkillSet(name="devtools", skills=[s1, s2])
        assert ss.find("review") is s1
        assert ss.find("debug") is s2
        assert ss.find("nonexistent") is None

    def test_filter_by_tag(self) -> None:
        s1 = Skill(
            name="review",
            description="Review",
            metadata=SkillMetadata(tags=("python", "security")),
        )
        s2 = Skill(
            name="debug",
            description="Debug",
            metadata=SkillMetadata(tags=("python",)),
        )
        s3 = Skill(name="plain", description="No tags")
        ss = SkillSet(name="all", skills=[s1, s2, s3])

        python_skills = ss.filter_by_tag("python")
        assert len(python_skills) == 2

        security_skills = ss.filter_by_tag("security")
        assert len(security_skills) == 1
        assert security_skills[0].name == "review"

        no_skills = ss.filter_by_tag("nonexistent")
        assert len(no_skills) == 0


class TestSkillActivation:
    """Test SkillActivation enum."""

    def test_values(self) -> None:
        assert SkillActivation.EAGER == "eager"
        assert SkillActivation.LAZY == "lazy"

    def test_string_comparison(self) -> None:
        assert SkillActivation.EAGER == "eager"
        assert SkillActivation.LAZY != "eager"
