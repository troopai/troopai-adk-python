"""Tests for SkillDiscoveryToolset."""

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from troopai.adk.skills import Skill, SkillDiscoveryToolset, SkillMetadata
from troopai.adk.skills.discovery import resource_is_file_path
from troopai.adk.tools import FunctionTool
from troopai.adk.tools.tool_context import ToolContext


async def _invoke(tool: FunctionTool, args: dict[str, str]) -> str:
    """Invoke a discovery tool with a real (minimal) ``ToolContext``."""
    assert tool.on_invoke is not None
    ctx = ToolContext[Any](
        tool_name=tool.name,
        tool_call_id=f"call_{tool.name}",
        tool_arguments={},
        raw_arguments="",
    )
    result = await tool.on_invoke(ctx, json.dumps(args))
    assert isinstance(result, str)
    return result


@pytest.fixture
def review_skill() -> Skill:
    tool = FunctionTool(
        name="analyze_code",
        schema={"type": "object", "properties": {}},
        description="Analyze code",
    )
    return Skill(
        name="code-review",
        description="Expert code review",
        instructions="Review code carefully",
        tools=[tool],
        metadata=SkillMetadata(version="1.0", tags=("python", "security")),
        resources={"references/guide.md": "/path/to/guide.md"},
    )


@pytest.fixture
def debug_skill() -> Skill:
    return Skill(
        name="debugging",
        description="Systematic debugging",
        instructions="Debug step by step",
    )


class TestSkillDiscoveryToolset:
    """Test SkillDiscoveryToolset tool generation."""

    def test_tools_generated(
        self,
        review_skill: Skill,
        debug_skill: Skill,
    ) -> None:
        discovery = SkillDiscoveryToolset(skills=[review_skill, debug_skill])
        tools = discovery.tools()
        names = [t.name for t in tools]
        assert "list_skills" in names
        assert "load_skill" in names
        assert "load_skill_resource" in names  # review_skill has resources

    def test_no_resource_tool_when_no_resources(
        self,
        debug_skill: Skill,
    ) -> None:
        discovery = SkillDiscoveryToolset(skills=[debug_skill])
        tools = discovery.tools()
        names = [t.name for t in tools]
        assert "list_skills" in names
        assert "load_skill" in names
        assert "load_skill_resource" not in names

    def test_script_tool_opt_in(
        self,
        review_skill: Skill,
    ) -> None:
        no_scripts = SkillDiscoveryToolset(skills=[review_skill])
        assert "run_skill_script" not in [t.name for t in no_scripts.tools()]

        with_scripts = SkillDiscoveryToolset(
            skills=[review_skill],
            enable_scripts=True,
        )
        tools = with_scripts.tools()
        names = [t.name for t in tools]
        assert "run_skill_script" in names
        # Script tool requires approval
        script_tool = next(t for t in tools if t.name == "run_skill_script")
        assert script_tool.requires_approval is True

    @pytest.mark.asyncio
    async def test_list_skills(
        self,
        review_skill: Skill,
        debug_skill: Skill,
    ) -> None:
        discovery = SkillDiscoveryToolset(skills=[review_skill, debug_skill])
        list_tool = next(t for t in discovery.tools() if t.name == "list_skills")

        result = await _invoke(list_tool, {})
        data = json.loads(result)
        assert len(data) == 2
        assert data[0]["name"] == "code-review"
        assert data[0]["description"] == "Expert code review"
        assert data[0]["tags"] == ["python", "security"]
        assert data[0]["version"] == "1.0"
        assert data[1]["name"] == "debugging"

    @pytest.mark.asyncio
    async def test_load_skill_found(self, review_skill: Skill) -> None:
        discovery = SkillDiscoveryToolset(skills=[review_skill])
        load_tool = next(t for t in discovery.tools() if t.name == "load_skill")

        result = await _invoke(load_tool, {"name": "code-review"})
        data = json.loads(result)
        assert data["name"] == "code-review"
        assert data["instructions"] == "Review code carefully"
        assert "analyze_code" in data["tools"]
        assert "references/guide.md" in data["resources"]

    @pytest.mark.asyncio
    async def test_load_skill_not_found(self, review_skill: Skill) -> None:
        discovery = SkillDiscoveryToolset(skills=[review_skill])
        load_tool = next(t for t in discovery.tools() if t.name == "load_skill")

        result = await _invoke(load_tool, {"name": "nonexistent"})
        data = json.loads(result)
        assert "error" in data
        assert "code-review" in data["available"]

    @pytest.mark.asyncio
    async def test_load_skill_resource_found(
        self,
        review_skill: Skill,
        tmp_path,
    ) -> None:
        # Create actual file
        guide = tmp_path / "guide.md"
        guide.write_text("# Guide Content")

        # Update skill resources to point to real file
        review_skill.resources = {"references/guide.md": str(guide)}

        discovery = SkillDiscoveryToolset(skills=[review_skill])
        resource_tool = next(t for t in discovery.tools() if t.name == "load_skill_resource")

        result = await _invoke(resource_tool, {"skill_name": "code-review", "resource_id": "references/guide.md"})
        assert "Guide Content" in result

    @pytest.mark.asyncio
    async def test_load_skill_resource_not_found(
        self,
        review_skill: Skill,
    ) -> None:
        discovery = SkillDiscoveryToolset(skills=[review_skill])
        resource_tool = next(t for t in discovery.tools() if t.name == "load_skill_resource")

        result = await _invoke(resource_tool, {"skill_name": "code-review", "resource_id": "nonexistent"})
        data = json.loads(result)
        assert "error" in data


class TestLoadSkillResourceInlineContent:
    """``Skill.resources`` values may be inline content, not just paths."""

    @pytest.mark.asyncio
    async def test_inline_content_returned_verbatim(self) -> None:
        """A resource value that is inline content (not an existing path)
        is returned as-is, not reported as a missing file."""
        content = "# Notes\n\nThis is inline content, not a file path."
        skill = Skill(
            name="inline-skill",
            description="inline content skill",
            resources={"notes.md": content},
        )
        discovery = SkillDiscoveryToolset(skills=[skill])
        resource_tool = next(t for t in discovery.tools() if t.name == "load_skill_resource")

        result = await _invoke(resource_tool, {"skill_name": "inline-skill", "resource_id": "notes.md"})
        assert result == content

    @pytest.mark.asyncio
    async def test_inline_content_with_nul_byte_does_not_crash(self) -> None:
        """Inline content that cannot be a filesystem path (embeds a NUL
        byte) is returned verbatim instead of raising an uncaught
        ValueError out of the tool handler."""
        content = "binary-ish\x00payload"
        skill = Skill(
            name="nul-skill",
            description="skill with NUL content",
            resources={"blob": content},
        )
        discovery = SkillDiscoveryToolset(skills=[skill])
        resource_tool = next(t for t in discovery.tools() if t.name == "load_skill_resource")

        result = await _invoke(resource_tool, {"skill_name": "nul-skill", "resource_id": "blob"})
        assert result == content

    @pytest.mark.asyncio
    async def test_real_file_path_still_read_from_disk(self, tmp_path: Path) -> None:
        """A value naming an existing file is still read from disk."""
        guide = tmp_path / "guide.md"
        guide.write_text("on-disk content")
        skill = Skill(
            name="file-skill",
            description="file-backed skill",
            resources={"references/guide.md": str(guide)},
        )
        discovery = SkillDiscoveryToolset(skills=[skill])
        resource_tool = next(t for t in discovery.tools() if t.name == "load_skill_resource")

        result = await _invoke(resource_tool, {"skill_name": "file-skill", "resource_id": "references/guide.md"})
        assert result == "on-disk content"


class TestResourceIsFilePath:
    """Unit coverage for the path-vs-inline discriminator."""

    def test_existing_file_is_a_path(self, tmp_path: Path) -> None:
        f = tmp_path / "x.txt"
        f.write_text("hi")
        assert resource_is_file_path(str(f), None) is True

    def test_nonexistent_string_is_inline(self) -> None:
        assert resource_is_file_path("not a path, just content", None) is False

    def test_nul_byte_is_inline(self) -> None:
        assert resource_is_file_path("a\x00b", None) is False

    def test_resource_root_set_forces_path(self, tmp_path: Path) -> None:
        # Directory-sourced skills always store real paths, so even a value
        # that is not currently on disk is treated as a (missing) file.
        assert resource_is_file_path(str(tmp_path / "gone.md"), tmp_path) is True


class TestRunSkillScriptInterpreter:
    @pytest.mark.asyncio
    async def test_python_script_uses_current_interpreter(self, tmp_path: Path) -> None:
        """Python scripts run under ``sys.executable``, not a bare
        ``python`` resolved from PATH (which may be missing or a different
        interpreter than the one running the ADK).

        The command is captured directly (not via stdout) so the assertion
        holds even where ``python`` happens to resolve to the same
        interpreter as ``sys.executable``.
        """
        script = tmp_path / "whoami.py"
        script.write_text("print('ok')\n")
        skill = Skill(
            name="py-skill",
            description="runs a python script",
            resources={"scripts/whoami.py": str(script)},
        )
        discovery = SkillDiscoveryToolset(skills=[skill], enable_scripts=True)
        script_tool = next(t for t in discovery.tools() if t.name == "run_skill_script")

        fake = MagicMock(stdout="ok\n", stderr="", returncode=0)
        with patch("subprocess.run", return_value=fake) as run_mock:
            await _invoke(script_tool, {"skill_name": "py-skill", "script_id": "scripts/whoami.py"})

        assert run_mock.call_count == 1
        cmd = run_mock.call_args.args[0]
        assert cmd[0] == sys.executable
        assert cmd[1] == str(script.resolve())
