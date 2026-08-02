"""Tests for DirectorySkillSource and SKILL.md parsing."""

from pathlib import Path

import pytest

from troopai.adk.skills import Skill
from troopai.adk.skills.sources.directory import (
    DirectorySkillSource,
    parse_skill_md,
)


class TestParseSkillMd:
    """Test SKILL.md frontmatter parser."""

    def test_basic(self) -> None:
        content = """---
name: test-skill
description: A test skill
---

## Instructions

Do something useful.
"""
        frontmatter, body = parse_skill_md(content)
        assert frontmatter["name"] == "test-skill"
        assert frontmatter["description"] == "A test skill"
        assert "Do something useful" in body

    def test_all_fields(self) -> None:
        content = """---
name: code-review
description: Expert code review
version: 1.0.0
author: TroopAI
tags: python, security, review
license: MIT
---

Review code carefully.
"""
        frontmatter, body = parse_skill_md(content)
        assert frontmatter["name"] == "code-review"
        assert frontmatter["version"] == "1.0.0"
        assert frontmatter["author"] == "TroopAI"
        assert frontmatter["tags"] == "python, security, review"
        assert frontmatter["license"] == "MIT"
        assert "Review code carefully" in body

    def test_no_frontmatter_raises(self) -> None:
        with pytest.raises(ValueError, match="YAML frontmatter"):
            parse_skill_md("Just some text without frontmatter")

    def test_empty_body(self) -> None:
        content = """---
name: minimal
description: Minimal skill
---
"""
        frontmatter, body = parse_skill_md(content)
        assert frontmatter["name"] == "minimal"
        assert body == ""

    def test_double_quoted_scalar_is_unquoted(self) -> None:
        content = '---\nname: "my-skill"\ndescription: "Quoted desc"\n---\nBody.\n'
        frontmatter, _ = parse_skill_md(content)
        assert frontmatter["name"] == "my-skill"
        assert frontmatter["description"] == "Quoted desc"

    def test_single_quoted_scalar_is_unquoted(self) -> None:
        content = "---\nname: 'my-skill'\ndescription: 'Quoted desc'\n---\nBody.\n"
        frontmatter, _ = parse_skill_md(content)
        assert frontmatter["name"] == "my-skill"
        assert frontmatter["description"] == "Quoted desc"

    def test_quoted_value_with_colon_preserved(self) -> None:
        # Quoting is the YAML idiom for a scalar containing a colon; the
        # inner colon must survive unquoting.
        content = '---\nname: x\ndescription: "Analyzes code: Python and JS"\n---\nBody.\n'
        frontmatter, _ = parse_skill_md(content)
        assert frontmatter["description"] == "Analyzes code: Python and JS"

    def test_unquoted_comma_value_not_mangled(self) -> None:
        content = "---\nname: x\ndescription: y\ntags: python, security, review\n---\nBody.\n"
        frontmatter, _ = parse_skill_md(content)
        assert frontmatter["tags"] == "python, security, review"

    def test_frontmatter_only_no_trailing_newline(self) -> None:
        # A structurally valid file whose closing '---' is the last byte
        # (no trailing newline) must parse, not raise.
        content = "---\nname: test\ndescription: A test skill\n---"
        frontmatter, body = parse_skill_md(content)
        assert frontmatter["name"] == "test"
        assert frontmatter["description"] == "A test skill"
        assert body == ""

    def test_directory_source_quoted_name_is_findable(self, tmp_path: Path) -> None:
        # A loaded skill with a quoted frontmatter name must carry the bare
        # name so equality-based lookups (discovery / SkillSet) can find it.
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text('---\nname: "my-skill"\ndescription: "Quoted skill"\n---\nDo work.\n')
        skill = DirectorySkillSource(path=tmp_path).load_sync()
        assert skill.name == "my-skill"
        assert skill.description == "Quoted skill"


class TestDirectorySkillSource:
    """Test loading skills from filesystem directories."""

    def test_load_full_skill(self, tmp_path: Path) -> None:
        # Create SKILL.md
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("""---
name: code-review
description: Expert code review with security focus
version: 1.0.0
author: TroopAI
tags: python, security
license: MIT
---

## Instructions

When reviewing code:
1. Check for security vulnerabilities
2. Suggest improvements
""")
        # Create resources
        (tmp_path / "scripts").mkdir()
        (tmp_path / "scripts" / "lint.py").write_text("import sys")
        (tmp_path / "references").mkdir()
        (tmp_path / "references" / "guide.md").write_text("# Guide")

        source = DirectorySkillSource(path=tmp_path)
        skill = source.load_sync()

        assert skill.name == "code-review"
        assert skill.description == "Expert code review with security focus"
        assert skill.instructions is not None
        assert "security vulnerabilities" in skill.instructions
        assert skill.metadata is not None
        assert skill.metadata.version == "1.0.0"
        assert skill.metadata.author == "TroopAI"
        assert "python" in skill.metadata.tags
        assert "security" in skill.metadata.tags
        assert skill.metadata.license == "MIT"
        assert skill.resources is not None
        assert "scripts/lint.py" in skill.resources
        assert "references/guide.md" in skill.resources

    def test_missing_skill_md_raises(self, tmp_path: Path) -> None:
        source = DirectorySkillSource(path=tmp_path)
        with pytest.raises(FileNotFoundError, match="SKILL.md not found"):
            source.load_sync()

    def test_missing_name_raises(self, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("""---
description: Missing name
---

Body text.
""")
        source = DirectorySkillSource(path=tmp_path)
        with pytest.raises(ValueError, match="missing required 'name'"):
            source.load_sync()

    def test_missing_description_raises(self, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("""---
name: test
---

Body text.
""")
        source = DirectorySkillSource(path=tmp_path)
        with pytest.raises(ValueError, match="missing required 'description'"):
            source.load_sync()

    def test_from_directory_classmethod(self, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("""---
name: from-classmethod
description: Test classmethod
---

Instructions here.
""")
        skill = Skill.from_directory(tmp_path)
        assert skill.name == "from-classmethod"
        assert skill.description == "Test classmethod"

    @pytest.mark.asyncio
    async def test_async_load(self, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("""---
name: async-test
description: Async load test
---

Async body.
""")
        source = DirectorySkillSource(path=tmp_path)
        skill = await source.load()
        assert skill.name == "async-test"

    @pytest.mark.asyncio
    async def test_load_metadata(self, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("""---
name: meta-test
description: Metadata only test
---

Full body that should not be returned.
""")
        source = DirectorySkillSource(path=tmp_path)
        name, description = await source.load_metadata()
        assert name == "meta-test"
        assert description == "Metadata only test"

    def test_no_resources(self, tmp_path: Path) -> None:
        skill_md = tmp_path / "SKILL.md"
        skill_md.write_text("""---
name: no-resources
description: No resource dirs
---

Just instructions.
""")
        source = DirectorySkillSource(path=tmp_path)
        skill = source.load_sync()
        assert skill.resources is None
