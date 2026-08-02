"""Tests pinning S2 + S9 — ``run_skill_script`` must not become a
remote-code-execution primitive.

Three attack surfaces are covered:

1. **Env injection (S2)** — the LLM-controlled ``arguments`` dict
   MUST NOT be able to set dangerous keys (``LD_PRELOAD``,
   ``PYTHONPATH``, …) or use non-identifier names, and the parent
   process's full ``os.environ`` (API keys, cloud creds) MUST NOT
   be forwarded.
2. **Symlink escape (S9, part 1 — load time)** —
   ``DirectorySkillSource.collect_resources`` MUST skip symlinks
   and any file whose resolved path leaves the skill directory.
3. **Symlink escape (S9, part 2 — execute time)** — even when the
   caller hand-builds ``Skill.resources`` with a symlink entry,
   ``run_skill_script`` MUST refuse to run it.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from troopai.adk.skills import Skill, SkillDiscoveryToolset
from troopai.adk.skills.discovery import build_script_env
from troopai.adk.skills.sources.directory import collect_resources
from troopai.adk.tools import FunctionTool
from troopai.adk.tools.tool_context import ToolContext


def _dummy_ctx(tool_name: str, raw_args: str) -> ToolContext[Any]:
    """Construct a minimal :class:`ToolContext` for discovery-handler tests.

    The discovery handlers all ``# noqa: ARG001`` their ``ctx`` parameter,
    so the handler body never dereferences the context. We still pass a
    real ``ToolContext`` (rather than ``cast(..., None)``) so the test
    stays honest about the signature and survives any future change that
    starts reading ``ctx`` fields.
    """
    return ToolContext[Any](
        tool_name=tool_name,
        tool_call_id=f"call_test_{tool_name}",
        tool_arguments={},
        raw_arguments=raw_args,
    )


async def _invoke(tool: FunctionTool, raw_args: str) -> str:
    """Invoke a discovery FunctionTool and assert a string return."""
    assert tool.on_invoke is not None
    result = await tool.on_invoke(_dummy_ctx(tool.name, raw_args), raw_args)
    assert isinstance(result, str), f"expected str result from {tool.name}, got {type(result)!r}"
    return result


# --------------------------------------------------------------------------
# S2 — env allowlist / blocklist / identifier pattern
# --------------------------------------------------------------------------


class TestBuildScriptEnv:
    def test_only_allowlisted_parent_env_vars_are_inherited(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        # Secret-shaped vars MUST NOT leak into the subprocess.
        monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")

        env = build_script_env({})

        assert env.get("PATH") == "/usr/bin:/bin"
        assert "OPENAI_API_KEY" not in env
        assert "AWS_SECRET_ACCESS_KEY" not in env

    def test_ld_preload_injection_rejected(self) -> None:
        with pytest.raises(ValueError, match="blocklist"):
            build_script_env({"LD_PRELOAD": "/tmp/evil.so"})

    def test_pythonpath_injection_rejected(self) -> None:
        with pytest.raises(ValueError, match="blocklist"):
            build_script_env({"PYTHONPATH": "/tmp/evil"})

    def test_dyld_insert_libraries_rejected(self) -> None:
        with pytest.raises(ValueError, match="blocklist"):
            build_script_env({"DYLD_INSERT_LIBRARIES": "/tmp/evil.dylib"})

    @pytest.mark.parametrize(
        "key,value",
        [
            ("PERL5LIB", "/tmp/evil-perl"),
            ("PERL5OPT", "-M/tmp/evil"),
            ("RUBYOPT", "-r/tmp/evil"),
            ("RUBYLIB", "/tmp/evil-ruby"),
            ("NODE_OPTIONS", "--require /tmp/evil.js"),
            ("NODE_PATH", "/tmp/evil-node"),
            ("JAVA_TOOL_OPTIONS", "-javaagent:/tmp/evil.jar"),
            ("_JAVA_OPTIONS", "-javaagent:/tmp/evil.jar"),
            ("CLASSPATH", "/tmp/evil.jar"),
            ("LUA_PATH", "/tmp/?.lua;;"),
            ("LUA_CPATH", "/tmp/?.so;;"),
            ("PROMPT_COMMAND", "curl evil.example.com"),
            ("PYTHONINSPECT", "1"),
        ],
    )
    def test_language_runtime_loader_vars_rejected(self, key: str, value: str) -> None:
        """Each non-Python language runtime has its own "load code at
        startup" hook. Block them all by name so that if a skill script
        happens to invoke Perl/Ruby/Node/Java/Lua, the LLM cannot
        hijack the runtime by naming the env var."""
        with pytest.raises(ValueError, match="blocklist"):
            build_script_env({key: value})

    def test_path_override_rejected(self) -> None:
        with pytest.raises(ValueError, match="blocklist"):
            build_script_env({"PATH": "/tmp/evil-bin"})

    def test_lowercase_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="POSIX identifier"):
            build_script_env({"lowercase_key": "value"})

    def test_key_with_special_chars_rejected(self) -> None:
        with pytest.raises(ValueError, match="POSIX identifier"):
            build_script_env({"BAD KEY": "value"})

    def test_key_with_equals_sign_rejected(self) -> None:
        with pytest.raises(ValueError, match="POSIX identifier"):
            build_script_env({"BAD=KEY": "value"})

    def test_key_starting_with_digit_rejected(self) -> None:
        with pytest.raises(ValueError, match="POSIX identifier"):
            build_script_env({"1BAD": "value"})

    def test_valid_arg_key_allowed(self) -> None:
        env = build_script_env({"SAFE_ARG": "hello"})
        assert env["SAFE_ARG"] == "hello"

    def test_non_string_key_rejected(self) -> None:
        with pytest.raises(ValueError, match="string"):
            build_script_env({42: "value"})  # type: ignore[dict-item]


# --------------------------------------------------------------------------
# S9 part 1 — symlinks skipped at collect_resources time
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="symlink semantics vary on Windows",
)
class TestCollectResourcesSymlinkEscape:
    def test_symlink_pointing_outside_is_skipped(self, tmp_path: Path) -> None:
        # Set up a skill dir with an escape-symlink in scripts/.
        skill_dir = tmp_path / "my-skill"
        (skill_dir / "scripts").mkdir(parents=True)
        external = tmp_path / "outside_secret.txt"
        external.write_text("hunter2")

        evil_link = skill_dir / "scripts" / "leak.sh"
        os.symlink(external, evil_link)

        # Plus one legit file, to prove the collector still runs.
        legit = skill_dir / "scripts" / "ok.sh"
        legit.write_text("#!/bin/bash\necho ok\n")

        resources = collect_resources(skill_dir)
        assert resources is not None
        # Legit file is present; symlink escape is NOT.
        assert "scripts/ok.sh" in resources
        assert "scripts/leak.sh" not in resources

    def test_symlink_pointing_inside_skill_dir_is_still_skipped(self, tmp_path: Path) -> None:
        """Even an in-tree symlink is skipped — simpler contract and
        removes any chance of the scanner doubling up a resource."""
        skill_dir = tmp_path / "my-skill"
        (skill_dir / "scripts").mkdir(parents=True)
        (skill_dir / "references").mkdir()
        real = skill_dir / "references" / "real.md"
        real.write_text("real content")

        inner_link = skill_dir / "scripts" / "alias.md"
        os.symlink(real, inner_link)

        resources = collect_resources(skill_dir)
        assert resources is not None
        assert "references/real.md" in resources
        assert "scripts/alias.md" not in resources


# --------------------------------------------------------------------------
# S9 part 2 + S2 integration — execute-time refusal
# --------------------------------------------------------------------------


def _make_skill(
    resources: dict[str, str],
    *,
    resource_root: Path | None = None,
) -> Skill:
    return Skill(
        name="test-skill",
        description="test",
        resources=resources,
        resource_root=resource_root,
    )


class TestRunSkillScriptRejections:
    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="symlink semantics vary on Windows",
    )
    async def test_refuses_symlinked_script(self, tmp_path: Path) -> None:
        real = tmp_path / "real.sh"
        real.write_text("#!/bin/bash\necho ok\n")
        link = tmp_path / "link.sh"
        os.symlink(real, link)

        skill = _make_skill({"scripts/x.sh": str(link)})
        discovery = SkillDiscoveryToolset(skills=[skill], enable_scripts=True)
        script_tool = next(t for t in discovery.tools() if t.name == "run_skill_script")

        result_json = await _invoke(
            script_tool,
            json.dumps({"skill_name": "test-skill", "script_id": "scripts/x.sh"}),
        )
        data: dict[str, Any] = json.loads(result_json)
        assert "error" in data
        assert "symlink" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_arguments_must_be_object(self, tmp_path: Path) -> None:
        real = tmp_path / "ok.sh"
        real.write_text("#!/bin/bash\necho ok\n")
        skill = _make_skill({"scripts/ok.sh": str(real)})
        discovery = SkillDiscoveryToolset(skills=[skill], enable_scripts=True)
        script_tool = next(t for t in discovery.tools() if t.name == "run_skill_script")

        result_json = await _invoke(
            script_tool,
            json.dumps(
                {
                    "skill_name": "test-skill",
                    "script_id": "scripts/ok.sh",
                    "arguments": ["not", "an", "object"],
                }
            ),
        )
        data: dict[str, Any] = json.loads(result_json)
        assert "error" in data
        assert "object" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_rejects_ld_preload_in_arguments(self, tmp_path: Path) -> None:
        real = tmp_path / "ok.sh"
        real.write_text("#!/bin/bash\necho ok\n")
        skill = _make_skill({"scripts/ok.sh": str(real)})
        discovery = SkillDiscoveryToolset(skills=[skill], enable_scripts=True)
        script_tool = next(t for t in discovery.tools() if t.name == "run_skill_script")

        result_json = await _invoke(
            script_tool,
            json.dumps(
                {
                    "skill_name": "test-skill",
                    "script_id": "scripts/ok.sh",
                    "arguments": {"LD_PRELOAD": "/tmp/evil.so"},
                }
            ),
        )
        data: dict[str, Any] = json.loads(result_json)
        assert "error" in data
        assert "blocklist" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_missing_script_returns_error(self, tmp_path: Path) -> None:
        skill = _make_skill({"scripts/nope.sh": str(tmp_path / "nope.sh")})
        discovery = SkillDiscoveryToolset(skills=[skill], enable_scripts=True)
        script_tool = next(t for t in discovery.tools() if t.name == "run_skill_script")

        result_json = await _invoke(
            script_tool,
            json.dumps({"skill_name": "test-skill", "script_id": "scripts/nope.sh"}),
        )
        data: dict[str, Any] = json.loads(result_json)
        assert "error" in data
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_unsupported_suffix_rejected(self, tmp_path: Path) -> None:
        binary = tmp_path / "evil.bin"
        binary.write_text("#!/bin/bash\necho pwned\n")
        skill = _make_skill({"scripts/evil.bin": str(binary)})
        discovery = SkillDiscoveryToolset(skills=[skill], enable_scripts=True)
        script_tool = next(t for t in discovery.tools() if t.name == "run_skill_script")

        result_json = await _invoke(
            script_tool,
            json.dumps(
                {
                    "skill_name": "test-skill",
                    "script_id": "scripts/evil.bin",
                }
            ),
        )
        data: dict[str, Any] = json.loads(result_json)
        assert "error" in data
        assert "Unsupported" in data["error"]


# --------------------------------------------------------------------------
# S9 part 3 — execute-time containment against Skill.resource_root
# --------------------------------------------------------------------------


class TestResourceRootContainment:
    """When a skill carries a ``resource_root`` (populated by
    ``DirectorySkillSource`` at load time), ``run_skill_script`` MUST
    refuse to execute any script whose resolved path escapes that
    root. This blocks the hand-crafted
    ``resources={"scripts/x.sh": "/etc/passwd"}`` case — an attack
    that the symlink and suffix checks alone cannot catch."""

    @pytest.mark.asyncio
    async def test_refuses_script_outside_skill_root(self, tmp_path: Path) -> None:
        # Legit skill root with nothing inside it.
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        # Attacker-controlled .sh outside the skill dir.
        outside = tmp_path / "attacker.sh"
        outside.write_text("#!/bin/bash\necho pwned\n")
        skill = _make_skill(
            {"scripts/attacker.sh": str(outside)},
            resource_root=skill_dir.resolve(),
        )
        discovery = SkillDiscoveryToolset(skills=[skill], enable_scripts=True)
        script_tool = next(t for t in discovery.tools() if t.name == "run_skill_script")

        result_json = await _invoke(
            script_tool,
            json.dumps(
                {
                    "skill_name": "test-skill",
                    "script_id": "scripts/attacker.sh",
                }
            ),
        )
        data: dict[str, Any] = json.loads(result_json)
        assert "error" in data
        assert "outside skill root" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_allows_script_inside_skill_root(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        (skill_dir / "scripts").mkdir(parents=True)
        legit = skill_dir / "scripts" / "ok.sh"
        legit.write_text("#!/bin/bash\necho ok\n")
        skill = _make_skill(
            {"scripts/ok.sh": str(legit)},
            resource_root=skill_dir.resolve(),
        )
        discovery = SkillDiscoveryToolset(skills=[skill], enable_scripts=True)
        script_tool = next(t for t in discovery.tools() if t.name == "run_skill_script")

        result_json = await _invoke(
            script_tool,
            json.dumps({"skill_name": "test-skill", "script_id": "scripts/ok.sh"}),
        )
        data: dict[str, Any] = json.loads(result_json)
        # Either it ran (returncode present) or it errored — but the
        # error MUST NOT be about containment, because this script is
        # inside the declared root.
        if "error" in data:
            assert "outside skill root" not in data["error"].lower()
        else:
            assert "returncode" in data

    @pytest.mark.asyncio
    async def test_no_resource_root_allows_arbitrary_path(self, tmp_path: Path) -> None:
        """Inline-built skills (``resource_root=None``) deliberately
        skip containment — the caller vouches for the paths. This is
        the existing hand-built behaviour that the previous tests in
        this file rely on; pinning it here so a future tightening
        doesn't silently break inline skill construction."""
        elsewhere = tmp_path / "anywhere.sh"
        elsewhere.write_text("#!/bin/bash\necho ok\n")
        skill = _make_skill({"scripts/anywhere.sh": str(elsewhere)})
        assert skill.resource_root is None
        discovery = SkillDiscoveryToolset(skills=[skill], enable_scripts=True)
        script_tool = next(t for t in discovery.tools() if t.name == "run_skill_script")

        result_json = await _invoke(
            script_tool,
            json.dumps(
                {
                    "skill_name": "test-skill",
                    "script_id": "scripts/anywhere.sh",
                }
            ),
        )
        data: dict[str, Any] = json.loads(result_json)
        # No containment error is emitted when resource_root is None.
        if "error" in data:
            assert "outside skill root" not in data["error"].lower()


# --------------------------------------------------------------------------
# S9 part 4 — DirectorySkillSource populates resource_root
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform.startswith("win"),
    reason="symlink resolution semantics vary on Windows",
)
class TestDirectorySourcePopulatesResourceRoot:
    def test_loaded_skill_has_resolved_resource_root(self, tmp_path: Path) -> None:
        """The skill carries its canonical (resolved) root so that
        execute-time containment has something to check against."""
        from troopai.adk.skills.sources.directory import DirectorySkillSource

        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: my-skill\ndescription: test\n---\nhello\n")
        source = DirectorySkillSource(path=skill_dir)
        skill = source.load_sync()

        assert skill.resource_root is not None
        assert skill.resource_root == skill_dir.resolve(strict=True)
        # The root is absolute — relative paths would break the
        # ``resolved.relative_to(resource_root)`` containment check.
        assert skill.resource_root.is_absolute()


# --------------------------------------------------------------------------
# S9 part 5 — load_skill_resource enforces the same containment as scripts
# --------------------------------------------------------------------------


class TestLoadSkillResourceContainment:
    """Symmetric with ``run_skill_script``: the read tool MUST refuse
    symlinks and paths escaping ``resource_root``. Without this, a
    caller who can influence ``skill.resources`` turns the tool into an
    arbitrary-file-read primitive."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="symlink semantics vary on Windows",
    )
    async def test_refuses_symlinked_resource(self, tmp_path: Path) -> None:
        real = tmp_path / "real.md"
        real.write_text("content")
        link = tmp_path / "link.md"
        os.symlink(real, link)

        skill = _make_skill({"references/link.md": str(link)})
        discovery = SkillDiscoveryToolset(skills=[skill])
        resource_tool = next(t for t in discovery.tools() if t.name == "load_skill_resource")

        result_json = await _invoke(
            resource_tool,
            json.dumps(
                {
                    "skill_name": "test-skill",
                    "resource_id": "references/link.md",
                }
            ),
        )
        data: dict[str, Any] = json.loads(result_json)
        assert "error" in data
        assert "symlink" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_refuses_resource_outside_skill_root(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        outside = tmp_path / "secret.txt"
        outside.write_text("hunter2")

        skill = _make_skill(
            {"references/secret.txt": str(outside)},
            resource_root=skill_dir.resolve(),
        )
        discovery = SkillDiscoveryToolset(skills=[skill])
        resource_tool = next(t for t in discovery.tools() if t.name == "load_skill_resource")

        result_json = await _invoke(
            resource_tool,
            json.dumps(
                {
                    "skill_name": "test-skill",
                    "resource_id": "references/secret.txt",
                }
            ),
        )
        data: dict[str, Any] = json.loads(result_json)
        assert "error" in data
        assert "outside skill" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_allows_resource_inside_skill_root(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "my-skill"
        (skill_dir / "references").mkdir(parents=True)
        legit = skill_dir / "references" / "guide.md"
        legit.write_text("legitimate")

        skill = _make_skill(
            {"references/guide.md": str(legit)},
            resource_root=skill_dir.resolve(),
        )
        discovery = SkillDiscoveryToolset(skills=[skill])
        resource_tool = next(t for t in discovery.tools() if t.name == "load_skill_resource")

        result_json = await _invoke(
            resource_tool,
            json.dumps(
                {
                    "skill_name": "test-skill",
                    "resource_id": "references/guide.md",
                }
            ),
        )
        assert result_json == "legitimate"
