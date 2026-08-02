"""LLM-driven skill discovery toolset.

Provides FunctionTools that allow an LLM to discover, load, and
interact with skills at runtime.  Similar to Google ADK's
SkillToolset but integrated with the TroopAI tool system.

Discovery tools are opt-in — the developer explicitly adds them
to the agent's tool list.  No hidden behavior.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from troopai.adk.skills.skill import Skill
    from troopai.adk.tools.function_tool import FunctionTool

logger = logging.getLogger(__name__)

# Script execution is the single most dangerous tool in the ADK: the
# LLM picks a resource key, the runner forks a subprocess, and any env
# var the LLM supplies lands in that subprocess. The constants below
# pin the safe defaults.

_SCRIPT_TIMEOUT_SECONDS = 30
"""Hard wall-clock cap for a single script invocation."""

# Allowlist of parent env vars forwarded into the subprocess. We
# deliberately exclude everything else — especially anything matching
# ``*_API_KEY``, ``*_SECRET``, AWS/GCP/Azure credentials, and so on —
# because the LLM cannot be trusted not to have the script exfiltrate
# them via ``echo $OPENAI_API_KEY`` and friends.
#
# Per-var rationale (change with care — each addition widens the
# script-execution attack surface):
#
# - ``PATH``:      script interpreters (python, bash) need a PATH to
#                  resolve themselves; no PATH → ENOENT.
# - ``HOME``:      many tools read config from ``~`` (e.g. pip cache,
#                  git); setting HOME=/nonexistent breaks them. Note
#                  that a malicious parent env can populate HOME to
#                  redirect a Python script into reading ``~/.pythonstartup``
#                  — mitigated by PYTHONSTARTUP being on the blocklist.
# - ``LANG``/``LC_ALL``/``LC_CTYPE``: locale settings; scripts that
#                  parse text need these to behave consistently.
# - ``USER``/``LOGNAME``: some tools (git, ssh config) check whoami.
# - ``TERM``:      curses-based tools key off this.
# - ``TMPDIR``:    tempfile.gettempdir() reads this; without it, tmpfile
#                  creation may fail on restricted systems.
# - ``SHELL``:     inherited so that tools spawning subshells (``git``,
#                  editor launchers) pick up the user's normal shell.
#                  Residual risk: a malicious parent env that controls
#                  SHELL could influence tools that exec $SHELL — but
#                  such tools already trust the parent env in any
#                  normal invocation.
_ENV_INHERIT_ALLOWLIST: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "USER",
        "LOGNAME",
        "TERM",
        "TMPDIR",
        "SHELL",
    }
)

# Dangerous env var names that can turn any subprocess into a
# code-execution primitive. The LLM MUST NOT be allowed to inject these
# even if its argument key happens to look like a valid identifier.
# Each group is a language runtime's "load this module / run this code
# at startup" hook — equivalent to ``LD_PRELOAD`` for native binaries.
_ENV_INJECTION_BLOCKLIST: frozenset[str] = frozenset(
    {
        # ----- Shell lookup path -----
        "PATH",
        # ----- Native dynamic linker (glibc) -----
        "LD_PRELOAD",
        "LD_LIBRARY_PATH",
        "LD_AUDIT",
        # ----- Native dynamic linker (macOS dyld) -----
        "DYLD_INSERT_LIBRARIES",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        # ----- Python interpreter -----
        "PYTHONPATH",
        "PYTHONHOME",
        "PYTHONSTARTUP",
        "PYTHONINSPECT",
        # ----- POSIX shell startup hooks -----
        "BASH_ENV",
        "ENV",
        "IFS",
        "PS4",
        "SHELLOPTS",
        "PROMPT_COMMAND",
        # ----- Perl interpreter -----
        "PERL5LIB",
        "PERL5OPT",
        "PERLLIB",
        # ----- Ruby interpreter -----
        "RUBYOPT",
        "RUBYLIB",
        "GEM_HOME",
        "GEM_PATH",
        # ----- Node.js -----
        "NODE_OPTIONS",
        "NODE_PATH",
        # ----- JVM (Java / Scala / Kotlin / …) -----
        "JAVA_TOOL_OPTIONS",
        "_JAVA_OPTIONS",
        "CLASSPATH",
        # ----- Lua -----
        "LUA_PATH",
        "LUA_CPATH",
        # ----- PHP interpreter -----
        "PHP_INI_SCAN_DIR",
        "PHPRC",
        # ----- Shell directory lookup -----
        "CDPATH",
    }
)

# POSIX identifier pattern for env var names — upper-case, underscore,
# digits (but not leading digit). Anything else is rejected so the
# LLM can't construct shell-exploit names like ``PATH=:`` or use
# control characters.
_ENV_KEY_PATTERN: re.Pattern[str] = re.compile(r"^[A-Z_][A-Z0-9_]{0,63}$")

_ENV_VALUE_MAX_LEN = 4096
"""Per-value length cap to keep env growth bounded."""


def resource_is_file_path(value: str, resource_root: Path | None) -> bool:
    """Return whether a resource value should be read from disk.

    ``Skill.resources`` values are EITHER a filesystem path OR inline
    content. Directory- and remote-sourced skills (``resource_root`` set)
    always store real, resolved file paths, so those are always read from
    disk. For hand-built skills the value is read as a file only when it
    names an existing filesystem entry; anything else — including a string
    that cannot be a path at all (e.g. one embedding a NUL byte) — is
    treated as inline content and returned verbatim by the caller.

    Args:
        value: The raw resource value from ``Skill.resources``.
        resource_root: The skill's ``resource_root`` (``None`` for
            hand-built skills whose values are trusted unconditionally).

    Returns:
        ``True`` when ``value`` should be resolved and read as a file;
        ``False`` when it is inline content.
    """
    try:
        is_fs_entry = Path(value).is_symlink() or Path(value).exists()
    except (OSError, ValueError):
        # Not a usable path (too long, embeds a NUL, ...) → inline content.
        return False
    if resource_root is not None:
        return True
    return is_fs_entry


def build_script_env(script_args: dict[str, Any]) -> dict[str, str]:
    """Build a safe subprocess env from the allowlist + sanitised args.

    The parent ``os.environ`` is filtered through
    :data:`_ENV_INHERIT_ALLOWLIST`; keys supplied by the LLM are
    validated against :data:`_ENV_KEY_PATTERN`, rejected if they are in
    :data:`_ENV_INJECTION_BLOCKLIST`, and truncated to
    :data:`_ENV_VALUE_MAX_LEN` chars.

    Args:
        script_args: Key-value pairs provided by the LLM to pass as
            environment variables into the subprocess.

    Returns:
        A sanitised ``dict[str, str]`` environment mapping suitable for
        passing directly to ``subprocess.run(env=...)``.

    Raises:
        ValueError: If a script-supplied key fails validation. The
            caller converts this into a JSON error response — we fail
            loud rather than silently skipping an injected key.
    """
    env: dict[str, str] = {k: os.environ[k] for k in _ENV_INHERIT_ALLOWLIST if k in os.environ}
    for key, value in script_args.items():
        if not isinstance(key, str):
            raise ValueError(f"Script arg key must be a string; got {type(key).__name__}")
        if _ENV_KEY_PATTERN.match(key) is None:
            raise ValueError(
                f"Script arg key {key!r} is not a valid POSIX identifier (must match {_ENV_KEY_PATTERN.pattern})"
            )
        if key in _ENV_INJECTION_BLOCKLIST:
            raise ValueError(
                f"Script arg key {key!r} is on the env-injection blocklist and cannot be set from skill arguments"
            )
        env[key] = str(value)[:_ENV_VALUE_MAX_LEN]
    return env


@dataclass
class SkillDiscoveryToolset:
    """Generates FunctionTools for LLM-driven skill discovery.

    When added to an agent's tool list, these tools allow the LLM
    to discover available skills, load their instructions, access
    resources, and run scripts.

    Attributes:
        skills: The skills available for discovery.
        enable_scripts: Whether to enable the ``run_skill_script``
            tool.  Disabled by default for security.

    Example::

        from troopai.adk.skills import Skill, SkillDiscoveryToolset

        skills = [code_review_skill, data_analysis_skill]
        discovery = SkillDiscoveryToolset(skills=skills)

        agent = Agent(
            name="Assistant",
            system_prompt="...",
            skills=skills,
            tools=[*discovery.tools()],
        )
    """

    skills: list[Skill] = field(default_factory=list)
    """The skills available for discovery."""

    enable_scripts: bool = False
    """Whether to enable the ``run_skill_script`` tool.

    Disabled by default.  When enabled, the LLM can execute
    scripts from skill ``resources`` (Python and Bash).
    """

    def tools(self) -> list[FunctionTool]:
        """Generate the discovery FunctionTools.

        Returns:
            A list of FunctionTools for skill discovery.
            Always includes ``list_skills`` and ``load_skill``.
            Includes ``load_skill_resource`` if any skill has resources.
            Includes ``run_skill_script`` if ``enable_scripts=True``.
        """
        result: list[FunctionTool] = [
            self._build_list_skills_tool(),
            self._build_load_skill_tool(),
        ]

        # Only add resource tool if any skill has resources
        has_resources = any(s.resources is not None and len(s.resources) > 0 for s in self.skills)
        if has_resources:
            result.append(self._build_load_skill_resource_tool())

        if self.enable_scripts:
            result.append(self._build_run_skill_script_tool())

        return result

    def _build_list_skills_tool(self) -> FunctionTool:
        """Build the ``list_skills`` tool."""
        from troopai.adk.tools.function_tool import FunctionTool

        skills_ref = self.skills

        async def list_skills_handler(ctx: Any, raw_args: str) -> str:  # noqa: ARG001
            """List all available skills with names and descriptions."""
            entries: list[dict[str, Any]] = []
            for skill in skills_ref:
                entry: dict[str, Any] = {"name": skill.name, "description": skill.description}
                if skill.metadata is not None:
                    if skill.metadata.tags:
                        entry["tags"] = list(skill.metadata.tags)
                    if skill.metadata.version is not None:
                        entry["version"] = skill.metadata.version
                entries.append(entry)
            return json.dumps(entries, separators=(",", ":"))

        return FunctionTool(
            name="list_skills",
            description="List all available skills with their names and descriptions.",
            schema={
                "type": "object",
                "properties": {},
            },
            on_invoke=list_skills_handler,
        )

    def _build_load_skill_tool(self) -> FunctionTool:
        """Build the ``load_skill`` tool."""
        from troopai.adk.tools.function_tool import FunctionTool

        skills_ref = self.skills

        async def load_skill_handler(ctx: Any, raw_args: str) -> str:  # noqa: ARG001
            """Load the full instructions for a named skill."""
            args = json.loads(raw_args) if len(raw_args) > 0 else {}
            name = args.get("name", "")
            skill = next((s for s in skills_ref if s.name == name), None)
            if skill is None:
                available = [s.name for s in skills_ref]
                return json.dumps(
                    {
                        "error": f"Skill '{name}' not found",
                        "available": available,
                    },
                    separators=(",", ":"),
                )
            return json.dumps(
                {
                    "name": skill.name,
                    "description": skill.description,
                    "instructions": skill.instructions or "(no instructions)",
                    "tools": [getattr(t, "name", str(t)) for t in skill.tools],
                    "resources": list(skill.resources.keys()) if skill.resources else [],
                },
                separators=(",", ":"),
            )

        return FunctionTool(
            name="load_skill",
            description=(
                "Load the full instructions and details for a named skill. "
                "Call list_skills first to see available skills."
            ),
            schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill to load.",
                    },
                },
                "required": ["name"],
            },
            on_invoke=load_skill_handler,
        )

    def _build_load_skill_resource_tool(self) -> FunctionTool:
        """Build the ``load_skill_resource`` tool."""
        from troopai.adk.tools.function_tool import FunctionTool

        skills_ref = self.skills

        async def load_resource_handler(ctx: Any, raw_args: str) -> str:  # noqa: ARG001
            """Load a resource file from a skill."""
            args = json.loads(raw_args) if len(raw_args) > 0 else {}
            skill_name = args.get("skill_name", "")
            resource_id = args.get("resource_id", "")

            skill = next((s for s in skills_ref if s.name == skill_name), None)
            if skill is None:
                return json.dumps({"error": f"Skill '{skill_name}' not found"}, separators=(",", ":"))
            if skill.resources is None or resource_id not in skill.resources:
                available = list(skill.resources.keys()) if skill.resources else []
                return json.dumps(
                    {
                        "error": f"Resource '{resource_id}' not found in skill '{skill_name}'",
                        "available": available,
                    },
                    separators=(",", ":"),
                )

            raw_value = skill.resources[resource_id]

            # Skill.resources values are either a filesystem path or inline
            # content. Inline content — including a value that cannot be a
            # path at all (e.g. one embedding a NUL byte) — is returned
            # verbatim; only real file paths go through the symlink,
            # containment, and read checks below.
            if not resource_is_file_path(raw_value, skill.resource_root):
                return raw_value

            resource_path = Path(raw_value)

            # Symmetric with run_skill_script: refuse symlinks and
            # enforce resource_root containment when it is set. Without
            # this, load_skill_resource is an arbitrary-file-read
            # primitive for any caller that can influence skill.resources.
            if resource_path.is_symlink():
                return json.dumps(
                    {"error": f"Refusing to read symlinked resource: {resource_path}"},
                    separators=(",", ":"),
                )
            try:
                resolved_path = resource_path.resolve(strict=True)
            except (OSError, RuntimeError):
                return json.dumps(
                    {"error": f"Resource not found: {resource_path}"},
                    separators=(",", ":"),
                )
            if skill.resource_root is not None:
                try:
                    resolved_path.relative_to(skill.resource_root)
                except ValueError:
                    return json.dumps(
                        {
                            "error": (
                                f"Refusing to read resource outside skill "
                                f"root: {resolved_path} is not under "
                                f"{skill.resource_root}"
                            )
                        },
                        separators=(",", ":"),
                    )
            try:
                return resolved_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                return json.dumps({"error": f"Failed to read resource: {e}"}, separators=(",", ":"))

        return FunctionTool(
            name="load_skill_resource",
            description="Load a resource file from a skill. Use load_skill first to see available resources.",
            schema={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The name of the skill.",
                    },
                    "resource_id": {
                        "type": "string",
                        "description": "The resource identifier (e.g. 'references/guide.md').",
                    },
                },
                "required": ["skill_name", "resource_id"],
            },
            on_invoke=load_resource_handler,
        )

    def _build_run_skill_script_tool(self) -> FunctionTool:
        """Build the ``run_skill_script`` tool."""
        from troopai.adk.tools.function_tool import FunctionTool

        skills_ref = self.skills

        async def run_script_handler(ctx: Any, raw_args: str) -> str:  # noqa: ARG001
            """Execute a script from a skill's resources."""
            args = json.loads(raw_args) if len(raw_args) > 0 else {}
            skill_name = args.get("skill_name", "")
            script_id = args.get("script_id", "")
            script_args_raw = args.get("arguments", {})
            if not isinstance(script_args_raw, dict):
                return json.dumps(
                    {"error": "'arguments' must be a JSON object"},
                    separators=(",", ":"),
                )
            script_args: dict[str, Any] = script_args_raw

            skill = next((s for s in skills_ref if s.name == skill_name), None)
            if skill is None:
                return json.dumps({"error": f"Skill '{skill_name}' not found"}, separators=(",", ":"))
            if skill.resources is None or script_id not in skill.resources:
                return json.dumps(
                    {"error": f"Script '{script_id}' not found in skill '{skill_name}'"}, separators=(",", ":")
                )

            script_path = Path(skill.resources[script_id])

            # Defence in depth against symlink trickery.
            #
            #   1. ``is_symlink()`` catches the common case where
            #      ``skill.resources[...]`` is itself a symlink.
            #   2. Comparing ``lstat`` (does not follow symlinks) with
            #      ``stat`` (follows them) catches the narrow TOCTOU
            #      window where the file flips between ``is_symlink()``
            #      and the stat call — the two return different inodes
            #      iff any link was followed.
            #
            # There is still a residual TOCTOU window between these
            # checks and ``subprocess.run``: an attacker with write
            # access to the skill directory could swap the file again.
            # That attacker can already overwrite the script body
            # directly, so the residual risk is bounded by "attacker
            # has skill-directory write access" — if that invariant
            # breaks, script execution is the smallest of our problems.
            if script_path.is_symlink():
                return json.dumps(
                    {"error": f"Refusing to run symlinked script: {script_path}"},
                    separators=(",", ":"),
                )
            try:
                lstat = script_path.lstat()
                followed = script_path.stat()
            except OSError:
                return json.dumps(
                    {"error": f"Script file not found: {script_path}"},
                    separators=(",", ":"),
                )
            if (lstat.st_dev, lstat.st_ino) != (followed.st_dev, followed.st_ino):
                return json.dumps(
                    {"error": f"Refusing to run symlinked script: {script_path}"},
                    separators=(",", ":"),
                )
            try:
                resolved_path = script_path.resolve(strict=True)
            except (OSError, RuntimeError):
                return json.dumps(
                    {"error": f"Script file not found: {script_path}"},
                    separators=(",", ":"),
                )
            if not resolved_path.is_file():
                return json.dumps(
                    {"error": f"Refusing to run non-regular file: {resolved_path}"},
                    separators=(",", ":"),
                )

            # Execute-time containment: when the skill carries a
            # ``resource_root`` (set by ``DirectorySkillSource`` to the
            # canonical skill directory), the resolved script path MUST
            # sit inside that root. This blocks both the hand-crafted
            # ``resources={"scripts/x.sh": "/etc/passwd"}`` case and any
            # post-load file relocation that moves the target outside
            # the skill tree. ``resource_root=None`` means "trust the
            # caller" — fine for inline-built skills in tests.
            if skill.resource_root is not None:
                try:
                    resolved_path.relative_to(skill.resource_root)
                except ValueError:
                    return json.dumps(
                        {
                            "error": (
                                f"Refusing to run script outside skill root: "
                                f"{resolved_path} is not under {skill.resource_root}"
                            )
                        },
                        separators=(",", ":"),
                    )

            suffix = resolved_path.suffix.lower()
            if suffix == ".py":
                # Use the running interpreter, not whatever "python" happens
                # to resolve to on PATH (which may be missing or a different
                # Python than the one executing the ADK).
                cmd = [sys.executable, str(resolved_path)]
            elif suffix in (".sh", ".bash"):
                cmd = ["bash", str(resolved_path)]
            else:
                return json.dumps({"error": f"Unsupported script type: {suffix}"}, separators=(",", ":"))

            try:
                env = build_script_env(script_args)
            except ValueError as exc:
                return json.dumps({"error": str(exc)}, separators=(",", ":"))

            try:
                result = await asyncio.to_thread(
                    subprocess.run,
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=_SCRIPT_TIMEOUT_SECONDS,
                    env=env,
                    cwd=str(resolved_path.parent),
                )
                return json.dumps(
                    {
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "returncode": result.returncode,
                    },
                    separators=(",", ":"),
                )
            except subprocess.TimeoutExpired:
                return json.dumps(
                    {"error": f"Script execution timed out ({_SCRIPT_TIMEOUT_SECONDS}s)"},
                    separators=(",", ":"),
                )
            except (OSError, subprocess.SubprocessError) as e:
                return json.dumps({"error": f"Script execution failed: {e}"}, separators=(",", ":"))

        return FunctionTool(
            name="run_skill_script",
            description=(
                "Execute a script from a skill's resources. "
                "Only available when enable_scripts=True. "
                "Supports Python (.py) and Bash (.sh/.bash) scripts."
            ),
            schema={
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "The name of the skill.",
                    },
                    "script_id": {
                        "type": "string",
                        "description": "The script resource identifier (e.g. 'scripts/analyze.py').",
                    },
                    "arguments": {
                        "type": "object",
                        "description": "Key-value arguments passed as environment variables.",
                    },
                },
                "required": ["skill_name", "script_id"],
            },
            on_invoke=run_script_handler,
            requires_approval=True,  # Scripts require human approval by default
        )
