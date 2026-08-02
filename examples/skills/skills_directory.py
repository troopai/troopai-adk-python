"""
Example: Loading Skills from SKILL.md Directories

Demonstrates loading skills from filesystem directories containing
SKILL.md files (compatible with LangChain/CrewAI/Google ADK format).

The loaded skill provides domain instructions and tools to a code
review agent that actually runs via Runner.

Flow:
1. Create a sample skill directory with SKILL.md + resources
2. Load the skill via Skill.from_directory()
3. Attach code analysis tools to the loaded skill
4. Run the agent on a code snippet
5. Clean up the temporary skill directory
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
import shutil
import tempfile
from pathlib import Path

from troopai.adk.agents import Agent
from troopai.adk.run import RunConfig, Runner
from troopai.adk.skills import Skill
from troopai.adk.tools import function_tool
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Create a sample skill directory (simulates a real skill package)
# =============================================================================


def create_skill_directory() -> Path:
    """Create a sample code-review skill directory."""
    skill_dir = Path(tempfile.mkdtemp()) / "code-review"
    skill_dir.mkdir()

    (skill_dir / "SKILL.md").write_text("""---
name: code-review
description: Expert Python code review with security and performance focus
version: 1.0.0
author: TroopAI
tags: python, security, performance
license: MIT
---

## Instructions

When reviewing Python code, follow this process:

1. Use `analyze_security` to check for security vulnerabilities
2. Use `check_complexity` to measure code complexity
3. Combine both analyses into a clear review with:
   - A severity rating (LOW, MEDIUM, HIGH, CRITICAL)
   - Specific line references when possible
   - Concrete fix suggestions

Be constructive — explain WHY something is an issue, not just WHAT.
""")

    # Resources the discovery toolset can expose
    (skill_dir / "references").mkdir()
    (skill_dir / "references" / "checklist.md").write_text(
        "# Code Review Checklist\n"
        "- [ ] No hardcoded secrets\n"
        "- [ ] Input validation on all user data\n"
        "- [ ] SQL queries use parameterized statements\n"
        "- [ ] Error handling doesn't leak stack traces\n"
    )

    return skill_dir


# =============================================================================
# Tools that complement the directory skill's instructions
# =============================================================================

# Patterns that indicate security issues
_DANGEROUS_PATTERNS = [
    ("hardcoded password", "password", "="),
    ("SQL injection risk", "SELECT", "%"),
    ("unsafe subprocess usage", "shell", "True"),
]


@function_tool(
    name="analyze_security",
    description="Analyze Python code for security vulnerabilities. Returns findings.",
)
def analyze_security(code: str) -> str:
    """Simulate a security analysis."""
    findings = []
    for label, pattern1, pattern2 in _DANGEROUS_PATTERNS:
        if pattern1 in code and pattern2 in code:
            findings.append(f"HIGH: {label} detected")
    if len(findings) == 0:
        findings.append("No security issues detected")
    return "Security Analysis:\n" + "\n".join(f"  - {f}" for f in findings)


@function_tool(
    name="check_complexity",
    description="Check code complexity metrics. Returns cyclomatic complexity and line count.",
)
def check_complexity(code: str) -> str:
    """Simulate complexity analysis."""
    lines = code.strip().split("\n")
    branch_keywords = ("if ", "elif ", "for ", "while ", "except ", "with ")
    branches = sum(1 for line in lines if any(kw in line for kw in branch_keywords))
    complexity = branches + 1
    return (
        f"Complexity Analysis:\n"
        f"  - Lines: {len(lines)}\n"
        f"  - Cyclomatic complexity: {complexity}\n"
        f"  - Rating: {'LOW' if complexity <= 5 else 'MEDIUM' if complexity <= 10 else 'HIGH'}"
    )


# =============================================================================
# Load skill from directory and run the agent
# =============================================================================


async def main() -> None:
    # 1. Create and load the skill from a directory
    skill_dir = create_skill_directory()
    try:
        skill = Skill.from_directory(skill_dir)

        logger.info(
            "Loaded skill '%s' v%s from %s", skill.name, skill.metadata.version if skill.metadata else "?", skill_dir
        )
        logger.info("Tags: %s", skill.metadata.tags if skill.metadata else [])
        logger.info("Resources: %s", list(skill.resources.keys()) if skill.resources else [])

        # 2. Add tools to the loaded skill (tools can come from code, not just SKILL.md)
        skill.tools = [analyze_security, check_complexity]

        # 3. Create agent with the directory-loaded skill
        agent = Agent(
            name="Code Reviewer",
            system_prompt=(
                "You are a senior Python code reviewer. Use your skill tools to "
                "analyze code, then provide a clear summary of findings."
            ),
            skills=[skill],
        )

        # 4. Run on a code snippet with intentional issues
        code_to_review = (
            "def login(username, password):\n"
            "    query = \"SELECT * FROM users WHERE name='%s'\" % (username,)\n"
            "    result = db.run(query)\n"
            "    if result:\n"
            '        token = "hardcoded_secret_key_123"\n'
            "        return token\n"
            "    return None\n"
        )

        logger.info("=" * 60)
        logger.info("Reviewing code snippet...")
        logger.info("=" * 60)

        result = await Runner.arun(
            agent,
            f"Review this Python code:\n```python\n{code_to_review}```",
            run_config=RunConfig(verbose=VerboseConfig()),
        )
        logger.info("Review:\n%s", result.final_output)
        logger.info("Tokens used: %s", result.context.usage.total_tokens)
    finally:
        # 5. Clean up the temporary skill directory
        shutil.rmtree(skill_dir.parent, ignore_errors=True)
        logger.info("Cleaned up temporary skill directory")


if __name__ == "__main__":
    asyncio.run(main())
