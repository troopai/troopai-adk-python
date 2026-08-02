"""
Example: LLM-Driven Skill Discovery with SkillDiscoveryToolset

Demonstrates giving an LLM tools to discover and explore available
skills at runtime.  The system prompt is intentionally generic — the
agent must figure out how to use list_skills, load_skill, and
load_skill_resource purely from their tool descriptions.

Two scenarios show discovery in action:

- **Scenario 1** (diagnostics): The agent naturally discovers and calls
  ``run_diagnostics`` — straightforward single-tool discovery.

- **Scenario 2** (rate limits): The answer is in a skill *resource*
  file, not in the search index.  The agent needs to chain:
  ``load_skill`` → see the resources list → ``load_skill_resource``.
  More capable models (Sonnet, Opus) handle this multi-step discovery
  well; smaller models (Haiku) may fall back to ``search_docs`` and
  miss the resource.  This is an honest trade-off between model
  capability and discovery depth.

The discovery tools are opt-in: added explicitly to Agent.tools.
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
from troopai.adk.skills import (
    Skill,
    SkillActivation,
    SkillDiscoveryToolset,
    SkillMetadata,
    prompt_with_skill_instructions,
)
from troopai.adk.tools import function_tool
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Skills with tools
# =============================================================================


@function_tool(
    name="search_docs",
    description="Search the documentation for a topic. Returns matching sections.",
)
def search_docs(query: str) -> str:
    """Simulate documentation search."""
    docs = {
        "authentication": "Use JWT tokens with short expiry. Rotate secrets monthly.",
        "database": "Use connection pooling. Max pool size: 20. Timeout: 30s.",
        "deployment": "Use blue-green deployments. Rollback window: 15 minutes.",
        "testing": "Minimum 80% coverage. All API endpoints must have integration tests.",
    }
    matches = [f"- **{k}**: {v}" for k, v in docs.items() if query.lower() in k]
    if len(matches) > 0:
        return f"Documentation results for '{query}':\n" + "\n".join(matches)
    return f"No documentation found for '{query}'. Indexed topics: {', '.join(docs)}."


@function_tool(
    name="run_diagnostics",
    description="Run system diagnostics. Returns health status of services.",
)
def run_diagnostics(service: str) -> str:
    """Simulate diagnostics."""
    services = {
        "api": {"status": "healthy", "latency_ms": 45, "error_rate": "0.1%"},
        "database": {"status": "healthy", "latency_ms": 12, "connections": 15},
        "cache": {"status": "degraded", "latency_ms": 120, "hit_rate": "72%"},
    }
    info = services.get(service.lower())
    if info is None:
        return f"Unknown service '{service}'. Available: {', '.join(services)}"
    return f"Diagnostics for {service}: {info}"


# Create a temporary resources directory for the docs skill
_tmp_dir = Path(tempfile.mkdtemp())
_references_dir = _tmp_dir / "references"
_references_dir.mkdir()
(_references_dir / "api-guide.md").write_text(
    "# API Guide\n\n"
    "## Authentication\n"
    "All endpoints require a Bearer token.\n\n"
    "## Rate Limits\n"
    "- Standard: 100 req/min\n"
    "- Premium: 1000 req/min\n"
)

documentation_skill = Skill(
    name="documentation",
    description="Search and retrieve internal documentation and API guides",
    instructions=(
        "When users ask about how things work, search the documentation first. "
        "If the docs don't have the answer, say so clearly."
    ),
    tools=[search_docs],
    metadata=SkillMetadata(
        version="2.1.0",
        tags=("docs", "knowledge", "reference"),
    ),
    resources={
        "references/api-guide.md": str(_references_dir / "api-guide.md"),
    },
)

operations_skill = Skill(
    name="operations",
    description="System diagnostics, health checks, and operational monitoring",
    instructions=(
        "When users report issues or ask about system status, run diagnostics on the relevant service first."
    ),
    tools=[run_diagnostics],
    metadata=SkillMetadata(
        version="1.0.0",
        tags=("ops", "monitoring", "health"),
    ),
)


# =============================================================================
# Agent with discovery toolset
# =============================================================================

all_skills = [documentation_skill, operations_skill]
discovery = SkillDiscoveryToolset(skills=all_skills)

BASE_PROMPT = "You are a platform engineering assistant. Use your available tools to help users."

# Approach 1: Generic prompt — the agent discovers tools from their
# descriptions alone.  Works well with capable models (Sonnet, Opus);
# smaller models (Haiku) may miss multi-step discovery chains.
agent_without_instructions = Agent(
    name="Platform Assistant (no instructions)",
    system_prompt=BASE_PROMPT,
    skills=all_skills,
    skill_activation=SkillActivation.EAGER,
    tools=[*discovery.tools()],
)

# Approach 2: prompt_with_skill_instructions() — opt-in prefix that
# teaches the LLM the discovery workflow (list → load → act).  Same
# pattern as handoff_prompt.py.  More reliable across all model sizes.
agent_with_instructions = Agent(
    name="Platform Assistant (with instructions)",
    system_prompt=prompt_with_skill_instructions(BASE_PROMPT),
    skills=all_skills,
    skill_activation=SkillActivation.EAGER,
    tools=[*discovery.tools()],
)


# =============================================================================
# Run scenarios — same query against both agents for comparison
# =============================================================================

QUERY = "What are our API rate limits?"
"""Multi-step discovery query.

The answer lives in references/api-guide.md (a skill resource),
NOT in the search_docs index.  The ideal tool chain is:
    search_docs (miss) → load_skill → load_skill_resource

Smaller models may stop at search_docs and give up.
The skill prompt prefix helps them chain deeper.
"""


async def run_diagnostics_scenario() -> None:
    """Simple single-hop discovery — both agents handle this well."""
    logger.info("=" * 60)
    logger.info("Scenario 1: Cache is slow (single-hop discovery)")
    logger.info("=" * 60)

    result = await Runner.arun(
        agent_with_instructions,
        "Our cache seems slow. Can you check what's going on?",
        run_config=RunConfig(verbose=VerboseConfig()),
    )
    logger.info("Response:\n%s", result.final_output)

    tool_calls = [i for i in result.new_items if i.type == "tool_call"]
    logger.info("Tools called: %s", [tc.raw.name for tc in tool_calls])
    logger.info("Tokens used: %s", result.context.usage.total_tokens)


async def run_without_instructions() -> None:
    """Multi-step discovery WITHOUT the opt-in prefix."""
    logger.info("=" * 60)
    logger.info("Scenario 2a: Rate limits — WITHOUT prompt_with_skill_instructions()")
    logger.info("=" * 60)

    result = await Runner.arun(agent_without_instructions, QUERY, run_config=RunConfig(verbose=VerboseConfig()))
    logger.info("Response:\n%s", result.final_output)

    tool_calls = [i for i in result.new_items if i.type == "tool_call"]
    logger.info("Tools called: %s", [tc.raw.name for tc in tool_calls])
    logger.info("Tokens used: %s", result.context.usage.total_tokens)


async def run_with_instructions() -> None:
    """Multi-step discovery WITH the opt-in prefix."""
    logger.info("=" * 60)
    logger.info("Scenario 2b: Rate limits — WITH prompt_with_skill_instructions()")
    logger.info("=" * 60)

    result = await Runner.arun(agent_with_instructions, QUERY, run_config=RunConfig(verbose=VerboseConfig()))
    logger.info("Response:\n%s", result.final_output)

    tool_calls = [i for i in result.new_items if i.type == "tool_call"]
    logger.info("Tools called: %s", [tc.raw.name for tc in tool_calls])
    logger.info("Tokens used: %s", result.context.usage.total_tokens)


async def main() -> None:
    try:
        await run_diagnostics_scenario()
        logger.info("")
        await run_without_instructions()
        logger.info("")
        await run_with_instructions()
    finally:
        shutil.rmtree(_tmp_dir, ignore_errors=True)
        logger.info("Cleaned up temporary resources")


if __name__ == "__main__":
    asyncio.run(main())
