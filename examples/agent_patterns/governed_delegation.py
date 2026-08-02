"""
Example: Governed Delegation with as_tool()

Demonstrates governance features for sub-agent delegation: timeout
enforcement, budget limits, result truncation, custom extractors,
and agent graph introspection.

Key concepts shown:
- ``timeout`` — time-bound sub-agent execution via ``asyncio.wait_for()``
- ``budget`` — per-delegation token limits via ``LLMUsageLimits``
- ``max_result_tokens`` — truncate sub-agent output before parent sees it
- ``extractor`` — custom output filtering from ``RunResult``
- ``get_agent_graph()`` — recursive topology introspection
- Context isolation — sub-agent intermediate results never enter parent context
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import json
import logging

from troopai.adk.agents import Agent
from troopai.adk.llms.llm_usage import LLMUsageLimits
from troopai.adk.run import RunConfig, Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Sub-agents with different roles
# =============================================================================

researcher = Agent(
    name="Researcher",
    description="Research topics and summarize key findings with statistics and sources.",
    system_prompt=(
        "You are a research specialist. When given a topic, provide a "
        "thorough analysis with key findings, statistics, and sources. "
        "Be comprehensive but concise."
    ),
)

writer = Agent(
    name="Writer",
    description="Write clear, well-structured summaries for a business audience.",
    system_prompt=(
        "You are a technical writer. Take research findings and write "
        "a clear, well-structured summary suitable for a business audience. "
        "Keep it under 200 words."
    ),
)

fact_checker = Agent(
    name="Fact Checker",
    description="Review content for accuracy and flag unsupported claims.",
    system_prompt=(
        "You are a fact checker. Review the given content for accuracy. "
        "Flag any claims that seem unsupported or incorrect. "
        "Output only your verdict: PASS or FAIL with brief reasoning."
    ),
)


# =============================================================================
# Supervisor with governed delegation
# =============================================================================
#
# Each sub-agent is wrapped via ``as_tool()`` with governance params:
# - ``timeout`` — time-bound the sub-agent run
# - ``budget`` — cap token spend per delegation
# - ``max_result_tokens`` — truncate output before parent sees it
#
# The ``description`` field on each Agent flows as the default tool
# description, giving the supervisor LLM good routing signal.

supervisor = Agent(
    name="Supervisor",
    system_prompt=(
        "You coordinate a research team. For any research request:\n"
        "1. Use the researcher tool to gather findings\n"
        "2. Use the writer tool to create a summary\n"
        "3. Use the fact_checker tool to verify accuracy\n"
        "Compile the final report from these results."
    ),
    tools=[
        # Researcher: generous budget, moderate timeout
        researcher.as_tool(
            timeout=60.0,
            budget=LLMUsageLimits(
                total_tokens_limit=20_000,
                request_limit=5,
            ),
        ),
        # Writer: smaller budget (simple task), tight timeout
        writer.as_tool(
            timeout=30.0,
            budget=LLMUsageLimits(
                total_tokens_limit=5_000,
                request_limit=2,
            ),
            max_result_tokens=500,  # Truncate long outputs
        ),
        # Fact checker: minimal budget, very tight timeout
        fact_checker.as_tool(
            timeout=15.0,
            budget=LLMUsageLimits(
                total_tokens_limit=3_000,
                request_limit=1,
            ),
        ),
    ],
)


# =============================================================================
# Agent graph introspection
# =============================================================================


def show_agent_graph() -> None:
    """Display the agent topology for governance visibility."""
    graph = supervisor.get_agent_graph()
    logger.info("Agent Topology:")
    logger.info(json.dumps(graph, indent=2))

    # Also show delegate tools specifically
    delegates = supervisor.get_delegate_tools()
    logger.info(
        "Delegate tools: %s",
        [t.name for t in delegates],
    )


# =============================================================================
# Main execution
# =============================================================================


async def main() -> None:
    logger.info("=" * 60)
    logger.info("Governed Delegation Example")
    logger.info("=" * 60)

    # Show the agent graph before running
    show_agent_graph()

    # Run the supervisor
    logger.info("\nRunning supervisor with governed sub-agents...")
    result = await Runner.arun(
        supervisor,
        "Research the impact of large language models on software engineering "
        "productivity. Focus on recent studies from 2024-2025.",
        run_config=RunConfig(verbose=VerboseConfig()),
    )

    logger.info("\n" + "=" * 60)
    logger.info("Final Report:")
    logger.info("=" * 60)
    logger.info(result.final_output)

    # Show usage stats
    logger.info("\n" + "=" * 60)
    logger.info("Usage Statistics:")
    logger.info("=" * 60)
    logger.info("Total requests: %d", result.context.usage.requests)
    logger.info("Total input tokens: %d", result.context.usage.input_tokens)
    logger.info("Total output tokens: %d", result.context.usage.output_tokens)


if __name__ == "__main__":
    asyncio.run(main())
