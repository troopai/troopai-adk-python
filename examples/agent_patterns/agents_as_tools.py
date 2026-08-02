"""
Example: Agents as Tools (LLM-Orchestrated Delegation)

This example demonstrates wrapping agents as FunctionTools using ``Agent.as_tool()``.
The parent LLM naturally calls these tools to delegate work to sub-agents, and the
results come back as tool output -- control stays with the parent agent.

Key differences from other multi-agent patterns:

| Pattern            | Who Decides  | Control    | History Shared       |
|--------------------|-------------|------------|----------------------|
| Handoff (code)     | Code        | Transfer   | Filtered             |
| Handoff (LLM)      | LLM         | Transfer   | Filtered             |
| **as_tool() (LLM)**| **LLM**     | **Retained**| **None (input only)**|

Three usage patterns are shown below:
1. Simple - Default input schema, two agents as tools for a supervisor
2. Structured input - Custom Pydantic schema with input_builder
3. Custom output - extractor to post-process sub-agent results
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from typing import Any

from pydantic import BaseModel, Field

from troopai.adk.agents import Agent
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools.tool_context import ToolContext
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Pattern 1: Simple (Default Input Schema)
# =============================================================================
# The simplest usage -- each sub-agent becomes a tool the supervisor can call
# with a plain text task description.

researcher = Agent(
    name="Researcher",
    system_prompt="You are a research specialist. Given a topic, provide a concise, factual summary with key points.",
)

writer = Agent(
    name="Writer",
    system_prompt=(
        "You are a writing specialist. Given content or instructions, produce polished, well-structured prose."
    ),
)

# The supervisor sees "researcher" and "writer" as regular tools.
# It calls them like any other tool -- the sub-agent runs, and the
# result comes back as tool output.
simple_supervisor = Agent(
    name="Simple Supervisor",
    system_prompt=(
        "You coordinate a research-and-writing pipeline.\n"
        "1. Use the researcher tool to gather information.\n"
        "2. Use the writer tool to produce a polished article.\n"
        "Synthesise the results into a final answer."
    ),
    tools=[
        researcher.as_tool(
            tool_description="Research a topic and return key findings.",
        ),
        writer.as_tool(
            tool_description="Write polished prose from provided content.",
        ),
    ],
)


# =============================================================================
# Pattern 2: Structured Input Schema with input_builder
# =============================================================================
# When you need the LLM to provide structured parameters (not just a string),
# supply a custom Pydantic schema and an input_builder to transform it.


class TranslationInput(BaseModel):
    """Structured input for the translator agent."""

    text: str = Field(description="The text to translate.")
    target_language: str = Field(description="Target language (e.g., 'French', 'Spanish').")
    formality: str = Field(
        default="neutral",
        description="Tone: 'formal', 'informal', or 'neutral'.",
    )


def build_translation_input(parsed: TranslationInput, ctx: ToolContext) -> str:
    """Transform structured input into a natural-language prompt for the agent."""
    return f"Translate the following text to {parsed.target_language} using a {parsed.formality} tone:\n\n{parsed.text}"


translator = Agent(
    name="Translator",
    system_prompt=(
        "You are a professional translator. Translate the given text "
        "accurately, preserving meaning and adapting tone as requested."
    ),
)

structured_supervisor = Agent(
    name="Structured Supervisor",
    system_prompt=(
        "You help users with multilingual content.\n"
        "Use the translator tool to translate text. Specify the target "
        "language and formality level."
    ),
    tools=[
        translator.as_tool(
            tool_name="translate",
            tool_description="Translate text to a target language with specified formality.",
            input_schema=TranslationInput,
            input_builder=build_translation_input,
        ),
    ],
)


# =============================================================================
# Pattern 3: Custom Output Extractor
# =============================================================================
# When you need to post-process or summarise the sub-agent's RunResult before
# the parent LLM sees it, provide a extractor.

analyst = Agent(
    name="Data Analyst",
    system_prompt=(
        "You are a data analyst. Analyze the given data or question and "
        "provide a structured analysis with findings and recommendations."
    ),
)


def extract_analysis_summary(result: Any) -> str:
    """Extract a concise summary from the analyst's RunResult.

    This lets you control exactly what the parent LLM sees, e.g.
    trimming verbose output or adding metadata.
    """
    output = str(result.final_output)
    # Truncate very long outputs so the parent context isn't flooded
    if len(output) > 2000:
        output = output[:2000] + "\n... [truncated]"
    return f"[Analysis from {result.last_agent.name}]\n{output}"


custom_output_supervisor = Agent(
    name="Custom Output Supervisor",
    system_prompt=(
        "You manage data analysis requests.\n"
        "Use the analyze tool to run analysis, then interpret the results "
        "for the user in plain language."
    ),
    tools=[
        analyst.as_tool(
            tool_name="analyze",
            tool_description="Run data analysis and return structured findings.",
            extractor=extract_analysis_summary,
        ),
    ],
)


# =============================================================================
# Running the examples
# =============================================================================


async def main():
    """Demonstrate all three patterns."""
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    # Pattern 1: Simple
    logger.info("=" * 60)
    logger.info("Pattern 1: Simple (Default Input Schema)")
    logger.info("=" * 60)
    result = await Runner.arun(
        simple_supervisor,
        "Write a short article about the benefits of open-source software.",
        run_config=run_config,
    )
    logger.info(f"Output:\n{result.final_output}\n")

    # Pattern 2: Structured Input
    logger.info("=" * 60)
    logger.info("Pattern 2: Structured Input Schema")
    logger.info("=" * 60)
    result = await Runner.arun(
        structured_supervisor,
        "Translate 'Hello, how are you today?' to French with a formal tone.",
        run_config=run_config,
    )
    logger.info(f"Output:\n{result.final_output}\n")

    # Pattern 3: Custom Output
    logger.info("=" * 60)
    logger.info("Pattern 3: Custom Output Extractor")
    logger.info("=" * 60)
    result = await Runner.arun(
        custom_output_supervisor,
        "Analyze the trend of remote work adoption over the past 5 years.",
        run_config=run_config,
    )
    logger.info(f"Output:\n{result.final_output}\n")


if __name__ == "__main__":
    asyncio.run(main())
