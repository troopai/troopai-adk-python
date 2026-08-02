"""
Example: Deterministic Pipeline (Sequential with Structured Output Gating)

Demonstrates a code-orchestrated sequential pipeline where code controls
progression between agents based on structured output fields.

Pipeline: Outline Agent → Quality Gate Agent → Story Writer Agent

1. The **outline agent** generates a story outline (free text).
2. The **quality gate agent** evaluates the outline and returns a
   structured ``QualityAssessment`` with a ``passes`` boolean field.
3. Code checks ``passes``: if True, the **story writer** proceeds;
   if False, the pipeline stops and reports the issues.

This differs from other multi-agent patterns:

| Pattern              | Flow      | Decision Maker | Progression         |
|----------------------|-----------|----------------|---------------------|
| LLM as a Judge       | Iterative | LLM (loop)     | Feedback-driven     |
| Code-orchestrated    | Routing   | Code (intent)  | Branch on signals   |
| **Deterministic**    | **Linear**| **Code (gate)**| **Structured field**|

Key concepts shown:
- ``output_schema`` for structured agent output (Pydantic BaseModel)
- ``final_output_as(Type)`` for typed access to structured results
- Sequential ``Runner.arun()`` chaining with programmatic gating
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from pydantic import BaseModel, Field

from troopai.adk.agents import Agent
from troopai.adk.run import RunConfig, Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Structured output for the quality gate
# =============================================================================


class QualityAssessment(BaseModel):
    """Structured evaluation from the quality gate agent."""

    passes: bool = Field(
        description="True if the outline is good enough to proceed to writing.",
    )
    strengths: list[str] = Field(
        description="What the outline does well (1-3 bullet points).",
    )
    issues: list[str] = Field(
        description="Specific problems to fix (empty list if passes is True).",
    )
    recommendation: str = Field(
        description="Brief recommendation: either 'Proceed to writing' or a specific suggestion for improvement.",
    )


# =============================================================================
# Pipeline agents
# =============================================================================

outline_agent = Agent(
    name="Outline Writer",
    system_prompt=(
        "You are a story outline specialist. Given a story concept, produce "
        "a detailed outline with:\n"
        "- Setting and time period\n"
        "- Main characters (name, role, motivation)\n"
        "- Plot structure (beginning, rising action, climax, resolution)\n"
        "- Key scenes (3-5 scenes with brief descriptions)\n\n"
        "Output only the outline — no commentary or meta-notes."
    ),
)

quality_gate_agent = Agent(
    name="Quality Gate",
    system_prompt=(
        "You are a story outline reviewer. Evaluate the given outline for:\n"
        "- Structural completeness (setting, characters, plot arc)\n"
        "- Character depth (motivations, conflicts)\n"
        "- Plot coherence (logical progression, no gaps)\n"
        "- Scene specificity (concrete enough to write from)\n\n"
        "Be constructive but maintain standards. An outline should be "
        "detailed enough for a writer to produce a full story from it."
    ),
    output_schema=QualityAssessment,
)

story_writer_agent = Agent(
    name="Story Writer",
    system_prompt=(
        "You are a creative fiction writer. Given a story outline, write "
        "a compelling short story (3-5 paragraphs) that brings the outline "
        "to life. Focus on vivid descriptions, natural dialogue, and "
        "emotional resonance. Output only the story — no commentary."
    ),
)


# =============================================================================
# Sequential pipeline
# =============================================================================


async def main():
    task = (
        "A retired lighthouse keeper discovers that the lighthouse's "
        "light has been sending coded messages for decades — messages "
        "meant for someone who never arrived."
    )
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    # Step 1: Generate outline
    logger.info("=" * 60)
    logger.info("Step 1: Generating outline")
    logger.info("=" * 60)
    outline_result = await Runner.arun(outline_agent, task, run_config=run_config)
    outline = outline_result.final_output
    logger.info(f"\n{outline}\n")

    # Step 2: Quality gate — structured assessment
    logger.info("=" * 60)
    logger.info("Step 2: Quality gate evaluation")
    logger.info("=" * 60)
    gate_result = await Runner.arun(
        quality_gate_agent,
        f"Evaluate this story outline:\n\n{outline}",
        run_config=run_config,
    )
    assessment: QualityAssessment = gate_result.final_output_as(QualityAssessment)

    logger.info(f"\nPasses: {assessment.passes}")
    for strength in assessment.strengths:
        logger.info(f"  + {strength}")
    for issue in assessment.issues:
        logger.info(f"  - {issue}")
    logger.info(f"  Recommendation: {assessment.recommendation}\n")

    # Step 3: Gate decision — proceed or stop
    if not assessment.passes:
        logger.info("=" * 60)
        logger.info("Pipeline stopped: outline did not pass quality gate")
        logger.info("=" * 60)
        return

    logger.info("=" * 60)
    logger.info("Step 3: Writing story (outline passed)")
    logger.info("=" * 60)
    story_result = await Runner.arun(
        story_writer_agent,
        f"Write a short story based on this outline:\n\n{outline}",
        run_config=run_config,
    )
    logger.info(f"\n{story_result.final_output}\n")


if __name__ == "__main__":
    asyncio.run(main())
