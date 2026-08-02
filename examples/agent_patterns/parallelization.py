"""
Example: Parallel Agent Execution with Judge Selection

Demonstrates running multiple agents concurrently with ``asyncio.gather()``
and selecting the best result via a judge agent with structured output.

Three translator agents with different styles (literal, natural, creative)
translate the same text in parallel. A judge agent then evaluates all
translations and picks the best one.

Key concepts shown:
- ``asyncio.gather()`` for true concurrent ``Runner.arun()`` execution
- Independent agent runs — no shared context or history between them
- Judge agent with ``output_schema`` for structured verdict
- ``final_output_as(Type)`` for typed access to the judge's decision
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
# Translator agents (different styles)
# =============================================================================

translator_literal = Agent(
    name="Literal Translator",
    system_prompt=(
        "You are a literal translator. Translate the given text to French "
        "as faithfully as possible, preserving the original sentence "
        "structure and word order. Prioritize accuracy over fluency. "
        "Output only the translation — no commentary."
    ),
)

translator_natural = Agent(
    name="Natural Translator",
    system_prompt=(
        "You are a natural translator. Translate the given text to French "
        "using idiomatic expressions and natural phrasing that a native "
        "speaker would use. Prioritize fluency and readability. "
        "Output only the translation — no commentary."
    ),
)

translator_creative = Agent(
    name="Creative Translator",
    system_prompt=(
        "You are a creative translator. Translate the given text to French "
        "with artistic flair — adapt metaphors, use literary devices, and "
        "capture the emotional essence rather than just the words. "
        "Output only the translation — no commentary."
    ),
)


# =============================================================================
# Judge agent
# =============================================================================


class JudgeVerdict(BaseModel):
    """Structured verdict from the judge agent."""

    best_index: int = Field(
        ge=0,
        lt=3,
        description="Index of the best translation (0 = literal, 1 = natural, 2 = creative).",
    )
    reasoning: str = Field(
        description="Brief explanation of why this translation was chosen, noting strengths and weaknesses of each.",
    )


judge = Agent(
    name="Translation Judge",
    system_prompt=(
        "You are an expert bilingual judge evaluating English-to-French "
        "translations. Compare the translations for accuracy, fluency, "
        "and style. Consider:\n"
        "- Accuracy: Does the meaning match the original?\n"
        "- Fluency: Does it read naturally in French?\n"
        "- Style: Is the tone appropriate for the content?\n\n"
        "Pick the best overall translation."
    ),
    output_schema=JudgeVerdict,
)


# =============================================================================
# Parallel execution
# =============================================================================

STYLE_LABELS = ["Literal", "Natural", "Creative"]


async def main():
    source_text = (
        "The old lighthouse keeper watched the storm roll in, knowing "
        "that tonight would test every beam of light he could muster "
        "against the darkness."
    )
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    logger.info("=" * 60)
    logger.info("Source text:")
    logger.info("=" * 60)
    logger.info(f"{source_text}\n")

    # Run all three translators concurrently
    logger.info("=" * 60)
    logger.info("Running 3 translators in parallel...")
    logger.info("=" * 60)

    results = await asyncio.gather(
        Runner.arun(translator_literal, source_text, run_config=run_config),
        Runner.arun(translator_natural, source_text, run_config=run_config),
        Runner.arun(translator_creative, source_text, run_config=run_config),
    )

    translations = []
    for i, result in enumerate(results):
        translation = result.final_output
        translations.append(translation)
        logger.info(f"\n[{STYLE_LABELS[i]}]\n{translation}")

    # Judge evaluates all translations
    logger.info(f"\n{'=' * 60}")
    logger.info("Judge evaluation:")
    logger.info("=" * 60)

    comparison_prompt = (
        f"Original English text:\n{source_text}\n\n"
        f"Translation 0 (Literal):\n{translations[0]}\n\n"
        f"Translation 1 (Natural):\n{translations[1]}\n\n"
        f"Translation 2 (Creative):\n{translations[2]}"
    )

    judge_result = await Runner.arun(judge, comparison_prompt, run_config=run_config)
    verdict: JudgeVerdict = judge_result.final_output_as(JudgeVerdict)

    logger.info(f"\nBest translation: {STYLE_LABELS[verdict.best_index]} (index {verdict.best_index})")
    logger.info(f"Reasoning: {verdict.reasoning}")


if __name__ == "__main__":
    asyncio.run(main())
