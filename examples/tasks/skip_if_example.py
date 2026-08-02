"""Conditional task execution with ``Task.skip_if`` in a pipeline.

Demonstrates:

- A three-step :class:`TaskPipeline` where the middle step is skipped
  conditionally based on the first step's classification output.
- Skipped tasks remain in :attr:`TaskPipelineResult.task_outputs` with
  ``skipped=True`` — positional indexing stays stable.
- ``skip_if`` predicate inspecting prior :class:`TaskOutput` content.

Each task's ``description`` is fed verbatim as the user prompt; the
framework does not rewrite prompts at runtime. The classifier sees a
prompt that already embeds the text-to-classify; the translator and
reviewer each have self-contained descriptions. If you need a
downstream task to consume an upstream task's output, see
``sequential_pipeline.py`` for the explicit-chaining pattern.

Run from the project root::

    python examples/tasks/skip_if_example.py

Requires ``ANTHROPIC_API_KEY`` (or another litellm-supported key) set
in the environment or ``.env``.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from collections.abc import Sequence

from troopai.adk.agents import Agent
from troopai.adk.run import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.tasks import Task, TaskPipeline
from troopai.adk.tasks.task_output import TaskOutput
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


USER_TEXT = "Hola, ¿cómo estás hoy?"  # Spanish — translator will run


classifier = Agent(
    name="classifier",
    system_prompt=(
        "You are a classifier. Reply with ONE WORD only: either "
        "'english' or 'other', based on the language of the user "
        "message. No punctuation, no explanation."
    ),
)

translator = Agent(
    name="translator",
    system_prompt="You are a translator. Translate the user message to English. Reply with the translation only.",
)

reviewer = Agent(
    name="reviewer",
    system_prompt="You are a reviewer. Briefly comment on the user message in ONE sentence (under 25 words).",
)


def skip_translation_if_already_english(prior: Sequence[TaskOutput]) -> bool:
    """Skip translation when the classifier said 'english'.

    ``prior`` is the immutable tuple of every prior :class:`TaskOutput`
    in pipeline order. Pure function of inputs — no side effects.
    """
    last_output = prior[-1].final_output
    if not isinstance(last_output, str):
        return False
    return last_output.strip().lower().startswith("english")


async def main() -> None:
    # Each task's description IS its user prompt. Classifier and
    # reviewer reference USER_TEXT inline; translator's description
    # is self-contained (the LLM will use its own conversation
    # context if anything was forwarded — here it's a fresh task).
    classify = Task(
        description=f"Classify the language of this text:\n\n{USER_TEXT}",
        agent=classifier,
        name="detect-language",
    )
    translate = Task(
        description=f"Translate this text to English:\n\n{USER_TEXT}",
        agent=translator,
        name="translate-to-english",
        skip_if=skip_translation_if_already_english,
    )
    review = Task(
        description=f"Comment briefly on this message in one sentence:\n\n{USER_TEXT}",
        agent=reviewer,
        name="review-text",
    )

    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())
    result = await Runner.arun_task_pipeline(TaskPipeline(tasks=(classify, translate, review)), run_config=run_config)

    logger.info("=" * 60)
    logger.info("Pipeline outputs (slots stable, skipped slots preserved):")
    for i, output in enumerate(result.task_outputs):
        marker = "SKIPPED" if output.skipped else "RAN    "
        logger.info("  [%d] %s %s: %s", i, marker, output.task_name, output.final_output)
    logger.info("=" * 60)
    logger.info("Final output: %s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
