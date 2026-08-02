"""
Example: LLM as a Judge (Iterative Generate–Evaluate–Refine)

Two agents collaborate in a feedback loop: a *generator* produces a draft,
and a *judge* evaluates it and returns a structured verdict. When the judge
is not satisfied, its feedback is fed back to the generator for another
pass — up to a bounded number of iterations.

Unlike the deterministic pipeline (a one-shot structured-output quality
gate), this pattern loops: the judge's verdict drives whether to refine or
stop.

| Pattern            | Flow      | Decision Maker | Progression       |
|--------------------|-----------|----------------|-------------------|
| **LLM as a Judge** | Iterative | LLM (loop)     | Feedback-driven   |
| Deterministic      | Linear    | Code (gate)    | Structured field  |

Key concepts shown:
- ``output_schema`` for a structured judge verdict (score + feedback)
- ``final_output_as(Type)`` for typed access to the verdict
- A bounded refine loop driven by the judge's score
- Feeding the stateless judge the attempt number + its own prior feedback, so
  its cross-call instructions ("not the first attempt", "addressed my earlier
  feedback") rest on context it was actually given rather than memory it lacks
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from typing import Literal

from pydantic import BaseModel, Field

from troopai.adk.agents import Agent
from troopai.adk.run import RunConfig, Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

# Bounded refine loop — the judge drives early exit on "pass".
MAX_ITERATIONS = 5


# =============================================================================
# Structured verdict from the judge
# =============================================================================


class EvaluationFeedback(BaseModel):
    """Structured verdict returned by the judge agent."""

    score: Literal["pass", "needs_improvement", "fail"] = Field(
        description="'pass' when the outline is ready; otherwise it needs another pass.",
    )
    feedback: str = Field(
        description="Specific, actionable guidance for improving the outline.",
    )


# =============================================================================
# Generator + judge agents
# =============================================================================

story_outline_generator = Agent(
    name="Story Outline Generator",
    system_prompt=(
        "You generate a very short story outline from the user's idea. "
        "If feedback from a previous attempt is included, use it to improve "
        "the outline. Output only the outline — no commentary."
    ),
)

evaluator = Agent(
    name="Outline Judge",
    system_prompt=(
        "You evaluate a story outline and decide whether it is good enough. "
        "Be demanding: return 'pass' only when the outline is genuinely "
        "strong (clear arc, concrete characters, a hook). Otherwise return "
        "'needs_improvement' or 'fail' with specific, actionable feedback. "
        "Do not pass on the first attempt, but once a revision has clearly "
        "addressed your earlier feedback, return 'pass'."
    ),
    output_schema=EvaluationFeedback,
)


# =============================================================================
# Refine loop
# =============================================================================


def _generator_prompt(idea: str, outline: str | None, feedback: str | None) -> str:
    """Build the generator prompt, folding in prior feedback when present."""
    if outline is None or feedback is None:
        return f"Story idea: {idea}"
    return f"Story idea: {idea}\n\nPrevious outline:\n{outline}\n\nJudge feedback to address:\n{feedback}"


def _evaluator_prompt(outline: str, attempt: int, prior_feedback: str | None) -> str:
    """Build the judge prompt with the attempt number and its prior feedback.

    Each judge call is stateless, so the cross-call context its instructions
    lean on ("do not pass on the first attempt", "once a revision has addressed
    your earlier feedback") must be supplied explicitly — the attempt counter
    and the feedback the judge gave last round.
    """
    header = f"Evaluate this story outline (attempt {attempt}):\n\n{outline}"
    if prior_feedback is None:
        return header
    return f"{header}\n\nYour feedback on the previous attempt was:\n{prior_feedback}"


async def main() -> None:
    idea = "A clockmaker in a town where time has stopped."
    outline: str | None = None
    feedback: str | None = None
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    for iteration in range(1, MAX_ITERATIONS + 1):
        logger.info("=" * 60)
        logger.info("Iteration %d: generating outline", iteration)
        logger.info("=" * 60)
        gen_result = await Runner.arun(
            story_outline_generator, _generator_prompt(idea, outline, feedback), run_config=run_config
        )
        outline = gen_result.final_output
        logger.info("\n%s\n", outline)

        # ``feedback`` still holds the previous iteration's verdict here (None on
        # the first pass); it becomes this iteration's feedback only at the loop's
        # end, so the judge sees exactly the note it gave last round.
        eval_result = await Runner.arun(
            evaluator, _evaluator_prompt(outline, iteration, feedback), run_config=run_config
        )
        verdict = eval_result.final_output_as(EvaluationFeedback)
        logger.info("Judge score:    %s", verdict.score)
        logger.info("Judge feedback: %s\n", verdict.feedback)

        if verdict.score == "pass":
            logger.info("Outline accepted after %d iteration(s).", iteration)
            break
        feedback = verdict.feedback
    else:
        logger.info("Reached the %d-iteration cap without a 'pass'.", MAX_ITERATIONS)


if __name__ == "__main__":
    asyncio.run(main())
