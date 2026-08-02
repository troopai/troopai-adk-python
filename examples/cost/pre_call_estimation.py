"""Pre-call cost estimation with LLM.estimate_cost.

Demonstrates how to estimate the cost of an LLM call before making it,
using the :meth:`LLM.estimate_cost` method. No real API call is made —
estimation is pure token counting combined with the provider's cost table.

Two cases are shown:

1. **Bounded estimate** — ``max_output_tokens`` is provided; the returned
   :class:`CostEstimate` reflects both input tokens and the upper-bound
   output tokens, and ``output_bounded`` is ``True``.
2. **Unbounded estimate** — no ``max_output_tokens``; the estimate is an
   input-only floor (the developer has not declared a cap) and
   ``output_bounded`` is ``False``.

Usage::

    python examples/cost/pre_call_estimation.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import logging

from troopai.adk.llms import LiteLLM
from troopai.adk.types.input import LLMInputContentItem, LLMInputEasyMessage

logger = logging.getLogger(__name__)

MODEL = "claude-haiku-4-5-20251001"


def main() -> None:
    llm = LiteLLM(model=MODEL)

    # LLMInputEasyMessage is part of the LLMInputContentItem union; the explicit
    # cast makes the invariant list checker happy without using Sequence or Any.
    messages: list[LLMInputContentItem] = [
        LLMInputEasyMessage(
            role="user",
            content="Summarise the history of machine learning in three paragraphs.",
        ),
    ]

    # --- Case 1: bounded estimate (max_output_tokens declared) ---
    bounded = llm.estimate_cost(messages, MODEL, max_output_tokens=500)
    logger.info("=== Bounded estimate (max_output_tokens=500) ===")
    logger.info("  model                  : %s", bounded.model)
    logger.info("  input_tokens           : %d", bounded.input_tokens)
    logger.info("  estimated_output_tokens: %d", bounded.estimated_output_tokens)
    logger.info(
        "  estimated_cost_usd     : %s",
        f"${bounded.estimated_cost_usd:.6f}" if bounded.estimated_cost_usd is not None else "unknown",
    )
    logger.info("  output_bounded         : %s", bounded.output_bounded)

    # --- Case 2: unbounded estimate (no max_output_tokens) ---
    unbounded = llm.estimate_cost(messages, MODEL)
    logger.info("=== Unbounded estimate (no max_output_tokens) ===")
    logger.info("  model                  : %s", unbounded.model)
    logger.info("  input_tokens           : %d", unbounded.input_tokens)
    logger.info("  estimated_output_tokens: %d (floor — output excluded)", unbounded.estimated_output_tokens)
    logger.info(
        "  estimated_cost_usd     : %s (input-only floor)",
        f"${unbounded.estimated_cost_usd:.6f}" if unbounded.estimated_cost_usd is not None else "unknown",
    )
    logger.info("  output_bounded         : %s", unbounded.output_bounded)

    if bounded.estimated_cost_usd is not None and unbounded.estimated_cost_usd is not None:
        logger.info(
            "Bounded adds $%.6f for up to %d output tokens",
            bounded.estimated_cost_usd - unbounded.estimated_cost_usd,
            bounded.estimated_output_tokens,
        )


if __name__ == "__main__":
    main()
