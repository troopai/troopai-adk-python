"""Pre-call cost estimation types for the LLM ABC."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostEstimate:
    """A pre-call estimate of an LLM request's cost.

    The pre-flight twin of :meth:`LLM.cost`. Input tokens are counted
    precisely; output is bounded by the resolved ``max_output_tokens``
    when the developer set one, otherwise excluded (``output_bounded`` is
    ``False`` and the estimate is an input-only floor).

    Attributes:
        model: The model the estimate is for.
        input_tokens: Precisely counted input tokens.
        estimated_output_tokens: Assumed output tokens (0 when unbounded).
        estimated_cost_usd: Estimated USD, or ``None`` when the provider
            has no cost table (callers MUST treat ``None`` as "unknown").
        output_bounded: ``True`` when ``max_output_tokens`` bounded output.
    """

    model: str
    """The model the estimate is for."""

    input_tokens: int
    """Precisely counted input tokens."""

    estimated_output_tokens: int
    """Assumed output tokens (0 when unbounded)."""

    estimated_cost_usd: float | None
    """Estimated USD, or ``None`` when the provider has no cost table."""

    output_bounded: bool
    """``True`` when ``max_output_tokens`` bounded output."""
