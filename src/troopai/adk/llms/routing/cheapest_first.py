"""Cheapest-first router: order candidates by estimated USD ascending."""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import override

from troopai.adk.llms.routing.router import LLMRouter, RoutedModel, RoutingContext

logger = logging.getLogger(__name__)

__all__ = ["CheapestFirstRouter"]


class CheapestFirstRouter(LLMRouter):
    """Try the cheapest candidate first; the loop escalates on failure.

    Candidates whose cost is unknown (``estimated_cost_usd is None``) sort
    last — priced models are tried before unpriced ones.

    Note: :meth:`candidates` calls :meth:`LLM.estimate_cost` once per
    candidate on every invocation (token counting is O(N x message_tokens)),
    so keep the candidate list small on hot paths.
    """

    def __init__(self, models: Sequence[RoutedModel]) -> None:
        """
        Args:
            models: Candidate models to rank by estimated cost.  Must
                contain at least one entry.

        Raises:
            ValueError: If ``models`` is empty.
        """
        if len(models) == 0:
            raise ValueError("CheapestFirstRouter requires at least one candidate")
        self._models = list(models)

    @override
    def candidates(self, ctx: RoutingContext) -> Sequence[RoutedModel]:
        def sort_key(rm: RoutedModel) -> float:
            # Input-only floor: output length is unknown pre-call, so the
            # estimate excludes output tokens (max_output_tokens unset).
            est = rm.llm.estimate_cost(ctx.messages, rm.model).estimated_cost_usd
            return math.inf if est is None else est

        ordered = sorted(self._models, key=sort_key)
        logger.debug("CheapestFirstRouter ordered candidates: %s", [rm.model for rm in ordered])
        return ordered
