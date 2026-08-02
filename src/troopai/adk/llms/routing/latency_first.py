"""Latency-first router: order candidates by recorded latency ascending."""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from typing import override

from troopai.adk.llms.routing.router import LLMRouter, RoutedModel, RoutingContext

logger = logging.getLogger(__name__)

__all__ = ["LatencyFirstRouter"]


class LatencyFirstRouter(LLMRouter):
    """Try the fastest candidate first (by a developer-supplied latency map).

    ``latencies`` maps model name -> observed latency (ms); the
    ``troopai.agent.turn.duration_ms`` histogram is one source (turn-level
    wall-clock, not call-level latency); for model-only latency, record LLM
    response times separately. Models with no recorded latency sort last.
    """

    def __init__(self, models: Sequence[RoutedModel], latencies: Mapping[str, float]) -> None:
        """
        Args:
            models: Candidate models to rank by recorded latency.  Must
                contain at least one entry.
            latencies: Map of model name → observed latency in milliseconds.
                Models absent from this map sort last.

        Raises:
            ValueError: If ``models`` is empty.
        """
        if len(models) == 0:
            raise ValueError("LatencyFirstRouter requires at least one candidate")
        self._models = list(models)
        self._latencies = dict(latencies)

    @override
    def candidates(self, ctx: RoutingContext) -> Sequence[RoutedModel]:
        del ctx
        ordered = sorted(self._models, key=lambda rm: self._latencies.get(rm.model, math.inf))
        logger.debug("LatencyFirstRouter ordered candidates: %s", [rm.model for rm in ordered])
        return ordered
