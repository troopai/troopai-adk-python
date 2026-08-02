from __future__ import annotations

from unittest.mock import patch

import pytest

from troopai.adk.llms.cost import CostEstimate
from troopai.adk.llms.litellm import LiteLLM
from troopai.adk.llms.routing import CheapestFirstRouter, LatencyFirstRouter, RoutedModel, RoutingContext


def _rm(model: str) -> RoutedModel:
    return RoutedModel(llm=LiteLLM(model=model), model=model)


def _ctx() -> RoutingContext:
    return RoutingContext(messages=[{"role": "user", "content": "x"}], tenant_id=None, run_cost=0.0)


def test_cheapest_first_orders_by_estimate() -> None:
    cheap, mid, none_cost = _rm("cheap"), _rm("mid"), _rm("unknown")
    costs = {"cheap": 0.01, "mid": 0.05, "unknown": None}

    def fake_estimate(self, messages, model, *, max_output_tokens=None):
        return CostEstimate(
            model=model,
            input_tokens=1,
            estimated_output_tokens=0,
            estimated_cost_usd=costs[model],
            output_bounded=False,
        )

    with patch.object(LiteLLM, "estimate_cost", fake_estimate):
        ordered = CheapestFirstRouter([mid, none_cost, cheap]).candidates(_ctx())
    assert [c.model for c in ordered] == ["cheap", "mid", "unknown"]  # None-cost sorts last


def test_latency_first_orders_by_latency() -> None:
    a, b = _rm("a"), _rm("b")
    ordered = LatencyFirstRouter([b, a], latencies={"a": 10.0, "b": 99.0}).candidates(_ctx())
    assert [c.model for c in ordered] == ["a", "b"]


def test_latency_first_unknown_model_sorts_last() -> None:
    a, b = _rm("a"), _rm("b")
    ordered = LatencyFirstRouter([a, b], latencies={"b": 5.0}).candidates(_ctx())
    assert [c.model for c in ordered] == ["b", "a"]  # 'a' has no recorded latency → last


def test_empty_candidates_rejected() -> None:
    with pytest.raises(ValueError):
        CheapestFirstRouter([])
    with pytest.raises(ValueError):
        LatencyFirstRouter([], latencies={})


def test_cheapest_first_preserves_input_order_for_unknown_cost_ties() -> None:
    a, b = _rm("a"), _rm("b")

    def fake_estimate(self, messages, model, *, max_output_tokens=None):
        return CostEstimate(
            model=model,
            input_tokens=1,
            estimated_output_tokens=0,
            estimated_cost_usd=None,  # both unknown → tie
            output_bounded=False,
        )

    with patch.object(LiteLLM, "estimate_cost", fake_estimate):
        ordered = CheapestFirstRouter([a, b]).candidates(_ctx())
    assert [c.model for c in ordered] == ["a", "b"]  # stable sort keeps input order on ties
