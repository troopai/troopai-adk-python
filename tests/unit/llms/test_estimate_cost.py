from __future__ import annotations

from unittest.mock import patch

from troopai.adk.llms.cost import CostEstimate
from troopai.adk.llms.litellm import LiteLLM


def test_estimate_cost_input_only_when_no_output_bound() -> None:
    llm = LiteLLM(model="claude-haiku-4-5-20251001")
    with (
        patch("litellm.token_counter", return_value=120),
        patch.object(LiteLLM, "cost", return_value=0.0012),
    ):
        est = llm.estimate_cost([{"role": "user", "content": "hi"}], "claude-haiku-4-5-20251001")
    assert isinstance(est, CostEstimate)
    assert est.input_tokens == 120
    assert est.estimated_output_tokens == 0
    assert est.output_bounded is False
    assert est.estimated_cost_usd == 0.0012  # input-only floor still carries a USD figure


def test_estimate_cost_bounds_output_when_max_set() -> None:
    llm = LiteLLM(model="claude-haiku-4-5-20251001")
    with (
        patch("litellm.token_counter", return_value=100),
        patch.object(LiteLLM, "cost", return_value=0.0042),
    ):
        est = llm.estimate_cost(
            [{"role": "user", "content": "hi"}],
            "claude-haiku-4-5-20251001",
            max_output_tokens=500,
        )
    assert est.estimated_output_tokens == 500
    assert est.output_bounded is True
    assert est.estimated_cost_usd == 0.0042


def test_estimate_cost_none_when_no_cost_table() -> None:
    llm = LiteLLM(model="made-up-model")
    with (
        patch("litellm.token_counter", return_value=50),
        patch.object(LiteLLM, "cost", return_value=None),
    ):
        est = llm.estimate_cost([{"role": "user", "content": "hi"}], "made-up-model", max_output_tokens=100)
    assert est.input_tokens == 50
    assert est.estimated_output_tokens == 100
    assert est.output_bounded is True
    assert est.estimated_cost_usd is None
