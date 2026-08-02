from unittest.mock import patch

from troopai.adk.llms.litellm.litellm_model import LiteLLM
from troopai.adk.types.tokens.llm_usage import LLMUsage


def _usage(inp: int, out: int) -> LLMUsage:
    return LLMUsage(requests=1, input_tokens=inp, output_tokens=out, total_tokens=inp + out)


def test_litellm_cost_sums_prompt_and_completion():
    llm = LiteLLM(model="gpt-4o-mini")
    with patch("litellm.cost_per_token", return_value=(0.001, 0.002)) as m:
        cost = llm.cost("gpt-4o-mini", _usage(10, 5))
    assert cost == 0.003
    m.assert_called_once_with(model="gpt-4o-mini", prompt_tokens=10, completion_tokens=5)


def test_litellm_cost_unknown_model_returns_none():
    llm = LiteLLM(model="gpt-4o-mini")
    with patch("litellm.cost_per_token", side_effect=Exception("unknown model")):
        assert llm.cost("made-up-model", _usage(1, 1)) is None
