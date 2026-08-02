from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import pytest

from troopai.adk.llms.litellm import LiteLLM
from troopai.adk.llms.routing import LLMRouter, RoutedModel, RoutingContext


def test_routed_model_holds_llm_and_name() -> None:
    llm = LiteLLM(model="claude-haiku-4-5-20251001")
    rm = RoutedModel(llm=llm, model="claude-haiku-4-5-20251001")
    assert rm.model == "claude-haiku-4-5-20251001"
    assert rm.llm is llm


def test_router_is_abstract_and_subclassable() -> None:
    class Static(LLMRouter):
        def __init__(self, models: Sequence[RoutedModel]) -> None:
            self._models = list(models)

        def candidates(self, ctx: RoutingContext) -> Sequence[RoutedModel]:
            return self._models

    llm = LiteLLM(model="claude-haiku-4-5-20251001")
    r = Static([RoutedModel(llm=llm, model="claude-haiku-4-5-20251001")])
    ctx = RoutingContext(messages=[{"role": "user", "content": "x"}], tenant_id=None, run_cost=0.0)
    assert len(r.candidates(ctx)) == 1
    assert r.should_escalate(None) is False  # default predicate: no content-based escalation

    with pytest.raises(TypeError):
        LLMRouter()  # type: ignore[abstract]  # abstract candidates() blocks direct instantiation


def test_routed_model_and_context_are_frozen() -> None:
    llm = LiteLLM(model="claude-haiku-4-5-20251001")
    rm = RoutedModel(llm=llm, model="m")
    with pytest.raises(dataclasses.FrozenInstanceError):
        rm.model = "other"  # type: ignore[misc]
    ctx = RoutingContext(messages=[], tenant_id=None, run_cost=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.run_cost = 1.0  # type: ignore[misc]
