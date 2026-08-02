"""Tests for call_llm_with_routing escalation driver.

Proves:
(a) escalation to next candidate on a retryable (non-TroopAIError) exception.
(b) NoRoutingCandidateError raised when all candidates fail, chaining the last error.
(c) TroopAIError (framework error) is NOT retried and propagates immediately.
(d) should_escalate predicate drives escalation to the next candidate.

Fake-LLM pattern mirrors test_budget_enforcement.py: minimal ``LLM``
subclass with controlled ``acomplete`` behaviour; ``cost``/``estimate_cost``
use defaults (None) since the routing driver never calls them.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, override
from unittest.mock import patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.exceptions import NoRoutingCandidateError, UserError
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.llms.cost import CostEstimate
from troopai.adk.llms.llm import LLM
from troopai.adk.llms.routing import LLMRouter, RoutedModel, RoutingContext
from troopai.adk.run.config import RunConfig
from troopai.adk.run.llm_calls import call_llm_with_routing
from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText
from troopai.adk.types.tokens.llm_usage import LLMUsage

# ---------------------------------------------------------------------------
# Fake LLM helpers
# ---------------------------------------------------------------------------


def _text_response(content: str, model: str = "fake") -> LLMResponse:
    return LLMResponse(
        response_id="resp-routing-test",
        model=model,
        response=[LLMResponseText(text=content)],
        usage=LLMUsage(requests=1, input_tokens=5, output_tokens=3, total_tokens=8),
    )


class GoodFakeLLM(LLM):
    """Always returns a successful LLMResponse with the given content."""

    def __init__(self, content: str = "good") -> None:
        self._content = content
        self.acomplete_called: bool = False

    @override
    async def acomplete(  # type: ignore[override]
        self,
        messages: Any,
        llm_config: Any = None,
        tools: Any = None,
        output_schema: Any = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[Any]:
        self.acomplete_called = True
        return _text_response(self._content)

    @override
    def cost(self, model: str, usage: LLMUsage) -> float | None:
        del model, usage
        return None

    @override
    def estimate_cost(
        self,
        messages: Any,
        model: str,
        *,
        max_output_tokens: int | None = None,
    ) -> CostEstimate:
        del messages, max_output_tokens
        return CostEstimate(
            model=model,
            input_tokens=5,
            estimated_output_tokens=0,
            estimated_cost_usd=None,
            output_bounded=False,
        )


class RaisingFakeLLM(LLM):
    """Always raises a RuntimeError (non-TroopAIError — routing-retryable)."""

    @override
    async def acomplete(  # type: ignore[override]
        self,
        messages: Any,
        llm_config: Any = None,
        tools: Any = None,
        output_schema: Any = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[Any]:
        raise RuntimeError("provider timeout")

    @override
    def cost(self, model: str, usage: LLMUsage) -> float | None:
        del model, usage
        return None

    @override
    def estimate_cost(
        self,
        messages: Any,
        model: str,
        *,
        max_output_tokens: int | None = None,
    ) -> CostEstimate:
        del messages, max_output_tokens
        return CostEstimate(
            model=model,
            input_tokens=5,
            estimated_output_tokens=0,
            estimated_cost_usd=None,
            output_bounded=False,
        )


class FrameworkErrorFakeLLM(LLM):
    """Always raises a UserError (TroopAIError subclass — NOT retryable)."""

    @override
    async def acomplete(  # type: ignore[override]
        self,
        messages: Any,
        llm_config: Any = None,
        tools: Any = None,
        output_schema: Any = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[Any]:
        raise UserError("invalid request — framework error")

    @override
    def cost(self, model: str, usage: LLMUsage) -> float | None:
        del model, usage
        return None

    @override
    def estimate_cost(
        self,
        messages: Any,
        model: str,
        *,
        max_output_tokens: int | None = None,
    ) -> CostEstimate:
        del messages, max_output_tokens
        return CostEstimate(
            model=model,
            input_tokens=5,
            estimated_output_tokens=0,
            estimated_cost_usd=None,
            output_bounded=False,
        )


# ---------------------------------------------------------------------------
# Static router for tests
# ---------------------------------------------------------------------------


class StaticRouter(LLMRouter):
    """Returns a fixed candidate list; optionally overrides should_escalate."""

    def __init__(
        self,
        candidates_list: list[RoutedModel],
        escalate_content: str | None = None,
    ) -> None:
        self._candidates = candidates_list
        # When set, should_escalate returns True for responses whose content matches.
        self._escalate_content = escalate_content

    @override
    def candidates(self, ctx: RoutingContext) -> Sequence[RoutedModel]:
        del ctx
        return self._candidates

    @override
    def should_escalate(self, response: LLMResponse | None) -> bool:
        if self._escalate_content is None or response is None:
            return False
        return response.content == self._escalate_content


# ---------------------------------------------------------------------------
# Shared agent + config
# ---------------------------------------------------------------------------


def _agent(name: str = "routing-test-agent") -> Agent:
    return Agent(name=name, system_prompt="test agent")


def _config() -> RunConfig:
    return RunConfig()


def _hooks() -> RunHooks:
    return RunHooks()


# ---------------------------------------------------------------------------
# Patch helpers: bypass enforce_tenant_budget (no-op in these tests)
# and build_tools (no real tools needed).
# ---------------------------------------------------------------------------


def _noop_patches() -> Any:
    from unittest.mock import AsyncMock

    return patch(
        "troopai.adk.run.llm_calls.build_tools",
        new=AsyncMock(return_value=None),
    )


# ---------------------------------------------------------------------------
# Test 1 — escalate to next candidate on non-TroopAIError exception
# ---------------------------------------------------------------------------


async def test_escalates_to_next_on_exception() -> None:
    """Router [raising, good] → outcome from the good candidate."""
    raising = RaisingFakeLLM()
    good = GoodFakeLLM(content="hello from good")

    router = StaticRouter(
        [
            RoutedModel(llm=raising, model="raising"),
            RoutedModel(llm=good, model="good"),
        ]
    )
    agent = _agent()
    config = _config()

    with _noop_patches():
        outcome = await call_llm_with_routing(
            router,
            agent,
            [],
            config,
            _hooks(),
        )

    assert outcome.model == "good"
    assert outcome.response.content == "hello from good"
    assert outcome.llm is good


# ---------------------------------------------------------------------------
# Test 2 — all candidates fail → NoRoutingCandidateError chaining last error
# ---------------------------------------------------------------------------


async def test_all_fail_raises_no_candidate() -> None:
    """Router [raising, raising] → NoRoutingCandidateError whose __cause__ is RuntimeError."""
    router = StaticRouter(
        [
            RoutedModel(llm=RaisingFakeLLM(), model="raise-1"),
            RoutedModel(llm=RaisingFakeLLM(), model="raise-2"),
        ]
    )
    agent = _agent()
    config = _config()

    with _noop_patches(), pytest.raises(NoRoutingCandidateError) as exc_info:
        await call_llm_with_routing(router, agent, [], config, _hooks())

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, RuntimeError)


# ---------------------------------------------------------------------------
# Test 3 — TroopAIError is NOT retried; propagates immediately
# ---------------------------------------------------------------------------


async def test_framework_error_not_retried() -> None:
    """A UserError (TroopAIError) from the first candidate propagates immediately.

    The second (good) candidate must never be reached.
    """
    framework_err_llm = FrameworkErrorFakeLLM()
    good = GoodFakeLLM(content="should not reach me")

    router = StaticRouter(
        [
            RoutedModel(llm=framework_err_llm, model="framework-err"),
            RoutedModel(llm=good, model="good"),
        ]
    )
    agent = _agent()
    config = _config()

    with _noop_patches(), pytest.raises(UserError):
        await call_llm_with_routing(router, agent, [], config, _hooks())

    # The UserError propagated immediately; the second (good) candidate was never reached.
    assert good.acomplete_called is False


# ---------------------------------------------------------------------------
# Test 4 — should_escalate predicate triggers escalation
# ---------------------------------------------------------------------------


async def test_should_escalate_predicate_triggers_escalation() -> None:
    """Router escalates when should_escalate returns True for the first response.

    First candidate returns content "bad"; router escalates (predicate matches).
    Second candidate returns "good"; router accepts.
    """
    bad_llm = GoodFakeLLM(content="bad")
    good_llm = GoodFakeLLM(content="good")

    # escalate_content="bad" → should_escalate returns True when content == "bad"
    router = StaticRouter(
        [
            RoutedModel(llm=bad_llm, model="bad-model"),
            RoutedModel(llm=good_llm, model="good-model"),
        ],
        escalate_content="bad",
    )
    agent = _agent()
    config = _config()

    with _noop_patches():
        outcome = await call_llm_with_routing(router, agent, [], config, _hooks())

    assert outcome.model == "good-model"
    assert outcome.response.content == "good"
    assert outcome.llm is good_llm


# ---------------------------------------------------------------------------
# Test 5 — output-schema validation failure triggers escalation
# ---------------------------------------------------------------------------


async def test_schema_validation_failure_escalates() -> None:
    """First candidate response fails output-schema validation → escalates to second.

    Exercises the _routed_response_ok schema path in call_llm_with_routing.
    """
    from pydantic import BaseModel

    class Out(BaseModel):
        value: int

    # First candidate returns content that is not valid JSON for Out schema.
    invalid_llm = GoodFakeLLM(content="not valid json")
    # Second candidate returns valid JSON for Out schema.
    valid_llm = GoodFakeLLM(content='{"value": 1}')

    router = StaticRouter(
        [
            RoutedModel(llm=invalid_llm, model="invalid-model"),
            RoutedModel(llm=valid_llm, model="valid-model"),
        ]
    )
    agent = Agent(name="schema-test-agent", system_prompt="test", output_schema=Out)
    config = _config()

    with _noop_patches():
        outcome = await call_llm_with_routing(router, agent, [], config, _hooks())

    assert outcome.model == "valid-model"
    assert outcome.llm is valid_llm
