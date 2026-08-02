"""Tests for call_llm_streamed_with_routing escalation driver.

Proves:
(a) Escalation to the next candidate when a pre-token exception is retryable.
(b) No escalation once streaming has begun (has_emitted_tokens is True).
(c) NoRoutingCandidateError when all candidates fail before emitting any token.

Streaming fake-LLM pattern: minimal ``LLM`` subclass whose ``acomplete``
(with ``stream=True``) returns an ``AsyncIterator[LLMStreamEvent]``.

- ``RaisingPreTokenLLM``: raises immediately inside the async iterator (before
  any ``part_delta``), simulating a provider error before the first byte.
- ``RaisingMidStreamLLM``: emits one text ``part_delta`` then raises, simulating
  a mid-stream disconnection.
- ``GoodStreamLLM``: emits a text ``part_start`` + ``part_delta`` + ``done``
  sequence and tracks whether its stream was started.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
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
from troopai.adk.run.llm_calls import call_llm_streamed_with_routing
from troopai.adk.run.stream import RunResultStreaming
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseText,
    LLMStreamEvent,
)
from troopai.adk.types.tokens.llm_usage import LLMUsage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _done_response(model: str = "fake-stream") -> LLMResponse:
    return LLMResponse(
        response_id="resp-stream-test",
        model=model,
        response=[LLMResponseText(text="streamed content")],
        usage=LLMUsage(requests=1, input_tokens=5, output_tokens=3, total_tokens=8),
    )


def _text_part_start_event(index: int = 0) -> LLMStreamEvent:
    return LLMStreamEvent(type="part_start", index=index, part=LLMResponseText(text=""))


def _text_part_delta_event(delta: str, index: int = 0) -> LLMStreamEvent:
    return LLMStreamEvent(type="part_delta", index=index, delta=delta)


# ---------------------------------------------------------------------------
# Fake LLM implementations
# ---------------------------------------------------------------------------


class RaisingPreTokenLLM(LLM):
    """Raises RuntimeError before yielding any event (pre-token failure)."""

    @override
    async def acomplete(  # type: ignore[override]
        self,
        messages: Any,
        llm_config: Any = None,
        tools: Any = None,
        output_schema: Any = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
        if stream:

            async def gen() -> AsyncIterator[LLMStreamEvent]:
                raise RuntimeError("provider timeout before first token")
                yield  # pragma: no cover — unreachable; satisfies AsyncGenerator protocol

            return gen()
        raise RuntimeError("non-streaming not used in routing tests")

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


class RaisingMidStreamLLM(LLM):
    """Emits one text part_delta (so has_emitted_tokens becomes True) then raises."""

    @override
    async def acomplete(  # type: ignore[override]
        self,
        messages: Any,
        llm_config: Any = None,
        tools: Any = None,
        output_schema: Any = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
        if stream:

            async def gen() -> AsyncIterator[LLMStreamEvent]:
                yield _text_part_start_event()
                yield _text_part_delta_event("hello")
                raise RuntimeError("mid-stream disconnection")

            return gen()
        raise RuntimeError("non-streaming not used in routing tests")

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


class FrameworkErrorPreTokenLLM(LLM):
    """Raises a UserError (TroopAIError subclass) before yielding any token."""

    @override
    async def acomplete(  # type: ignore[override]
        self,
        messages: Any,
        llm_config: Any = None,
        tools: Any = None,
        output_schema: Any = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
        if stream:

            async def gen() -> AsyncIterator[LLMStreamEvent]:
                raise UserError("framework error before first token")
                yield  # pragma: no cover — unreachable; satisfies AsyncGenerator protocol

            return gen()
        raise RuntimeError("non-streaming not used in routing tests")

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


class GoodStreamLLM(LLM):
    """Emits part_start + part_delta + done successfully. Tracks if it was started."""

    def __init__(self, model_name: str = "good-stream") -> None:
        self._model_name = model_name
        self.stream_started: bool = False

    @override
    async def acomplete(  # type: ignore[override]
        self,
        messages: Any,
        llm_config: Any = None,
        tools: Any = None,
        output_schema: Any = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
        self.stream_started = True
        if stream:
            response = _done_response(self._model_name)

            async def gen() -> AsyncIterator[LLMStreamEvent]:
                yield _text_part_start_event()
                yield _text_part_delta_event("streamed content")
                yield LLMStreamEvent(type="done", response=response)

            return gen()
        return _done_response(self._model_name)

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
# Static router
# ---------------------------------------------------------------------------


class StaticRouter(LLMRouter):
    """Returns a fixed candidate list. No escalation predicate."""

    def __init__(self, candidates_list: list[RoutedModel]) -> None:
        self._candidates = candidates_list

    @override
    def candidates(self, ctx: RoutingContext) -> list[RoutedModel]:
        del ctx
        return self._candidates

    @override
    def should_escalate(self, response: LLMResponse | None) -> bool:
        return False


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _agent(name: str = "stream-routing-test") -> Agent:
    return Agent(name=name, system_prompt="test agent")


def _config() -> RunConfig:
    return RunConfig()


def _hooks() -> RunHooks:
    return RunHooks()


def _make_result(agent: Agent) -> RunResultStreaming:
    return RunResultStreaming(current_agent=agent)


def _noop_patches() -> Any:
    from unittest.mock import AsyncMock

    return patch(
        "troopai.adk.run.llm_calls.build_tools",
        new=AsyncMock(return_value=None),
    )


# ---------------------------------------------------------------------------
# Test 1 — escalate to next candidate when the first raises before any token
# ---------------------------------------------------------------------------


async def test_streamed_escalates_before_first_token() -> None:
    """Router [raising-pre-token, good] — outcome comes from the good candidate.

    The first LLM raises before emitting any token, so has_emitted_tokens
    remains False and escalation to the second candidate is safe.
    """
    raising = RaisingPreTokenLLM()
    good = GoodStreamLLM(model_name="good-stream")

    router = StaticRouter(
        [
            RoutedModel(llm=raising, model="raising-pre-token"),
            RoutedModel(llm=good, model="good-stream"),
        ]
    )
    agent = _agent()
    result = _make_result(agent)

    with _noop_patches():
        outcome = await call_llm_streamed_with_routing(
            router=router,
            agent=agent,
            messages=[],
            config=_config(),
            hooks=_hooks(),
            result=result,
        )

    assert outcome.model == "good-stream"
    assert outcome.llm is good
    assert result.has_emitted_tokens is True


# ---------------------------------------------------------------------------
# Test 2 — no escalation once a token has been emitted to the consumer
# ---------------------------------------------------------------------------


async def test_streamed_no_escalation_after_first_token() -> None:
    """Router [mid-stream-raiser, good] — raises RuntimeError and does NOT escalate.

    The first LLM emits one text delta (setting has_emitted_tokens=True)
    then raises. The routing layer must re-raise rather than calling the
    second (good) candidate, whose stream_started must remain False.
    """
    mid_raiser = RaisingMidStreamLLM()
    good = GoodStreamLLM(model_name="should-not-run")
    good.stream_started = False

    router = StaticRouter(
        [
            RoutedModel(llm=mid_raiser, model="mid-stream-raiser"),
            RoutedModel(llm=good, model="should-not-run"),
        ]
    )
    agent = _agent()
    result = _make_result(agent)

    with _noop_patches(), pytest.raises(RuntimeError, match="mid-stream disconnection"):
        await call_llm_streamed_with_routing(
            router=router,
            agent=agent,
            messages=[],
            config=_config(),
            hooks=_hooks(),
            result=result,
        )

    # Token was emitted by the first candidate before it raised.
    assert result.has_emitted_tokens is True
    # The second candidate's stream was never started.
    assert good.stream_started is False


# ---------------------------------------------------------------------------
# Test 3 — all candidates fail pre-token → NoRoutingCandidateError
# ---------------------------------------------------------------------------


async def test_streamed_all_fail_raises() -> None:
    """Router [raising, raising] → NoRoutingCandidateError chaining last RuntimeError."""
    router = StaticRouter(
        [
            RoutedModel(llm=RaisingPreTokenLLM(), model="raise-1"),
            RoutedModel(llm=RaisingPreTokenLLM(), model="raise-2"),
        ]
    )
    agent = _agent()
    result = _make_result(agent)

    with _noop_patches(), pytest.raises(NoRoutingCandidateError) as exc_info:
        await call_llm_streamed_with_routing(
            router=router,
            agent=agent,
            messages=[],
            config=_config(),
            hooks=_hooks(),
            result=result,
        )

    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert result.has_emitted_tokens is False


# ---------------------------------------------------------------------------
# Test 4 — TroopAIError pre-token: NOT escalated, propagates immediately
# ---------------------------------------------------------------------------


async def test_streamed_framework_error_not_escalated() -> None:
    """Router [framework-error, good] — UserError propagates; good candidate never starts.

    A first streaming candidate whose acomplete raises a UserError (TroopAIError
    subclass) before any token is NOT escalated to the next candidate.
    Framework errors are deliberate stops, not transient failures.
    """
    framework_err = FrameworkErrorPreTokenLLM()
    good = GoodStreamLLM(model_name="should-not-run")
    good.stream_started = False

    router = StaticRouter(
        [
            RoutedModel(llm=framework_err, model="framework-err-stream"),
            RoutedModel(llm=good, model="should-not-run"),
        ]
    )
    agent = _agent()
    result = _make_result(agent)

    with _noop_patches(), pytest.raises(UserError):
        await call_llm_streamed_with_routing(
            router=router,
            agent=agent,
            messages=[],
            config=_config(),
            hooks=_hooks(),
            result=result,
        )

    # The UserError propagated immediately; the second candidate was never started.
    assert good.stream_started is False
    assert result.has_emitted_tokens is False
