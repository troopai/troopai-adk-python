"""Tests for ``RunConfig.error_handlers`` recovery mechanism.

Covers both the non-streaming ``arun`` path and the streaming path,
confirming:

1. A registered handler for the raised exception type recovers with a
   fallback final_output.
2. An unhandled exception type propagates unchanged.
3. Subclass matching: a handler keyed on a base class matches a derived
   exception.
4. MRO: when both a specific and a base class handler are registered,
   the most-derived match wins.
5. Streaming path parity: recovery works the same on the streaming path.
6. Async handlers are awaited correctly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.exceptions.exceptions import ModelRefusalError, TroopAIError
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner, _resolve_error_handler
from troopai.adk.run.stream import RunResultStreaming
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
)

# ── Helpers ──────────────────────────────────────────────────────────


def _final_text_response(text: str = "done") -> LLMResponse:
    """LLMResponse that produces a final text output (no tool call)."""
    from troopai.adk.types.responses.llm_response import LLMResponseText

    return LLMResponse(
        response_id="resp-final",
        model="fake",
        response=[LLMResponseText(text=text)],
    )


def _make_agent() -> Agent:
    return Agent(name="test-agent", system_prompt="You are a test agent.")


async def _patched_arun_raises(
    agent: Agent,
    prompt: str,
    *,
    run_config: RunConfig,
    stream: bool = False,
    session: Any | None = None,
) -> Any:
    """Run ``Runner.arun`` with call_llm stubbed to raise ModelRefusalError."""

    async def fake_call_llm(*_args: Any, **_kwargs: Any) -> LLMResponse:
        raise ModelRefusalError("refused")

    async def fake_call_llm_streamed(*_args: Any, **_kwargs: Any) -> LLMResponse:
        raise ModelRefusalError("refused")

    with (
        patch(
            "troopai.adk.run.loop.call_llm",
            new=AsyncMock(side_effect=fake_call_llm),
        ),
        patch(
            "troopai.adk.run.loop.call_llm_streamed",
            new=AsyncMock(side_effect=fake_call_llm_streamed),
        ),
        patch(
            "troopai.adk.run.runner.run_blocking_input_guardrails",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "troopai.adk.run.runner.run_parallel_input_guardrails",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "troopai.adk.run.runner.run_output_guardrails",
            new=AsyncMock(return_value=[]),
        ),
    ):
        if stream:
            streaming: RunResultStreaming = await Runner.arun(
                agent,
                prompt,
                run_config=run_config,
                stream=True,
                session=session,
            )
            async for _ in streaming.stream_events():
                pass
            return streaming
        return await Runner.arun(
            agent,
            prompt,
            run_config=run_config,
            stream=False,
        )


# ── Unit tests for _resolve_error_handler ────────────────────────────


class TestResolveErrorHandler:
    def test_exact_match(self) -> None:
        handler = lambda e: "exact"  # noqa: E731
        handlers: dict[type[Exception], Any] = {ModelRefusalError: handler}
        exc = ModelRefusalError("x")
        assert _resolve_error_handler(exc, handlers) is handler

    def test_base_class_match(self) -> None:
        handler = lambda e: "base"  # noqa: E731
        handlers: dict[type[Exception], Any] = {TroopAIError: handler}
        exc = ModelRefusalError("x")  # subclass of TroopAIError
        assert _resolve_error_handler(exc, handlers) is handler

    def test_no_match_returns_none(self) -> None:
        handlers: dict[type[Exception], Any] = {ValueError: lambda e: "val"}
        exc = ModelRefusalError("x")
        assert _resolve_error_handler(exc, handlers) is None

    def test_mro_most_derived_wins(self) -> None:
        specific = lambda e: "specific"  # noqa: E731
        base = lambda e: "base"  # noqa: E731
        handlers: dict[type[Exception], Any] = {
            TroopAIError: base,
            ModelRefusalError: specific,
        }
        exc = ModelRefusalError("x")
        assert _resolve_error_handler(exc, handlers) is specific


# ── Non-streaming path ────────────────────────────────────────────────


class TestErrorHandlersNonStreaming:
    @pytest.mark.asyncio
    async def test_refusal_recovered_to_default_value(self) -> None:
        """ModelRefusalError is caught, handler returns 'fallback'."""
        agent = _make_agent()
        config = RunConfig(error_handlers={ModelRefusalError: lambda e: "fallback"})

        result = await _patched_arun_raises(agent, "go", run_config=config)

        assert result.final_output == "fallback"

    @pytest.mark.asyncio
    async def test_unhandled_exception_type_propagates(self) -> None:
        """error_handlers maps ModelRefusalError but a different exception is raised."""
        agent = _make_agent()

        async def fake_llm(*_args: Any, **_kwargs: Any) -> LLMResponse:
            raise ValueError("not a refusal")

        config = RunConfig(error_handlers={ModelRefusalError: lambda e: "fallback"})

        with (
            patch("troopai.adk.run.loop.call_llm", new=AsyncMock(side_effect=fake_llm)),
            patch(
                "troopai.adk.run.loop.call_llm_streamed",
                new=AsyncMock(side_effect=fake_llm),
            ),
            patch(
                "troopai.adk.run.runner.run_blocking_input_guardrails",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "troopai.adk.run.runner.run_parallel_input_guardrails",
                new=AsyncMock(return_value=[]),
            ),
            patch(
                "troopai.adk.run.runner.run_output_guardrails",
                new=AsyncMock(return_value=[]),
            ),
            pytest.raises(ValueError, match="not a refusal"),
        ):
            await Runner.arun(agent, "go", run_config=config)

    @pytest.mark.asyncio
    async def test_subclass_matching(self) -> None:
        """error_handlers maps Exception (base); ModelRefusalError raised → resolved."""
        agent = _make_agent()
        config = RunConfig(error_handlers={Exception: lambda e: "base-fallback"})

        result = await _patched_arun_raises(agent, "go", run_config=config)

        assert result.final_output == "base-fallback"

    @pytest.mark.asyncio
    async def test_mro_most_derived_wins(self) -> None:
        """error_handlers has both ModelRefusalError and TroopAIError; specific wins."""
        agent = _make_agent()
        config = RunConfig(
            error_handlers={
                ModelRefusalError: lambda e: "specific-fallback",
                TroopAIError: lambda e: "base-fallback",
            }
        )

        result = await _patched_arun_raises(agent, "go", run_config=config)

        assert result.final_output == "specific-fallback"

    @pytest.mark.asyncio
    async def test_async_handler(self) -> None:
        """Async handler is awaited; its return value becomes final_output."""
        agent = _make_agent()

        async def async_handler(exc: Exception) -> str:
            return "async-fallback"

        config = RunConfig(error_handlers={ModelRefusalError: async_handler})

        result = await _patched_arun_raises(agent, "go", run_config=config)

        assert result.final_output == "async-fallback"


# ── Streaming path ────────────────────────────────────────────────────


class TestErrorHandlersStreaming:
    @pytest.mark.asyncio
    async def test_streaming_path_parity(self) -> None:
        """Streaming run with ModelRefusalError recovered → final_output == 'fallback'."""
        agent = _make_agent()
        config = RunConfig(error_handlers={ModelRefusalError: lambda e: "fallback"})

        streaming = await _patched_arun_raises(agent, "go", run_config=config, stream=True)

        assert streaming.final_output == "fallback"


class TestRecoverySkipsPersistence:
    async def test_streaming_recovery_skips_session_persistence(self) -> None:
        """A recovered streamed run must not persist its truncated turn.

        Persisting the partial items would seed the next turn with a
        half-formed exchange; the documented contract is skip-on-recovery
        in both paths.
        """
        session = MagicMock()
        session.settings = None
        session.get = AsyncMock(return_value=[])
        session.add = AsyncMock()
        session.save_state = AsyncMock()
        session.id = "s-recover"

        agent = _make_agent()
        config = RunConfig(error_handlers={ModelRefusalError: lambda e: "fallback"})

        streaming = await _patched_arun_raises(agent, "go", run_config=config, stream=True, session=session)

        assert streaming.recovered is True
        assert streaming.final_output == "fallback"
        session.add.assert_not_awaited()
        session.save_state.assert_not_awaited()

    async def test_non_streaming_recovered_result_carries_context_and_flag(self) -> None:
        agent = _make_agent()
        config = RunConfig(error_handlers={ModelRefusalError: lambda e: "fallback"})

        result = await _patched_arun_raises(agent, "go", run_config=config)

        assert result.recovered is True
        assert result.context is not None
