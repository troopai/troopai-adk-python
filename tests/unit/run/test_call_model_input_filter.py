"""Tests for ``RunConfig.call_model_input_filter``.

Covers the file-private helper ``_apply_call_model_input_filter`` in
``troopai.adk.run.loop`` plus one end-to-end integration via
``Runner.arun`` with a stubbed ``call_llm`` that captures what the
filter actually produced.

The cross-module underscore import of ``_apply_call_model_input_filter``
below is a deliberate test-only exception: we are unit-testing a
file-private helper from the same package, not using it as a public
API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.run.config import (
    CallModelData,
    ModelInputData,
    RunConfig,
)
from troopai.adk.run.context import RunContext
from troopai.adk.run.loop import _apply_call_model_input_filter  # test-only import
from troopai.adk.types.input import LLMInputContentItem
from troopai.adk.types.input.llm_input_easy_message import LLMInputEasyMessage

# ── Helpers ──────────────────────────────────────────────────────────


@dataclass
class _FakeAgent:
    """Minimal Agent stand-in for helper-level tests.

    The helper only reads ``agent.name``, so a real ``Agent`` instance
    isn't required.
    """

    name: str = "test-agent"


def _make_messages(n: int = 2) -> list[LLMInputContentItem]:
    msgs: list[LLMInputContentItem] = []
    sys_msg: LLMInputEasyMessage = {"role": "system", "content": "You are helpful."}
    msgs.append(sys_msg)
    for i in range(n):
        user_msg: LLMInputEasyMessage = {"role": "user", "content": f"Question {i}"}
        msgs.append(user_msg)
    return msgs


def _make_ctx(context: Any = None) -> RunContext:
    return RunContext(context=context)


# ── Helper-level tests ───────────────────────────────────────────────


class TestCallModelInputFilterHelper:
    """Direct unit tests of ``_apply_call_model_input_filter``."""

    @pytest.mark.asyncio
    async def test_none_filter_is_passthrough(self) -> None:
        """With no filter, the helper returns the input list unchanged (identity)."""
        config = RunConfig(call_model_input_filter=None)
        messages = _make_messages()
        result = await _apply_call_model_input_filter(
            agent=_FakeAgent(),  # type: ignore[arg-type]
            config=config,
            ctx_wrapper=_make_ctx(),
            messages=messages,
        )
        assert result is messages  # identity — no copy on pass-through

    @pytest.mark.asyncio
    async def test_sync_filter_mutates_input(self) -> None:
        """A sync filter returning a new ModelInputData is applied."""

        def sync_filter(payload: CallModelData) -> ModelInputData:
            new_input = list(payload.model_data.input)
            extra: LLMInputEasyMessage = {"role": "user", "content": "injected"}
            new_input.append(extra)
            return ModelInputData(input=new_input)

        config = RunConfig(call_model_input_filter=sync_filter)
        messages = _make_messages(n=2)

        result = await _apply_call_model_input_filter(
            agent=_FakeAgent(),  # type: ignore[arg-type]
            config=config,
            ctx_wrapper=_make_ctx(),
            messages=messages,
        )
        assert len(result) == len(messages) + 1
        assert result[-1]["content"] == "injected"  # type: ignore[typeddict-item]

    @pytest.mark.asyncio
    async def test_async_filter_mutates_input(self) -> None:
        """An async filter is awaited and its return value is applied."""

        async def async_filter(payload: CallModelData) -> ModelInputData:
            new_input = list(payload.model_data.input)
            extra: LLMInputEasyMessage = {"role": "user", "content": "async-injected"}
            new_input.append(extra)
            return ModelInputData(input=new_input)

        config = RunConfig(call_model_input_filter=async_filter)
        messages = _make_messages(n=1)

        result = await _apply_call_model_input_filter(
            agent=_FakeAgent(),  # type: ignore[arg-type]
            config=config,
            ctx_wrapper=_make_ctx(),
            messages=messages,
        )
        assert len(result) == len(messages) + 1
        assert result[-1]["content"] == "async-injected"  # type: ignore[typeddict-item]

    @pytest.mark.asyncio
    async def test_filter_receives_agent_and_context(self) -> None:
        """The payload carries the exact agent and unwrapped user context."""
        agent = _FakeAgent(name="inspector")
        user_ctx = {"user_id": "u-123", "role": "admin"}
        seen: dict[str, Any] = {}

        def inspecting_filter(payload: CallModelData) -> ModelInputData:
            seen["agent"] = payload.agent
            seen["context"] = payload.context
            return payload.model_data

        config = RunConfig(call_model_input_filter=inspecting_filter)
        messages = _make_messages()

        await _apply_call_model_input_filter(
            agent=agent,  # type: ignore[arg-type]
            config=config,
            ctx_wrapper=_make_ctx(user_ctx),
            messages=messages,
        )
        assert seen["agent"] is agent
        assert seen["context"] == user_ctx

    @pytest.mark.asyncio
    async def test_filter_wrong_return_type_raises_typeerror(self) -> None:
        """Returning something other than ModelInputData raises TypeError."""

        def bad_filter(_payload: CallModelData) -> Any:
            return ["not", "a", "ModelInputData"]

        config = RunConfig(call_model_input_filter=bad_filter)
        messages = _make_messages()

        with pytest.raises(TypeError, match="must return ModelInputData"):
            await _apply_call_model_input_filter(
                agent=_FakeAgent(),  # type: ignore[arg-type]
                config=config,
                ctx_wrapper=_make_ctx(),
                messages=messages,
            )

    @pytest.mark.asyncio
    async def test_filter_exception_propagates(self) -> None:
        """An exception inside the filter is re-raised to the caller."""

        def exploding_filter(_payload: CallModelData) -> ModelInputData:
            raise RuntimeError("kaboom")

        config = RunConfig(call_model_input_filter=exploding_filter)
        messages = _make_messages()

        with pytest.raises(RuntimeError, match="kaboom"):
            await _apply_call_model_input_filter(
                agent=_FakeAgent(),  # type: ignore[arg-type]
                config=config,
                ctx_wrapper=_make_ctx(),
                messages=messages,
            )

    @pytest.mark.asyncio
    async def test_filter_receives_shallow_copy(self) -> None:
        """In-place mutation of ``payload.model_data.input`` must not leak back.

        The helper hands the filter a shallow copy so a filter that
        mutates the wrapper's list in place (instead of returning a new
        ModelInputData) cannot silently corrupt the pre-filter messages.
        """

        def in_place_filter(payload: CallModelData) -> ModelInputData:
            mutation: LLMInputEasyMessage = {"role": "user", "content": "mutated"}
            payload.model_data.input.append(mutation)
            return payload.model_data

        config = RunConfig(call_model_input_filter=in_place_filter)
        original = _make_messages(n=2)
        original_len = len(original)

        result = await _apply_call_model_input_filter(
            agent=_FakeAgent(),  # type: ignore[arg-type]
            config=config,
            ctx_wrapper=_make_ctx(),
            messages=original,
        )
        # Original list is untouched — the filter mutated the copy
        assert len(original) == original_len
        # Result reflects the mutation
        assert len(result) == original_len + 1
        assert result[-1]["content"] == "mutated"  # type: ignore[typeddict-item]


# ── End-to-end integration via Runner.arun ───────────────────────────


class TestCallModelInputFilterIntegration:
    """Integration: the filter's output is what actually reaches the LLM."""

    @pytest.mark.asyncio
    async def test_filter_output_reaches_call_llm(self) -> None:
        """With a filter set, ``call_llm`` receives the filtered messages."""
        from troopai.adk.agents.agent import Agent
        from troopai.adk.run.runner import Runner
        from troopai.adk.types.responses.llm_response import (
            LLMResponse,
            LLMResponseText,
        )

        calls: list[list[LLMInputContentItem]] = []

        async def fake_call_llm(_agent, messages, _config, **_kwargs):
            calls.append(list(messages))
            return LLMResponse(
                response_id="test",
                model="fake",
                response=[LLMResponseText(text="final answer")],
            )

        def injector(payload: CallModelData) -> ModelInputData:
            new_input = list(payload.model_data.input)
            marker: LLMInputEasyMessage = {
                "role": "user",
                "content": "[injected-by-filter]",
            }
            new_input.append(marker)
            return ModelInputData(input=new_input)

        agent = Agent(name="test", system_prompt="You are helpful.")
        config = RunConfig(call_model_input_filter=injector)

        with (
            patch("troopai.adk.run.loop.call_llm", new=AsyncMock(side_effect=fake_call_llm)),
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
            result = await Runner.arun(agent, "hello", run_config=config)

        assert result.final_output == "final answer"
        assert len(calls) >= 1
        # The filter must have run on EVERY call_llm invocation, so each
        # captured message list must end with the injected marker.
        for i, messages in enumerate(calls):
            last = messages[-1]
            assert last.get("content") == "[injected-by-filter]", (  # type: ignore[union-attr]
                f"call #{i} did not end with the filter's injected marker"
            )
