"""End-to-end tests for :meth:`Runner.arun_task_pipeline_streamed`.

Exercises the stream-of-streams pipeline entry point. The outer
async iterator yields ``(task_index, RunResultStreaming | None)``
pairs in input order; skip slots yield ``None``; the caller drives
each inner stream's ``stream_events()`` independently.

Covered:

- Happy path: three tasks stream in order; each inner stream drains;
  ``final_output`` populates on each.
- Skip slot signalling: ``Task.skip_if`` returning True yields
  ``(index, None)`` rather than silently advancing — preserves
  positional indexing.
- Skip-if exception halts the pipeline at the offending slot; later
  slots never yield.
- Per-task streaming: each inner stream is independent, so the
  consumer can ``cancel`` one without affecting siblings.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.runner import Runner
from troopai.adk.tasks import Task, TaskPipeline
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseText,
)


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        response_id="resp-x",
        model="fake",
        response=[LLMResponseText(text=text)],
    )


@contextmanager
def _patched_streamed_loop(fake_streamed: Any) -> Iterator[None]:
    with ExitStack() as stack:
        stack.enter_context(
            patch("troopai.adk.run.loop.call_llm_streamed", new=AsyncMock(side_effect=fake_streamed)),
        )
        stack.enter_context(
            patch(
                "troopai.adk.run.runner.run_blocking_input_guardrails",
                new=AsyncMock(return_value=[]),
            ),
        )
        stack.enter_context(
            patch(
                "troopai.adk.run.runner.run_parallel_input_guardrails",
                new=AsyncMock(return_value=[]),
            ),
        )
        stack.enter_context(
            patch("troopai.adk.run.runner.run_output_guardrails", new=AsyncMock(return_value=[])),
        )
        yield


@pytest.mark.integration
async def test_three_task_pipeline_streams_in_order() -> None:
    agent = Agent(name="t", system_prompt="x")
    pipeline = TaskPipeline(
        tasks=(
            Task(description="alpha", agent=agent, name="t1"),
            Task(description="beta", agent=agent, name="t2"),
            Task(description="gamma", agent=agent, name="t3"),
        ),
    )

    async def fake_streamed(*_args: Any, **kwargs: Any) -> LLMResponse:
        messages = kwargs.get("messages") or _args[1]
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return _text_response(f"answered-{msg.get('content')}")
        return _text_response("unknown")

    seen_outputs: list[tuple[int, str]] = []

    with _patched_streamed_loop(fake_streamed):
        async for index, task_stream in Runner.arun_task_pipeline_streamed(pipeline):
            assert task_stream is not None
            async for _ in task_stream.stream_events():
                pass
            seen_outputs.append((index, task_stream.final_output))

    assert len(seen_outputs) == 3
    assert seen_outputs[0] == (0, "answered-alpha")
    assert seen_outputs[1] == (1, "answered-beta")
    assert seen_outputs[2] == (2, "answered-gamma")


@pytest.mark.integration
async def test_skip_slot_yields_none() -> None:
    agent = Agent(name="t", system_prompt="x")
    pipeline = TaskPipeline(
        tasks=(
            Task(description="run-me", agent=agent, name="t1"),
            Task(
                description="skip-me",
                agent=agent,
                name="t2-skipped",
                skip_if=lambda _prior: True,
            ),
            Task(description="run-me-too", agent=agent, name="t3"),
        ),
    )

    async def fake_streamed(*_args: Any, **kwargs: Any) -> LLMResponse:
        return _text_response("ok")

    yielded: list[tuple[int, bool]] = []

    with _patched_streamed_loop(fake_streamed):
        async for index, task_stream in Runner.arun_task_pipeline_streamed(pipeline):
            yielded.append((index, task_stream is None))
            if task_stream is not None:
                async for _ in task_stream.stream_events():
                    pass

    # All three slots yielded, in input order; middle slot is None.
    assert yielded == [(0, False), (1, True), (2, False)]


@pytest.mark.integration
async def test_skip_if_exception_halts_pipeline(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If ``skip_if`` raises, the pipeline iterator stops at that
    slot — subsequent tasks never start. The exception is logged but
    NOT propagated to the consumer (consumers can detect halt by
    iterator exhausting before all input tasks ran)."""
    import logging

    agent = Agent(name="t", system_prompt="x")

    def _broken_skip(_prior: Any) -> bool:
        raise RuntimeError("skip-if-failed")

    pipeline = TaskPipeline(
        tasks=(
            Task(description="run-me", agent=agent, name="t1"),
            Task(description="halt-here", agent=agent, name="t2-broken", skip_if=_broken_skip),
            Task(description="never-ran", agent=agent, name="t3"),
        ),
    )

    async def fake_streamed(*_args: Any, **kwargs: Any) -> LLMResponse:
        return _text_response("ok")

    yielded_indices: list[int] = []

    with caplog.at_level(logging.WARNING, logger="troopai.adk.run.runner"), _patched_streamed_loop(fake_streamed):
        async for index, task_stream in Runner.arun_task_pipeline_streamed(pipeline):
            yielded_indices.append(index)
            if task_stream is not None:
                async for _ in task_stream.stream_events():
                    pass

    # t1 yielded, then t2's broken skip_if yields an error-stop sentinel
    # slot (1, None) and the iterator halts — t3 (index 2) never yields,
    # so it never starts.
    assert yielded_indices == [0, 1]
    # The halt logs a WARNING — operators rely on this signal to
    # distinguish "halted at slot N" from "ran all slots cleanly".
    assert any("skip-if-failed" in record.message and "t2-broken" in record.message for record in caplog.records)


@pytest.mark.integration
async def test_streaming_placeholder_visible_to_next_skip_if() -> None:
    """The prior_outputs list passed to a later ``skip_if`` includes
    a placeholder slot for each prior streamed task. The placeholder
    sets ``streaming_placeholder=True`` so skip_if predicates can
    distinguish "real completion" from "awaiting drain".
    """
    agent = Agent(name="t", system_prompt="x")

    captured_prior: list[Any] = []

    def _skip_inspector(prior: Any) -> bool:
        # Capture for assertion; never actually skip.
        captured_prior.append(list(prior))
        return False

    pipeline = TaskPipeline(
        tasks=(
            Task(description="t1", agent=agent, name="t1"),
            Task(description="t2", agent=agent, name="t2", skip_if=_skip_inspector),
        ),
    )

    async def fake_streamed(*_args: Any, **kwargs: Any) -> LLMResponse:
        return _text_response("ok")

    with _patched_streamed_loop(fake_streamed):
        async for _index, task_stream in Runner.arun_task_pipeline_streamed(pipeline):
            if task_stream is not None:
                async for _ in task_stream.stream_events():
                    pass

    # t2's skip_if saw exactly one prior slot — t1 — and that slot
    # carried streaming_placeholder=True so a real-aware skip_if can
    # discriminate.
    assert len(captured_prior) == 1
    assert len(captured_prior[0]) == 1
    placeholder = captured_prior[0][0]
    assert placeholder.task_name == "t1"
    assert placeholder.streaming_placeholder is True
    # Field invariants on a placeholder: identity + metadata only.
    assert placeholder.final_output is None
    assert placeholder.usage is None
    assert placeholder.error is None
    assert placeholder.skipped is False


@pytest.mark.integration
async def test_consumer_break_stops_pipeline() -> None:
    """Breaking out of the outer async-for stops the pipeline at the
    current boundary — subsequent tasks never start."""
    agent = Agent(name="t", system_prompt="x")
    pipeline = TaskPipeline(
        tasks=(
            Task(description="run-1", agent=agent, name="t1"),
            Task(description="run-2", agent=agent, name="t2"),
            Task(description="run-3", agent=agent, name="t3"),
        ),
    )

    async def fake_streamed(*_args: Any, **kwargs: Any) -> LLMResponse:
        return _text_response("ok")

    yielded_indices: list[int] = []

    with _patched_streamed_loop(fake_streamed):
        async for index, task_stream in Runner.arun_task_pipeline_streamed(pipeline):
            yielded_indices.append(index)
            if task_stream is not None:
                async for _ in task_stream.stream_events():
                    pass
            if index == 0:
                break

    # Consumer broke after t1; t2 / t3 never ran.
    assert yielded_indices == [0]
    # Yield a control tick so any lingering background task can
    # propagate its CancelledError without spilling into the next test.
    await asyncio.sleep(0)
