"""Tests for :class:`~troopai.adk.workflows.temporal.streaming.TemporalStreamingLLM`
and :class:`~troopai.adk.workflows.temporal.streaming.StreamTokenEvent`.

Covers:
- StreamTokenEvent defaults and frozen semantics.
- TemporalStreamingLLM subclass relationship with TemporalLLM.
- stream_topic_prefix field existence and default value.
"""

from __future__ import annotations

import dataclasses

import pytest

pytest.importorskip("temporalio")


# ---------------------------------------------------------------------------
# StreamTokenEvent
# ---------------------------------------------------------------------------


class TestStreamTokenEventDefaults:
    def test_stream_token_event_defaults(self) -> None:
        """StreamTokenEvent has correct default field values."""
        from troopai.adk.workflows.temporal.streaming import StreamTokenEvent

        event = StreamTokenEvent()

        assert event.content == ""
        assert event.is_final is False
        assert event.metadata == {}

    def test_stream_token_event_accepts_values(self) -> None:
        """StreamTokenEvent stores provided field values."""
        from troopai.adk.workflows.temporal.streaming import StreamTokenEvent

        event = StreamTokenEvent(content="hello", is_final=True, metadata={"k": "v"})

        assert event.content == "hello"
        assert event.is_final is True
        assert event.metadata == {"k": "v"}


class TestStreamTokenEventIsFrozen:
    def test_stream_token_event_is_frozen(self) -> None:
        """StreamTokenEvent is a frozen dataclass (mutation raises)."""
        from troopai.adk.workflows.temporal.streaming import StreamTokenEvent

        event = StreamTokenEvent(content="x")

        with pytest.raises((dataclasses.FrozenInstanceError, AttributeError)):
            event.content = "y"  # type: ignore[misc]

    def test_stream_token_event_is_dataclass(self) -> None:
        """StreamTokenEvent is registered as a dataclass."""
        from troopai.adk.workflows.temporal.streaming import StreamTokenEvent

        assert dataclasses.is_dataclass(StreamTokenEvent)


# ---------------------------------------------------------------------------
# TemporalStreamingLLM subclass relationship
# ---------------------------------------------------------------------------


class TestTemporalStreamingLLMInheritsTemporalLLM:
    def test_temporal_streaming_llm_inherits_temporal_llm(self) -> None:
        """TemporalStreamingLLM is a subclass of TemporalLLM."""
        from troopai.adk.workflows.temporal.llm import TemporalLLM
        from troopai.adk.workflows.temporal.streaming import TemporalStreamingLLM

        assert issubclass(TemporalStreamingLLM, TemporalLLM)

    def test_temporal_streaming_llm_instance_is_temporal_llm(self) -> None:
        """A TemporalStreamingLLM instance satisfies isinstance checks for TemporalLLM."""
        from unittest.mock import MagicMock

        from troopai.adk.llms.llm import LLM
        from troopai.adk.workflows.engine import ModelActivityConfig
        from troopai.adk.workflows.temporal.llm import TemporalLLM
        from troopai.adk.workflows.temporal.streaming import TemporalStreamingLLM

        wrapped = MagicMock(spec=LLM)
        llm = TemporalStreamingLLM(
            wrapped=wrapped,
            activity_config=ModelActivityConfig(),
            model_name="test-model",
        )

        assert isinstance(llm, TemporalLLM)


# ---------------------------------------------------------------------------
# stream_topic_prefix field
# ---------------------------------------------------------------------------


class TestTemporalStreamingLLMHasStreamTopicName:
    def test_temporal_streaming_llm_has_stream_topic_name(self) -> None:
        """TemporalStreamingLLM exposes the stream_topic_prefix field."""
        from unittest.mock import MagicMock

        from troopai.adk.llms.llm import LLM
        from troopai.adk.workflows.engine import ModelActivityConfig
        from troopai.adk.workflows.temporal.streaming import TemporalStreamingLLM

        wrapped = MagicMock(spec=LLM)
        llm = TemporalStreamingLLM(
            wrapped=wrapped,
            activity_config=ModelActivityConfig(),
            model_name="test-model",
        )

        assert hasattr(llm, "stream_topic_prefix")
        assert llm.stream_topic_prefix == "llm-stream"

    def test_stream_topic_prefix_can_be_overridden(self) -> None:
        """stream_topic_prefix accepts a custom value at construction time."""
        from unittest.mock import MagicMock

        from troopai.adk.llms.llm import LLM
        from troopai.adk.workflows.engine import ModelActivityConfig
        from troopai.adk.workflows.temporal.streaming import TemporalStreamingLLM

        wrapped = MagicMock(spec=LLM)
        llm = TemporalStreamingLLM(
            wrapped=wrapped,
            activity_config=ModelActivityConfig(),
            model_name="test-model",
            stream_topic_prefix="my-prefix",
        )

        assert llm.stream_topic_prefix == "my-prefix"


# ---------------------------------------------------------------------------
# acomplete_streamed — outside workflow path
# ---------------------------------------------------------------------------


class TestTemporalStreamingLLMOutsideWorkflowBug:
    async def test_acomplete_streamed_awaits_wrapped_acomplete_outside_workflow(self) -> None:
        """acomplete_streamed awaits wrapped.acomplete(stream=True), then iterates it.

        The real LLM.acomplete is an ``async def`` that RETURNS an
        ``AsyncIterator`` (``return self._stream(...)``), so calling it yields a
        coroutine that must be awaited before iterating — mirroring the
        production streaming path (``await llm.acomplete(...)`` then ``async for``).
        """
        import sys
        from collections.abc import AsyncIterator
        from unittest.mock import MagicMock

        from troopai.adk.llms.llm import LLM
        from troopai.adk.types.responses.llm_response import LLMStreamEvent
        from troopai.adk.workflows.engine import ModelActivityConfig
        from troopai.adk.workflows.temporal.streaming import TemporalStreamingLLM

        async def _wrapped_acomplete(
            messages,
            llm_config=None,
            tools=None,
            output_schema=None,
            stream: bool = False,
        ) -> AsyncIterator[LLMStreamEvent]:
            """Mirror a real LLM: an async def that RETURNS an async iterator."""

            async def _gen() -> AsyncIterator[LLMStreamEvent]:
                yield LLMStreamEvent(type="done")

            return _gen()

        wrapped = MagicMock(spec=LLM)
        wrapped.acomplete = _wrapped_acomplete

        llm = TemporalStreamingLLM(
            wrapped=wrapped,
            activity_config=ModelActivityConfig(),
            model_name="test-model",
        )

        fake_workflow = MagicMock()
        fake_workflow.in_workflow.return_value = False
        original = sys.modules.get("temporalio.workflow")
        sys.modules["temporalio.workflow"] = fake_workflow
        try:
            events = [e async for e in llm.acomplete_streamed("hello")]
        finally:
            if original is None:
                del sys.modules["temporalio.workflow"]
            else:
                sys.modules["temporalio.workflow"] = original

        assert len(events) == 1
        assert events[0].type == "done"


class TestTemporalStreamingLLMOutsideWorkflow:
    async def test_acomplete_streamed_delegates_outside_workflow(self) -> None:
        """acomplete_streamed forwards to wrapped.acomplete(stream=True) outside a workflow.

        acomplete_streamed is an async generator — callers iterate it directly
        without ``await``.
        """
        import sys
        from collections.abc import AsyncIterator
        from unittest.mock import MagicMock

        from troopai.adk.llms.llm import LLM
        from troopai.adk.types.responses.llm_response import LLMStreamEvent
        from troopai.adk.workflows.engine import ModelActivityConfig
        from troopai.adk.workflows.temporal.streaming import TemporalStreamingLLM

        seen_calls: list[tuple] = []

        async def _fake_acomplete(
            messages,
            llm_config=None,
            tools=None,
            output_schema=None,
            stream: bool = False,
        ) -> AsyncIterator[LLMStreamEvent]:
            # Real LLM.acomplete is an async def that RETURNS an async iterator,
            # so callers await it before iterating.
            seen_calls.append((messages, stream))

            async def _gen() -> AsyncIterator[LLMStreamEvent]:
                yield LLMStreamEvent(type="done")

            return _gen()

        wrapped = MagicMock(spec=LLM)
        wrapped.acomplete = _fake_acomplete

        llm = TemporalStreamingLLM(
            wrapped=wrapped,
            activity_config=ModelActivityConfig(),
            model_name="test-model",
        )

        fake_workflow = MagicMock()
        fake_workflow.in_workflow.return_value = False
        original = sys.modules.get("temporalio.workflow")
        sys.modules["temporalio.workflow"] = fake_workflow
        try:
            # acomplete_streamed is now an async generator — no await needed.
            events = [e async for e in llm.acomplete_streamed("hello")]
        finally:
            if original is None:
                del sys.modules["temporalio.workflow"]
            else:
                sys.modules["temporalio.workflow"] = original

        assert len(seen_calls) == 1
        assert seen_calls[0] == ("hello", True)
        assert len(events) == 1
        assert events[0].type == "done"

    async def test_acomplete_streamed_yields_done_inside_workflow(self) -> None:
        """acomplete_streamed yields a single done event when inside a workflow.

        acomplete_streamed is an async generator — callers iterate it directly
        without ``await``.
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from temporalio import workflow as temporal_workflow

        from troopai.adk.llms.llm import LLM
        from troopai.adk.types.responses.llm_response import LLMResponse
        from troopai.adk.workflows.engine import ModelActivityConfig
        from troopai.adk.workflows.temporal.streaming import TemporalStreamingLLM

        fake_response = LLMResponse(response_id="r1", model="gpt-4o")

        wrapped = MagicMock(spec=LLM)
        llm = TemporalStreamingLLM(
            wrapped=wrapped,
            activity_config=ModelActivityConfig(),
            model_name="test-model",
        )

        with (
            patch.object(temporal_workflow, "in_workflow", return_value=True),
            patch.object(llm, "_execute_as_activity", new=AsyncMock(return_value=fake_response)),
        ):
            # acomplete_streamed is now an async generator — no await needed.
            events = [e async for e in llm.acomplete_streamed("hello")]

        assert len(events) == 1
        assert events[0].type == "done"
        assert events[0].response is fake_response
