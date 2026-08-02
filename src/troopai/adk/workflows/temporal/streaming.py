"""TemporalStreamingLLM — streaming LLM bridge for Temporal workflows.

Extends :class:`~troopai.adk.workflows.temporal.llm.TemporalLLM` with a
dedicated streaming method.  Outside a Temporal workflow the call is
forwarded directly to the wrapped LLM with ``stream=True``.  Inside a
workflow, the activity executes non-streaming and the complete
response is surfaced as a single ``"done"`` :class:`LLMStreamEvent`;
native token-level streaming applies only outside a workflow.

Usage::

    from troopai.adk.llms import LiteLLM
    from troopai.adk.workflows.engine import ModelActivityConfig
    from troopai.adk.workflows.temporal.streaming import TemporalStreamingLLM

    llm = TemporalStreamingLLM(
        wrapped=LiteLLM(model="gpt-4o"),
        activity_config=ModelActivityConfig(),
        model_name="gpt-4o",
    )

    async for event in llm.acomplete_streamed(messages="Hello!"):
        if event.type == "done":
            response = event.response

References:
    Temporal Python SDK workflow API:
    https://docs.temporal.io/develop/python/core-application#develop-workflows
    Temporal execute_activity docs:
    https://python.temporal.io/temporalio.workflow.html#execute_activity
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from troopai.adk.workflows.temporal.llm import TemporalLLM

if TYPE_CHECKING:
    from troopai.adk.llms.llm_config import LLMConfig
    from troopai.adk.schemas import AgentOutputSchemaBase
    from troopai.adk.tools import Tool
    from troopai.adk.types.responses.llm_response import LLMStreamEvent

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True, kw_only=True)
class StreamTokenEvent:
    """A single token or completion signal from a streaming LLM session.

    Carries an incremental text fragment (``content``) plus a flag that
    marks the final event of a stream (``is_final``).  Arbitrary
    provider-specific metadata can be attached via ``metadata``.

    Attributes:
        content: Token text fragment.  Empty string for non-text events
            such as the final marker.
        is_final: ``True`` on the last event of the stream.
        metadata: Optional provider-specific extras.  Intentionally
            ``dict[str, Any]`` because the shape is genuinely dynamic
            (provider-controlled, not framework-defined).
    """

    content: str = ""
    """Token text fragment."""

    is_final: bool = False
    """``True`` when this is the last event in the stream."""

    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
    """Optional provider-specific extras.

    Intentionally ``dict[str, Any]`` — shape is provider-controlled and
    genuinely dynamic at runtime.
    """


@dataclasses.dataclass
class TemporalStreamingLLM(TemporalLLM):
    """LLM bridge with streaming support for Temporal workflows.

    Extends :class:`~troopai.adk.workflows.temporal.llm.TemporalLLM` with
    :meth:`acomplete_streamed`, which delegates to the wrapped LLM's native
    streaming path when called outside a workflow and falls back to a
    single-event non-streaming activity call inside a workflow.

    The ``stream_topic_prefix`` field is a namespace prefix for
    WorkflowStream topic names; it is currently unused by the
    non-streaming activity fallback.

    Attributes:
        wrapped: The underlying :class:`~troopai.adk.llms.llm.LLM` that
            handles provider communication.
        activity_config: Timeout and retry policy for each Temporal
            activity execution.
        model_name: Registry key for worker-side model lookup.  When
            empty at construction time, ``__post_init__`` sets this to
            ``str(wrapped)``.
        stream_topic_prefix: Prefix for WorkflowStream topic names.
            Defaults to ``"llm-stream"``.

    References:
        Temporal Python SDK workflow API:
        https://docs.temporal.io/develop/python/core-application#develop-workflows
        Temporal execute_activity docs:
        https://python.temporal.io/temporalio.workflow.html#execute_activity
    """

    stream_topic_prefix: str = "llm-stream"
    """Prefix for WorkflowStream topic names.

    Currently unused — the in-workflow path falls back to a single
    non-streaming activity call.
    """

    async def acomplete_streamed(
        self,
        messages: str | list[Any],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Stream an LLM response, routing through Temporal when inside a workflow.

        Outside a Temporal workflow the call is forwarded directly to the
        wrapped LLM with ``stream=True``.  Inside a workflow the activity
        executes non-streaming and yields a single ``"done"``
        :class:`~troopai.adk.types.responses.llm_response.LLMStreamEvent`
        wrapping the complete response.

        This method is an async generator — callers iterate it directly::

            async for event in llm.acomplete_streamed(messages):
                ...

        Args:
            messages: Conversation input — plain string or list of
                provider-agnostic content items.
            llm_config: Optional LLM parameters (temperature, max tokens, …).
            tools: Optional pre-filtered tool list.
            output_schema: Optional structured output schema.

        Yields:
            :class:`~troopai.adk.types.responses.llm_response.LLMStreamEvent`
            objects.

        References:
            Temporal execute_activity:
            https://python.temporal.io/temporalio.workflow.html#execute_activity
        """
        from troopai.adk.types.responses.llm_response import LLMStreamEvent

        try:
            from temporalio import workflow

            in_workflow = workflow.in_workflow()
        except ImportError:
            in_workflow = False

        if not in_workflow:
            # acomplete is an async def, so the call returns a coroutine that
            # must be awaited to obtain the AsyncIterator before iterating.
            stream_iter = await self.wrapped.acomplete(
                messages,
                llm_config,
                tools,
                output_schema,
                stream=True,
            )
            async for event in stream_iter:
                yield event
            return

        logger.warning(
            "TemporalStreamingLLM: streaming not supported inside Temporal workflows — "
            "falling back to non-streaming activity. Callers receive a single 'done' event. "
            "model=%r",
            self.model_name,
        )
        response = await self._execute_as_activity(messages, llm_config, tools, output_schema)
        yield LLMStreamEvent(type="done", response=response)
