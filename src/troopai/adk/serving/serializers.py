"""JSON serialization of Runner outputs for the HTTP serving surfaces.

Turns the framework's Layer-3 results (:class:`RunResult`,
:class:`RunResultStreaming`) and streaming events into JSON-able dicts.
Conversation items are converted through the framework's existing
Layer-3 → Layer-1 path (:meth:`ItemHelpers.run_items_to_params`); the
Layer-2 provider wire types never appear here, so the REST surface
speaks only the provider-agnostic layers.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from troopai.adk.run.stream import (
    AgentUpdatedStreamEvent,
    HookLifecycleEvent,
    RawResponseStreamEvent,
    RunItemStreamEvent,
)
from troopai.adk.types.items.items import ItemHelpers, RunItemBase
from troopai.adk.types.responses.llm_response import LLMStreamEvent

if TYPE_CHECKING:
    from troopai.adk.run.stream import RunResultStreaming, StreamEvent
    from troopai.adk.types.run.run_result import RunResult
    from troopai.adk.types.tokens.llm_usage import LLMUsage

logger = logging.getLogger(__name__)


def usage_to_dict(usage: LLMUsage) -> dict[str, int]:
    """Project an :class:`LLMUsage` accumulator to its scalar JSON shape.

    Args:
        usage: The cumulative usage tracked on the run context.

    Returns:
        The request/token counters as a flat JSON-able dict.
    """
    return {
        "requests": usage.requests,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "total_tokens": usage.total_tokens,
    }


def _output_to_jsonable(value: Any) -> Any:
    """Coerce a developer-controlled ``final_output`` into JSON.

    ``final_output`` carries whatever the agent's ``output_type`` produced
    (a string, a Pydantic model, a dataclass, or an arbitrary object), so
    this is a genuinely dynamic value rather than a closed union. Known
    structured shapes are encoded faithfully; anything else is stringified
    so the HTTP response stays valid JSON.

    Args:
        value: The run's ``final_output``.

    Returns:
        A JSON-serializable representation of ``value``.
    """
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    return str(value)


def run_result_to_dict(result: RunResult[Any]) -> dict[str, Any]:
    """Serialize a completed (or interrupted) :class:`RunResult` to JSON.

    Args:
        result: The result returned by a non-streaming ``Runner.arun``.

    Returns:
        A JSON-able dict: final output, Layer-1 item params, the HITL
        ``requires_action`` flag, the last agent's name, and usage.
    """
    return {
        "final_output": _output_to_jsonable(result.final_output),
        "new_items": ItemHelpers.run_items_to_params(result.new_items),
        "requires_action": result.requires_action,
        "last_agent": result.last_agent.name if result.last_agent is not None else None,
        "usage": usage_to_dict(result.context.usage),
    }


def streaming_result_to_dict(streaming: RunResultStreaming) -> dict[str, Any]:
    """Serialize the terminal state of a streaming run to JSON.

    Args:
        streaming: The streaming result, after its event iterator drained.

    Returns:
        A JSON-able summary: final output, Layer-1 item params, the
        active agent's name, and usage (``None`` until a context exists).
    """
    usage = usage_to_dict(streaming.context.usage) if streaming.context is not None else None
    return {
        "final_output": _output_to_jsonable(streaming.final_output),
        "new_items": ItemHelpers.run_items_to_params(streaming.new_items),
        "agent": streaming.current_agent.name if streaming.current_agent is not None else None,
        "usage": usage,
    }


def stream_event_to_dict(event: StreamEvent) -> dict[str, Any] | None:
    """Serialize one streaming event for Server-Sent Events.

    Args:
        event: A single event yielded by ``stream_events()``.

    Returns:
        A JSON-able dict for the event, or ``None`` when the event
        carries nothing worth forwarding to an HTTP client (e.g. a raw
        chunk with no text delta).
    """
    if isinstance(event, RawResponseStreamEvent):
        delta = _raw_text_delta(event.data)
        return {"type": "raw_response_event", "delta": delta} if delta is not None else None
    if isinstance(event, RunItemStreamEvent):
        return {"type": "run_item_stream_event", "name": str(event.name), "item": _item_to_param(event.item)}
    if isinstance(event, AgentUpdatedStreamEvent):
        return {"type": "agent_updated_stream_event", "agent": event.new_agent.name}
    # The RawResponseStreamEvent arm above produces the ``| None`` return (a
    # raw chunk with no text delta). StreamEvent's four arms are exhaustive,
    # so the fall-through past this last arm is unreachable.
    if isinstance(event, HookLifecycleEvent):
        return {
            "type": "hook_lifecycle_event",
            "kind": str(event.kind),
            "agent_name": event.agent_name,
            "payload": event.payload,
        }


def _raw_text_delta(data: Any) -> str | None:
    """Pull the incremental text fragment from a raw stream chunk.

    Args:
        data: The opaque payload on a :class:`RawResponseStreamEvent`.

    Returns:
        The text delta when the chunk is a ``part_delta`` carrying one,
        else ``None``.
    """
    if isinstance(data, LLMStreamEvent) and data.delta is not None and len(data.delta) > 0:
        return data.delta
    return None


def _item_to_param(item: Any) -> Any:
    """Convert a streamed run item to its Layer-1 param form.

    Args:
        item: The ``item`` payload on a :class:`RunItemStreamEvent`
            (a Layer-3 :class:`RunItemBase` in practice).

    Returns:
        The Layer-1 param dict, or ``item`` unchanged when it is not a
        run item.
    """
    if isinstance(item, RunItemBase):
        return item.to_param()
    return item
