"""Tests for the Runner-output → JSON serializers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from troopai.adk.agents.agent import Agent
from troopai.adk.run.stream import (
    AgentUpdatedStreamEvent,
    HookEventKind,
    HookLifecycleEvent,
    RawResponseStreamEvent,
    RunItemStreamEvent,
    RunItemType,
)
from troopai.adk.serving.serializers import (
    run_result_to_dict,
    stream_event_to_dict,
    usage_to_dict,
)
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseText,
    LLMStreamEvent,
)
from troopai.adk.types.run.run_result import RunResult
from troopai.adk.types.tokens.llm_usage import LLMUsage


def test_usage_to_dict_projects_scalar_counters() -> None:
    usage = LLMUsage(requests=2, total_tokens=30, input_tokens=20, output_tokens=10)
    assert usage_to_dict(usage) == {
        "requests": 2,
        "input_tokens": 20,
        "output_tokens": 10,
        "total_tokens": 30,
    }


def test_run_result_to_dict_basic_shape() -> None:
    result: RunResult[Any] = RunResult(final_output="hi there", user_prompt="hi")
    data = run_result_to_dict(result)
    assert data["final_output"] == "hi there"
    assert data["new_items"] == []
    assert data["requires_action"] is False
    assert data["last_agent"] is None
    assert set(data["usage"]) == {"requests", "input_tokens", "output_tokens", "total_tokens"}


class _Weather(BaseModel):
    city: str
    high_c: int


def test_run_result_to_dict_serializes_pydantic_output() -> None:
    result: RunResult[Any] = RunResult(final_output=_Weather(city="Paris", high_c=21), user_prompt="weather?")
    assert run_result_to_dict(result)["final_output"] == {"city": "Paris", "high_c": 21}


def test_stream_event_run_item_passes_dict_item_through() -> None:
    event = RunItemStreamEvent(name=RunItemType.TOOL_OUTPUT, item={"role": "tool", "content": "42"})
    assert stream_event_to_dict(event) == {
        "type": "run_item_stream_event",
        "name": "tool_output",
        "item": {"role": "tool", "content": "42"},
    }


def test_stream_event_agent_updated(scripted_agent: Agent[None]) -> None:
    event = AgentUpdatedStreamEvent(new_agent=scripted_agent)
    assert stream_event_to_dict(event) == {"type": "agent_updated_stream_event", "agent": "support"}


def test_stream_event_raw_delta_extracted() -> None:
    event = RawResponseStreamEvent(data=LLMStreamEvent(type="part_delta", index=0, delta="lo"))
    assert stream_event_to_dict(event) == {"type": "raw_response_event", "delta": "lo"}


def test_stream_event_raw_without_delta_is_dropped() -> None:
    response = LLMResponse(response_id="r", model="m", response=[LLMResponseText(text="x")])
    event = RawResponseStreamEvent(data=LLMStreamEvent(type="done", response=response))
    assert stream_event_to_dict(event) is None


def test_stream_event_hook_lifecycle() -> None:
    event = HookLifecycleEvent(
        kind=HookEventKind.TOOL_START,
        agent_name="support",
        payload={"tool_name": "search"},
    )
    assert stream_event_to_dict(event) == {
        "type": "hook_lifecycle_event",
        "kind": "tool_start",
        "agent_name": "support",
        "payload": {"tool_name": "search"},
    }
