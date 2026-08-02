"""Tests for ``run/turn_resolution.py`` decision helpers.

Focus: the deterministic (``HandoffRoute``) structured-output handoff path
must serialize an assistant turn's thinking parts into the wire-format
``thinking_blocks`` shape (``list[dict]`` keyed ``type``/``thinking``/
``signature``) — NOT the raw ``LLMResponseReasoning`` dataclasses. The
converter passthrough and ``ItemHelpers.messages_to_run_items`` both expect
dicts; dataclasses would be dropped from history and break extended-thinking
replay.
"""

from __future__ import annotations

from typing import Any, Literal
from unittest.mock import AsyncMock

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.handoffs.handoff_route import HandoffRoute
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run import turn_resolution
from troopai.adk.run.config import DEFAULT_RUN_CONFIG
from troopai.adk.run.context import RunContext
from troopai.adk.run.next_step import NextStepHandoff
from troopai.adk.run.turn_resolution import resolve_structured_output_step
from troopai.adk.types.intents import Intent
from troopai.adk.types.items.items import ReasoningItem
from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseReasoning, LLMResponseText


class _RouteIntent(Intent):
    kind: Literal["route"] = "route"


def _build_route(target_agent: Agent) -> HandoffRoute[Any, Any]:
    route: HandoffRoute[Any, Any] = HandoffRoute(name="triage-route")
    route.when(_RouteIntent).to(target_agent)
    return route


def _patch_handoff_machinery(monkeypatch: pytest.MonkeyPatch, target_agent: Agent) -> None:
    """Stub the post-triage handoff execution so the test isolates the
    triage-message construction at the top of the deterministic path."""
    from troopai.adk.handoffs.handoff_input_data import HandoffInputData

    handoff_data = HandoffInputData(intent=None, context=(), output=())

    async def _fake_exec(**_kwargs: Any) -> tuple[Agent, HandoffInputData]:
        return target_agent, handoff_data

    async def _passthrough_prepare(*_args: Any, **_kwargs: Any) -> list[Any]:
        return []

    async def _passthrough_budget(messages: list[Any], *_args: Any, **_kwargs: Any) -> list[Any]:
        return messages

    async def _passthrough_inject(_agent: Any, messages: list[Any], _ctx: Any) -> list[Any]:
        return messages

    monkeypatch.setattr(turn_resolution, "execute_deterministic_handoff", AsyncMock(side_effect=_fake_exec))
    monkeypatch.setattr(turn_resolution, "prepare_handoff_input", _passthrough_prepare)
    monkeypatch.setattr(turn_resolution, "apply_handoff_budget", _passthrough_budget)
    monkeypatch.setattr(turn_resolution, "resolve_model_name", lambda *a, **k: "gpt-4o")
    monkeypatch.setattr(turn_resolution, "resolve_compaction_llm", lambda *a, **k: None)
    monkeypatch.setattr(turn_resolution, "_inject_system_prompt_impl", _passthrough_inject)


# JSON the LLM "emits" under ``response_format`` — validates into ``_RouteIntent``
# so the deterministic route resolves to a target.
_ROUTE_JSON = '{"kind": "route"}'


def _make_response(thinking: str, signature: str | None) -> LLMResponse:
    return LLMResponse(
        response_id="r1",
        model="gpt-4o",
        response=[
            LLMResponseReasoning(thinking=thinking, signature=signature),
            LLMResponseText(text=_ROUTE_JSON),
        ],
    )


async def test_deterministic_handoff_serializes_thinking_blocks_as_wire_dicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The triage message's ``thinking_blocks`` must be wire-format dicts,
    not ``LLMResponseReasoning`` dataclasses."""
    target_agent = Agent(name="target", system_prompt="t")
    source_agent = Agent(
        name="triage",
        system_prompt="t",
        output_schema=_RouteIntent,
        handoffs=_build_route(target_agent),
    )
    _patch_handoff_machinery(monkeypatch, target_agent)

    messages: list[Any] = []
    new_items: list[Any] = []
    response = _make_response(thinking="weighing the options", signature="sig-abc")

    result = await resolve_structured_output_step(
        current_agent=source_agent,
        response=response,
        messages=messages,
        new_items=new_items,
        context_end=0,
        context=None,
        ctx_wrapper=RunContext(context=None),
        hooks=RunHooks(),
        config=DEFAULT_RUN_CONFIG,
    )

    assert isinstance(result, NextStepHandoff)
    # The triage message remains in ``messages`` on the success path.
    triage_msg = messages[-1]
    blocks = triage_msg["thinking_blocks"]
    assert isinstance(blocks, list) and len(blocks) == 1
    block = blocks[0]
    # Wire shape — a plain dict, NOT an LLMResponseReasoning dataclass.
    assert isinstance(block, dict)
    assert not isinstance(block, LLMResponseReasoning)
    assert block == {"type": "thinking", "thinking": "weighing the options", "signature": "sig-abc"}


async def test_deterministic_handoff_thinking_blocks_round_trip_to_reasoning_item(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The triage message must round-trip through ``message_to_run_items``
    into a ``ReasoningItem`` — proving the ``isinstance(block, dict)`` gate
    in the converter is satisfied (it silently drops dataclass blocks)."""
    target_agent = Agent(name="target", system_prompt="t")
    source_agent = Agent(
        name="triage",
        system_prompt="t",
        output_schema=_RouteIntent,
        handoffs=_build_route(target_agent),
    )
    _patch_handoff_machinery(monkeypatch, target_agent)

    new_items: list[Any] = []
    response = _make_response(thinking="careful reasoning", signature="sig-xyz")

    await resolve_structured_output_step(
        current_agent=source_agent,
        response=response,
        messages=[],
        new_items=new_items,
        context_end=0,
        context=None,
        ctx_wrapper=RunContext(context=None),
        hooks=RunHooks(),
        config=DEFAULT_RUN_CONFIG,
    )

    reasoning_items = [it for it in new_items if isinstance(it, ReasoningItem)]
    assert len(reasoning_items) == 1
    assert "careful reasoning" in reasoning_items[0].raw.thinking


async def test_deterministic_handoff_omits_thinking_blocks_when_no_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no thinking parts, ``thinking_blocks`` must not be injected at
    all (the previous always-true guard wrote an empty list)."""
    target_agent = Agent(name="target", system_prompt="t")
    source_agent = Agent(
        name="triage",
        system_prompt="t",
        output_schema=_RouteIntent,
        handoffs=_build_route(target_agent),
    )
    _patch_handoff_machinery(monkeypatch, target_agent)

    messages: list[Any] = []
    response = LLMResponse(
        response_id="r1",
        model="gpt-4o",
        response=[LLMResponseText(text=_ROUTE_JSON)],
    )

    await resolve_structured_output_step(
        current_agent=source_agent,
        response=response,
        messages=messages,
        new_items=[],
        context_end=0,
        context=None,
        ctx_wrapper=RunContext(context=None),
        hooks=RunHooks(),
        config=DEFAULT_RUN_CONFIG,
    )

    assert "thinking_blocks" not in messages[-1]
