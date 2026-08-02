"""Integration test: all three middleware layers active in one run.

Asserts the complete event ordering across the three middleware
chains during a real ``Runner.arun`` call:

    AgentLoggingMiddleware enters
      LLMLoggingMiddleware enters (turn 1)
        — acomplete returns a tool_call —
      LLMLoggingMiddleware exits (turn 1)
      ToolLoggingMiddleware enters
        — tool runs —
      ToolLoggingMiddleware exits
      LLMLoggingMiddleware enters (turn 2)
        — acomplete returns final text —
      LLMLoggingMiddleware exits (turn 2)
    AgentLoggingMiddleware exits

The test uses a stub ``LLM`` subclass that returns a scripted
sequence of ``LLMResponse`` objects, bypassing litellm. The point is
to wire all three layers together and verify they compose into the
expected order.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.agents.middleware import Middleware
from troopai.adk.llms.llm import LLM
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.llms.llm_middleware import LLMLoggingMiddleware
from troopai.adk.run.agent_middleware import AgentLoggingMiddleware
from troopai.adk.run.runner import Runner
from troopai.adk.schemas import AgentOutputSchemaBase
from troopai.adk.tools import (
    Tool,
    ToolLoggingMiddleware,
    function_tool,
)
from troopai.adk.types.input import LLMInputContentItem
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
    LLMResponseText,
    LLMStreamEvent,
)


@function_tool(name="echo", description="echo a string")
def echo(text: str) -> str:
    return f"echoed:{text}"


def _tool_call_response(call_id: str, name: str, args: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        response_id=f"resp-{call_id}",
        model="fake-model",
        response=[
            LLMResponseFunctionToolCall(
                call_id=call_id,
                name=name,
                arguments=json.dumps(args),
            )
        ],
    )


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        response_id="resp-final",
        model="fake-model",
        response=[LLMResponseText(text=text)],
    )


class _ScriptedLLM(LLM):
    """Returns the given LLMResponses in order on each ``acomplete`` call.

    For ``stream=True``, wraps each scripted response in a single
    terminal ``"done"`` event. The streaming consumer in
    ``call_llm_streamed`` extracts the response from the ``done``
    event verbatim — rich part-by-part streaming isn't needed to
    verify middleware integration.
    """

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)
        self._index = 0

    def _next(self) -> LLMResponse:
        if self._index >= len(self._responses):
            return _text_response("script-tail")
        response = self._responses[self._index]
        self._index += 1
        return response

    # Stub narrows the overloaded ``acomplete`` to one fixed return
    # union; mypy cannot see the overload chain resolves the same way.
    async def acomplete(  # type: ignore[override]
        self,
        messages: str | list[LLMInputContentItem],
        llm_config: LLMConfig | None = None,
        tools: list[Tool] | None = None,
        output_schema: AgentOutputSchemaBase | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
        if stream:
            response = self._next()

            async def gen() -> AsyncIterator[LLMStreamEvent]:
                yield LLMStreamEvent(type="done", response=response)

            return gen()
        return self._next()


@pytest.mark.integration
async def test_three_middleware_layers_fire_in_order(
    caplog: pytest.LogCaptureFixture,
) -> None:
    scripted_llm = _ScriptedLLM(
        [
            _tool_call_response("c1", "echo", {"text": "hi"}),
            _text_response("all done"),
        ]
    )

    agent = Agent(
        name="Composer",
        system_prompt="You are a test agent.",
        llm=scripted_llm,
        tools=[echo],
        middleware=Middleware(
            tools=[ToolLoggingMiddleware()],
            agents=[AgentLoggingMiddleware()],
            llms=[LLMLoggingMiddleware()],
        ),
    )

    with caplog.at_level(logging.INFO):
        result = await Runner.arun(agent, "go")

    assert result.final_output == "all done"

    # Filter records by middleware logger origin so we don't pick up
    # incidental loggers (verbose hooks, run loop, etc.).
    interesting = [
        (r.name, r.message)
        for r in caplog.records
        if r.name
        in (
            "troopai.adk.run.agent_middleware",
            "troopai.adk.llms.llm_middleware",
            "troopai.adk.tools.tool_middleware",
        )
    ]

    # Compute the order of high-level events by mapping each record
    # to a stable token. We don't require an exact byte-for-byte match
    # because token counts / model labels would make the test brittle.
    def _classify(name: str, message: str) -> str | None:
        if name == "troopai.adk.run.agent_middleware":
            if "starting" in message:
                return "agent_start"
            if "completed" in message:
                return "agent_end"
        if name == "troopai.adk.llms.llm_middleware":
            if "starting" in message:
                return "llm_start"
            if "completed" in message:
                return "llm_end"
        if name == "troopai.adk.tools.tool_middleware":
            if "starting" in message:
                return "tool_start"
            if "completed" in message:
                return "tool_end"
        return None

    ordered = [tag for tag in (_classify(n, m) for n, m in interesting) if tag is not None]

    assert ordered == [
        "agent_start",
        "llm_start",
        "llm_end",
        "tool_start",
        "tool_end",
        "llm_start",
        "llm_end",
        "agent_end",
    ]


@pytest.mark.integration
async def test_agent_middleware_re_fires_per_handoff(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two-agent handoff: AgentMiddleware fires twice (once per block)."""
    target_agent = Agent(
        name="Beta",
        system_prompt="You are Beta.",
        middleware=Middleware(agents=[AgentLoggingMiddleware()]),
    )
    target_agent.llm = _ScriptedLLM([_text_response("from beta")])

    source_agent = Agent(
        name="Alpha",
        system_prompt="You are Alpha.",
        handoffs=[target_agent],
        middleware=Middleware(agents=[AgentLoggingMiddleware()]),
    )
    # Alpha's LLM emits a handoff tool call to transfer to Beta.
    source_agent.llm = _ScriptedLLM(
        [
            _tool_call_response("c1", "transfer_to_beta", {}),
        ]
    )

    with caplog.at_level(logging.INFO, logger="troopai.adk.run.agent_middleware"):
        result = await Runner.arun(source_agent, "go")

    assert result.final_output == "from beta"

    starting_records = [
        r for r in caplog.records if r.name == "troopai.adk.run.agent_middleware" and "starting" in r.message
    ]
    completed_records = [
        r for r in caplog.records if r.name == "troopai.adk.run.agent_middleware" and "completed" in r.message
    ]
    # AgentMiddleware fires once per per-agent block: once for Alpha,
    # once for Beta after the handoff.
    assert len(starting_records) == 2
    assert len(completed_records) == 2
    assert "Alpha" in starting_records[0].message
    assert "Beta" in starting_records[1].message


@pytest.mark.integration
async def test_agent_middleware_fires_on_streaming_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``Agent.middleware.agents`` runs on ``Runner.arun(stream=True)``.

    Before the streaming-loop refactor, the streaming path bypassed
    the agent-middleware chain. After the refactor, the chain is
    composed inside ``run_agent_loop_streamed`` exactly the same way
    as the non-streaming driver.
    """
    scripted_llm = _ScriptedLLM([_text_response("done-streaming")])

    agent = Agent(
        name="StreamComposer",
        system_prompt="You are a streaming test agent.",
        llm=scripted_llm,
        middleware=Middleware(agents=[AgentLoggingMiddleware()]),
    )

    with caplog.at_level(logging.INFO, logger="troopai.adk.run.agent_middleware"):
        result = Runner.run(agent, "go", stream=True)
        async for _ in result.stream_events():
            pass

    assert result.final_output == "done-streaming"

    starting_records = [
        r for r in caplog.records if r.name == "troopai.adk.run.agent_middleware" and "starting" in r.message
    ]
    completed_records = [
        r for r in caplog.records if r.name == "troopai.adk.run.agent_middleware" and "completed" in r.message
    ]
    assert len(starting_records) == 1, f"expected 1 starting record, got {[r.message for r in starting_records]}"
    assert len(completed_records) == 1, f"expected 1 completed record, got {[r.message for r in completed_records]}"
    assert "StreamComposer" in starting_records[0].message


@pytest.mark.integration
async def test_agent_middleware_re_fires_per_handoff_on_streaming_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two-agent handoff under streaming: middleware fires twice."""
    target_agent = Agent(
        name="StreamBeta",
        system_prompt="You are StreamBeta.",
        middleware=Middleware(agents=[AgentLoggingMiddleware()]),
    )
    target_agent.llm = _ScriptedLLM([_text_response("from stream beta")])

    source_agent = Agent(
        name="StreamAlpha",
        system_prompt="You are StreamAlpha.",
        handoffs=[target_agent],
        middleware=Middleware(agents=[AgentLoggingMiddleware()]),
    )
    source_agent.llm = _ScriptedLLM(
        [
            # Handoff tool names are snake_cased from the target agent name
            # ("StreamBeta" -> transfer_to_stream_beta), matching get_name().
            _tool_call_response("c1", "transfer_to_stream_beta", {}),
        ]
    )

    with caplog.at_level(logging.INFO, logger="troopai.adk.run.agent_middleware"):
        result = Runner.run(source_agent, "go", stream=True)
        async for _ in result.stream_events():
            pass

    assert result.final_output == "from stream beta"

    starting_records = [
        r for r in caplog.records if r.name == "troopai.adk.run.agent_middleware" and "starting" in r.message
    ]
    assert len(starting_records) == 2, f"expected 2 starting records, got {[r.message for r in starting_records]}"
    assert "StreamAlpha" in starting_records[0].message
    assert "StreamBeta" in starting_records[1].message


@pytest.mark.integration
async def test_llm_stream_middleware_fires_on_streaming_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``Agent.middleware.stream_llms`` runs on streaming LLM calls."""
    from troopai.adk.llms.llm_stream_middleware import LLMStreamLoggingMiddleware

    scripted_llm = _ScriptedLLM([_text_response("streamed-with-mw")])

    agent = Agent(
        name="StreamMW",
        system_prompt="You are a streaming-LLM-middleware test agent.",
        llm=scripted_llm,
        middleware=Middleware(stream_llms=[LLMStreamLoggingMiddleware()]),
    )

    with caplog.at_level(logging.INFO, logger="troopai.adk.llms.llm_stream_middleware"):
        result = Runner.run(agent, "go", stream=True)
        async for _ in result.stream_events():
            pass

    assert result.final_output == "streamed-with-mw"

    starting = [r for r in caplog.records if "stream call starting" in r.message]
    completed = [r for r in caplog.records if "stream call completed" in r.message]
    assert len(starting) == 1
    assert len(completed) == 1


@pytest.mark.integration
async def test_non_streaming_llm_middleware_warns_on_streaming_run(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-streaming ``llms`` registered alone on a streaming run → warning."""
    scripted_llm = _ScriptedLLM([_text_response("warn-case")])

    agent = Agent(
        name="WarnAgent",
        system_prompt="Test that the path-mismatch warning fires.",
        llm=scripted_llm,
        middleware=Middleware(llms=[LLMLoggingMiddleware()]),
    )

    with caplog.at_level(logging.WARNING, logger="troopai.adk.run.llm_calls"):
        result = Runner.run(agent, "go", stream=True)
        async for _ in result.stream_events():
            pass

    assert result.final_output == "warn-case"

    mismatch = [r for r in caplog.records if "Non-streaming LLMMiddleware does not apply" in r.message]
    assert len(mismatch) == 1, f"expected 1 mismatch warning, got {[r.message for r in caplog.records]}"
