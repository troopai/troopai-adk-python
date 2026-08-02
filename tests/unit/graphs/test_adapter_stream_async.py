"""AgentExecutable.stream_async parity + default stream_async behavior.

Covers:
- The default ``stream_async`` on a plain callable yields exactly ONE terminal
  ``{"type": "result", "result": NodeResult}`` event.
- ``AgentExecutable.stream_async`` yields >= 1 agent interior events followed
  by a terminal ``{"type": "result", "result": NodeResult}`` whose ``output``
  and ``usage`` are byte-identical to those returned by ``invoke`` on the same
  agent with the same input (parity via shared ``_run_agent_node_result``).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from unittest.mock import AsyncMock, patch

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs.adapters import AgentExecutable, to_executable
from troopai.adk.orchestration.executable import ExecutableInput, NodeResult
from troopai.adk.run.config import DEFAULT_RUN_CONFIG
from troopai.adk.run.context import RunContext
from troopai.adk.types.responses.llm_response import LLMResponse, LLMResponseText

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_input() -> ExecutableInput:
    """Construct a minimal ``ExecutableInput`` with one user message."""
    return ExecutableInput(content=[{"role": "user", "content": "hello"}])  # type: ignore[list-item]


def _text_response(text: str) -> LLMResponse:
    return LLMResponse(
        response_id="resp-test",
        model="fake",
        response=[LLMResponseText(text=text)],
    )


@contextmanager
def _patched_llm(text: str) -> Iterator[None]:
    """Patch both the non-streaming and streaming LLM call sites.

    Non-streaming path: ``call_llm`` in ``troopai.adk.run.loop``.
    Streaming path: ``call_llm_streamed`` in ``troopai.adk.run.loop``.
    Both guardrail patches suppress the network calls in each respective
    runner path.
    """
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "troopai.adk.run.loop.call_llm",
                new=AsyncMock(side_effect=lambda *args, **kwargs: _text_response(text)),
            )
        )
        stack.enter_context(
            patch(
                "troopai.adk.run.loop.call_llm_streamed",
                new=AsyncMock(side_effect=lambda *args, **kwargs: _text_response(text)),
            )
        )
        stack.enter_context(
            patch("troopai.adk.run.runner.run_blocking_input_guardrails", new=AsyncMock(return_value=[]))
        )
        stack.enter_context(
            patch("troopai.adk.run.runner.run_parallel_input_guardrails", new=AsyncMock(return_value=[]))
        )
        stack.enter_context(patch("troopai.adk.run.runner.run_output_guardrails", new=AsyncMock(return_value=[])))
        yield


# ---------------------------------------------------------------------------
# Default stream_async: callable node yields one terminal event
# ---------------------------------------------------------------------------


async def test_callable_default_stream_async_yields_single_terminal() -> None:
    """The default ``stream_async`` on a plain callable yields exactly ONE terminal."""
    ex = to_executable(lambda: "done")
    inp = ExecutableInput(content=[])
    ctx: RunContext[None] = RunContext(context=None)
    items = [it async for it in ex.stream_async(inp, ctx, DEFAULT_RUN_CONFIG)]
    assert len(items) == 1
    assert items[0]["type"] == "result"
    result: NodeResult[None] = items[0]["result"]
    assert result.output == "done"


# ---------------------------------------------------------------------------
# AgentExecutable.stream_async parity
# ---------------------------------------------------------------------------


async def test_agent_stream_async_yields_agent_events_then_terminal() -> None:
    """``AgentExecutable.stream_async`` emits interior events then a terminal result."""
    agent = Agent(name="stream-test", system_prompt="you are helpful")
    ex = AgentExecutable(agent=agent)
    inp = _minimal_input()
    ctx: RunContext[None] = RunContext(context=None)

    with _patched_llm("streamed-answer"):
        items = [it async for it in ex.stream_async(inp, ctx, DEFAULT_RUN_CONFIG)]

    assert len(items) >= 1, "stream_async must yield at least the terminal event"

    # Terminal event must be last
    terminal = items[-1]
    assert terminal["type"] == "result"
    nr_stream: NodeResult[None] = terminal["result"]
    assert nr_stream.output == "streamed-answer"

    # There must be at least one interior agent_event before the terminal
    interior = [it for it in items if it.get("type") == "agent_event"]
    assert len(interior) >= 1, "at least one agent_event must precede the terminal"


async def test_agent_stream_async_invoke_parity() -> None:
    """``stream_async`` and ``invoke`` must produce byte-identical output & usage."""
    agent = Agent(name="parity-test", system_prompt="you are helpful")
    ex = AgentExecutable(agent=agent)
    inp = _minimal_input()

    # --- invoke path ---
    with _patched_llm("parity-answer"):
        ctx_invoke: RunContext[None] = RunContext(context=None)
        nr_invoke: NodeResult[None] = await ex.invoke(inp, ctx_invoke, DEFAULT_RUN_CONFIG)

    # --- stream_async path ---
    with _patched_llm("parity-answer"):
        ctx_stream: RunContext[None] = RunContext(context=None)
        items = [it async for it in ex.stream_async(inp, ctx_stream, DEFAULT_RUN_CONFIG)]

    terminal = items[-1]
    assert terminal["type"] == "result"
    nr_stream: NodeResult[None] = terminal["result"]

    # output parity
    assert nr_stream.output == nr_invoke.output

    # usage parity (both come from the inner RunContext after the agent loop)
    assert nr_stream.usage == nr_invoke.usage

    # final_text parity
    assert nr_stream.final_text == nr_invoke.final_text

    # new_items parity
    assert len(nr_stream.new_items) == len(nr_invoke.new_items)

    # metadata parity
    assert nr_stream.metadata["adapter"] == nr_invoke.metadata["adapter"]
    assert nr_stream.metadata["agent_name"] == nr_invoke.metadata["agent_name"]


async def test_agent_stream_async_max_turns_override() -> None:
    """When ``max_turns`` is set on ``AgentExecutable``, ``stream_async`` honours it."""
    agent = Agent(name="max-turns-test", system_prompt="you are helpful")
    ex = AgentExecutable(agent=agent, max_turns=2)
    inp = _minimal_input()
    ctx: RunContext[None] = RunContext(context=None)

    with _patched_llm("ok"):
        items = [it async for it in ex.stream_async(inp, ctx, DEFAULT_RUN_CONFIG)]

    terminal = items[-1]
    assert terminal["type"] == "result"
    nr: NodeResult[None] = terminal["result"]
    assert nr.output == "ok"


async def test_swarm_keeps_default_terminal_only_stream_async() -> None:
    """``SwarmExecutable`` keeps the default stream_async (one terminal event only)."""
    from troopai.adk.graphs.adapters import SwarmExecutable
    from troopai.adk.swarms.policy import RoundRobinPolicy
    from troopai.adk.swarms.swarm import Swarm
    from troopai.adk.swarms.termination import MaxTurnsTermination

    member = Agent(name="swarmmember", system_prompt="hello")
    swarm = Swarm(
        members=(member,),
        entry=member,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(1),
    )
    ex = SwarmExecutable(swarm=swarm)
    inp = ExecutableInput(content=[])
    ctx: RunContext[None] = RunContext(context=None)

    # stream_async on SwarmExecutable should call invoke internally and yield ONE event
    with _patched_llm("swarm-out"):
        items = [it async for it in ex.stream_async(inp, ctx, DEFAULT_RUN_CONFIG)]

    assert len(items) == 1
    assert items[0]["type"] == "result"
