"""Shared fixtures for CLI tests.

Provides a scripted, network-free agent exposed as an importable module
in ``tmp_path`` so commands can load it via ``--agent module:var`` with
the working directory importable — the same path real users take.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

STUB_MODULE_NAME = "cli_stub_agents"

STUB_MODULE_SOURCE = textwrap.dedent(
    '''
    """Scripted, network-free agents for CLI tests."""

    from collections.abc import AsyncIterator

    from troopai.adk.agents.agent import Agent
    from troopai.adk.llms.llm import LLM
    from troopai.adk.llms.llm_config import LLMConfig
    from troopai.adk.schemas import AgentOutputSchemaBase
    from troopai.adk.tools import Tool
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.responses.llm_response import (
        LLMResponse,
        LLMResponseText,
        LLMStreamEvent,
    )


    class ScriptedLLM(LLM):
        """Replies with a fixed text on every completion call."""

        def __init__(self, reply: str = "scripted reply") -> None:
            self.reply = reply

        # Stub narrows the overloaded ``acomplete`` to one fixed return
        # union; the checker cannot see the overload chain resolves the
        # same way.
        async def acomplete(  # type: ignore[override]
            self,
            messages: str | list[LLMInputContentItem],
            llm_config: LLMConfig | None = None,
            tools: list[Tool] | None = None,
            output_schema: AgentOutputSchemaBase | None = None,
            stream: bool = False,
        ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
            response = LLMResponse(
                response_id="resp-1",
                model="scripted-model",
                response=[LLMResponseText(text=self.reply)],
            )
            if stream:

                async def gen() -> AsyncIterator[LLMStreamEvent]:
                    yield LLMStreamEvent(type="done", response=response)

                return gen()
            return response


    class DeltaScriptedLLM(ScriptedLLM):
        """Streams the reply as real part deltas before the done event."""

        async def acomplete(  # type: ignore[override]
            self,
            messages: str | list[LLMInputContentItem],
            llm_config: LLMConfig | None = None,
            tools: list[Tool] | None = None,
            output_schema: AgentOutputSchemaBase | None = None,
            stream: bool = False,
        ) -> LLMResponse | AsyncIterator[LLMStreamEvent]:
            response = LLMResponse(
                response_id="resp-1",
                model="scripted-model",
                response=[LLMResponseText(text=self.reply)],
            )
            if not stream:
                return response
            half = len(self.reply) // 2

            async def gen() -> AsyncIterator[LLMStreamEvent]:
                yield LLMStreamEvent(type="part_start", index=0, part=LLMResponseText(text=""))
                yield LLMStreamEvent(type="part_delta", index=0, delta=self.reply[:half])
                yield LLMStreamEvent(type="part_delta", index=0, delta=self.reply[half:])
                yield LLMStreamEvent(type="done", response=response)

            return gen()


    support = Agent(name="support", system_prompt="Help politely.", llm=ScriptedLLM())
    reviewer = Agent(name="reviewer", system_prompt="Review tersely.", llm=ScriptedLLM("looks good"))
    delta_support = Agent(name="delta_support", system_prompt="Help politely.", llm=DeltaScriptedLLM("streamed reply"))

    from troopai.adk.swarms import MaxTurnsTermination, RoundRobinPolicy, Swarm

    team = Swarm(
        members=(support, reviewer),
        entry=support,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(limit=2),
    )

    from troopai.adk.graphs import Graph

    flow = Graph.new("stub_flow").node("support", support).entry("support").terminal("support").compile()
    '''
)


@pytest.fixture
def stub_agent_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Write the scripted-agent module into ``tmp_path`` and chdir there."""
    (tmp_path / f"{STUB_MODULE_NAME}.py").write_text(STUB_MODULE_SOURCE, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path
