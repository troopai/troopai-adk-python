"""Shared fixtures for serving-layer tests.

A scripted, network-free :class:`LLM` exposed through ``Agent`` fixtures
so the REST/SSE surfaces can be exercised end-to-end without touching a
provider.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

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
    """Replies with a fixed text on every completion (streamed or not)."""

    def __init__(self, reply: str = "hello from agent") -> None:
        self.reply = reply

    # The stub narrows the overloaded ``acomplete`` to one fixed return
    # union; the checker cannot see the overload chain resolves the same way.
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
                yield LLMStreamEvent(type="part_start", index=0, part=LLMResponseText(text=""))
                yield LLMStreamEvent(type="part_delta", index=0, delta=self.reply)
                yield LLMStreamEvent(type="done", response=response)

            return gen()
        return response


@pytest.fixture
def scripted_agent() -> Agent[None]:
    """A network-free agent that always replies ``"hello from agent"``."""
    return Agent(name="support", system_prompt="Help.", llm=ScriptedLLM())


@pytest.fixture
def streaming_agent() -> Agent[None]:
    """A network-free agent whose reply text is ``"streamed reply"``."""
    return Agent(name="support", system_prompt="Help.", llm=ScriptedLLM("streamed reply"))
