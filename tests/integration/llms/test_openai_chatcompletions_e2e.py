"""End-to-end live integration test for ``OpenAIChatCompletionsLLM``.

Skipped unless ``OPENAI_API_KEY`` is present in the environment. The
test executes exactly one real roundtrip against the OpenAI Chat
Completions API and asserts on non-empty output plus non-zero usage.

Run with::

    OPENAI_API_KEY=sk-... pytest tests/integration/llms/test_openai_chatcompletions_e2e.py -v
"""

from __future__ import annotations

import os

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.llms.openai import OpenAIChatCompletionsLLM
from troopai.adk.run.runner import Runner

pytestmark = pytest.mark.skipif(
    os.environ.get("OPENAI_API_KEY") is None,
    reason="Set OPENAI_API_KEY to run this integration test.",
)


@pytest.mark.integration
async def test_openai_chatcompletions_basic_roundtrip() -> None:
    agent = Agent(
        name="Assistant",
        system_prompt="You are concise. Answer in one short sentence.",
        llm=OpenAIChatCompletionsLLM("gpt-4o-mini"),
    )

    result = await Runner.arun(agent, "In one sentence, what colour is the sky on a clear day?")

    assert result.final_output is not None
    assert len(str(result.final_output)) > 0
    assert result.context.usage.total_tokens > 0
