"""LLM middleware — deterministic cache that short-circuits the chain.

Demonstrates:

1. ``LLMLoggingMiddleware`` — the shipped baseline that logs each
   call's model and token totals.
2. A custom ``CacheMiddleware`` that stores
   ``LLMResponse`` keyed on the agent name + first user message.
   When a cached response exists, the middleware short-circuits via
   ``LLMMiddlewareTermination`` — the underlying ``LLM.acomplete()``
   is never called.

The example issues two ``Runner.arun`` calls with the same prompt
to the same agent. The first call goes to the stub LLM; the second
hits the cache.

Usage:
    python examples/llms/llm_middleware/llm_caching.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from troopai.adk.agents.agent import Agent
from troopai.adk.agents.middleware import Middleware
from troopai.adk.llms.llm import LLM
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.llms.llm_middleware import (
    LLMLoggingMiddleware,
    LLMMiddlewareNext,
    LLMMiddlewareTermination,
)
from troopai.adk.run.config import RunConfig
from troopai.adk.run.context import RunContext
from troopai.adk.run.runner import Runner
from troopai.adk.schemas import AgentOutputSchemaBase
from troopai.adk.tools import Tool
from troopai.adk.types.input import LLMInputContentItem
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseText,
    LLMStreamEvent,
)
from troopai.adk.types.tokens import LLMUsage
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom middleware: deterministic cache
# ---------------------------------------------------------------------------


class CacheMiddleware:
    """Keys on (agent name, first user message text). Short-circuits on hit."""

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], LLMResponse] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _first_user_text(messages: list[LLMInputContentItem]) -> str:
        for msg in messages:
            role = msg.get("role")
            if role == "user":
                content = msg.get("content")
                if isinstance(content, str):
                    return content
        return ""

    async def __call__(
        self,
        agent: Agent,
        messages: list[LLMInputContentItem],
        llm_config: LLMConfig | None,
        context: RunContext[Any],
        next: LLMMiddlewareNext,
    ) -> LLMResponse:
        key = (agent.name, self._first_user_text(messages))
        cached = self._store.get(key)
        if cached is not None:
            self.hits += 1
            logger.info("cache HIT for agent=%s key=%r", agent.name, key[1])
            # Short-circuit: outer middleware unwinds via the typed
            # termination exception so we get a single, clean exit
            # path rather than a silent return.
            raise LLMMiddlewareTermination(cached)
        self.misses += 1
        logger.info("cache MISS for agent=%s key=%r", agent.name, key[1])
        response = await next(messages, llm_config)
        self._store[key] = response
        return response


# ---------------------------------------------------------------------------
# Stub LLM (no provider API needed)
# ---------------------------------------------------------------------------


class _ScriptedLLM(LLM):
    """Returns the same final-text response every call."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.call_count = 0

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
            raise NotImplementedError("stub does not stream")
        self.call_count += 1
        return LLMResponse(
            response_id=f"r-{self.call_count}",
            model="stub",
            response=[LLMResponseText(text=self._text)],
            usage=LLMUsage(requests=1, input_tokens=4, output_tokens=4, total_tokens=8),
        )


# ---------------------------------------------------------------------------
# Compose and run
# ---------------------------------------------------------------------------


async def main() -> None:
    cache = CacheMiddleware()
    stub = _ScriptedLLM("hello cached world")
    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    agent = Agent(
        name="Cached",
        system_prompt="You are a deterministic agent.",
        llm=stub,
        middleware=Middleware(llms=[LLMLoggingMiddleware(), cache]),
    )

    logger.info("--- first call (expect MISS, stub fires once) ---")
    r1 = await Runner.arun(agent, "ping", run_config=run_config)
    logger.info("final_output: %r", r1.final_output)
    logger.info("stub call_count: %d", stub.call_count)

    logger.info("--- second call, same prompt (expect HIT, stub stays at 1) ---")
    r2 = await Runner.arun(agent, "ping", run_config=run_config)
    logger.info("final_output: %r", r2.final_output)
    logger.info("stub call_count: %d", stub.call_count)

    logger.info("=== cache stats: hits=%d misses=%d ===", cache.hits, cache.misses)


if __name__ == "__main__":
    asyncio.run(main())
