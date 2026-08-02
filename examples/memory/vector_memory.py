"""Semantic memory with VectorMemory + the existing recall/remember tools.

Run: python -m examples.memory.vector_memory
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.llms.litellm.litellm_embedder import LiteLLMEmbedder
from troopai.adk.memory import MemoryConfig, VectorMemory
from troopai.adk.memory.stores.in_memory import InMemoryVectorStore
from troopai.adk.tools import RecallMemoryTool, RememberMemoryTool

logger = logging.getLogger(__name__)


async def main() -> None:
    memory = VectorMemory(
        store=InMemoryVectorStore(),
        embedder=LiteLLMEmbedder(model="text-embedding-3-small"),
    )
    await memory.add("The user prefers dark mode and concise answers.", namespace="user:42")
    results = await memory.search("what are the UI preferences?", namespace="user:42")
    for result in results:
        logger.info("score=%.3f  %s", result.score, result.entry.content)

    # The same VectorMemory drives the recall/remember tools and Runner injection:
    tools = [
        RememberMemoryTool(memory=memory, namespace="user:42"),
        RecallMemoryTool(memory=memory, namespace="user:42"),
    ]
    config = MemoryConfig(memory=memory, namespace="user:42", inject=True)
    logger.info("Configured %d memory tools; injection=%s", len(tools), config.inject)


if __name__ == "__main__":
    asyncio.run(main())
