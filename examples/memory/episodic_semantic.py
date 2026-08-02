"""Distill raw episodic turns into semantic facts (explicit, opt-in).

Run: python -m examples.memory.episodic_semantic
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
from troopai.adk.llms.litellm.litellm_model import LiteLLM
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.memory import (
    MemoryKind,
    MemoryMetadata,
    MemorySearchFilter,
    MemorySource,
    VectorMemory,
    distill_to_semantic,
)
from troopai.adk.memory.extractor import LLMExtractor
from troopai.adk.memory.stores.in_memory import InMemoryVectorStore

logger = logging.getLogger(__name__)


async def main() -> None:
    memory = VectorMemory(
        store=InMemoryVectorStore(),
        embedder=LiteLLMEmbedder(model="text-embedding-3-small"),
    )
    # Episodic: raw turns.
    for turn in ["I just moved to Berlin.", "I'm vegetarian.", "I prefer trains over flights."]:
        await memory.add(
            turn,
            namespace="user:7",
            metadata=MemoryMetadata(source=MemorySource.MANUAL, kind=MemoryKind.EPISODIC),
        )

    # Semantic: distill on demand (explicit — costs tokens).
    episodic = await memory.search("user", namespace="user:7", limit=20)
    facts = await distill_to_semantic(
        [r.entry for r in episodic],
        into=memory,
        extractor=LLMExtractor(llm=LiteLLM(model="gpt-4o-mini"), llm_config=LLMConfig(temperature=0.0)),
        namespace="user:7",
    )
    logger.info("Distilled %d semantic facts", len(facts))
    semantic = await memory.search(
        "dietary and travel preferences",
        namespace="user:7",
        filter=MemorySearchFilter(kind=MemoryKind.SEMANTIC),
    )
    for result in semantic:
        logger.info("FACT %s", result.entry.content)


if __name__ == "__main__":
    asyncio.run(main())
