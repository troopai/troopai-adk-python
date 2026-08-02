"""Ground an agent's answers in a document corpus with DocumentSearchTool.

A ``TXTSearchTool`` indexes a small local text corpus (embedded on first use)
and the agent calls it to retrieve relevant passages before answering. Swap
``TXTSearchTool`` for ``PDFSearchTool`` / ``WebsiteSearchTool`` / … to index
other source types; pass ``DocumentSearchTool`` a mixed ``sources`` list to let
it auto-dispatch a loader per source.

Requires an embedding key (``OPENAI_API_KEY`` for ``text-embedding-3-small``)
and an LLM key for the agent.

Run: python examples/rag/document_search.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
import tempfile
from pathlib import Path

from troopai.adk import Agent, RunConfig, Runner
from troopai.adk.llms.litellm.litellm_embedder import LiteLLMEmbedder
from troopai.adk.tools import TXTSearchTool
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)

CORPUS = """\
# Acme Cloud — Service Tiers

The Free tier includes 5 GB of storage and community support only.
The Pro tier includes 1 TB of storage, email support, and a 99.9% uptime SLA.
The Enterprise tier adds SSO, a dedicated success manager, and a 99.99% SLA.

Backups run nightly on Pro and Enterprise; Free-tier data is not backed up.
"""


async def main() -> None:
    with tempfile.TemporaryDirectory() as workdir:
        corpus_path = Path(workdir) / "service_tiers.txt"
        corpus_path.write_text(CORPUS, encoding="utf-8")

        # Sources are bound here, at construction — the LLM only supplies queries.
        search = TXTSearchTool(
            sources=[str(corpus_path)],
            embedder=LiteLLMEmbedder(model="text-embedding-3-small"),
        )

        agent = Agent(
            name="Docs Assistant",
            system_prompt=(
                "Answer questions about Acme Cloud using the txt_search tool. "
                "Ground every claim in retrieved passages and name the tier involved."
            ),
            tools=[search],
        )

        result = await Runner.arun(
            agent,
            "Which tiers back up my data, and what uptime does Enterprise promise?",
            run_config=RunConfig(verbose=VerboseConfig()),
        )
        logger.info("Final answer: %s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
