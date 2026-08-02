"""Persistent memory example using SQLiteMemory + auto-extraction.

Demonstrates that memories survive across separate runs by simulating
two independent sessions against the same database file:

  Run 1 — User shares facts, agent stores them via auto-extraction.
  Run 2 — Fresh SQLiteMemory instance opens the same DB and recalls
           everything from Run 1, proving file-backed persistence.
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from pathlib import Path

from troopai.adk import Agent, RunConfig, Runner, VerboseConfig
from troopai.adk.llms.litellm.litellm_model import LiteLLM
from troopai.adk.memory import (
    LLMExtractor,
    MemoryConfig,
    SQLiteMemory,
)
from troopai.adk.tools.builtin.memory_tool import ForgetMemoryTool, RecallMemoryTool, RememberMemoryTool

logger = logging.getLogger(__name__)

DB_PATH = Path("agent_memory.db")


def build_agent(memory: SQLiteMemory, namespace: str) -> Agent:
    return Agent(
        name="Assistant",
        system_prompt=(
            "You are a helpful assistant with long-term memory. "
            "You automatically remember important facts from conversations. "
            "You can also explicitly save or recall memories using your tools."
        ),
        tools=[
            RememberMemoryTool(memory=memory, namespace=namespace),
            RecallMemoryTool(memory=memory, namespace=namespace),
            ForgetMemoryTool(memory=memory, namespace=namespace),
        ],
    )


def build_memory_config(memory: SQLiteMemory, namespace: str) -> MemoryConfig:
    return MemoryConfig(
        memory=memory,
        namespace=namespace,
        inject=True,
        inject_limit=5,
        auto_extract=True,
        extractor=LLMExtractor(llm=LiteLLM(model="gemini/gemini-3-flash-preview")),
    )


async def run_session(
    prompts: list[str],
    memory: SQLiteMemory,
    namespace: str,
    run_config: RunConfig,
) -> None:
    """Run a sequence of prompts against a single memory instance."""
    agent = build_agent(memory, namespace)
    config = build_memory_config(memory, namespace)

    for prompt in prompts:
        logger.info(f"\nUser: {prompt}")
        result = await Runner.arun(agent, prompt, memory=config, run_config=run_config)
        logger.info(f"Agent: {result.final_output or '[Used tools only, no text response]'}")


async def show_memories(memory: SQLiteMemory, namespace: str) -> None:
    """Print all stored memories."""
    results = await memory.search("user", namespace=namespace, limit=10)
    if len(results) > 0:
        logger.info("\n--- Stored Memories ---")
        for r in results:
            logger.info(f"  [{r.entry.metadata.importance}/5] {r.entry.content}")
    else:
        logger.info("\n--- No memories stored ---")


async def main():
    namespace = "user:demo"

    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    # Start with a clean slate so the example is reproducible
    DB_PATH.unlink(missing_ok=True)

    # ── Run 1: Share information ──────────────────────────────
    logger.info("=" * 55)
    logger.info("RUN 1 — Sharing information (stored to agent_memory.db)")
    logger.info("=" * 55)

    memory = SQLiteMemory(path=DB_PATH)
    await run_session(
        [
            "Hi! I'm a Python developer working on a machine learning project.",
            "I prefer using PyTorch over TensorFlow.",
        ],
        memory,
        namespace,
        run_config,
    )
    await show_memories(memory, namespace)
    await memory.close()  # Close DB — simulates process exit

    # ── Run 2: Recall from persistence ────────────────────────
    logger.info("")
    logger.info("=" * 55)
    logger.info("RUN 2 — Fresh instance, same DB file (proving persistence)")
    logger.info("=" * 55)

    memory = SQLiteMemory(path=DB_PATH)  # New instance, same file
    await run_session(
        ["What do you remember about me?"],
        memory,
        namespace,
        run_config,
    )
    await show_memories(memory, namespace)
    await memory.close()

    # Clean up so re-runs start fresh
    DB_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
