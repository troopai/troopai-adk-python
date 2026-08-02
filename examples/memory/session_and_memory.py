"""Session + Memory: conversation continuity meets long-term knowledge.

Demonstrates both mechanisms working together:

  Session — gives the agent conversation history within a single chat
            (the agent can reference earlier messages in the same session).
  Memory  — gives the agent extracted knowledge across sessions
            (facts learned in Session 1 are available in Session 2).

Flow:
  Session 1 — Multi-turn conversation. The agent sees full chat history
              via Session, and auto-extraction stores knowledge to Memory.
  Session 2 — New conversation (fresh session, no chat history).
              The agent has no session history, but Memory injects
              knowledge extracted from Session 1.
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
from troopai.adk.session import SQLiteMultiSessions
from troopai.adk.tools.builtin.memory_tool import RecallMemoryTool, RememberMemoryTool

logger = logging.getLogger(__name__)

SESSION_DB = Path("session_demo.db")
MEMORY_DB = Path("memory_demo.db")


def build_agent(memory: SQLiteMemory, namespace: str) -> Agent:
    return Agent(
        name="Assistant",
        system_prompt=(
            "You are a helpful assistant with long-term memory. "
            "You remember important facts from conversations automatically. "
            "You can also use the recall tool to search your memory."
        ),
        tools=[
            RememberMemoryTool(memory=memory, namespace=namespace),
            RecallMemoryTool(memory=memory, namespace=namespace),
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


async def main():
    namespace = "user:alice"

    # Console output comes from the verbose event stream; logger lines
    # land in the rotating .log file configured at import time.
    run_config = RunConfig(verbose=VerboseConfig())

    # Clean slate for reproducibility
    SESSION_DB.unlink(missing_ok=True)
    MEMORY_DB.unlink(missing_ok=True)

    sessions = SQLiteMultiSessions(path=SESSION_DB, app_name="demo")
    memory = SQLiteMemory(path=MEMORY_DB)
    memory_config = build_memory_config(memory, namespace)
    agent = build_agent(memory, namespace)

    # ── Session 1: Multi-turn conversation ────────────────────
    logger.info("=" * 60)
    logger.info("SESSION 1 — Multi-turn chat (session provides history)")
    logger.info("=" * 60)

    session1 = await sessions.create(session_id="chat-1", user_id="alice")

    prompts_session1 = [
        "Hi! I'm Alice, a data scientist at TechCorp.",
        "I'm building a recommendation engine using collaborative filtering.",
        "Can you remind me what I just said I'm working on?",
    ]

    for prompt in prompts_session1:
        logger.info(f"\nUser: {prompt}")
        result = await Runner.arun(
            agent,
            prompt,
            session=session1,
            memory=memory_config,
            run_config=run_config,
        )
        logger.info(f"Agent: {result.final_output or '[Used tools only, no text response]'}")

    # Show what session captured (chronological log)
    events = await session1.get()
    logger.info(f"\n--- Session 1: {len(events)} events stored ---")

    # Show what memory extracted (semantic knowledge)
    results = await memory.search("Alice", namespace=namespace, limit=10)
    logger.info(f"--- Memory: {len(results)} facts extracted ---")
    for r in results:
        logger.info(f"  [{r.entry.metadata.importance}/5] {r.entry.content}")

    # ── Session 2: New conversation, same user ────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("SESSION 2 — New chat (no history, but memory persists)")
    logger.info("=" * 60)

    session2 = await sessions.create(session_id="chat-2", user_id="alice")

    prompts_session2 = [
        "Hey, what do you know about me from our previous conversations?",
        "Based on what you just told me, what kind of project am I working on?",
    ]

    for prompt in prompts_session2:
        logger.info(f"\nUser: {prompt}")
        result = await Runner.arun(
            agent,
            prompt,
            session=session2,
            memory=memory_config,
            run_config=run_config,
        )
        logger.info(f"Agent: {result.final_output or '[Used tools only, no text response]'}")

    # Final state
    events2 = await session2.get()
    logger.info(f"\n--- Session 2: {len(events2)} events (own history only) ---")
    results = await memory.search("Alice", namespace=namespace, limit=10)
    logger.info(f"--- Memory: {len(results)} total facts (accumulated across sessions) ---")
    for r in results:
        logger.info(f"  [{r.entry.metadata.importance}/5] {r.entry.content}")

    # Clean up
    await memory.close()
    await sessions.close()
    SESSION_DB.unlink(missing_ok=True)
    MEMORY_DB.unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(main())
