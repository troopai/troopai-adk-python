"""Basic memory example using TemporaryMemory + MemoryTool.

Demonstrates:
- TemporaryMemory for prototyping
- MemoryTool for agent-facing memory access
- Manual memory population
"""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk import Agent, RunConfig, Runner, VerboseConfig
from troopai.adk.memory import MemoryConfig, TemporaryMemory
from troopai.adk.tools.builtin.memory_tool import ForgetMemoryTool, RecallMemoryTool, RememberMemoryTool

logger = logging.getLogger(__name__)


async def main():
    # Create an in-memory store (keyword-based search)
    memory = TemporaryMemory()

    # Pre-populate with some knowledge
    await memory.add("User prefers dark mode", namespace="user:1")
    await memory.add("User is a Python developer", namespace="user:1")
    await memory.add("User works at Acme Corp", namespace="user:1")

    # Create an agent with MemoryTool for explicit memory access
    agent = Agent(
        name="Assistant",
        system_prompt=(
            "You are a helpful assistant. Use the recall tool to look up "
            "what you know about the user. Use the remember tool to save "
            "new important information."
        ),
        tools=[
            RememberMemoryTool(memory=memory, namespace="user:1"),
            RecallMemoryTool(memory=memory, namespace="user:1"),
            ForgetMemoryTool(memory=memory, namespace="user:1"),
        ],
    )

    # Also configure auto-injection so memories appear in context
    memory_config = MemoryConfig(
        memory=memory,
        namespace="user:1",
        inject=True,
        inject_limit=5,
    )

    # Run the agent — memories are injected + tools available
    result = await Runner.arun(
        agent,
        "What do you know about me?",
        memory=memory_config,
        run_config=RunConfig(verbose=VerboseConfig()),
    )
    logger.info(f"Agent: {result.final_output}")


if __name__ == "__main__":
    asyncio.run(main())
