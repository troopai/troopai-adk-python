"""Wrap a remote A2A agent as a FunctionTool inside a local Agent.

This is the dual-surface option: instead of calling the remote
directly, we let the local agent's LLM invoke it mid-turn alongside
any other tools it has. The remote agent shows up in the LLM's tool
list with a snake-cased name derived from the agent name.

Usage::

    pip install 'troopai-adk-python[a2a]'
    python examples/a2a/client_as_tool.py [REMOTE_URL]
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
import sys

from troopai.adk.a2a import A2AAgent
from troopai.adk.agents import Agent
from troopai.adk.run import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


async def main(remote_url: str) -> None:
    remote = A2AAgent(
        name="Researcher",
        description="Looks up authoritative information from a remote knowledge base.",
        url=remote_url,
    )
    try:
        local = Agent(
            name="Coordinator",
            system_prompt=(
                "You coordinate research tasks. When the user asks a "
                "question that requires fresh authoritative information, "
                "delegate to the researcher tool. Summarise the result."
            ),
            tools=[
                # max_result_tokens caps the remote response that lands
                # back in the parent's context — important when the
                # remote can return long documents.
                remote.as_tool(max_result_tokens=2_000, timeout=60.0),
            ],
        )

        result = await Runner.arun(
            local,
            user_prompt="Find recent papers on retrieval augmentation and summarise.",
            run_config=RunConfig(verbose=VerboseConfig()),
        )
        logger.info("Coordinator output:\n%s", result.final_output)
    finally:
        await remote.close()


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    asyncio.run(main(url))
