"""Example: SandboxAgent + DockerSandboxClient + ShellCapability.

Runs a sandboxed agent inside a Docker container so the model can
``ls`` / ``cat`` / ``python`` against the container filesystem.

Prerequisites:
- ``pip install 'troopai-adk-python[sandbox-docker]'``
- A reachable Docker daemon (``docker info`` should succeed).
- ``ANTHROPIC_API_KEY`` (or your provider's key) in the environment.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.agents.agent import Agent
from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.sandbox.agent import SandboxAgent
from troopai.adk.sandbox.capabilities.shell import ShellCapability
from troopai.adk.sandbox.clients.docker import (
    DockerSandboxClient,
    DockerSandboxClientOptions,
)
from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


async def main() -> None:
    agent: Agent = SandboxAgent(
        name="docker-shell-demo",
        system_prompt="You have shell access to a Linux sandbox. Run `ls /` then summarize the output.",
        capabilities=[ShellCapability()],
    )
    client = DockerSandboxClient()
    options = DockerSandboxClientOptions(
        image="python:3.12-slim",
        memory_mb=512,
    )
    sandbox = SandboxRunConfig(client=client, options=options)
    result = await Runner.arun(
        agent,
        "List the root of the sandbox filesystem and summarize what you see.",
        run_config=RunConfig(sandbox=sandbox, verbose=VerboseConfig()),
    )
    logger.info("Agent final output:")
    logger.info(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
