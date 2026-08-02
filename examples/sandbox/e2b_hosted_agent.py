"""Example: SandboxAgent + E2bSandboxClient (hosted bridge).

Requires the [sandbox-e2b] extra and an E2B account.

Prerequisites:
- ``pip install 'troopai-adk-python[sandbox-e2b]'``
- ``E2B_API_KEY`` set in the environment.
- ``ANTHROPIC_API_KEY`` (or your provider's key) set.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
import os

from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.sandbox.agent import SandboxAgent
from troopai.adk.sandbox.capabilities.shell import ShellCapability
from troopai.adk.sandbox.clients.hosted.e2b import (
    E2bSandboxClient,
    E2bSandboxClientOptions,
)
from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


async def main() -> None:
    api_key = os.environ.get("E2B_API_KEY")
    if api_key is None or len(api_key) == 0:
        raise SystemExit("E2B_API_KEY missing — set it in the environment.")
    agent = SandboxAgent(
        name="e2b-shell-demo",
        system_prompt="You have shell access. Run `python --version` and report it.",
        capabilities=[ShellCapability()],
    )
    client = E2bSandboxClient()
    options = E2bSandboxClientOptions(api_key=api_key, template_id="base")
    sandbox = SandboxRunConfig(client=client, options=options)
    result = await Runner.arun(
        agent,
        "What Python version does the sandbox have?",
        run_config=RunConfig(sandbox=sandbox, verbose=VerboseConfig()),
    )
    logger.info("Agent final output:")
    logger.info(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
