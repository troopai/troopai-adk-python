"""Example: SandboxAgent + MemoryCapability across two runs.

First run: agent solves a task. The session-stop hook persists raw
memory + a consolidated summary into ``memories/MEMORY.md`` inside
the workspace.

Second run: a verifier agent reads MEMORY.md (via ShellCapability)
and confirms the prior context is available.

Uses the LocalSubprocess backend so this example is self-contained
(no Docker daemon required).

Prerequisites:
- ``pip install 'troopai-adk-python[examples]'`` (for python-dotenv)
- ``ANTHROPIC_API_KEY`` set.
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

from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.sandbox.agent import SandboxAgent
from troopai.adk.sandbox.capabilities.filesystem import FilesystemCapability
from troopai.adk.sandbox.capabilities.memory import MemoryCapability
from troopai.adk.sandbox.capabilities.shell import ShellCapability
from troopai.adk.sandbox.clients.local.subprocess_client import (
    LocalSandboxClientOptions,
    LocalSubprocessSandboxClient,
)
from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


async def main() -> None:
    # A single shared workspace directory persists memories/MEMORY.md across
    # the two runs. (The local subprocess backend has no snapshot store; the
    # shared working directory is what carries memory between sessions.)
    with tempfile.TemporaryDirectory() as workspace_dir:
        client = LocalSubprocessSandboxClient()
        options = LocalSandboxClientOptions(working_directory=workspace_dir)

        author = SandboxAgent(
            name="memory-author",
            system_prompt=(
                "You have shell access and persistent memory. "
                "Write a one-line note to memories/MEMORY.md saying "
                "'Bug fix landed: NULL handling in run.py.'"
            ),
            capabilities=[FilesystemCapability(), ShellCapability(), MemoryCapability()],
        )
        # Console output comes from the verbose event stream; logger lines
        # land in the rotating .log file configured at import time.
        sandbox = SandboxRunConfig(client=client, options=options)
        run_config = RunConfig(sandbox=sandbox, verbose=VerboseConfig())
        await Runner.arun(
            author,
            "Record the bug fix in memory.",
            run_config=run_config,
        )

        verifier = SandboxAgent(
            name="memory-verifier",
            system_prompt=(
                "You have shell + persistent memory. Read memories/MEMORY.md and report what prior runs recorded."
            ),
            capabilities=[FilesystemCapability(), ShellCapability(), MemoryCapability()],
        )
        result = await Runner.arun(
            verifier,
            "What does prior memory say?",
            run_config=run_config,
        )
        logger.info("Verifier output:")
        logger.info(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
