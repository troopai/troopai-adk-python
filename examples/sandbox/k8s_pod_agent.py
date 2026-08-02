"""Example: SandboxAgent + K8sPodSandboxClient + ShellCapability.

Spawns an ephemeral pod with restricted PodSecurity in the configured
namespace, runs the agent's shell commands inside it via
kubernetes.stream.

Prerequisites:
- ``pip install 'troopai-adk-python[sandbox-k8s]'``
- A reachable cluster (``kubectl cluster-info`` should succeed); the
  client loads in-cluster config if running inside a pod, else
  ``~/.kube/config``.
- The configured namespace must have the ``restricted`` PodSecurity
  admission profile available.
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

from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.sandbox.agent import SandboxAgent
from troopai.adk.sandbox.capabilities.shell import ShellCapability
from troopai.adk.sandbox.clients.k8s import (
    K8sPodSandboxClient,
    K8sSandboxClientOptions,
)
from troopai.adk.sandbox.config import SandboxRunConfig
from troopai.adk.types.sandbox.resource_limits import SandboxResourceLimits
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


async def main() -> None:
    agent = SandboxAgent(
        name="k8s-shell-demo",
        system_prompt="You have shell access to a Linux pod sandbox. Run `uname -a` then summarize.",
        capabilities=[ShellCapability()],
    )
    client = K8sPodSandboxClient()
    options = K8sSandboxClientOptions(
        image="python:3.12-slim",
        namespace="default",
        resource_limits=SandboxResourceLimits(cpu_cores=0.5, memory_mb=256),
    )
    sandbox = SandboxRunConfig(client=client, options=options)
    result = await Runner.arun(
        agent,
        "What kernel is the sandbox running?",
        run_config=RunConfig(sandbox=sandbox, verbose=VerboseConfig()),
    )
    logger.info("Agent final output:")
    logger.info(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
