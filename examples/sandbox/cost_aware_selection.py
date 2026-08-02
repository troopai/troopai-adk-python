"""Cost-aware sandbox selection + observability demo.

Demonstrates cost-aware sandbox selection with ``CheapestFirstSelector``:
given two candidates — a free local backend and a priced E2B hosted
backend — the selector picks the cheapest one that satisfies the run's
requirements.  The full
``SandboxRunConfig`` is then wired into ``RunConfig`` together with
``LoggingAuditSink`` and tracing so every lifecycle event lands in the
structured log.

The agent loop is guarded behind an API-key check so the example
completes with exit 0 in an offline environment and still surfaces all
config and selection logic.
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

logger = logging.getLogger(__name__)


async def main() -> None:
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.runner import Runner
    from troopai.adk.sandbox.agent import SandboxAgent
    from troopai.adk.sandbox.capabilities.shell import ShellCapability
    from troopai.adk.sandbox.clients.hosted.e2b.e2b_client import E2bSandboxClient
    from troopai.adk.sandbox.clients.local import LocalSubprocessSandboxClient
    from troopai.adk.sandbox.config import SandboxRunConfig
    from troopai.adk.sandbox.observability.audit_sink import LoggingAuditSink
    from troopai.adk.sandbox.selector import CheapestFirstSelector, SandboxCandidate
    from troopai.adk.types.sandbox.cost import SandboxRequirements
    from troopai.adk.verbose import VerboseConfig

    # -----------------------------------------------------------------
    # 1. Cost-aware selection (offline, no LLM, no live session)
    #
    #    Two candidates: E2B (priced at $0.06/min) listed first, then
    #    LocalSubprocess (free). CheapestFirstSelector filters by
    #    capability (both support network=True) and ranks by rate.
    #    The free backend wins regardless of list order.
    # -----------------------------------------------------------------
    local_client = LocalSubprocessSandboxClient()
    e2b_client = E2bSandboxClient()

    pricey_e2b = SandboxCandidate(client=e2b_client)
    free_local = SandboxCandidate(client=local_client)

    requirements = SandboxRequirements(network=True)
    chosen = CheapestFirstSelector().select([pricey_e2b, free_local], requirements)

    logger.info(
        "Selection result: backend=%s  (e2b=$0.06/min, local=free)",
        chosen.client.backend_id,
    )

    # -----------------------------------------------------------------
    # 2. Wire the full run config
    #
    #    selector= + candidates= + requirements= drive backend selection
    #    inside Runner.arun.  audit_sink= routes lifecycle events to the
    #    Python logger.  tracing_enabled=True emits framework spans.
    #    capture_live_cost stays False (the cost-conservative default;
    #    enabling it triggers a provider billing API call after the run).
    # -----------------------------------------------------------------
    sandbox_config = SandboxRunConfig(
        selector=CheapestFirstSelector(),
        candidates=[pricey_e2b, free_local],
        requirements=requirements,
        audit_sink=LoggingAuditSink(),
        # capture_live_cost=False is the default; the developer must
        # opt in to the provider billing API call explicitly.
    )
    run_config = RunConfig(sandbox=sandbox_config, tracing_enabled=True, verbose=VerboseConfig())

    logger.info(
        "RunConfig wired: selector=CheapestFirstSelector, audit_sink=LoggingAuditSink, "
        "tracing_enabled=True, capture_live_cost=False (cost-conservative default)",
    )

    # -----------------------------------------------------------------
    # 3. Build the agent
    # -----------------------------------------------------------------
    agent = SandboxAgent(
        name="coder",
        system_prompt="You can run shell commands in a sandboxed workspace.",
        capabilities=[ShellCapability()],
    )

    # -----------------------------------------------------------------
    # 4. Guarded end-to-end run
    #
    #    Runner.arun is called only when an API key is present so the
    #    example completes offline with exit 0.  On success, the final
    #    output and sandbox_usage (exec count + computed_cost_usd) are
    #    logged.
    # -----------------------------------------------------------------
    api_key_present = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))

    if api_key_present:
        result = await Runner.arun(
            agent,
            "Run `echo hello from sandbox` and report the output.",
            run_config=run_config,
        )
        logger.info("Agent final output: %s", result.final_output)
        if result.sandbox_usage is not None:
            logger.info(
                "Sandbox usage: exec_count=%d  computed_cost_usd=%.6f",
                result.sandbox_usage.exec_count,
                result.sandbox_usage.computed_cost_usd,
            )
    else:
        logger.info(
            "No API key detected. "
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY to run the full agent loop "
            "and surface result.sandbox_usage (exec_count, computed_cost_usd).",
        )


if __name__ == "__main__":
    asyncio.run(main())
