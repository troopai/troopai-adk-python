"""Verbose output — CI-safe auto-mode example.

Demonstrates that ``mode="auto"`` is the responsible default for any
script that might run in a non-interactive environment (CI, piped
stdout, file-redirected output, ``TERM=dumb``).

The resolution ladder in :func:`troopai.adk.verbose.mode.resolve_mode`
silently downgrades ``panel`` to ``line`` in each hostile environment,
so operators get coloured output on their laptop and plain lines in
GitHub Actions — with zero changes to application code.

Run it several ways to see the auto-downgrade at work::

    # Interactive TTY — expect Rich panels
    python examples/verbose/ci_safe.py

    # CI mode — expect plain line output
    CI=1 python examples/verbose/ci_safe.py

    # NO_COLOR — expect plain text, no escape codes
    NO_COLOR=1 python examples/verbose/ci_safe.py

    # Piped — expect line mode (no TTY on stdout)
    python examples/verbose/ci_safe.py | cat
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk import Agent, RunConfig, Runner, VerboseConfig
from troopai.adk.verbose import is_ci, is_no_color, is_rich_available, is_tty, resolve_mode

logger = logging.getLogger(__name__)


async def main() -> None:
    # Resolve and report the effective mode before running. This is
    # what the runner does internally — calling it explicitly here just
    # makes the decision visible for the example.
    cfg = VerboseConfig(mode="auto")
    resolved = resolve_mode(cfg)
    logger.info("Auto-mode decision:")
    logger.info("  TTY attached: %s", is_tty(cfg.resolve_output()))
    logger.info("  CI env set:   %s", is_ci())
    logger.info("  NO_COLOR:     %s", is_no_color())
    logger.info("  Rich avail:   %s", is_rich_available())
    logger.info("  resolved:     %s", resolved)

    agent = Agent(
        name="Assistant",
        llm="gpt-4o-mini",
        system_prompt="Answer concisely.",
    )

    result = await Runner.arun(
        agent,
        "Name three cloud-native CI/CD tools.",
        run_config=RunConfig(verbose=cfg),
    )

    logger.info("Final output: %s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
