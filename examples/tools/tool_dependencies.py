"""Tool runtime dependencies — declare what a tool needs to run.

Demonstrates ``FunctionTool.requires_env`` and
``FunctionTool.requires_packages``: declare environment-variable and
package requirements as tool metadata. ``Agent.__post_init__`` validates
the declarations at construction and raises ``ToolDependencyError``
listing every unsatisfied requirement.

Two scenarios:

1. **Healthy startup** — every requirement is satisfied; the agent is
   constructed normally.
2. **Missing env var** — the agent construction fails fast with a
   clear, aggregated error message.

Usage:
    python examples/tools/tool_dependencies.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import logging
import os

from troopai.adk.agents import Agent
from troopai.adk.exceptions import ToolDependencyError
from troopai.adk.tools import function_tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# A tool that needs a Slack token and the slack-sdk package
# ---------------------------------------------------------------------------


@function_tool(
    name="slack_notify",
    description="Post a notification to Slack.",
    requires_env=("SLACK_TOKEN",),
    # Use a known-installed package for the example so scenario 1 succeeds
    # without an extra pip install. In real code this would be
    # ``("slack-sdk>=3.0",)``.
    requires_packages=("pydantic>=2",),
)
def slack_notify(message: str) -> str:
    # In production this would import slack_sdk and post the message.
    # We just echo it for the example.
    return f"would post to slack: {message}"


# ---------------------------------------------------------------------------
# Scenario 1 — healthy startup
# ---------------------------------------------------------------------------


def scenario_healthy() -> None:
    logger.info("=== Scenario 1: Healthy startup ===")
    os.environ["SLACK_TOKEN"] = "sk-fake-token-for-example"
    try:
        agent = Agent(
            name="Notifier",
            system_prompt="Send notifications when asked.",
            tools=[slack_notify],
        )
        logger.info("agent constructed: %s (tools: %d)", agent.name, len(agent.tools))
    except ToolDependencyError as e:
        logger.error("unexpected: %s", e)


# ---------------------------------------------------------------------------
# Scenario 2 — missing env var, fail fast
# ---------------------------------------------------------------------------


def scenario_missing_env() -> None:
    logger.info("\n=== Scenario 2: Missing env var ===")
    os.environ.pop("SLACK_TOKEN", None)
    try:
        Agent(
            name="Notifier",
            system_prompt="Send notifications when asked.",
            tools=[slack_notify],
        )
        logger.error("expected ToolDependencyError but agent was constructed")
    except ToolDependencyError as e:
        logger.info("caught ToolDependencyError as expected:")
        logger.info("  agent: %s", e.agent_name)
        logger.info("  missing: %s", dict(e.missing))
        logger.info("  full message:\n%s", e)


def main() -> None:
    scenario_healthy()
    scenario_missing_env()


if __name__ == "__main__":
    main()
