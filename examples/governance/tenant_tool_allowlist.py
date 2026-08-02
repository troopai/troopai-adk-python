"""Per-tenant tool allowlists.

A forbidden tool call fails fast with ToolNotPermittedForTenant (the tool
body never runs). Needs an LLM API key to actually drive the agent.

Run: python -m examples.governance.tenant_tool_allowlist
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.agents import Agent
from troopai.adk.exceptions import ToolNotPermittedForTenant
from troopai.adk.run import Runner
from troopai.adk.run.config import RunConfig
from troopai.adk.tools import function_tool
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


@function_tool
def delete_account(user_id: str) -> str:
    """Delete a user account (privileged)."""
    return f"deleted {user_id}"


async def main() -> None:
    agent = Agent(
        name="ops",
        system_prompt="You are an operations assistant. Use the available tools to fulfil requests.",
        tools=[delete_account],
    )
    # Tenant "free" may call no tools; "admin" may call delete_account.
    config = RunConfig(
        tenant_id="free",
        tenant_tool_allowlist={"free": set(), "admin": {"delete_account"}},
        verbose=VerboseConfig(),
    )
    try:
        await Runner.arun(agent, "Delete account u1", run_config=config)
    except ToolNotPermittedForTenant as exc:
        logger.info("blocked as expected: %s", exc)


if __name__ == "__main__":
    asyncio.run(main())
