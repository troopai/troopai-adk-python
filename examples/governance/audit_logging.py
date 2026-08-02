"""Append-only tool-call audit logging to a JSONL file.

Run: python examples/governance/audit_logging.py
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
from pathlib import Path

from troopai.adk.agents import Agent
from troopai.adk.audit import JsonlFileAuditSink
from troopai.adk.run import Runner
from troopai.adk.run.config import RunConfig
from troopai.adk.tools import function_tool
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


@function_tool
def lookup(query: str) -> str:
    """Look something up."""
    return f"result for {query}"


async def main() -> None:
    agent = Agent(
        name="search",
        system_prompt="You answer questions by calling the lookup tool.",
        tools=[lookup],
    )
    # Write the audit log to a throwaway directory so the demo leaves no
    # files behind, then read it back to show what was recorded.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "audit.jsonl"
        config = RunConfig(tenant_id="acme", audit_sink=JsonlFileAuditSink(path), verbose=VerboseConfig())
        await Runner.arun(agent, "look up widgets", run_config=config)
        entries = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
        logger.info("audit log: %d tool-call event(s) recorded", len(entries))
        for entry in entries:
            logger.info("  %s", entry)


if __name__ == "__main__":
    asyncio.run(main())
