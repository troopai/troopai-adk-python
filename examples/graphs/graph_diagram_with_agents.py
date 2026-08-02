"""Graph topology diagram for an Agent-based review pipeline.

Builds a Graph whose nodes wrap Agents (via :class:`AgentExecutable`),
emits Mermaid + DOT before running, and renders to disk via the
optional ``viz`` / ``mermaid`` extras.

Requires an LLM API key (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``).
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from pathlib import Path

from troopai.adk import Agent, Runner
from troopai.adk.graphs import AgentExecutable
from troopai.adk.graphs.graph import Graph
from troopai.adk.run import RunConfig
from troopai.adk.verbose import VerboseConfig
from troopai.adk.visualization import render_dot, render_mermaid

logger = logging.getLogger(__name__)


def _reviewer_agent() -> Agent:
    """Agent that reviews an article for factual issues."""
    return Agent(
        name="fact_reviewer",
        system_prompt="Review the article for factual accuracy. Be concise.",
    )


def _editor_agent() -> Agent:
    """Agent that polishes the article based on review feedback."""
    return Agent(
        name="editor",
        system_prompt="Polish the article using the reviewer's feedback. Keep it short.",
    )


async def main() -> None:
    """Build an agent-backed Graph, render its diagram, then run it."""
    graph = (
        Graph.new("review-pipeline", description="Fact-check then edit")
        .node("reviewer", AgentExecutable(agent=_reviewer_agent()), description="Fact review")
        .node("editor", AgentExecutable(agent=_editor_agent()), description="Polish article")
        .edge("reviewer", "editor", label="reviewed")
        .entry("reviewer")
        .terminal("editor")
        .compile()
    )
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    render_dot(graph.to_dot(), out_dir / "review_graph")
    render_mermaid(graph.to_mermaid(), out_dir / "review_graph")
    logger.info("Running the graph…")
    result = await Runner.arun_graph(
        graph,
        user_prompt="Article: The Eiffel Tower is in Berlin and was built in 1889.",
        run_config=RunConfig(verbose=VerboseConfig()),
    )
    logger.info("status=%s, output=%r", result.status, str(result.final_output)[:160])


if __name__ == "__main__":
    asyncio.run(main())
