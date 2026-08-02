"""Flow topology diagram for an Agent-based research workflow.

Builds a small but realistic Flow that calls Agents inline from step
bodies, then emits the topology as Mermaid + DOT and renders to disk
via the optional ``viz`` / ``mermaid`` extras.

The example DOES make LLM calls (one per agent invocation) — it
exists to show that ``Flow.to_mermaid()`` / ``Flow.to_dot()`` read
the *static* topology, so they work both before AND after the flow
runs. We render the diagram before running so the topology is
captured even if the run later fails.

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

from pydantic import BaseModel

from troopai.adk import Agent, Flow, Runner, flow_listen, flow_router, flow_start
from troopai.adk.visualization import render_dot, render_mermaid

logger = logging.getLogger(__name__)


class ResearchState(BaseModel):
    """Typed shared state for the research flow."""

    topic: str = ""
    classification: str = ""
    summary: str = ""
    citations: str = ""
    final_report: str = ""


def _classifier_agent() -> Agent:
    """Agent that decides whether the topic is technical or non-technical."""
    return Agent(
        name="classifier",
        system_prompt="Classify the topic into exactly one of 'technical' or 'non_technical'. Reply with only the label.",
    )


def _summary_agent() -> Agent:
    """Agent that writes a one-paragraph summary of the topic."""
    return Agent(
        name="summary_writer",
        system_prompt="Write a short, factual one-paragraph summary of the topic.",
    )


def _citations_agent() -> Agent:
    """Agent that produces a citation list for the topic."""
    return Agent(
        name="citations_writer",
        system_prompt="List 3 high-quality sources that cover the topic. Each on its own line.",
    )


def _editor_agent() -> Agent:
    """Agent that combines summary + citations into a final report."""
    return Agent(
        name="editor",
        system_prompt="Combine the summary and citations into a polished report. Keep it concise.",
    )


class ResearchFlow(Flow[ResearchState]):
    """A research flow: classify, then fan-out summary + citations, then synthesise."""

    @flow_start(description="Receive research topic")
    async def receive(self) -> None:
        self.state.topic = "the impact of LLMs on software engineering"

    @flow_router("receive", description="Classify topic")
    async def classify(self) -> str:
        result = await Runner.arun(_classifier_agent(), self.state.topic)
        label = (result.final_output or "").strip().lower()
        self.state.classification = label
        if "technical" in label and "non" not in label:
            return "technical_branch"
        return "non_technical_branch"

    @flow_listen("technical_branch", description="Summarise (technical)")
    async def technical_summary(self) -> None:
        result = await Runner.arun(_summary_agent(), self.state.topic)
        self.state.summary = result.final_output or ""

    @flow_listen("non_technical_branch", description="Summarise (non-technical)")
    async def non_technical_summary(self) -> None:
        result = await Runner.arun(_summary_agent(), self.state.topic)
        self.state.summary = result.final_output or ""

    @flow_listen(technical_summary | non_technical_summary, description="Gather citations")
    async def gather_citations(self) -> None:
        result = await Runner.arun(_citations_agent(), self.state.topic)
        self.state.citations = result.final_output or ""

    @flow_listen("gather_citations", description="Synthesise final report")
    async def synthesise(self) -> None:
        prompt = f"Topic: {self.state.topic}\n\nSummary:\n{self.state.summary}\n\nCitations:\n{self.state.citations}"
        result = await Runner.arun(_editor_agent(), prompt)
        self.state.final_report = result.final_output or ""


async def main() -> None:
    """Emit + render the diagram, then run the flow."""
    flow = ResearchFlow(ResearchState)
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    # Render BEFORE running — the topology is static.
    render_dot(flow.to_dot(), out_dir / "research_flow")
    render_mermaid(flow.to_mermaid(), out_dir / "research_flow")
    logger.info("Running the flow…")
    result = await Runner.arun_flow(flow)
    logger.info("status=%s, decision=%r", result.status, flow.state.final_report[:120])


if __name__ == "__main__":
    asyncio.run(main())
