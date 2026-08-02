"""Print a Flow's topology as Mermaid + Graphviz DOT, optionally render to disk.

Builds a small Flow with starts, listeners, an AND gate, and a router,
then emits the topology in both Mermaid and DOT formats. Renders to
image files when the optional ``viz`` / ``mermaid`` extras are
installed; falls back to saving raw source on missing CLI / network.

Synthetic-only — no Agent / LLM call.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic import BaseModel

from troopai.adk.flows import Flow, flow_listen, flow_router, flow_start
from troopai.adk.visualization import render_dot, render_mermaid

logger = logging.getLogger(__name__)


class ReviewState(BaseModel):
    """State for a small content-review flow."""

    article: str = ""
    facts_ok: bool = False
    style_ok: bool = False
    decision: str = ""


class ReviewFlow(Flow[ReviewState]):
    """Two parallel reviewers gate a routing decision."""

    @flow_start(description="Receive article")
    async def receive(self) -> None:
        self.state.article = "draft content"

    @flow_listen("receive", description="Fact-check the article")
    async def fact_check(self) -> None:
        self.state.facts_ok = True

    @flow_listen("receive", description="Style-check the article")
    async def style_check(self) -> None:
        self.state.style_ok = True

    @flow_listen(fact_check & style_check, description="Merge reviewer verdicts")
    async def merge(self) -> None:
        self.state.decision = "merged"

    @flow_router("merge", description="Route by combined verdict")
    async def route(self) -> str:
        if self.state.facts_ok and self.state.style_ok:
            return "approve"
        return "reject"

    @flow_listen("approve", description="Publish the article")
    async def publish(self) -> None:
        self.state.decision = "published"

    @flow_listen("reject", description="Reject the article")
    async def reject_step(self) -> None:
        self.state.decision = "rejected"


def main() -> None:
    """Build the flow, print both renderings, and try to render to disk."""
    flow = ReviewFlow(ReviewState)
    mermaid_src = flow.to_mermaid()
    dot_src = flow.to_dot()
    logger.info("--- Mermaid ---\n%s", mermaid_src)
    logger.info("--- DOT ---\n%s", dot_src)
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    dot_outcome = render_dot(dot_src, out_dir / "flow")
    mermaid_outcome = render_mermaid(mermaid_src, out_dir / "flow")
    logger.info("Render outcomes: dot=%s, mermaid=%s", dot_outcome, mermaid_outcome)


if __name__ == "__main__":
    main()
