"""Print a Graph's topology as Mermaid + DOT, optionally render to disk.

See ``examples/flows/flow_diagram.py`` for the rendering contract —
both examples share the helpers in ``examples/viz_render.py``.

Synthetic-only — no Agent / LLM call.
"""

from __future__ import annotations

import logging
from pathlib import Path

from troopai.adk.graphs.graph import Graph
from troopai.adk.visualization import render_dot, render_mermaid

logger = logging.getLogger(__name__)


def _noop() -> str:
    """Trivial callable node body for the example."""
    return "ok"


def main() -> None:
    """Compile a small graph, print + render Mermaid + DOT."""
    graph = (
        Graph.new("review-pipeline", description="Content review with conditional publish")
        .node("intake", _noop, description="Receive draft")
        .node("fact_check", _noop, description="Fact-check")
        .node("style_check", _noop, description="Style review")
        .node("decide", _noop, description="Combined verdict")
        .node("publish", _noop, description="Publish article")
        .node("reject", _noop, description="Reject article")
        .edge("intake", "fact_check")
        .edge("intake", "style_check")
        .edge("fact_check", "decide")
        .edge("style_check", "decide")
        .edge("decide", "publish", label="approved")
        .edge("decide", "reject", label="rejected", when=lambda _r: True)
        .entry("intake")
        .terminal("publish")
        .terminal("reject")
        .compile()
    )
    mermaid_src = graph.to_mermaid()
    dot_src = graph.to_dot()
    logger.info("--- Mermaid ---\n%s", mermaid_src)
    logger.info("--- DOT ---\n%s", dot_src)
    out_dir = Path(__file__).resolve().parent / "out"
    out_dir.mkdir(exist_ok=True)
    dot_outcome = render_dot(dot_src, out_dir / "graph")
    mermaid_outcome = render_mermaid(mermaid_src, out_dir / "graph")
    logger.info("Render outcomes: dot=%s, mermaid=%s", dot_outcome, mermaid_outcome)


if __name__ == "__main__":
    main()
