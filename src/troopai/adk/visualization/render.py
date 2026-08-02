"""Rendering helpers that turn emitter strings into image files on disk.

Optional convenience layer over :func:`flow_to_mermaid` /
:func:`flow_to_dot` / :func:`graph_to_mermaid` / :func:`graph_to_dot`.
Pull in the matching extras to get image output:

- ``pip install 'troopai-adk-python[viz]'`` — adds the ``graphviz`` Python
  package, which shells out to the local ``dot`` CLI. Used by
  :func:`render_dot` to produce SVG.
- ``pip install 'troopai-adk-python[mermaid]'`` — adds ``mermaid-py``, which
  renders Mermaid strings via the Mermaid Live online API. Used by
  :func:`render_mermaid` to produce PNG.

Both helpers return a typed three-way :data:`RenderOutcome` so callers
can distinguish "produced an image" from "wrote raw source" from
"no-op (extra not installed)". The string emitters themselves never
require either extra — the framework still works fully without them.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

RenderOutcome = Literal["rendered", "raw_fallback", "skipped"]
"""Three-way outcome distinguishing image / raw-source / no-op."""


def render_dot(dot_source: str, out_path: Path) -> RenderOutcome:
    """Render a DOT string to SVG using the optional ``viz`` extra.

    Falls back to saving the raw ``.dot`` source when the Graphviz
    ``dot`` CLI is missing or fails to spawn (``ExecutableNotFound``
    or ``CalledProcessError``).

    Args:
        dot_source: DOT digraph source — typically the return value
            of :meth:`Flow.to_dot` or :meth:`Graph.to_dot`.
        out_path: Target path (without extension). The extension is
            chosen by the outcome (``.svg`` on success, ``.dot`` on
            fallback).

    Returns:
        ``"rendered"`` when an SVG was produced; ``"raw_fallback"``
        when a raw ``.dot`` file was written; ``"skipped"`` when the
        ``graphviz`` package is not installed.
    """
    try:
        from graphviz import ExecutableNotFound, Source
    except ImportError:
        logger.info("Install 'troopai-adk-python[viz]' to render DOT.")
        return "skipped"
    try:
        Source(dot_source).render(filename=str(out_path), format="svg", cleanup=True)
        logger.info("DOT rendered to %s.svg", out_path)
        return "rendered"
    except (ExecutableNotFound, subprocess.CalledProcessError) as exc:
        raw_path = out_path.with_suffix(".dot")
        raw_path.write_text(dot_source, encoding="utf-8")
        logger.info(
            "Graphviz unavailable (%s: %s); saved raw DOT to %s. "
            "Install Graphviz (apt: graphviz / brew: graphviz) to render images.",
            type(exc).__name__,
            str(exc)[:200],
            raw_path,
        )
        return "raw_fallback"


def render_mermaid(mermaid_source: str, out_path: Path) -> RenderOutcome:
    """Render a Mermaid string to PNG using the optional ``mermaid`` extra.

    Uses the Mermaid Live online renderer (network required). Falls
    back to saving the raw ``.mmd`` source on any renderer failure
    other than ``KeyboardInterrupt`` / ``SystemExit`` which are
    re-raised.

    Args:
        mermaid_source: Mermaid flowchart source — typically the
            return value of :meth:`Flow.to_mermaid` or
            :meth:`Graph.to_mermaid`.
        out_path: Target path (without extension).

    Returns:
        ``"rendered"`` when PNG was produced; ``"raw_fallback"`` when
        a raw ``.mmd`` file was written; ``"skipped"`` when the
        ``mermaid-py`` package is not installed.

    Raises:
        KeyboardInterrupt: Re-raised unchanged when the renderer raises it,
            so interactive interrupts are never swallowed.
        SystemExit: Re-raised unchanged when the renderer raises it,
            so process-exit signals are never swallowed.
    """
    try:
        from mermaid import Mermaid
    except ImportError:
        logger.info("Install 'troopai-adk-python[mermaid]' to render Mermaid.")
        return "skipped"
    try:
        Mermaid(mermaid_source).to_png(f"{out_path}.png")
        logger.info("Mermaid rendered to %s.png", out_path)
        return "rendered"
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raw_path = out_path.with_suffix(".mmd")
        raw_path.write_text(mermaid_source, encoding="utf-8")
        logger.info(
            "Mermaid Live renderer unreachable (%s: %s); saved raw Mermaid to %s. "
            "Paste the content at https://mermaid.live to view.",
            type(exc).__name__,
            str(exc)[:200],
            raw_path,
        )
        return "raw_fallback"
