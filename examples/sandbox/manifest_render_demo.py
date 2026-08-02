"""Render a manifest as an ASCII filesystem tree.

The sandbox runtime feeds this rendering into the agent's system
prompt so the model can grep for paths and descriptions before
making tool calls. This script reproduces the same rendering
outside the agent loop so you can iterate on manifest layouts
locally.

No external API key required.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import logging
from pathlib import Path

from troopai.adk.sandbox.manifest_render import (
    coerce_rel_path,
    render_manifest_description,
)
from troopai.adk.types.sandbox.entries import BaseEntry, Dir, File
from troopai.adk.types.sandbox.permissions import Permissions

logger = logging.getLogger(__name__)


def _coerce(value: str | Path) -> Path:
    return coerce_rel_path(value)


def build_demo_entries() -> dict[str | Path, BaseEntry]:
    """Construct a small but realistic manifest tree."""
    src_children: dict[str, BaseEntry] = {
        "app.py": File(content=b"# entrypoint", description="entry point"),
        "lib.py": File(content=b"# helpers", description="shared helpers"),
    }
    docs_children: dict[str, BaseEntry] = {
        "README.md": File(content=b"# Docs", description="user-facing docs"),
    }
    entries: dict[str | Path, BaseEntry] = {
        "src": Dir(permissions=Permissions(directory=True), children=src_children),
        "docs": Dir(permissions=Permissions(directory=True), children=docs_children),
        "Makefile": File(content=b"all:\n\techo hi", description="task runner"),
    }
    return entries


def main() -> None:
    entries = build_demo_entries()
    logger.info("=== depth=1 (one level deep, default) ===")
    out_d1 = render_manifest_description(
        root="/workspace",
        entries=entries,
        coerce_rel_path=_coerce,
        depth=1,
    )
    logger.info("\n%s", out_d1)

    logger.info("=== depth=2 (children of children) ===")
    out_d2 = render_manifest_description(
        root="/workspace",
        entries=entries,
        coerce_rel_path=_coerce,
        depth=2,
    )
    logger.info("\n%s", out_d2)
    assert "app.py" in out_d2

    logger.info("=== max_chars=200 (truncation) ===")
    out_capped = render_manifest_description(
        root="/workspace",
        entries=entries,
        coerce_rel_path=_coerce,
        depth=None,
        max_chars=200,
    )
    logger.info("\n%s", out_capped)
    assert "truncated" in out_capped or len(out_capped) <= 250


if __name__ == "__main__":
    main()
