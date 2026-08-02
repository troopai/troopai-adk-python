"""Direct usage of the V4A diff engine — no agent loop, no sandbox session.

Demonstrates the ``troopai.adk.sandbox.apply_diff.apply_diff`` pure
function: feed it the current contents of a file and a V4A patch
body, get back the rewritten contents.

Useful when you want to:
* test patch behavior without spinning up a sandbox
* drive a custom editor backend (your own ``PatchFormat``)
* roundtrip diffs produced by an LLM through your own pipeline

No external API key required.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import logging

from troopai.adk.sandbox.apply_diff import apply_diff

logger = logging.getLogger(__name__)


def demo_update() -> None:
    """Apply an update diff to existing content."""
    original = "alpha\nbeta\ngamma\ndelta"
    diff = "@@\n alpha\n-beta\n+BETA\n+inserted\n gamma\n delta"

    updated = apply_diff(original, diff, mode="default")
    logger.info("=== Update diff ===")
    logger.info("Before:\n%s", original)
    logger.info("After:\n%s", updated)
    assert "BETA" in updated
    assert "inserted" in updated
    assert "beta" not in updated


def demo_create() -> None:
    """Apply a create-mode diff to an empty input."""
    diff = "+#!/usr/bin/env python\n+\n+print('hello from V4A')"
    created = apply_diff("", diff, mode="create")
    logger.info("=== Create-file diff ===")
    logger.info("Result:\n%s", created)
    assert created.startswith("#!/usr/bin/env python")


def demo_fuzzy_match() -> None:
    """Three-tier fuzzy matching: exact / rstrip / strip."""
    # Input has trailing whitespace on "alpha  " — exact equality would miss
    # it; the rstrip tier (fuzz score 1) lets the diff still apply.
    original = "alpha  \nbeta\ngamma"
    diff = "@@\n alpha\n-beta\n+BETA\n gamma"
    out = apply_diff(original, diff)
    logger.info("=== Fuzzy (rstrip) match ===")
    logger.info("Output:\n%s", out)
    assert "BETA" in out


def demo_crlf_preserved() -> None:
    """The engine preserves the source file's line-ending style."""
    original = "a\r\nb\r\nc"
    diff = "@@\n a\n-b\n+B\n c"
    out = apply_diff(original, diff)
    logger.info("=== Newline preservation ===")
    logger.info("Output has CRLF: %s", "\\r\\n" in repr(out))
    assert "\r\n" in out


def main() -> None:
    demo_update()
    demo_create()
    demo_fuzzy_match()
    demo_crlf_preserved()
    logger.info("All apply_diff demos passed.")


if __name__ == "__main__":
    main()
