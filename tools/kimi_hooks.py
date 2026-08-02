#!/usr/bin/env python3
"""Kimi Code hook entrypoint for this repository.

Wired from the project-scoped ``.kimi-code/config.toml`` ``[[hooks]]`` table
(active when launching with ``KIMI_CODE_HOME="$PWD/.kimi-code" kimi``):

- ``Stop``         -> ``python3 -m tools.kimi_hooks stop``
- ``SubagentStop`` -> ``python3 -m tools.kimi_hooks subagent-stop``

Kimi passes the event payload as JSON on stdin and runs the command with the
project directory as cwd. Both subcommands are advisory only: they drain the
payload, always exit 0 (Kimi treats other non-zero exits as fail-open allow),
and never emit a blocking decision.

``stop`` reuses the docstring-drift detection from the Claude hook
``.claude/hooks/docs_sync_reminder.py`` (single source of truth, same as the
Codex hook symlinks) but reports in Kimi's accepted form: plain text on
stdout, which Kimi appends to the context at the turn boundary. The Claude
``{"systemMessage": ...}`` JSON schema is not part of Kimi's hook contract.

``subagent-stop`` is a reserved no-op: ``SubagentStop`` is observation-only in
Kimi, so there is nothing to enforce at that point.
"""

from __future__ import annotations

import contextlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

_CLAUDE_HOOK = Path(".claude/hooks/docs_sync_reminder.py")
_MAX_SHOWN = 5


def _load_docs_sync_reminder() -> ModuleType | None:
    """Load the Claude docs-sync hook module; ``None`` when unavailable (fail-open)."""
    if not _CLAUDE_HOOK.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("docs_sync_reminder", _CLAUDE_HOOK)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except (OSError, SyntaxError, ImportError):
        return None
    return module


def _stop() -> None:
    reminder = _load_docs_sync_reminder()
    if reminder is None:
        return
    root = reminder._repo_root()
    if root is None:
        return
    try:
        changed = [path for path in reminder._changed_py(root) if reminder._docstring_changed(root, path)]
    except (subprocess.TimeoutExpired, OSError):
        return  # a hook must never break the session
    if len(changed) == 0:
        return
    shown = ", ".join(changed[:_MAX_SHOWN]) + (" …" if len(changed) > _MAX_SHOWN else "")
    print(
        f"⚠ Docstrings changed in {len(changed)} file(s) under {reminder.SRC} ({shown}). "
        "The published docs and their FR/DE translations are pulled from docstrings "
        "via Sphinx autodoc, so they may be stale — dispatch the docs-author agent "
        "(or run the sphinx-i18n skill) to re-sync. Editing the docstrings themselves "
        "stays with docstring-completer."
    )


def main(argv: list[str]) -> None:
    with contextlib.suppress(OSError, ValueError):
        sys.stdin.read()  # drain the event payload; its content is not needed
    if len(argv) > 1 and argv[1] == "stop":
        _stop()
    # "subagent-stop" (and anything unknown) is a deliberate no-op.


if __name__ == "__main__":
    main(sys.argv)
