#!/usr/bin/env python3
"""Stop hook: remind to sync the docs when src docstrings change.

Advisory only. Exits 0 and (when relevant) surfaces a ``systemMessage``; it
never returns a ``block`` decision, so it cannot interrupt, delay, or override
any in-flight work — including a docstring-completer dispatch. It is
registered on the main agent's ``Stop`` event (not ``SubagentStop``), so it
lands at a turn boundary, never mid-subagent.

What it does: at turn end, detect whether any *docstring* under
``src/troopai/adk`` changed versus ``HEAD`` (including new, untracked modules),
using an AST comparison so pure-logic edits don't trigger it. If a docstring
changed, remind the controller to dispatch the docs-author agent — because the
published docs and their FR/DE translations are pulled from docstrings via
Sphinx autodoc and may now be stale. Editing the docstrings themselves remains
docstring-completer's job; this only flags the downstream docs sync.
"""

from __future__ import annotations

import ast
import contextlib
import json
import os
import subprocess
import sys

SRC = "src/troopai/adk"
_MAX_SHOWN = 5


def _git(args: list[str], cwd: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)


def _docstrings(source: str) -> list[str] | None:
    """Return the sorted docstrings in ``source``, or ``None`` if unparseable."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    docs: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docs.append(doc)
    return sorted(docs)


def _changed_py(root: str) -> list[str]:
    """Tracked-modified + new-untracked ``.py`` paths under SRC (vs HEAD)."""
    tracked = _git(["diff", "--name-only", "HEAD", "--", SRC], root)
    untracked = _git(["ls-files", "--others", "--exclude-standard", "--", SRC], root)
    files: set[str] = set()
    for result in (tracked, untracked):
        for line in result.stdout.splitlines():
            path = line.strip()
            if path.endswith(".py"):
                files.add(path)
    return sorted(files)


def _docstring_changed(root: str, relpath: str) -> bool:
    """Whether ``relpath``'s docstrings differ from HEAD (True for new files with docs)."""
    try:
        with open(os.path.join(root, relpath), encoding="utf-8") as handle:
            new = _docstrings(handle.read())
    except OSError:
        return False
    if new is None:  # mid-edit syntax error — do not nag on a broken file
        return False
    head = _git(["show", f"HEAD:{relpath}"], root)
    if head.returncode != 0:  # new file, absent from HEAD
        return len(new) > 0
    old = _docstrings(head.stdout)
    if old is None:
        return False
    return old != new


def _repo_root() -> str | None:
    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    rev = _git(["rev-parse", "--show-toplevel"], root)
    if rev.returncode == 0:
        return rev.stdout.strip()
    return None


def main() -> None:
    with contextlib.suppress(OSError, ValueError):
        sys.stdin.read()  # drain the Stop payload; its content is not needed
    root = _repo_root()
    if root is None:
        return  # not a git repo — nothing to do
    try:
        changed = [path for path in _changed_py(root) if _docstring_changed(root, path)]
    except (subprocess.TimeoutExpired, OSError):
        return  # a hook must never break the session
    if len(changed) == 0:
        return
    shown = ", ".join(changed[:_MAX_SHOWN]) + (" …" if len(changed) > _MAX_SHOWN else "")
    message = (
        f"⚠ Docstrings changed in {len(changed)} file(s) under {SRC} ({shown}). "
        "The published docs and their FR/DE translations are pulled from docstrings "
        "via Sphinx autodoc, so they may be stale — dispatch the docs-author agent "
        "(or run the sphinx-i18n skill) to re-sync. Editing the docstrings themselves "
        "stays with docstring-completer."
    )
    print(json.dumps({"systemMessage": message}))


if __name__ == "__main__":
    main()
