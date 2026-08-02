---
paths:
  - "src/**/*.py"
  - "docs/**/*.md"
  - "examples/**/*.py"
  - "README.md"
  - "CHANGELOG.md"
---

# No Memory-to-Code Leakage — CRITICAL

Memory layer = `.claude/` and any `CLAUDE.md`. Its content MUST NOT leak
into shipped artifacts. Unconditional.

## Four Forbidden Forms

1. The literal path `.claude/` (e.g. `.claude/rules/foo.md`).
2. The literal `CLAUDE.md` (with or without a path).
3. A backtick-wrapped playbook basename (any file under `.claude/rules/`,
   with or without `.md`).
4. The human-readable rule title / bolded principle name cited as a
   governance pointer ("violates the no-hidden-behavior rule", "see X",
   "per X", "preserves X"). Bare descriptive prose ("the pytest-only
   suite") is fine; *governance-pointer* citations are leaks.

Out of scope (cross-references allowed): files under `.claude/`; any file
named `CLAUDE.md`.

Why: memory-layer content is authoring-environment material — not on PyPI,
not vendored. Leaking it produces dead pointers in user copies, doc-surface
pollution, and harness-state bleed into forks.

## Fix

When tempted to cite a playbook/`CLAUDE.md` inside shipped content:
**(1) delete** if surrounding prose is self-explanatory; **(2) inline** the
one-sentence claim in its own words without naming the artifact;
**(3) redirect** to a sibling `docs/<module>/<topic>.md`.

Types modeling Claude Code's on-disk contract describe the scope semantic,
not the filename (e.g. "local-machine scope, not shared with the
repository", not the `.claude/settings.local.json` path).
