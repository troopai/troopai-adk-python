---
name: ruff-format-code
description: Apply ruff formatting (and safe import-sort autofixes) to the Python files you changed. Use after editing code, before a commit or the code-hygiene-gate, or when `ruff format --check` reports drift.
allowed-tools: Bash(ruff *) Bash(git diff *)
---

# Ruff Format Code

Apply formatting to the code you changed — the fix side of the
`code-hygiene-gate`'s `ruff format --check`. Format only your own changes;
the gate verifies the rest of the tree.

## 1. Scope to the files you changed

Use the explicit paths you edited, or derive them:

```bash
git diff --name-only -- '*.py'           # unstaged
git diff --name-only --cached -- '*.py'  # staged
```

Never run a repo-wide `ruff format` — it churns code you did not touch.

## 2. Apply formatting

```bash
ruff format <paths>
```

## 3. Apply safe autofixes

```bash
ruff check --fix <paths>
```

Fixes autofixable lint (e.g. import sorting, `I001`). It only touches rules
ruff marks safe; if it removes something (an unused import), confirm that was
intended.

## 4. Confirm clean

```bash
ruff format --check <paths>
ruff check <paths>
```

Both must pass. This skill does NOT run mypy / pyright / IDE diagnostics —
those belong to the `code-hygiene-gate`, run before declaring work done.
