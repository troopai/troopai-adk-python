---
paths:
  - "**/CLAUDE.md"
---

# CLAUDE.md Conventions — CRITICAL

- `CLAUDE.md` files are concise navigation guides + architectural
  decisions — NEVER documentation, tutorials, or example repositories.
- Target **under 200 lines** (shorter = better adherence).
- NEVER code examples — point to `examples/<module>/`.
- NEVER detailed usage patterns — point to `docs/<module>/<topic>.md`.
- NEVER test instructions — point to `tests/`.

Required sections: (1) module purpose (1–2 sentences); (2) file listing
with one-line descriptions; (3) key architectural decisions for that
module; (4) pointers to `docs/` and `examples/`.

| Content | Location |
|---|---|
| Architecture, decisions, conventions | `CLAUDE.md` |
| Code examples, usage patterns | `examples/` |
| API docs, detailed explanations | `docs/<module>/<topic>.md` |
| Test patterns | `tests/` |
| Path-scoped rules | `.claude/rules/**/*.md` |

NEVER create `README.md` in `docs/` — use descriptive filenames.

## Self-Check on Any CLAUDE.md Edit

1. Now exceeds 200 lines? — trim or split.
2. Added code examples / detailed usage? — move to `examples/` / `docs/`.
3. Created `docs/<x>/README.md`? — rename to a descriptive filename.
