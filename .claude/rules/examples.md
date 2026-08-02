---
paths:
  - "examples/**/*.py"
---

# Examples Discipline — CRITICAL

Two non-negotiable contracts per example file.

## 1. Load `.env` First

Any example reading env vars MUST `load_dotenv()` at the top BEFORE any
import that captures env at module-load time (litellm, `troopai.adk.llms`).
Imports above `load_dotenv()` read empty values silently.

```python
"""Module docstring."""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio  # rest of imports follow
```

`try/except ImportError` is REQUIRED (`python-dotenv` is the optional
`[examples]` group). Synthetic-only Flow/Graph/Task examples that never
construct an `Agent`/`Swarm` make no provider calls and are exempt.

## 2. Run End-to-End

Any commit adding/modifying `examples/` MUST be preceded by ACTUALLY
RUNNING the affected example end-to-end. Type-checked + unit-tested +
never-run is NEVER verified.

- Server examples: boot + hit documented endpoints + ≥1 real request.
- Client examples: run against a live server, confirm documented output.
- Integration: exercise the full pipeline. CLI: run the documented invocation.

If a run fails for missing API key / credits / rate limits (NOT a code
bug), surface it explicitly to the user and do NOT mark verified. NEVER
silently skip the run.

## Self-Check

1. `load_dotenv()` at module top (or no env vars needed)?
2. Relevant key present; example ran end-to-end with documented output?
3. Failure → code bug fixed, or credit/auth surfaced to user?
