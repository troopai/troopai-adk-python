---
name: run-examples
description: Run the examples suite with examples/run_examples.py (auto mode), triage PASSED/FAILED/SKIPPED/TIMEOUT, and fix the broken examples end-to-end. Use when verifying that examples run, hunting for broken examples, or before a release that ships examples.
---

# Run Examples

Drive `examples/run_examples.py` to find and fix examples that fail when
run. This is the procedure the **examples-auto-runner** agent follows in its
own context for a large sweep (the agent preloads this skill); run the steps
inline for a single example. Run from the repo root with the conda env active.

## 1. Preview (no cost)

Always classify first — it makes zero API calls:

```bash
python examples/run_examples.py --list                 # whole suite
python examples/run_examples.py --list --filter flows  # one topic
```

Read the verdicts: `RUN` vs `SKIP (reason)`, and the per-example prereqs
(which keys / infrastructure each needs). This previews cost before paying.

## 2. Run

```bash
python examples/run_examples.py --auto-mode --filter <topic>   # scoped
python examples/run_examples.py --auto-mode                    # everything
```

`--auto-mode` sets `TROOPAI_EXAMPLES_INTERACTIVE_MODE=auto`, so examples
using the `examples/auto_mode.py` helpers (`is_auto_mode`,
`input_with_fallback`, `confirm_with_fallback`) run unattended. Examples
needing an absent key/server/daemon are skipped with a reason; pass
`--include-server`, `--include-interactive`, or `--force` to override.
Per-example logs are written under `logs/run_examples/<timestamp>/`; a
failure list is written to `logs/run_examples/latest_failures.txt`.

## 3. Triage failures

For each FAILED / TIMEOUT, open its per-example log and read the example
source. Classify the cause:

- **Code bug** (stale import, wrong API, missing `load_dotenv()`, a stray
  `print()`, an unbounded wait) → fix it (step 4).
- **Missing prerequisite** (no API key/credit; no Docker / Temporal /
  Restate / live server / MCP server) → NOT a code bug. Confirm the
  classification is right and surface it; do not fabricate infrastructure.

## 4. Fix and re-confirm

Fix code bugs at the source, minimally and in the surrounding style. Keep
the two example contracts: `load_dotenv()` before any env-capturing import,
and `logger` (never `print()`). Then re-run just that example and confirm
it exits 0 with meaningful output — a clean early exit that did no work is
not a pass:

```bash
python examples/run_examples.py --auto-mode --filter <path/to/example.py>
```

Re-run only the prior failures with:

```bash
python examples/run_examples.py --auto-mode \
  --rerun-file logs/run_examples/latest_failures.txt
```

## 5. Add or change an example

When adding a new example or editing one, you MUST run it end-to-end before
calling it done (type-checked + never-run is not verified). New interactive
examples should use the `auto_mode` helpers so the runner can drive them.

## 6. Verify

Format the files you changed with the `ruff-format-code` skill and confirm
`ruff check` is clean. Before declaring the whole task done, run the full
`code-hygiene-gate` (it adds mypy / pyright / IDE diagnostics). Report a
per-example PASSED/FAILED/SKIPPED summary, the fixes applied with the run
that confirms each, and the prereq-skips left for the maintainer.
