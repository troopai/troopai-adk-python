---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
  - "docs/**/*.md"
  - "examples/**/*.py"
  - "README.md"
---

# No Version Language In Shipped Code — CRITICAL · NON-NEGOTIABLE

Scope: shipped artifacts only — NOT `.claude/`, NOT any `CLAUDE.md`.

Shipped code MUST NEVER carry implementation-version language:

- NO `*_SCHEMA_VERSION` constant / `schema_version` field / `if version !=
  X: raise` mismatch check.
- NO `v1`/`v2`/`Phase N` in docstrings, comments, names, or error messages.
- NO `Scope (v1):` / `Future work:` / `Version history:` sections.
- NO `backward[s]-compat` / `legacy` note justifying a current decision by
  reference to a prior API shape.

Why: these are project-management metadata (plan / commit / CHANGELOG / PR,
never the code surface). In code they decay into lies the moment the next
version ships; a reader of today's build has no frame of reference for "v1".
Describe what the code does *today*.

| NEVER | ALWAYS |
|---|---|
| `FOO_SCHEMA_VERSION` | Delete it. New fields get defaults; old payloads load. |
| `if recorded != FOO_VERSION: raise` | Delete — trust the dataclass + `dict.get(k, default)`. Hard break ⇒ rename the class/loader. |
| `Resolverv2`, `arun_v1_legacy` | Descriptive names: `StableResolver`, `arun_handoffs_only`. |
| `"kept for backward compatibility"` | State the current rationale directly. |

Persisted formats evolve with NO version field: new fields default safely,
loaders are tolerant, hard breaks rename the loader.

## Exceptions (only three)

1. `pyproject.toml` / `setup.cfg` — package version, dep floors,
   `requires-python`.
2. `CHANGELOG.md` / release notes.
3. Test fixtures pinning a wire format (`test_loads_legacy_payload.py`) —
   production code MUST NOT carry the label.

Sanctioned externals (not first-party version language): K8s `apiVersion`,
`anthropic-version`, `pydantic v2`, JSON-Schema `Draft-07`, MCP-spec
deprecations, third-party SDK names.

## Self-Check

Grep added lines for `_SCHEMA_VERSION`, `schema_version`, `\bv[12]\b`,
`Phase [0-9]`, `backward-compat`, `legacy`. Any first-party survivor is a
violation → fix before commit.
