---
paths:
  - "src/**/*.py"
  - "tests/**/*.py"
  - "examples/**/*.py"
  - "scripts/**/*.py"
---

# Python Conventions — CRITICAL · Loaded When Editing `.py`

Style, idioms, and NASA-Power-of-Ten correctness invariants. Core
architectural invariants are always-on in `architecture.md`.

## Type Annotations

- **PEP 604 unions only**: `T | None`, `X | Y`. NEVER `Optional`/`Union`
  from `typing`. Remove stale imports when you touch the file.
- **No `dict[str, Any]`** when shape is known — TypedDict / dataclass / BaseModel.
- **Explicit checks**: `T | None`, collections, strings need explicit
  comparisons. Exempt: boolean-returning funcs (`isinstance`, `callable`,
  `hasattr`, `.startswith`), explicitly-typed `bool` vars.

| Avoid | Prefer |
|---|---|
| `if x:` for `T \| None` | `if x is not None:` |
| `if not items:` | `if len(items) == 0:` |
| `if s:` for str | `if len(s) > 0:` |

## Idioms

- **Named parameters.** NEVER `**kwargs` spreading for core params. NEVER
  `_build_kwargs()` / `to_litellm_kwargs()` — provider mapping lives in the
  LLM impl.
- `TYPE_CHECKING` for lazy imports (circular-dep avoidance). Tool calls use
  dict access: `tool_call["function"]["name"]`.
- **Single-word attribute names** (`handler`, `enforcement`); multi-word only
  for clarity.
- Dataclass/BaseModel attributes get BOTH class-level Args: AND inline
  docstrings. Provider doc links for provider-specific code. `logger.info` /
  `logger.debug` at key execution points.
- **Remove dead code immediately.** No commented-out blocks, no unused
  funcs/imports/params. No `_`-prefix on unused params — remove them.
- No single-call-site helpers that don't isolate a `cast()` or factor 10+
  lines. No checker-inferable annotations. No `_val`/`_tmp` without a real
  collision. No defensive `getattr(typed_obj, "field", default)`.
- TypedDict union narrowing: direct subscript narrows, `.get()` does not.
- Pydantic v2 union narrowing: `isinstance(item, X)`, NOT `.type == "lit"`.

## No `print()`

NEVER `print()` in any scoped `.py`. ALWAYS `logger =
logging.getLogger(__name__)` per module. Ruff `T201` stays enabled.
`print(msg)`→`logger.info(msg)`; `print(f"Error: {e}")`→`logger.error("Error:
%s", e)`; debug→`logger.debug`. Levels: info (high-level ops), debug
(internals), warning (recoverable), error (result-affecting failures).

## No Cross-Module Underscore / Cosmetic Underscores

- No cross-module import of `_`-prefixed names — if shared, rename to public
  and export in `__init__.py`.
- No `_`-prefixed Python filenames (`_helpers.py`) — privacy is communicated
  by omission from `__all__`. Exception: dunders.
- No cosmetic `from X import Y as _Y` / `_FooAlias`. Underscore MUST mean
  "genuinely private to this file." Exceptions: dunders, test scaffolding
  (`_FakeClient`), in-file helpers used several times, private dataclass
  fields surfaced via a public accessor. If it fits none, drop the underscore.

## Internal Metadata Via Methods

Internal metadata MUST be encapsulated behind methods/properties. External
code NEVER touches `obj._attr` directly. Internal state (`_cache`, `_agent`)
→ init in `__post_init__`, access via `get_cached()`/`get_delegate_agent()`.
Every new `_field` on a dataclass needs a public accessor; frozen-dataclass
writes route through `object.__setattr__(obj, "_field", value)`.

## Function Shape (R1–R6)

- **R1** No unbounded recursion — every recursive site has a framework limit
  (`max_turns`, `max_handoffs`, `max_retries`).
- **R2** Fixed loop bounds — no `while True` without a provably reachable break.
- **R3** No unbounded runtime growth — no history growth without
  compaction/eviction; no unbounded hot-path collections.
- **R4** Functions ≤ 60 lines. Existing violations are tracked debt, not a
  license to add more.
- **R5** Explicit boundary checks, NOT `assert` (stripped under `-O`) —
  raise `ValueError`/`TypeError`. ≥2 explicit guards per non-trivial function.
- **R6** Minimal scope. No module-level mutable state except loggers/true
  constants. Never reuse a variable for different types across branches.

## Async, Indirection, Codegen (R7–R9)

- **R7** Check all async results — every awaited result consumed or
  explicitly discarded with a comment. No fire-and-forget. Guardrail
  verdicts, tool errors, handoff results MUST propagate.
- **R8** No runtime codegen — no `eval`/`exec`/`compile`/dynamic
  `__import__`. Dynamic `getattr(typed, computed)` → direct access or typed
  dispatch table.
- **R9** Restrict deep callable indirection — no >1 anonymous lambda/untyped
  callable through a framework boundary without a typed `Protocol`; no
  `functools.partial` with >1 unfilled positional through a public API; no
  `setattr` with a computed name on typed dataclasses.

## Zero Warnings (R10) + Escape Hatches

ALL mypy + pyright + ruff warnings MUST be zero. Run BOTH mypy and pyright —
they catch different bugs. `cast()` / `# type: ignore` / `# pyright: ignore`
/ `# noinspection` are LAST RESORT.

Hierarchy: **(1) fix at source** (adjust annotation, add branch, narrow via
`isinstance`/`is not None`, typed local, discriminator) → **(2) delete dead
code** → **(3) if a marker survives** it MUST be narrow (single error
code/inspection ID, NEVER bare) with a one-line comment naming the invariant
the runtime upholds that the checker cannot see. mypy is the reference,
pyright second, PyCharm below both. `cast()` is an escape hatch, NOT a
conversion.

Rejected: bare `# type: ignore`; defensive `isinstance(x, dict)` on a
TypedDict-only union; `elif ctype == "<lit>"` for values absent from the
union (unless input is external); fallback `return str(x)` after exhaustive
`isinstance` chain; `cast("str", already_str)`; stale markers.
`--warn-unused-ignores` fails the build on stale ignores; drain marker
batches within the same session.

## Self-Check

1. PEP 604 unions, explicit conditionals, named params, no `print()`?
2. No cross-module `_name` import, no `obj._field` from another module?
3. R1–R10 respected; any surviving marker narrow + justified inline?
