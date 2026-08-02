# Architecture Invariants — CRITICAL · Always Loaded

The non-negotiable rules. Detailed style/module rules are path-scoped
siblings in this directory and load only when you touch matching files.

## Agent / Runner / LLM

- **Agent = config.** NEVER add `run()` / `arun()` on `Agent`. The Runner executes.
- **`ToolContext` ≠ `RunContext`.** Tools MUST NOT reach execution-wide state.
- Runner MUST go through the `LLM` ABC. NEVER call litellm directly.
- `LLMConfig` MUST be provider-agnostic — NO litellm/openai/anthropic/gemini
  references. ALL provider code lives in `llms/<provider>/`.
- NEVER adopt OpenAI's `AnyLLMModel` / `Model` ABC. Their `Model` is typed
  against `openai.types.*`, costing 3 conversion hops per turn plus dead
  params on every non-OpenAI impl. Our `LLM` ABC is typed against
  framework-owned types (1 conversion per direction, inside each provider).
  Reject any "adopt OpenAI canonical types to simplify" proposal on sight.

## No Implicit Behavior

- The framework MUST NOT auto-inject prompts, system messages, instructions,
  or tokens without explicit developer opt-in. Every framework-added token
  (system text, prompt suffix, tool description, default value) is opt-in.
- Default values that affect token cost MUST be cost-conservative
  (off / smallest / bounded). The developer NEVER opts *out* of a cost they
  did not choose. Applies to `max_turns`, `max_handoffs`, `*_retries`,
  `max_result_tokens`, `*_budget`, `prompt_caching`, compaction flags, retry
  bounds, history windows. **NON-NEGOTIABLE.**

## Type System

- ALL type definitions live in `src/troopai/adk/types/`.
- Three layers: Layer 1 (`LLMInputContentItem`, provider-agnostic) /
  Layer 2 (`ChatCompletion*`, wire) / Layer 3 (`RunItem`, developer-facing).
  Developer-facing APIs use Layer 1 or 3 — NEVER Layer 2. Wire conversion
  stays inside `llms/<provider>/`. Detail: `type-layers.md`.
- Wire-type TypedDicts MUST NEVER be converted to dataclass / BaseModel.
- Matrix: `@dataclass` for framework types; Pydantic `@dataclass` for stream
  types; `BaseModel` for validation-heavy / LLM-output / received types;
  `TypedDict` for LLM-input / sent / `*Param` replay types.

## Shipped-Code Prohibitions

- **No version language** in `src/`, `tests/`, `docs/`, `examples/`,
  `cookbook/`, `evals/`, `README.md`: no `*_SCHEMA_VERSION`, no
  `schema_version`, no `v1`/`v2`/`Phase N`, no version-mismatch `raise`, no
  `backward[s]-compat`/`legacy` justifications. Persisted formats evolve with
  NO version field (new fields default, loaders are tolerant, hard breaks
  rename the loader). Detail + exceptions: `no-version-language.md`.
- **No memory-layer leakage** into shipped artifacts: never the literal
  `.claude/`, `CLAUDE.md`, a playbook basename, or a rule title in shipped
  code/prose/comments. Inline the claim in its own words instead. Detail:
  `no-memory-in-shipped-code.md`.
- **No `print()`** anywhere — always `logging`. Detail: `python-conventions.md`.

## Upstream References

ALWAYS consult the closest upstream reference before designing a subsystem or
chasing a hard bug — cite a fresh read, then adapt; never guess. The
`consult-upstream-references` skill carries the subsystem→source map and how
to fetch + cite (context7 MCP, the installed SDK, WebFetch).

## Terminology

This codebase is an **ADK** (Agent Development Kit), NOT an SDK. Use "ADK" in
prose, commits, docstrings, comments. Reserve "SDK" for third parties
(OpenAI Agents SDK, Anthropic SDK).

## Self-Check

1. New `run()`/`arun()` on `Agent`, or litellm called outside `llms/`? — forbidden.
2. Provider SDK imported outside `llms/<provider>/`? — forbidden.
3. Framework-added token the developer did not opt into? — forbidden.
4. Wire-type TypedDict converted to dataclass/BaseModel? — forbidden.
5. Version language or memory-layer path in a shipped file? — forbidden.
