---
name: add-llm-provider
description: Step-by-step procedure to add a new native LLM provider to the TroopAI ADK (a new src/troopai/adk/llms/<provider>/ package). Use when adding/wiring a provider's native SDK as an LLM implementation.
---

# Add a Native LLM Provider

Constraints live in `.claude/rules/llms.md` and `architecture.md` (they
load when you edit `llms/`). This is the ordered procedure.

## 0. Reference first

Read the closest existing provider end-to-end as the template — pick the
one whose SDK shape is nearest. Existing packages:
`llms/litellm/`, `llms/anthropic/`, `llms/openai/`, `llms/gemini/`.
Also read the upstream SDK's own typed client from site-packages
(`python -c "import <sdk>; print(<sdk>.__file__)"`) — do NOT infer the
API from docs.

## 1. Create `src/troopai/adk/llms/<provider>/`

Mirror the sibling package's file set (see `llms/anthropic/`):

- `<provider>_model.py` — the `class <Provider>LLM(LLM)` impl. `acomplete`
  overloads (stream `True`/`False`). Owns the single SDK boundary call.
- `<provider>_converter.py` — stateless `@classmethod` converter,
  Layer 1 ⇄ the SDK's wire types. `items_to_*`, `convert_tools`,
  `convert_tool_choice`, `response_to_llm_response`, `_parse_usage`.
- `<provider>_config.py` — `class <Provider>Config(LLMConfig)` with typed
  provider-specific fields only (LLMConfig stays provider-agnostic).
- `<provider>_boundary.py` — `headers_as_sdk` / `sanitize_for_log`;
  single source of truth for the auth-header blocklist.
- `<provider>_retry.py` — `<provider>_exception_to_kind` classifier +
  `call_with_retry` shim over the generic `llms/retry.py` loop.
- Optional, only if the SDK has it: `<provider>_reasoning_resolver.py`,
  `<provider>_cache_applicator.py`.
- `__init__.py` — export the public `<Provider>LLM` / `<Provider>Config`.

## 2. Honor the LLM-layer contract

- `acomplete()` is the single entry point. NEVER call the SDK outside
  this package; NEVER `_build_kwargs()` / `**config` spreading — one SDK
  call with explicit named params.
- The boundary helper's return type MUST be the EXACT union the SDK's
  `.pyi` declares (e.g. `Response | AsyncStream[...]`). Narrow at the
  call site (branch on `stream`, or `isinstance` assertion), never in
  the helper.
- Structured output: implement all three tiers (json_schema /
  json_object+validation / prompt+validation); `_parse_response()`
  validates with `output_schema.validate_json()` in every tier.
- Usage: populate `LLMUsage` via `LLMSingleRequestUsage` with cache-aware
  fields; import from `types/tokens/llm_usage.py`, never the shim.

## 3. Wire provider selection

Grep how existing providers are constructed/selected (no central
registry — providers are instantiated directly). Add the new impl the
same way the siblings are reached. Do NOT add provider strings to
`LLMConfig`.

## 4. Complete the implementation

Code + tests (`tests/unit/llms/`, patterned on a sibling) + a
`docs/llms/<provider>.md` + a runnable `examples/llm_providers/<provider>/`
example. No `NotImplementedError` on any method the class exists to
perform. Add the new SDK row to the return-type inventory in
`.claude/rules/llms.md`.

## 5. Verify

Run the `code-hygiene-gate` skill, then run the example end-to-end (surface missing
credentials to the user — never mark verified without a real run).
