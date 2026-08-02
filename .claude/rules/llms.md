---
paths:
  - "src/troopai/adk/llms/**/*.py"
---

# LLM Layer — CRITICAL

## Single Entry Point

- `LLM.acomplete()` is the SINGLE entry point for all LLM communication.
  `complete()` is a sync wrapper via `asyncio.run()`. Runner NEVER calls
  litellm directly.
- ALL litellm code lives in `llms/litellm/`. Provider-agnostic abstractions
  (`LLM`, `LLMConfig`) live at `llms/` root. Provider-specific code lives in
  `llms/<provider>/` with provider doc links in docstrings.
- LiteLLM impl uses ONE `litellm.acompletion()` call with explicit named
  parameters. NEVER `_build_kwargs()`, intermediate dicts, or `**config`
  spreading. Mapping table lives in `llms/litellm/litellm_model.py`
  (`max_output_tokens → max_tokens`, `stop_sequences → stop`, …).
- `LLMUsage` canonically lives in `types/tokens/llm_usage.py`.
  `llms/llm_usage.py` is a re-export shim — NEVER add new token types to the
  shim. `LLMUsage` tracks per-request breakdown via `usage:
  list[LLMSingleRequestUsage]` (cache-aware: `cached_tokens`,
  `cache_creation_input_tokens`, `reasoning_tokens`), accumulated via `__add__`.

## Structured Output Tiers

- **Tier 1** — `supports_response_schema()` ⇒ json_schema (model constrained).
- **Tier 2** — `response_format` supported ⇒ json_object + validation.
- **Tier 3** — no JSON support ⇒ prompt-based + validation.

`_parse_response()` MUST validate with `output_schema.validate_json()` in
ALL tiers.

## SDK Return Types

Any helper wrapping a provider SDK network call MUST return the EXACT type
the SDK declares — read it from the SDK's `.pyi` in site-packages
(`python -c "import <sdk>; print(<sdk>.__file__)"`). NEVER pre-narrow or
`cast()` a wider value into a narrower one. For `@overload`-on-`stream` SDKs
the fallback returns a UNION; the helper return type MUST be that union.
Narrow at the **call site** (branch on `stream`, or `isinstance` assertion
documenting the runtime invariant), NEVER in the helper.

| Provider | SDK call | Helper return MUST be |
|---|---|---|
| LiteLLM | `litellm.acompletion(...)` | `ModelResponse \| CustomStreamWrapper` |
| Anthropic | `client.messages.create(...)` | `Message \| AsyncMessageStream` |
| OpenAI Responses | `client.responses.create(...)` | `Response \| AsyncStream[ResponseStreamEvent]` |
| OpenAI Chat | `client.chat.completions.create(...)` | `ChatCompletion \| AsyncStream[ChatCompletionChunk]` |

Why: pre-narrowing hides runtime mismatch (next layer `AttributeError`),
loses checker signal, and accumulates stale `cast()`s on SDK bumps.

## Self-Check

1. New caller bypasses `LLM.acomplete()`?
2. New token field on the `llms/llm_usage.py` shim instead of
   `types/tokens/llm_usage.py`?
3. Provider code outside `llms/<provider>/`?
4. Helper return type pre-narrowed vs the SDK `.pyi`?
