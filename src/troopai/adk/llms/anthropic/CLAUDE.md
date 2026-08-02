# Anthropic LLM Module

Native Anthropic implementation using the `anthropic` SDK directly. One
class — `AnthropicLLM` — backed by `client.messages.create()` for both
streaming and non-streaming, with full coverage of extended thinking,
prompt caching, structured output, retry policies, and usage tracking.

## Files

- `anthropic_model.py` — `AnthropicLLM(LLM)`: end-to-end implementation
  of `acomplete()`. `_call_messages` is the SDK boundary (returns the
  exact `Message | AsyncStream[RawMessageStreamEvent]` SDK union, narrowed
  at each call site via `isinstance`). `_stream` rebuilds parts in
  block-index order with a per-block accumulator.
- `anthropic_converter.py` — Stateless `@classmethod` converter:
  Layer 1 ↔ `anthropic.types.*`. Owns `items_to_messages`, `convert_tools`,
  `convert_tool_choice`, `response_to_llm_response`, the synthetic-tool
  helpers (`build_structured_output_tool`, `parse_structured_output`),
  and `_parse_usage`.
- `anthropic_config.py` — `AnthropicConfig(LLMConfig)` with typed
  Anthropic-specific fields: `thinking`, `service_tier`,
  `auto_cache_control`, `cache_control_ttl`.
- `anthropic_boundary.py` — `metadata_as_sdk` / `headers_as_sdk` /
  `sanitize_for_log`. Mirrors `openai/openai_boundary.py`. Single
  source of truth for the auth-header blocklist
  (`x-api-key` / `anthropic-api-key`).
- `anthropic_cache_applicator.py` — `apply_cache_control` injects
  ephemeral `cache_control` markers at the three canonical Anthropic
  caching positions (last system block, last tool definition, last
  user-message text block) when `AnthropicConfig.auto_cache_control`
  is `True`.
- `anthropic_reasoning_resolver.py` — `resolve_thinking` reads
  `AnthropicConfig.thinking` first, falls back to
  `LLMConfig.extra_args["thinking"]`, validates `budget_tokens >= 1024`.
- `anthropic_retry.py` — `anthropic_exception_to_kind` classifier +
  `call_with_retry` shim around `llms/retry.py::call_with_retry`.
  Maps `RateLimitError` (429) and `APIStatusError` 529 (overload) to
  `"rate_limit"`, 408/504 to `"timeout"`, 500/502/503 to
  `"server_error"`.

## Key Architectural Decisions

1. **Single class on the Messages API.** Anthropic exposes one main
   endpoint — `client.messages.create()` (with a `messages.stream()`
   manager that this module no longer uses). One class covers both
   `stream=False` and `stream=True` via the SDK union.
2. **No `**kwargs` spread at the SDK boundary.** All optional params
   are passed as explicit named arguments with `omit` (the new
   sentinel) or `NOT_GIVEN` (for `timeout` only — that field
   predates the Omit migration). User-supplied `extra_args` and
   `extra_body` are merged into a single `extra_body=` payload, which
   the SDK accepts as `object | None`. Spreading would degrade the
   typed return to `Any`.
3. **Structured output via synthetic tool.** `output_schema` triggers
   a single `ToolParam(name="structured_output")` whose `input_schema`
   is the requested JSON schema. `tool_choice` forces the model onto
   it; the resulting `ToolUseBlock.input` is validated through
   `AgentOutputSchemaBase.validate_json`. Same wrapping/unwrapping
   rules as the litellm and OpenAI paths.
4. **Per-block streaming accumulator.** `_BlockAccumulator` is keyed
   on the SDK's `event.index`. Text, tool-input JSON, thinking, and
   signature deltas all land in the same slot — ordering between
   event types no longer matters and the prior `signature_delta`
   IndexError is impossible.
5. **`auto_cache_control` instead of explicit injection points.** The
   litellm path exposes `cache_control_injection_points`; the native
   path inverts the contract — opting in via a single bool covers the
   95% case (cache the long context). Power users who need finer
   control still have `extra_body`.

## Provider-Native Capabilities

Provider-native tools (web search, computer use, text editor, etc.)
are NOT wrapped as framework tool classes in this codebase. To enable
them, pass the raw Anthropic tool JSON through `LLMConfig.extra_body`
/ `extra_args` — the model merges `extra_args` (minus `thinking`) and
`extra_body` into the single `extra_body=` payload the SDK accepts
unchanged.

## Type Strategy

Uses `anthropic.types.*` directly — no custom TypedDicts. These types
NEVER leak outside this module. The converter bridges to/from Layer 1
types (`LLMInputContentItem`, `LLMResponse`).
