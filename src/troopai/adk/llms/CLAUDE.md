# LLMs Module

Provider-agnostic LLM abstraction.

## Architecture

```
llms/
├── llm.py            # LLM ABC (provider-agnostic)
├── llm_config.py     # LLMConfig @dataclass (provider-agnostic)
├── llm_usage.py      # Re-export shim → types/tokens/llm_usage.py
├── litellm/          # LiteLLM (100+ providers)
│   ├── litellm_model.py        # LiteLLM class
│   ├── litellm_converter.py    # Layer 1 ↔ litellm wire types
│   └── litellm_provider.py     # Provider detection
├── anthropic/        # Anthropic native (anthropic SDK)
│   ├── anthropic_model.py
│   └── anthropic_converter.py
└── openai/           # OpenAI native (Responses + Chat Completions)
    ├── openai_responses_model.py
    └── openai_chatcompletions_model.py
```

Each provider uses its SDK types directly — no shared wire-format types.

## Design Principles

Provider agnosticism, the `LLM` ABC boundary, and structured-output
stance live in `.claude/rules/architecture.md` and `.claude/rules/llms.md`. Module-specific:

- **Runner = orchestration, LLM = communication.** Runner: tool
  enablement, handoff conversion, system prompt, history. LLM:
  parameter mapping, structured output, parsing, streaming.
- **Explicit parameter mapping** — `LiteLLM` maps `LLMConfig` fields
  via explicit named arguments; no `**kwargs` sinks.
- **Provider-specific code is isolated** — litellm in `llms/litellm/`,
  anthropic in `llms/anthropic/`, openai in `llms/openai/`.
  Provider-agnostic types at `llms/` root.

See `docs/llms/`, `examples/llm_providers/`.

## LLM ABC (`llm.py`)

`acomplete()` (async) + `complete()` (sync wrapper). Returns
`LLMResponse` (non-streaming) or `AsyncIterator[LLMStreamEvent]`
(streaming).

## LiteLLM (`litellm/litellm_model.py`)

Default impl using litellm for 100+ providers.

| Method | Purpose |
|---|---|
| `acomplete()` | Single `litellm.acompletion()` with explicit named params; `model` set in `__init__` |
| `_resolve_response_format()` | Three-tier capability check: json_schema → json_object → None |
| `_build_response_format()` | Builds the json_schema response_format dict |
| `_parse_response()` | `ModelResponse` → `LLMResponse` (validates output in all tiers) |
| `_parse_usage()` | Extracts usage (handles OpenAI + Anthropic styles) |
| `_stream()` / `_build_stream_response()` | Streaming chunk processing + reconstruction |

### Structured Output (Three Tiers)

When `output_schema` is provided, `_resolve_response_format()` checks
capabilities:

1. **JSON Schema** — model natively schema-constrained (best reliability)
2. **JSON Object** — model emits valid JSON, client-side validation
3. **No JSON support** — prompt-based, client-side validation

`_parse_response()` validates output against schema in all tiers.

### Parameter Mapping

| `LLMConfig` field | litellm parameter | Notes |
|---|---|---|
| `max_output_tokens` | `max_tokens` | Explicit named arg |
| `stop_sequences` | `stop` | |
| `response_logprobs` | `logprobs` (bool) | |
| `top_logprobs` | `top_logprobs` (int) | |
| `temperature` / `top_p` / `top_k` | Same name | Pass-through |
| `num_retries` | `num_retries` | API-level (429, 500, timeout) |
| `fallbacks` | `fallbacks` | Alternative models on total failure |
| `reasoning.effort` | `reasoning_effort` | Via `litellm_reasoning_resolver` |
| `reasoning.budget` | `thinking` (dict) | Anthropic-specific |
| `extra_body` / `extra_args` | `**extra_kwargs` | |
| `tool_choice` | `tool_choice` | `"auto"` / `"required"` / `"none"` / tool name |
| `tool_execution_mode` | `parallel_tool_calls` | `SEQUENTIAL` or `PARALLEL` |

Provider-native capabilities (web search, computer use, etc.) are
exposed as typed `HostedTool` subclasses (`tools/hosted/`) where
applicable; `extra_body` / `extra_args` is the escape hatch for
beta/esoteric shapes.

## LLMConfig (`llm_config.py`)

Provider-agnostic `@dataclass`, all fields `Optional[T] = None`.
Supports `resolve()` for merging (override > base).

## Usage Tracking

`LLMUsage` accumulates via `__add__`. Per-request breakdown (`usage`
list), cache-aware fields (`cached_tokens`,
`cache_creation_input_tokens`), reasoning tokens. Canonical lives in
`types/tokens/llm_usage.py`; `llms/llm_usage.py` is a re-export shim
for the historical import path.

## Prompt Caching (`litellm/litellm_cache_applicator.py`)

Caching fields live on `LiteLLMConfig` (all default off — the caller opts INTO
the cache-write premium). `litellm_cache_applicator.resolve_cache_control_injection_points`
turns the `auto_cache_control` one-bool opt-in into the canonical
`cache_control_injection_points` (system message + last input message), mirroring
the native path's `AnthropicConfig.auto_cache_control`. An explicit
`cache_control_injection_points` wins over the opt-in.

- **Anthropic** — `auto_cache_control` / explicit `cache_control_injection_points`,
  applied by litellm's `AnthropicCacheControlHook` (min ~1024-token prefix).
- **Gemini** — `cached_content` ID (reference mode).
- **OpenAI** — automatic prefix caching; `prompt_cache_key` /
  `prompt_cache_retention` are routing hints (markers ignored).

## Reasoning (`litellm/litellm_reasoning_resolver.py`)

Resolves `Reasoning` into `LiteLLMReasoningParam`:

- `reasoning_effort` — effort level string (all providers)
- `thinking` — budget control dict (Anthropic-specific)

Response parsing extracts `reasoning_content` (unified string) and
`thinking_blocks` (structured, required for multi-turn tool use).

## API-Level Retries & Fallbacks

| Layer | Field | Scope | Retries |
|---|---|---|---|
| SDK | `LLMConfig.num_retries` | Per LLM call | Transient HTTP (429, 500, timeouts) inside provider |
| Framework | `LLMConfig.retry_policy` | Per LLM call | Classified errors (`rate_limit` / `server_error` / `timeout`) outside SDK with backoff + jitter. Non-streaming only |
| SDK | `LLMConfig.fallbacks` | Per LLM call | Alternative models on total failure |
| Tool | `FunctionTool.max_retries` | Per tool per run | LLM-driven (errors go back to LLM) |

See `docs/llms/retry_policy.md`, `litellm/litellm_retry.py`.

## Adding a New Provider

1. Subclass `LLM`, implement `acomplete()`.
2. Return `LLMResponse` (non-streaming) or yield `LLMStreamEvent` (streaming).
3. Map `LLMConfig` fields via explicit named args.
4. Parse usage into `LLMUsage` with `InputTokensDetails` /
   `OutputTokensDetails`.
5. Set the agent's `llm` field to your instance.
