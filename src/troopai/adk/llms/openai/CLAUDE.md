# OpenAI LLM Module

Native OpenAI implementations using the `openai` SDK directly — no
litellm indirection. Two first-class `LLM` subclasses:
`OpenAIResponsesLLM` (Responses API) and `OpenAIChatCompletionsLLM`
(Chat Completions API).

## Files

- `openai_responses_model.py` — `OpenAIResponsesLLM(LLM)`: calls
  `client.responses.create()` with explicit named params.
- `openai_responses_converter.py` — Stateless `@classmethod` converter:
  Layer 1 ↔ `openai.types.responses.*`.
- `openai_responses_config.py` — `OpenAIResponsesConfig(LLMConfig)`
  with Responses-API-specific fields (reasoning, include, metadata,
  previous_response_id, truncation, …).
- `openai_chatcompletions_model.py` —
  `OpenAIChatCompletionsLLM(LLM)`: calls
  `client.chat.completions.create()` with explicit named params.
- `openai_chatcompletions_converter.py` — Stateless `@classmethod`
  converter: Layer 1 ↔ `openai.types.chat.*`.
- `openai_chatcompletions_config.py` —
  `OpenAIChatCompletionsConfig(LLMConfig)` with CC-specific fields
  (audio, modalities, web_search_options, prediction, …).
- `openai_retry.py` — `openai_exception_to_kind` classifier +
  `call_with_retry` wrapper delegating to `llms/retry.py`.

## Key Architectural Decisions

1. **Two classes, not one.** Responses and Chat Completions have
   different wire inputs (`ResponseInputParam` vs
   `list[ChatCompletionMessageParam]`), different output shapes
   (`Response.output` vs `ChatCompletion.choices[0].message`), and
   different `tool_choice` shapes — flat on Responses
   (`{"type":"function","name":"x"}`) vs nested on Chat Completions
   (`{"type":"function","function":{"name":"x"}}`). One class with a
   mode flag would branch every call site; two classes mirror the
   existing `LiteLLM` / `AnthropicLLM` split.
2. **No `FAKE_RESPONSES_ID`.** The OpenAI Agents SDK synthesises
   `"__fake_id__"` on the Chat Completions path because its ABC is
   typed against Responses-API identity semantics. This codebase's
   ABC returns framework-owned `LLMResponse` — items land directly
   in `LLMResponsePart` variants.
3. **Wire types from the installed SDK, never re-defined.** Configs
   and converters import from `openai.types.responses.*` /
   `openai.types.chat.*` / `openai.types.shared_params.*` verbatim.
   Provider-specific config fields on the two OpenAI configs are
   typed against the SDK's own TypedDicts — no framework-side
   mirrors.
4. **Hosted tools as typed `HostedTool` subclasses; `extra_body` is
   the escape hatch.** Per `tools-guardrails` rule. Converters
   dispatch typed `HostedTool` subclasses (web search, file search,
   code execution, etc.) to the matching wire shape and raise
   `UnsupportedHostedToolError` for unsupported subclasses. Raw
   `extra_body` / `extra_args` is forwarded unchanged for genuinely
   beta or esoteric shapes lacking a typed class.
5. **Generic retry loop shared across providers.** `llms/retry.py`
   holds the provider-agnostic loop; `openai_retry.py` supplies the
   openai-specific exception classifier.
6. **`LLMResponseProviderItem` variant for hosted tool outputs.** The
   Responses API emits ~15 non-function output items
   (`ResponseFileSearchToolCall`, `ResponseFunctionWebSearch`,
   `ImageGenerationCall`, `McpCall`, etc.) that don't fit the four
   "classic" `LLMResponsePart` variants. The 5th variant carries them
   with `item_type: str` + `raw: dict[str, Any]` — the "genuinely
   dynamic provider-specific extension" exception to the strong-typing
   rule.

## Streaming Strategy

- **Responses API**: `ResponseStreamEvent` has 53 named variants with
  explicit `item_added` / `item_delta` / `item_done` boundaries.
  Mapping to the framework's four stream events is mostly a 1:1
  dispatch by `event.type`.
- **Chat Completions**: `ChatCompletionChunk` delivers raw deltas;
  part boundaries must be reconstructed client-side. The converter
  tracks `text_index`, `refusal_index`, and a
  `dict[int, _ToolCallAccumulator]` keyed by the SDK's
  `choice.delta.tool_calls[i].index`.

## Provider-Native Capabilities

Web search, file search, computer use, image generation, code
interpreter, hosted MCP — pass the raw provider JSON through
`LLMConfig.extra_body`. The converters merge `extra_body` /
`extra_args` into the SDK call unchanged.

## Type Strategy

`openai.types.*` never leaks outside this module. The converters
bridge to/from Layer 1 types (`LLMInputContentItem`, `LLMResponse`).
Downstream code (Runner, Agent, run_impl) sees only Layer 1 / Layer 3
types.

## See Also

- `docs/llms/openai.md` — usage, config field tables, auth,
  hosted-tool passthrough, reasoning threading, structured output.
- `examples/llm_providers/openai/` — runnable examples.
- `tools-guardrails` rule — typed `HostedTool` subclass authoring contract.
- `llms` rule — LLM ABC contract.
