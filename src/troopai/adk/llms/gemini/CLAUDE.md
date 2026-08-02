# Gemini LLM Module

Native Google Gemini implementation using the `google-genai` SDK
directly. One class — `GeminiLLM` — backed by
`client.aio.models.generate_content` / `generate_content_stream`,
supporting both the Gemini Developer API (api_key auth) and Vertex AI
(project + location + credentials) via the SDK's built-in dispatch.

## Files

- `gemini_model.py` — `GeminiLLM(LLM)`: `acomplete` overloads,
  `_get_client` dispatching on `vertexai`, `_build_generate_content_config`,
  `_stream` per-part accumulator. Owns the SDK boundary call.
- `gemini_converter.py` — Stateless `@classmethod` converter:
  Layer 1 ↔ `google.genai.types.*`. Owns `items_to_contents`,
  `convert_tools` (function tools + Phase A hosted tools folded into
  one `Tool` instance), `convert_tool_choice`, `response_to_llm_response`,
  `_parse_usage`.
- `gemini_config.py` — `GeminiConfig(LLMConfig)` with typed
  Gemini-specific fields: `thinking_config`, `safety_settings`,
  `cached_content_name`, `response_modalities`.
- `gemini_boundary.py` — `headers_as_sdk` / `sanitize_for_log`. Mirrors
  the OpenAI / Anthropic boundary modules. Single source of truth for
  the auth-header blocklist (`x-goog-api-key`, `x-google-api-key`,
  `authorization`, `proxy-authorization`).
- `gemini_reasoning_resolver.py` — `resolve_thinking` reads
  `GeminiConfig.thinking_config` first, falls back to
  `LLMConfig.extra_args["thinking_config"]`, validates
  `thinking_budget >= 1024` for explicit-positive budgets.
- `gemini_retry.py` — `gemini_exception_to_kind` classifier +
  `call_with_retry` shim. Maps `ClientError(429)` to `"rate_limit"`,
  `ServerError(504)` to `"timeout"`, other 5xx to `"server_error"`.

## Key Architectural Decisions

1. **Single class, two backend modes.** `vertexai=False` (default)
   uses the Gemini Developer API with `api_key`; `vertexai=True` uses
   Vertex AI with `project` + `location` + optional `credentials`. The
   SDK's `Client.__init__` dispatches naturally; we just forward.
2. **Native `response_schema` for structured output.** No synthetic-tool
   pattern. When `output_schema` is supplied (and not plain text),
   `GenerateContentConfig.response_mime_type="application/json"` plus
   `response_schema=output_schema.json_schema()` constrains the model
   server-side. The Runner reads the JSON text part as usual.
3. **Reference-by-name caching.** `GeminiConfig.cached_content_name`
   accepts a pre-created cache resource name (`"cachedContents/<id>"`).
   The framework does NOT manage cache lifecycle — the developer
   creates the cache via `client.aio.caches.create` externally.
4. **Per-part streaming accumulator.** Each Gemini streaming chunk is
   a full `GenerateContentResponse` whose `candidates[0].content.parts`
   carries the delta text for that chunk. The `_PartAccumulator` keyed
   on the part index buffers fragments; the final `LLMResponse` is
   built when the stream closes.
5. **Hosted tools folded into one `Tool` instance.** Gemini's wire
   format expects a list of `Tool` objects where each instance can
   bundle `function_declarations` + `google_search` + `code_execution`
   + `url_context`. The converter collects every framework tool into a
   single `Tool` per request.

## Provider-Hosted Tools

Gemini's hosted capabilities are wired via the framework's typed
hosted-tool classes per `.claude/rules/tools-guardrails.md`:

- `WebSearchTool` → `Tool(google_search=GoogleSearch())`
- `CodeExecutionTool` → `Tool(code_execution=ToolCodeExecution())`
- `URLContextTool` → `Tool(url_context=UrlContext())`

`FileSearchTool` and `ImageGenerationTool` raise
`UnsupportedHostedToolError` — they are OpenAI-Responses-only.

## Type Strategy

Uses `google.genai.types.*` directly — no custom TypedDicts. These
types NEVER leak outside this module. The converter bridges to/from
Layer 1 types (`LLMInputContentItem`, `LLMResponse`).
