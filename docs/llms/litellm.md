# LiteLLM Provider

The default LLM provider in TroopAI Agents, using [litellm](https://github.com/BerriAI/litellm) to access 100+ language models through a unified API.

## Quick Start

```python
from troopai.adk.agents import Agent
from troopai.adk.llms import LiteLLM, LLMConfig

agent = Agent(
    name="Assistant",
    system_prompt="You are helpful.",
    llm=LiteLLM(),
)
```

If no `llm` is specified on an Agent, the Runner uses `LiteLLM()` by default.

## Authentication

LiteLLM reads API keys from environment variables:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
```

Or pass explicitly:

```python
llm = LiteLLM(api_key="sk-...", base_url="https://custom-endpoint.com")
```

## Package Structure

All litellm-specific code lives in `troopai.adk/llms/litellm/`:

```
llms/
├── llm.py               # LLM abstract base class (provider-agnostic)
├── llm_config.py         # LLMConfig @dataclass (provider-agnostic)
├── llm_usage.py          # Token usage tracking (provider-agnostic)
└── litellm/              # LiteLLM-specific implementation
    ├── litellm_model.py              # LiteLLM class
    ├── litellm_param_types.py        # *Param TypedDicts for wire-format dicts
    ├── litellm_prompt_caching_resolver.py # Prompt caching resolver
    ├── litellm_reasoning_resolver.py # Reasoning parameter resolver
    └── litellm_provider.py           # Provider detection
```

Provider-agnostic types (`LLM`, `LLMConfig`, `LLMUsage`) stay at the `llms/` root. This separation means a future direct provider implementation (e.g., `AnthropicProvider(LLM)`) wouldn't inherit any litellm-specific code.

## Configuration

`LLMConfig` is a provider-agnostic `@dataclass`. The `LiteLLM` class maps its fields to litellm parameter names via explicit named arguments — no dict spreading.

```python
from troopai.adk.llms import LLMConfig

config = LLMConfig(
    temperature=0.7,
    max_output_tokens=2000,
    top_p=0.9,
)
```

### Parameter Mapping

| LLMConfig field | litellm parameter | Notes |
|-----------------|-------------------|-------|
| `temperature` | `temperature` | Direct pass-through |
| `top_p` | `top_p` | Direct pass-through |
| `top_k` | `top_k` | Direct pass-through |
| `max_output_tokens` | `max_tokens` | Renamed |
| `stop_sequences` | `stop` | Renamed |
| `response_logprobs` | `logprobs` (bool) | Renamed |
| `logprobs` | `top_logprobs` (int) | Renamed |
| `num_retries` | `num_retries` | Transient error retries |
| `fallbacks` | `fallbacks` | Alternative models on failure |

### Resilience

```python
config = LLMConfig(
    num_retries=3,                                      # Retry on 429, 500, timeouts
    fallbacks=["gpt-4o-mini", "claude-sonnet-4-20250514"],  # Try these on total failure
)
```

## Prompt Caching

Prompt caching reduces costs by reusing previously computed token representations. Each provider has a different mechanism.

### Anthropic

Anthropic uses explicit `cache_control` breakpoints injected into messages. litellm's `AnthropicCacheControlHook` handles injection automatically.

```python
from troopai.adk.types.caching import AnthropicPromptCaching, CacheTTL

config = LLMConfig(
    prompt_caching=AnthropicPromptCaching(
        ttl=CacheTTL.ONE_HOUR,        # 1h cache (2x write, 0.1x read)
        max_cache_breakpoints=2,
        apply_to_system_prompt=True,
    ),
)
```

**How it works in litellm:**
1. `resolve_cache_params()` builds `cache_control_injection_points` (e.g., system message + last user message)
2. litellm's `AnthropicCacheControlHook` injects `{"type": "ephemeral"}` at those points
3. Anthropic's API caches content at the breakpoints

**Cost savings:**
- Cache write: 1.25x base input token price
- Cache read (hit): 0.1x base input token price
- 5-minute TTL is free; 1-hour TTL costs 2x

**Limitations:**
- litellm currently ignores the `ttl` field — it always uses Anthropic's default (5 minutes). A future direct `AnthropicProvider` would use `ttl` natively.
- Minimum tokens: varies by model (2048–4096)

### Google Gemini

Gemini uses **reference-based** caching: create a `CachedContent` resource via Google's API, then reference it by ID.

```python
from troopai.adk.types.caching import GeminiPromptCaching

config = LLMConfig(
    prompt_caching=GeminiPromptCaching(
        cached_content_id="cachedContents/abc123",
        cache_ttl=3600,  # 1 hour in seconds
    ),
)
```

**How it works in litellm:**
1. `resolve_cache_params()` returns the `cached_content` ID
2. litellm passes it to Gemini's `generateContent` API as `cachedContent`
3. Gemini reuses the pre-cached content

**Note:** Gemini also supports inline caching via message-level `cache_control` blocks, but that mechanism is handled by litellm's Vertex AI transformation layer separately — not through the injection points (which are Anthropic-only).

### OpenAI

OpenAI caches automatically for prompts >= 1024 tokens. Optional hints improve hit rates.

```python
from troopai.adk.types.caching import OpenAIPromptCaching

config = LLMConfig(
    prompt_caching=OpenAIPromptCaching(
        prompt_cache_key="my-app-stable",   # Routing hint
        prompt_cache_retention="24h",   # Cache duration
    ),
)
```

**Cost savings:** Cached tokens are 50% cheaper (0.5x).

## Structured Output

LiteLLM implements a three-tier strategy for structured output:

1. **JSON Schema mode** — Model natively constrains output to match the schema (best reliability)
2. **JSON Object mode** — Model outputs valid JSON, client validates against schema
3. **No JSON support** — Prompt-based + client-side validation

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    sentiment: str
    confidence: float

agent = Agent(
    name="Analyzer",
    output_schema=Analysis,
    llm=LiteLLM(),
)
```

## Wire-Format Types

All known-shape parameter dicts in `litellm_model.py` use `*Param` TypedDicts (defined in `litellm_param_types.py`) instead of `dict[str, Any]`. These are plain dicts at runtime — no conversion cost — but give static type checkers visibility into key names and value types.

| TypedDict | Used by | Purpose |
|-----------|---------|---------|
| `FunctionToolParam` | `_convert_tools()` | Function tool in litellm `tools` list |
| `FunctionDefinitionParam` | `_convert_tools()` | Inner function definition (name, description, parameters, strict) |
| `ResponseFormatParam` | `_build_response_format()` | Structured output `response_format` |
| `JsonSchemaResponseParam` | `_build_response_format()` | Inner `json_schema` dict |
| `BuiltinToolParam` | `_convert_builtin_tool()` | Built-in tool in litellm `tools` list |
| `WebSearchConfigParam` | `_convert_builtin_tool()` | Web search config |
| `FileSearchConfigParam` | `_convert_builtin_tool()` | File search config |
| `ComputerUseConfigParam` | `_convert_builtin_tool()` | Computer use config |
| `CodeInterpreterConfigParam` | `_convert_builtin_tool()` | Code interpreter config |
| `ImageGenerationConfigParam` | `_convert_builtin_tool()` | Image generation config |
| `ShellConfigParam` | `_convert_builtin_tool()` | Shell tool config |
| `ToolSearchConfigParam` | `_convert_builtin_tool()` | Tool search config |
| `StreamedToolCallParam` | `_stream()` | Streaming tool call accumulator |
| `WireToolParam` | `_convert_tools()` | Union: `FunctionToolParam \| BuiltinToolParam` |

`dict[str, Any]` is intentionally kept for `extra_params` / `extra_kwargs` (genuinely dynamic provider overrides) and JSON Schema fields (inherently unstructured).

## Streaming

```python
result = Runner.run(agent, "Tell me a story", stream=True)

async for event in result.stream_events():
    if event.type == "raw_response_event":
        if event.data.type == "content_delta":
            print(event.data.content_delta, end="", flush=True)
```

Usage tracking in streaming mode requires `LLMConfig.include_usage=True` (default), which sets `stream_options={"include_usage": True}` on the litellm call.

## Provider Detection

Utilities for detecting which provider backs a model:

```python
from troopai.adk.llms.litellm.litellm_provider import detect_provider, is_anthropic

detect_provider("claude-sonnet-4-20250514")  # "anthropic"
detect_provider("gpt-4o")                   # "openai"
detect_provider("gemini/gemini-2.5-flash")  # "vertex_ai"

is_anthropic("claude-sonnet-4-20250514")    # True
```

## Supported Models

LiteLLM supports 100+ models. Common examples:

| Provider | Model string |
|----------|-------------|
| Anthropic | `claude-opus-4-6`, `claude-sonnet-4-20250514`, `claude-haiku-4-5-20251001` |
| OpenAI | `gpt-4o`, `gpt-4o-mini`, `o1`, `o3-mini` |
| Google | `gemini/gemini-2.5-flash`, `gemini/gemini-2.5-pro` |
| Mistral | `mistral/mistral-large-latest` |
| Groq | `groq/llama-3.3-70b-versatile` |

See [litellm supported models](https://docs.litellm.ai/docs/providers) for the full list.
