# Native OpenAI Provider

Two first-class `LLM` subclasses call the `openai` SDK directly, no
litellm indirection:

- `OpenAIResponsesLLM` — backed by `client.responses.create()`.
- `OpenAIChatCompletionsLLM` — backed by
  `client.chat.completions.create()`.

Use these when you want every current OpenAI feature end-to-end
(structured output via `ResponseTextConfigParam`, Responses-API
`previous_response_id` state threading, hosted tools, per-call
prompt-cache routing hints, fully-typed usage) without paying
litellm's conversion hops.

The existing `LiteLLM` path still works for OpenAI models — it stays
available as the multi-provider convenience path.

## Quick Start

```python
from troopai.adk.agents import Agent
from troopai.adk.llms.openai import OpenAIResponsesLLM
from troopai.adk.run import Runner

agent = Agent(
    name="Assistant",
    system_prompt="You are helpful.",
    llm=OpenAIResponsesLLM("gpt-5.1"),
)

result = await Runner.arun(agent, "Hello!")
```

Chat Completions:

```python
from troopai.adk.llms.openai import OpenAIChatCompletionsLLM

agent = Agent(
    name="Assistant",
    system_prompt="You are helpful.",
    llm=OpenAIChatCompletionsLLM("gpt-4o"),
)
```

## Which API should I use?

| Feature | Responses | Chat Completions |
|---|---|---|
| State threading via `previous_response_id` | yes | no |
| Hosted tools (web_search, file_search, image_generation, MCP, …) | yes | limited |
| Structured output via `ResponseTextConfigParam` | yes | via `response_format={"type":"json_schema",…}` |
| Reasoning item replay | native | no |
| Broadest model coverage (older, third-party, Azure) | gpt-5+ only | yes |
| tool_choice shape | flat `{"type":"function","name":"x"}` | nested `{"type":"function","function":{"name":"x"}}` |

Default to `OpenAIResponsesLLM` for gpt-5.x / o-series. Use
`OpenAIChatCompletionsLLM` for gpt-4o, azure-deployed models, and
when you need the broader Chat Completions ecosystem.

## Authentication

Reads from environment by default:

```bash
export OPENAI_API_KEY=sk-...
export OPENAI_ORG_ID=org-...        # optional
export OPENAI_PROJECT_ID=proj-...   # optional
```

Or pass explicitly:

```python
llm = OpenAIResponsesLLM(
    "gpt-5.1",
    api_key="sk-...",
    base_url="https://custom-endpoint.openai.azure.com/...",
    organization="org-...",
    project="proj-...",
    max_retries=3,  # passes through to AsyncOpenAI
)
```

Azure OpenAI users: the framework uses the generic `AsyncOpenAI`
client — point `base_url` / `api_key` at your Azure endpoint. For
heavier Azure-specific features (API-version routing, token
authentication), construct your own client and keep the `LLM`
instance one step above.

## Provider-agnostic configuration

`LLMConfig` covers everything that is not OpenAI-specific:

```python
from troopai.adk.llms import LLMConfig

config = LLMConfig(
    temperature=0.3,
    top_p=0.9,
    max_output_tokens=2_000,
    stop_sequences=["END"],
    include_usage=True,  # default
)
```

`max_output_tokens` maps to `max_completion_tokens` on the Chat
Completions call (and to `max_output_tokens` on the Responses call).
`stop_sequences` maps to `stop` on both.

## Responses-API configuration

`OpenAIResponsesConfig` adds Responses-API-only fields. Types come
straight from `openai.types.*`:

```python
from openai.types.shared_params import Reasoning
from troopai.adk.llms.openai import OpenAIResponsesConfig

cfg = OpenAIResponsesConfig(
    # --- generic LLMConfig fields ---
    temperature=0.3,
    max_output_tokens=4_000,
    # --- Responses-API-specific ---
    reasoning=Reasoning(effort="high"),
    previous_response_id="resp_abc123",   # state threading
    store=True,                            # persist for later reference
    truncation="auto",                     # context window management
    parallel_tool_calls=True,
    service_tier="priority",
    prompt_cache_key="my-app-stable",
    prompt_cache_retention="24h",
    max_tool_calls=10,
    background=False,
    include=["reasoning.encrypted_content"],
    metadata={"session": "abc"},
)
```

| Field | Type | Maps to |
|---|---|---|
| `reasoning` | `openai.types.shared_params.Reasoning` | `reasoning=` |
| `include` | `list[ResponseIncludable]` | `include=` |
| `metadata` | `openai.types.shared_params.Metadata` (`Mapping[str, object]`) | `metadata=` (stringified at boundary) |
| `store` | `bool` | `store=` |
| `previous_response_id` | `str` | `previous_response_id=` |
| `conversation` | `str` | `conversation=` |
| `truncation` | `Literal["auto","disabled"]` | `truncation=` |
| `parallel_tool_calls` | `bool` | `parallel_tool_calls=` |
| `service_tier` | `Literal["auto","default","flex","scale","priority"]` | `service_tier=` |
| `prompt_cache_key` | `str` | `prompt_cache_key=` |
| `prompt_cache_retention` | `Literal["in_memory","24h"]` | `prompt_cache_retention=` |
| `max_tool_calls` | `int` | `max_tool_calls=` |
| `background` | `bool` | `background=` |

## Chat Completions configuration

```python
from troopai.adk.llms.openai import OpenAIChatCompletionsConfig

cfg = OpenAIChatCompletionsConfig(
    temperature=0.3,
    max_output_tokens=4_000,
    # --- CC-specific ---
    modalities=["text"],
    store=True,
    service_tier="flex",
    prompt_cache_key="my-app-stable",
    prompt_cache_retention="24h",
    verbosity="medium",
)
```

| Field | Type | Maps to |
|---|---|---|
| `metadata` | `Metadata` | `metadata=` (stringified at boundary) |
| `audio` | `completion_create_params.Audio` | `audio=` |
| `web_search_options` | `completion_create_params.WebSearchOptions` | `web_search_options=` |
| `prediction` | `completion_create_params.Prediction` | `prediction=` |
| `modalities` | `list[Literal["text","audio"]]` | `modalities=` |
| `store` | `bool` | `store=` |
| `service_tier` | `Literal["auto","default","flex"]` | `service_tier=` |
| `prompt_cache_key` | `str` | `prompt_cache_key=` |
| `prompt_cache_retention` | `Literal["in_memory","24h"]` | `prompt_cache_retention=` |
| `verbosity` | `Literal["low","medium","high"]` | `verbosity=` |

## Hosted tools via `extra_body`

Framework tool classes are NOT added for provider-native
capabilities. Pass the raw provider JSON through
`LLMConfig.extra_body` — the converter forwards it unchanged.

### Web search (Responses API)

```python
config = OpenAIResponsesConfig(
    extra_body={
        "tools": [
            {
                "type": "web_search",
                "user_location": {"type": "approximate", "country": "US"},
            }
        ],
    },
)

agent = Agent(
    name="SearchBot",
    system_prompt="Answer questions using web search.",
    llm=OpenAIResponsesLLM("gpt-5.1"),
    llm_config=config,
)
```

Hosted tool outputs arrive as `LLMResponseProviderItem` entries on
the `LLMResponse.parts` list — the Layer-3 consumer inspects
`item_type` (e.g., `"web_search_call"`, `"file_search_call"`,
`"image_generation_call"`, `"mcp_call"`) and the raw payload.

### File search, computer use, image generation, code interpreter, MCP

Same mechanism — pass the provider tool JSON through `extra_body`.
The converter never wraps or translates these; whatever the SDK
returns reaches your code via `LLMResponseProviderItem`.

## Structured output

Works the same on both classes: set `Agent.output_schema` and the
converter resolves it:

- On Responses: `text=ResponseTextConfigParam(...)` with a
  JSON-Schema `format`.
- On Chat Completions: `response_format=ResponseFormatJSONSchema(...)`.

`AgentOutputSchemaBase.is_strict_json_schema()` controls the `strict`
flag.

## Reasoning threading

Reasoning items returned on one turn are replayed verbatim on the
next turn as `LLMInputReasoning` entries. The Responses converter
preserves `encrypted_content` / `summary` / `content` fields exactly;
the Chat Completions converter logs a warning (CC does not accept
reasoning replay) and drops them.

## Streaming

```python
result = Runner.run(agent, "Tell me a story", stream=True)

async for event in result.stream_events():
    if event.type == "raw_response_event":
        if event.data.type == "content_delta":
            logger.info(event.data.content_delta)
```

Usage tracking in streaming mode: `LLMConfig.include_usage=True`
(default) injects `stream_options={"include_usage": True}` on the
Chat Completions call; on Responses the final
`response.completed` event always carries `ResponseUsage`.

## Retry policy

```python
from troopai.adk.types.llms.retry_policy import LLMRetryPolicy

config = LLMConfig(
    retry_policy=LLMRetryPolicy(
        max_retries=5,
        initial_delay_s=0.5,
        max_delay_s=30.0,
        jitter=True,
        retry_on=["rate_limit", "server_error", "timeout"],
    ),
)
```

Mapping (`openai_retry.openai_exception_to_kind`):

| Exception | Kind |
|---|---|
| `RateLimitError` | `"rate_limit"` |
| `APITimeoutError` | `"timeout"` |
| `APIConnectionError` | `"server_error"` |
| `APIStatusError` status 5xx / 529 | `"server_error"` |
| `APIStatusError` status 429 | `"rate_limit"` |
| `APIStatusError` status 408 | `"timeout"` |
| `APIStatusError` status other 4xx (400/401/403/404/422) | `None` (permanent) |

Non-streaming only — retries on a streaming call would either drop
tokens mid-stream or double-emit them.

## Supported models

| Series | Default class | Notes |
|---|---|---|
| `gpt-5`, `gpt-5.1`, `o1`, `o3`, `o3-mini`, `o4-mini` | `OpenAIResponsesLLM` | Full Responses-API feature set |
| `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `gpt-4.1-mini` | `OpenAIChatCompletionsLLM` | Chat Completions; also valid on Responses |
| Azure-deployed | `OpenAIChatCompletionsLLM` with custom `base_url` | Point `base_url` at Azure endpoint |

See the OpenAI API reference for the authoritative model list.
