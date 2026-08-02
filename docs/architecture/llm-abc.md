(architecture/llm-abc)=

# 🔌 The `LLM` ABC

All provider traffic flows through a framework-owned `LLM` abstract
base class. The Runner never imports a provider SDK directly. Provider
code lives in `src/troopai/adk/llms/<provider>/` and is the only place
allowed to import `litellm`, `anthropic`, `openai`, `google.genai`, etc.

## Shape

```python
class LLM(abc.ABC):
    @abc.abstractmethod
    async def acomplete(
        self,
        messages: Sequence[LLMInputContentItem | RunItem],
        *,
        tools: Sequence[Tool] | None = None,
        config: LLMConfig | None = None,
        stream: bool = False,
    ) -> LLMResponse | AsyncIterator[LLMStreamEvent]: ...
```

- **Input** is Layer 1 (or Layer 3 history materialised as Layer 1) —
  framework-owned.
- **Output** is `LLMResponse` (parts-based union of `TextPart`,
  `ThinkingPart`, `ToolCallPart`) when `stream=False`, or an
  `AsyncIterator[LLMStreamEvent]` when `stream=True`.
- **Config** is provider-agnostic. Provider-specific configs subclass
  it (see below).

## Why a framework-owned ABC

Adopting OpenAI's `Model` / `AnyLLMModel` as the canonical type would
cost three conversion hops per turn (developer → OpenAI shape → litellm
shape → provider shape) and would leak `openai.types.*` into every
non-OpenAI provider implementation. The framework-owned ABC means
**one conversion per direction**, inside each provider module.

## Available implementations

| Provider                  | Module                                  | Config class                     |
| ------------------------- | --------------------------------------- | -------------------------------- |
| LiteLLM (any)             | `src/troopai/adk/llms/litellm/`          | `LiteLLMConfig`                  |
| Anthropic native          | `src/troopai/adk/llms/anthropic/`        | `AnthropicConfig`                |
| OpenAI Responses native   | `src/troopai/adk/llms/openai/`           | `OpenAIResponsesConfig`          |
| OpenAI Chat Completions   | `src/troopai/adk/llms/openai/`           | `OpenAIChatCompletionsConfig`    |
| Gemini native             | `src/troopai/adk/llms/gemini/`           | `GeminiConfig`                   |

Each provider implementation owns its own wire layer (`ChatCompletion*`,
`Anthropic*`, etc.) and converts in/out on the boundary.

## Config layering

```{mermaid}
classDiagram
  class LLMConfig
  class LiteLLMConfig
  class AnthropicConfig
  class OpenAIResponsesConfig
  class OpenAIChatCompletionsConfig
  class GeminiConfig
  LLMConfig <|-- LiteLLMConfig
  LLMConfig <|-- AnthropicConfig
  LLMConfig <|-- OpenAIResponsesConfig
  LLMConfig <|-- OpenAIChatCompletionsConfig
  LLMConfig <|-- GeminiConfig
```

`LLMConfig` declares only provider-agnostic fields (`model`,
`temperature`, `max_tokens`, retry / cache strategy hooks). Provider
subclasses add typed fields for that provider's surface (Anthropic
hosted-tool params, OpenAI Responses API knobs, Gemini-specific
options, etc.).

## Invariants

- The Runner MUST NOT import a provider SDK.
- A provider implementation MUST NOT depend on another provider's
  module.
- `LLMConfig` MUST NOT reference any provider name.
- Wire-type `TypedDict`s MUST NOT be converted to `@dataclass` /
  `BaseModel`.
