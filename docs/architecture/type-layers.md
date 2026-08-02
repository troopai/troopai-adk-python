(architecture/type-layers)=

# 🧩 The Three Type Layers

The ADK uses three discrete type layers. Developer-facing APIs use
Layer 1 or Layer 3 — never Layer 2.

```{figure} ../_static/images/architecture/type-layers.svg
:alt: Three layers — Layer 1 LLMInputContentItem (in), Layer 2 ChatCompletion* (wire), Layer 3 RunItem (out).
:width: 80%
:class: themed
:align: center

One conversion per direction, inside each provider.
```

## Layer 1 — `LLMInputContentItem` (input, provider-agnostic)

The framework-owned input shape. A discriminated union over text,
image, audio, message, and tool-result payloads. Every developer-facing
prompt path accepts Layer 1. The major variants:

| Variant                          | Shape (discriminator + key fields)                                              |
| -------------------------------- | ------------------------------------------------------------------------------- |
| `LLMInputText`                   | `{ "type": "input_text", "text": str }`                                         |
| `LLMInputImage`                  | `{ "type": "input_image", "image_url": str, "detail": "low" \| "high" \| "auto" }` |
| `LLMInputAudio`                  | `{ "type": "input_audio", "input_audio": { "data": str, "format": "wav" \| "mp3" } }` |
| `LLMInputMessage`                | `{ "role": "user" \| "assistant" \| "system", "content": Sequence[...] }`        |
| `FunctionToolCallResultParam`    | `{ "type": "function_call_output", "call_id": str, "output": str \| Sequence[...] }` |

The full union also covers `LLMInputEasyMessage`,
`LLMResponseFunctionToolCallParam`, `LLMResponseMessageParam`,
`LLMResponseReasoningParam`, and `LLMResponseProviderItemParam` for
roundtripping prior assistant turns.

## Layer 2 — `ChatCompletion*` (wire, never developer-facing)

Provider-specific wire shapes. `ChatCompletionMessageParam`,
`ChatCompletionToolMessageParam`, etc. These live inside
`src/troopai/adk/llms/<provider>/` and never escape.

Layer 2 is `TypedDict`-shaped (sent-side, replay types). It is
**never** converted to `@dataclass` or `BaseModel`.

## Layer 3 — `RunItem` (output, developer-facing conversation history)

The framework-owned conversation-history shape. A discriminated union
where each variant maps to a turn artefact:

| Variant                | Meaning                                                  |
| ---------------------- | -------------------------------------------------------- |
| `UserItem`             | A user message.                                          |
| `SystemItem`           | A system / instruction message.                          |
| `MessageOutputItem`    | An assistant message produced by the model.              |
| `ToolCallItem`         | A tool call emitted by the model.                        |
| `ToolCallOutputItem`   | The execution result for a `ToolCallItem`.               |
| `HandoffCallItem`      | The model's tool call requesting a handoff.              |
| `HandoffOutputItem`    | The result item the next agent sees after a handoff.     |
| `ReasoningItem`        | Provider-emitted reasoning text (where surfaced).        |
| `CompactionItem`       | A compaction summary that replaced earlier turns.        |
| `MCPListToolsItem`     | The list of tools advertised by an MCP server.           |
| `MCPApprovalRequestItem` / `MCPApprovalResponseItem` | MCP human-approval round-trip. |
| `ProviderItem`         | Provider-specific opaque output item.                    |
| `ToolSearchCallItem`   | A hosted tool-search invocation.                         |

`RunResult.new_items`, `RunState.conversation_history`, and every
`HistoryProcessor` use Layer 3.

## Why three layers

Adopting OpenAI's `Model` ABC would force everything through their
`openai.types.*` shapes — three conversion hops per turn, with dead
parameters baked into every non-OpenAI provider implementation. The
ADK keeps a framework-owned input and output type, and a single
provider-local wire type. **One conversion per direction**, inside
the provider module that owns the wire format.

## The flow, end to end

```{mermaid}
flowchart LR
  d1([developer code]) -- "Layer 1" --> llm[LLM ABC]
  llm -- "convert (provider)" --> wire[Layer 2 wire]
  wire -- "HTTP" --> prov[(provider API)]
  prov -- "HTTP" --> wire2[Layer 2 wire]
  wire2 -- "convert (provider)" --> resp[LLMResponse]
  resp -- "Layer 3" --> d2([developer code])
```

The Runner only ever holds Layer 1 (going in) and Layer 3 (coming
out). The Layer 2 conversion is contained within each provider
implementation under `src/troopai/adk/llms/<provider>/`.

## Matrix — which Python construct for which type

| Use                    | Construct                |
| ---------------------- | ------------------------ |
| Framework type         | `@dataclass`             |
| Stream event type      | Pydantic `@dataclass`    |
| Validation-heavy / LLM-output / received | `BaseModel` |
| LLM-input / sent / `*Param` replay      | `TypedDict` |
