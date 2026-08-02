# Types Module

Type definitions for the framework — source of truth for every
framework-owned type used across the ADK.

## Directory Structure

```
types/
  agents/         # Agent-as-tool input/output
  common/         # Headers, Body, Query, Metadata, Reasoning config
  evals/          # Evaluation result types
  input/          # Layer 1 input (TypedDict, sent TO LLM)
  intents/        # Agent intent types
  items/          # Layer 3 RunItems + ItemHelpers (history)
  output/         # Layer 1 replay params + FunctionToolCallResult
  permissions/    # Reserved (not yet wired)
  responses/      # Layer 1 response @dataclass (received FROM LLM)
  run/            # RunResult and run-outcome types
  tokens/         # Token-tracking (InputTokensDetails, etc.)
  tools/          # Tool config + definition types
```

## Three-Layer Architecture (Canonical)

Three layers on one axis (developer-facing vs wire) × two directions
(sent TO vs received FROM the LLM). Every framework type belongs to
exactly one layer + one direction.

### Layer 1 — Provider-Agnostic (framework-owned)

Developer-facing and runner-internal. Lives in `types/input/`,
`types/output/`, `types/responses/`. **Never imports from any
provider SDK.**

#### 1a. Input (sent TO LLM) — `TypedDict`

Plain dicts at runtime (zero-cost hot paths) with TypedDict for shape
checking.

| Type | Purpose |
|---|---|
| `LLMInputText` / `LLMInputImage` / `LLMInputAudio` | Content parts |
| `LLMInputMessage` | Strict message: role + typed content list |
| `LLMInputEasyMessage` | Convenience: `content: str \| list` |
| `LLMResponseFunctionToolCallParam` | Tool-call replay |
| `FunctionToolCallResultParam` | Tool-result replay |
| `LLMResponseMessageParam` | Assistant-message replay |
| `LLMResponseReasoningParam` | Reasoning-block replay |

Union: `LLMInputContentItem` — accepted by `LLM.acomplete(messages: list[LLMInputContentItem])`.

Replay-param types live physically under `types/output/` (serialized
form of response parts) but flow in the **input** direction; re-exported
from `types/input/`.

#### 1b. Response (received FROM LLM) — `@dataclass`

Lives in `types/responses/llm_response.py`.

| Type | Purpose |
|---|---|
| `LLMResponseText` | Text content with annotations |
| `LLMResponseRefusal` | Refusal content |
| `LLMResponseFunctionToolCall` | Tool-call request |
| `LLMResponseReasoning` | Reasoning / thinking block |

Union: `LLMResponsePart`. Container: `LLMResponse` (`parts: list[LLMResponsePart]` + `usage` + stream metadata).

**Dual-type pattern:** every `LLMResponse*` dataclass has a matching
`*Param` TypedDict in `types/output/` and a `to_param()` method
annotated with the exact Param TypedDict so pyright validates the
round-trip statically.

### Layer 2 — Wire Types (provider-owned)

Each provider's SDK types. Confined to that provider's module. **Never**
appears in `types/`, `run/`, `agents/`, `tools/`, `context/`, or any
developer-facing surface.

| Provider module | Wire types | Converter |
|---|---|---|
| `llms/litellm/` | `litellm.types.llms.openai.*`, `litellm.types.utils.*` | `ChatCompletionConverter` |
| `llms/anthropic/` | `anthropic.types.*` | `AnthropicConverter` |
| `llms/openai/` | `openai.types.responses.*`, `openai.types.chat.*` | `OpenAIResponsesConverter`, `OpenAIChatCompletionsConverter` |

Constructed and consumed inside `LLM.acomplete()`; never leak. One
conversion hop per direction per provider module.

### Layer 3 — Items (framework-owned)

Conversation history entries. Lives in `types/items/`. Each `RunItem`
wraps a Layer 1 type via `raw: T` and adds framework metadata
(`type` discriminator, `agent_name`, extras like `source`/`target`).

`ItemHelpers` (`types/items/items.py`) owns Layer 1 ↔ Layer 3:

| Helper | Direction |
|---|---|
| `response_to_run_items(response, agent_name)` | Layer 1 → Layer 3 |
| `run_items_to_params(items)` | Layer 3 → Layer 1 (replay) |
| `message_to_run_items(message, agent_name)` | Layer 1 message → Layer 3 |

See `items/CLAUDE.md`.

## The Flow (one turn)

```
dev input: str | list[LLMInputContentItem]                  [Layer 1]
  → loop.py builds list[LLMInputContentItem]                [Layer 1]
  → LLM.acomplete(messages)                                 [boundary]
      | (inside llms/litellm or llms/anthropic only)
  → ChatCompletionConverter.items_to_messages(...)          [L1 → L2]
  → litellm.acompletion(...) | anthropic.messages.create()  [Layer 2]
  → _parse_response(...)                                    [L2 → L1]
  → LLMResponse (parts: list[LLMResponsePart])              [Layer 1]
  → ItemHelpers.response_to_run_items(...)                  [L1 → L3]
  → list[RunItem] in RunResult.new_items                    [Layer 3]
  → ItemHelpers.run_items_to_params(items)                  [L3 → L1]
  → next turn: list[LLMInputContentItem]                    [Layer 1]
```

**Invariant:** Layer 2 exists only inside one provider module, on
one side of the `LLM.acomplete()` boundary. Outside that boundary is
Layer 1 or Layer 3 only.

## Why Not `AnyLLMModel`?

See `architecture.md`.
Short version: OpenAI's `Model` ABC is typed against
`openai.types.responses.*` re-exports, forcing every non-OpenAI
provider into 3 conversion hops per turn. Our `LLMResponsePart` +
`LLMInputContentItem` duo is framework-owned; each provider does
exactly 1 hop per direction inside its boundary. Rejecting
`AnyLLMModel` is load-bearing.

## Other Notable Types

### Framework Types Adjacent to Replay Params (`types/output/`)

- `FunctionToolCallResult` (Pydantic BaseModel) — runtime tool execution
  result. Twin: `FunctionToolCallResultParam` (TypedDict, replay).

### Items (`types/items/`)

`RunItem` = Union of `@dataclass(frozen=True)` items. Shared:
`type: Literal[...]`, `agent_name: Optional[str]`, `raw: T`,
`to_param() -> LLMInputContentItem`.

### Responses (`types/responses/`)

- `LLMResponse` — container (`parts`, `usage`, refusal helpers)
- `LLMStreamEvent` — streaming events (text/reasoning/tool-call delta, usage, end)
- `LLMResponseAnnotation` — text annotations (citations, etc.)

### Tokens (`types/tokens/`)

| Type | Purpose |
|---|---|
| `InputTokensDetails` / `OutputTokensDetails` | Per-request breakdown (cached, cache-creation, reasoning) |
| `LLMSingleRequestUsage` | Single-request stats |
| `LLMUsage` | Cumulative accumulator (`__add__`) |
| `LLMUsageLimits` | Per-run limits (requests, tool calls, tokens) |

`llms/llm_usage.py` is a re-export shim preserving the historical
import path. Canonical lives here.

### Common (`types/common/`)

`Body`, `Query`, `Headers`, `Metadata` HTTP primitives;
`Reasoning` config (reasoning resolver inputs).

### Tool Types (`types/tools/`)

- `FunctionToolCallResult` — `FunctionTool` execution result
- Response-side dataclasses for provider-native tool calls
  (`WebSearchToolCall`, `FileSearchToolCall`, `ComputerToolCall`,
  `CodeInterpreterToolCall`, `ImageGenerationToolCall`,
  `ToolSearchToolCall`, `MCPCall`, …) — consumed by matching RunItem
  on provider-native tool traffic.

Request-side wrappers are NOT here — see `tools-guardrails` rule.

### Agents (`types/agents/`)

`AgentToolInput`, `AgentToolOutputExtractor`, `AgentToolInputBuilder`
— Agent-as-tool customization.

### Permissions (`types/permissions/`)

**Reserved.** Models the future structured-approval system. Current
runner uses `RunConfig.can_use_tool: Callable[..., bool]`.

See `docs/types/types.md` for usage examples.
