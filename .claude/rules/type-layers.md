---
paths:
  - "src/troopai/adk/run/**/*.py"
  - "src/troopai/adk/llms/**/*.py"
  - "src/troopai/adk/types/**/*.py"
---

# Type Layer Boundaries — CRITICAL

| Layer | Types | Audience |
|---|---|---|
| 1 | `LLMInputContentItem`, `LLMOutputContentItem` | Framework internals + developer API |
| 2 | `ChatCompletionMessageParam`, `ChatCompletionToolCall` | LLM wire format (litellm) |
| 3 | `RunItem` (`ToolCallItem`, `MessageOutputItem`, …) | Developer-facing conversation items |

- Developer-facing APIs (`Agent`, `Runner`, `RunResult`, `RunState`, evals)
  MUST use Layer 1 or Layer 3 — NEVER Layer 2.
- LLM implementations convert between Layer 1/3 and their wire format
  INTERNALLY. New developer-facing modules MUST NOT import `types/chat/`.
- Loop internals (`run/loop.py`, `turn_resolution.py`, `resumption.py`,
  `tools_executor.py`, `handoffs_executor.py`) work with Layer 1. Tool
  results use `FunctionToolCallResultParam` (Layer 1), NEVER
  `ChatCompletionToolMessageParam` (Layer 2). `new_items` and public fields
  are Layer 3.
- `HistoryProcessor` operates on `list[RunItem]` (Layer 3); the loop converts
  messages → RunItems → processor → back.

Layer 2 is provider-specific (Chat Completions / litellm). Layer 1 and 3 are
provider-agnostic. Conversion between layers happens only in the LLM impl and
`ChatCompletionConverter`.

## Self-Check

1. Developer-facing API accepts/returns a `types/chat/` type? — violation.
2. Loop internal accumulates wire-format messages instead of
   `RunItem`/`LLMInputContentItem`? — violation.
