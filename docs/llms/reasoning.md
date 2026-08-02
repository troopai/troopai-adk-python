# Reasoning / Extended Thinking

Configure chain-of-thought reasoning across OpenAI, Anthropic, and Google Gemini models.

## Quick Start

```python
from troopai.adk.agents import Agent
from troopai.adk.llms import LLMConfig
from troopai.adk.types.common import Reasoning

agent = Agent(
    name="Analyst",
    system_prompt="Think carefully before answering.",
    llm_config=LLMConfig(
        reasoning=Reasoning(effort="high"),
    ),
)
```

## Configuration

The `Reasoning` class on `LLMConfig` controls three dimensions:

| Field | Type | Description |
|-------|------|-------------|
| `effort` | `"none"` / `"minimal"` / `"low"` / `"medium"` / `"high"` / `"xhigh"` | How hard the model reasons |
| `mode` | `"auto"` / `"manual"` | Whether the model decides when to think, or the caller controls it |
| `budget` | `int` (tokens) | Explicit token budget for reasoning (required when `mode="manual"`) |
| `summary` | `"auto"` / `"concise"` / `"detailed"` | Level of detail for reasoning summaries |
| `include_thoughts` | `bool` | Whether to include reasoning content in responses |

### Effort Only (All Providers)

The simplest configuration — works across all reasoning-capable models:

```python
Reasoning(effort="high")
```

### Adaptive Mode (Anthropic)

Recommended for Claude Opus 4.6 and Sonnet 4.6. The model dynamically decides when and how much to think:

```python
Reasoning(mode="auto", effort="high")
```

### Explicit Budget (Anthropic / Gemini)

Control exactly how many tokens the model can use for reasoning:

```python
Reasoning(mode="manual", budget=8000)

# Or simply set budget — mode="manual" is inferred:
Reasoning(budget=8000)
```

### Reasoning Summaries (OpenAI)

Get a summary of the model's reasoning process:

```python
Reasoning(effort="high", summary="concise")
```

### Include Thoughts (Gemini)

Include thought content in the response:

```python
Reasoning(effort="high", include_thoughts=True)
```

## Provider Support

### OpenAI

| Feature | Support |
|---------|---------|
| `effort` | `"none"`, `"minimal"`, `"low"`, `"medium"`, `"high"`, `"xhigh"` |
| `mode` | Not applicable (reasoning is always on for o-series) |
| `budget` | Not applicable (tokens reserved internally) |
| `summary` | `"auto"`, `"concise"`, `"detailed"` |
| `include_thoughts` | Not applicable |
| Models | o3, o4-mini, o1, gpt-5, gpt-5-mini, gpt-5-nano |

**Notes:**
- Reserve minimum 25,000 tokens for `max_output_tokens` on complex tasks
- Reasoning tokens count as output tokens and occupy context window
- Avoid chain-of-thought prompts — the model handles this internally

### Anthropic

| Feature | Support |
|---------|---------|
| `effort` | `"low"`, `"medium"`, `"high"` (`"max"` Opus 4.6 only — use `extra_args`) |
| `mode` | `"auto"` (adaptive) or `"manual"` (explicit budget) |
| `budget` | Must be less than `max_output_tokens` |
| `summary` | Not directly used (Claude 4+ returns summarized thinking automatically) |
| `include_thoughts` | Not directly used (thinking blocks always included when enabled) |
| Models | Claude Opus 4.6, Sonnet 4.6, Opus 4.5, Sonnet 4 |

**Notes:**
- `mode="auto"` (adaptive) is recommended for Opus 4.6 / Sonnet 4.6
- `budget` is deprecated on Opus 4.6 / Sonnet 4.6 in favor of adaptive thinking
- When using thinking + tools, only `tool_choice: auto` or `none` is allowed
- Thinking blocks with signatures **must** be preserved between tool calls (handled automatically by the Runner)

### Google Gemini

| Feature | Support |
|---------|---------|
| `effort` | `"minimal"`, `"low"`, `"medium"`, `"high"` (Gemini 3 models) |
| `mode` | `"manual"` → `thinkingBudget` (Gemini 2.5) |
| `budget` | 0–32768 tokens (Gemini 2.5); `0` disables (Flash only) |
| `summary` | Not applicable (use `include_thoughts` instead) |
| `include_thoughts` | Maps to `thinkingConfig.includeThoughts` |
| Models | Gemini 3.1 Pro, Gemini 3.0, Gemini 2.5 Pro/Flash |

**Notes:**
- `"minimal"` effort not supported on Gemini 3.1 Pro
- Thinking cannot be disabled on Gemini 3.1 Pro and 2.5 Pro
- Thought signatures are **mandatory** on Gemini 3 function calls (400 error if omitted — handled automatically by the Runner)

## Response Data

When reasoning is enabled, responses include additional fields:

```python
result = await Runner.arun(agent, "Solve this problem...")

# Access via the LLM response in new_items
for item in result.new_items:
    if isinstance(item, dict) and item.get("role") == "assistant":
        # Reasoning content (unified string)
        reasoning = item.get("reasoning_content")

        # Thinking blocks (structured, with signatures)
        blocks = item.get("thinking_blocks")
```

### Token Usage

Reasoning tokens are tracked in usage statistics:

```python
usage = result.context.usage
print(f"Reasoning tokens: {usage.output_tokens_details.reasoning_tokens}")
```

## Streaming

Reasoning content is streamed as `"reasoning_delta"` events:

```python
result = Runner.run(agent, "Solve this...", stream=True, run_config=RunConfig(model="claude-opus-4-6"))

async for event in result.stream_events():
    if event.type == "raw_response_event":
        if event.data.type == "reasoning_delta":
            print(f"[thinking] {event.data.reasoning_delta}", end="")
        elif event.data.type == "content_delta":
            print(event.data.content_delta, end="")
```

## Context Management

Thinking blocks accumulate tokens in conversation history. The framework provides `clear_thinking_blocks()` to remove older thinking content while preserving recent turns:

```python
from troopai.adk.context import ContextConfig

# Keep thinking blocks from the last 2 turns, clear older ones
config = ContextConfig(
    clear_thinking_blocks=True,
    thinking_turns_to_keep=2,
)
```

## How It Works (Internal)

The `Reasoning` config on `LLMConfig` is provider-agnostic. The `LiteLLM` implementation resolves it into litellm-specific parameters via `litellm_reasoning_resolver.py`:

```
LLMConfig.reasoning (Reasoning)
    ↓
resolve_reasoning_params()
    ↓
LiteLLMReasoningParam
    ├── reasoning_effort: str  →  litellm.acompletion(reasoning_effort=...)
    └── thinking: dict         →  litellm.acompletion(thinking=...)
```

litellm then maps these to the appropriate provider-specific API parameters.
