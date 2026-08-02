# Prompts Module

Structured system prompt definitions for agents.

## Key Files

- `system_prompt.py` - `SystemPrompt`, `SystemPromptTone`, `DynamicSystemPrompt`

## SystemPrompt

A Pydantic `BaseModel` that captures system prompt characteristics as discrete fields
and renders them into a single prompt string via `generate()`.

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | str | Yes | Core identity and expertise |
| `context` | str | No | Background or domain context |
| `guidelines` | list[str] | No | Behavioral guidelines |
| `tone` | SystemPromptTone \| str | No | Response tone |
| `constraints` | list[str] | No | Hard constraints |
| `examples` | list[str] | No | Few-shot examples |
| `output_format` | str | No | Formatting instructions |
| `knowledge` | str | No | Domain-specific facts |

## SystemPromptTone

A `StrEnum` with predefined tone values:

| Value | Description |
|-------|-------------|
| `FORMAL` | Polished, no contractions or slang |
| `INFORMAL` | Relaxed, casual phrasing |
| `TECHNICAL` | Precise, domain-specific terminology |
| `CONVERSATIONAL` | Natural dialogue-style |
| `FRIENDLY` | Warm, approachable |
| `PROFESSIONAL` | Business-appropriate clarity |

Custom strings are also accepted: `tone="friendly but precise"`.

## DynamicSystemPrompt

Type alias for callables that receive a `DynamicSystemPromptData` bundle, returning a system prompt dynamically.

The callable receives a single `DynamicSystemPromptData` with:
- `data.context: RunContext` -- the execution context (carries user-provided context and usage metrics)
- `data.agent: Agent` -- the agent instance being run

Supports sync and async callables.

## Agent Integration

The `Agent.system_prompt` field accepts `str`, `SystemPrompt`, or `DynamicSystemPrompt`.

The `Runner` resolves the prompt via `_resolve_system_prompt(agent, ctx_wrapper)` before building LLM messages.
The callable is invoked with `DynamicSystemPromptData(context=ctx_wrapper, agent=agent)`.

See `docs/prompts/system_prompt.md` for usage examples.
