# Context Module

Provider-agnostic context management for conversation history.

## Key Files

- `context_config.py` - Configuration models (`CompactionConfig`, `ContextEditingConfig`, `ContextManagementConfig`, `CacheStrategy`)
- `token_counter.py` - Token estimation via `litellm.token_counter()`
- `context_editing.py` - Clear old tool results and thinking blocks
- `compaction.py` - LLM-based summarisation of older messages
- `context_manager.py` - Orchestrates all strategies
- `directives.py` - LLM-driven context directives (`DropDirective`, `CompactDirective`, `DirectiveStore`, `apply_directives`)

## Architecture

```
ContextManager.prepare_messages()
  1. Context Editing (cheap, no LLM call)
     - Clear old tool results -> placeholder (DEFAULT ON)
     - Clear old thinking blocks -> placeholder
  2. Token Budget Check
     - Warning at configurable threshold (default 80%)
  3. Compaction (expensive, LLM call)
     - Summarise old messages, preserve recent N turns
     - Summary stored as assistant message with _compaction flag
  4. Truncation (opt-in, `truncation=True`)
     - Hard enforcement: drop oldest non-system messages when over budget
     - Last resort after editing + compaction
  5. Pressure Feedback (opt-in, `pressure_feedback=True`)
     - Inject developer message: "[Context budget: 88% used...]"
     - LLM sees the pressure and can use manage_context/save_note
     - Removed when capacity drops below threshold
```

### Context Directives (LLM-driven)

```
apply_directives() — runs BEFORE ContextManager.prepare_messages()
  1. Consume pending directives from DirectiveStore
  2. DropDirective: keep system + last N messages, drop the rest
  3. CompactDirective: invoke ContextCompactor on old messages, keep last N
```

The `JITContextAwareTool.manage_context` tool emits directives; the Runner applies them. LLM strategizes, Runner executes.

See `docs/context/context_management.md` for usage examples.

## CacheStrategy

Controls how disabled/exhausted tools are handled in the tool list:

- `NONE` (default): Remove disabled tools entirely. Smaller payload but invalidates prompt cache.
- `STABLE`: Keep all tools, mark disabled ones as `[UNAVAILABLE]`. Preserves cache prefix for up to 90% savings.

## Key Design Decisions

- **Provider-agnostic**: Compaction routes through the `LLM` ABC (`llm.acomplete()`) so usage lands in `RunContext.usage` and middleware sees the call; `litellm.token_counter()` is used for name-based token counting only
- **No blind eviction**: Session stores everything; context management only controls what goes to the LLM
- **Layered strategy**: Editing first (cheap), compaction only when needed (expensive)
- **Opt-in tool result clearing**: `clear_tool_results=False` by default — developers opt INTO observation masking per the cost-conservative-defaults rule (silent data mutation of LLM inputs must be opted into, never opted out of)
- **Compaction stored as assistant message**: `{"role": "assistant", "content": summary, "_compaction": True}`

Framework-wide cost levers that interact with context management (JSON minification, `FunctionTool.max_result_tokens`) are defined in `.claude/rules/architecture.md`.
