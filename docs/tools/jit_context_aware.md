# JIT Context Aware Tool

Active context management for AI agents — the LLM manages its own context window via tools.

## Overview

`JITContextAwareTool` is a built-in tool (subclasses `BuiltinTool`) that gives agents active control over their context. Instead of passive threshold-based management (compaction, editing), the LLM decides when to save notes, search history, and monitor its budget.

Inspired by Anthropic's [Just-in-Time context](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) pattern.

## Quick Start

```python
from troopai.adk.agents import Agent
from troopai.adk.tools import JITContextAwareTool

agent = Agent(
    name="Research Assistant",
    system_prompt="Use save_note to preserve key findings.",
    tools=[JITContextAwareTool()],
)
```

The tool expands into 4 focused `FunctionTool` instances at runtime:

| Tool | Purpose | Context Type |
|------|---------|-------------|
| `save_note` | Store findings/decisions | `ToolContext` |
| `recall_notes` | Retrieve stored notes | `ToolContext` |
| `search_history` | Search conversation history | `HistoryAwareToolContext` |
| `context_stats` | Token usage and budget info | `ExecutionAwareToolContext` |

## Configuration

```python
JITContextAwareTool(
    max_notes=50,              # Max notes before LRU eviction
    default_importance=3,      # Default priority (1-5)
    include_stats=True,        # Include context_stats tool
    include_history_search=True,  # Include search_history tool
    note_store=None,           # Pluggable backend (default: InMemoryNoteStore)
)
```

## NoteStore Protocol

Storage is pluggable via the `NoteStore` protocol:

```python
class NoteStore(Protocol):
    def save(self, key: str, content: str, importance: int, turn: int) -> NoteEntry: ...
    def recall(self, query: Optional[str] = None) -> list[NoteEntry]: ...
    def delete(self, key: str) -> bool: ...
    def count(self) -> int: ...
    def keys(self) -> list[str]: ...
```

**Current:** `InMemoryNoteStore` — persists for a single `Runner.arun()` call.

**Not yet implemented:** `FileNoteStore` — writes to disk (like Claude Code's NOTES.md pattern). `SessionNoteStore` — backed by the framework's Session interface.

## HistoryAwareToolContext

A new level in the ToolContext hierarchy:

```
ToolContext → ExecutionAwareToolContext → HistoryAwareToolContext
```

`HistoryAwareToolContext` adds `history: tuple[RunItem, ...]` — a read-only, frozen snapshot of the conversation as Layer 3 RunItems (not raw wire types). The tools_executor converts messages at the boundary, preserving the three-layer type system.

Tools opt in via `history_aware=True` on `FunctionTool` or by annotating their first parameter as `HistoryAwareToolContext` with `@function_tool`.

## How It Works

1. Developer adds `JITContextAwareTool()` to an agent's tools
2. Runner's `build_tools()` detects it and calls `tool.build_tools()`
3. Generated `FunctionTool` instances are registered on the agent
4. During execution, tools_executor builds the appropriate context type
5. The LLM calls tools as needed — no auto-injection (JIT philosophy)

Notes survive context compaction because they're stored externally. The LLM retrieves them when needed via `recall_notes`.

## Examples

See `examples/tools/jit_context_aware.py`.
