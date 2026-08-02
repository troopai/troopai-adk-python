# Memory Module

The memory module provides extracted, searchable knowledge that persists across sessions.

## Session vs Memory

| | Session | Memory |
|---|---|---|
| **Purpose** | Chronological conversation log | Semantic knowledge store |
| **Scope** | Single conversation thread | Cross-session |
| **Access** | Append-only, get by recency | Search by relevance |
| **Content** | Raw messages and actions | Extracted facts and preferences |

The relationship: `Session → (extraction) → Memory → (injection) → Context → LLM`

## Quick Start

### TemporaryMemory (prototyping)

```python
from troopai.adk.memory import TemporaryMemory

memory = TemporaryMemory()
entry = await memory.add("User prefers dark mode", namespace="user:123")
results = await memory.search("dark mode", namespace="user:123")
```

### SQLiteMemory (production)

```python
from troopai.adk.memory import SQLiteMemory

memory = SQLiteMemory(path="memory.db")
entry = await memory.add("User prefers dark mode", namespace="user:123")
results = await memory.search("dark mode", namespace="user:123")
await memory.close()
```

## Runner Integration

### Auto-injection

Inject relevant memories into the agent's context before execution:

```python
from troopai.adk import Agent, Runner
from troopai.adk.memory import TemporaryMemory, MemoryConfig

memory = TemporaryMemory()
# Pre-populate with knowledge
await memory.add("User prefers concise answers", namespace="user:1")

config = MemoryConfig(
    memory=memory,
    namespace="user:1",
    inject=True,          # Auto-inject before agent loop
    inject_limit=5,       # Max memories to inject
)

result = await Runner.arun(agent, "Hello!", memory=config)
```

### Auto-extraction

Extract knowledge from conversations after each run:

```python
from troopai.adk.memory import MemoryConfig, LLMExtractor

config = MemoryConfig(
    memory=memory,
    namespace="user:1",
    auto_extract=True,
    extractor=LLMExtractor(model="gpt-4o-mini"),
)

result = await Runner.arun(agent, "I prefer dark mode", memory=config)
# Memory now contains: "User prefers dark mode"
```

### Profile Runner API

```python
result = await (
    Runner.configure()
    .agent(agent)
    .memory(
        MemoryConfig(
            memory=memory,
            namespace="user:1",
            inject=True,
            auto_extract=True,
            extractor=LLMExtractor(),
        )
    )
    .arun("Hello!")
)
```

## MemoryTool (agent-facing)

Give agents explicit tool access to memory:

```python
from troopai.adk import Agent
from troopai.adk.memory import TemporaryMemory, MemoryTool

memory = TemporaryMemory()
agent = Agent(
    name="Assistant",
    system_prompt="Use remember/recall to manage long-term memory.",
    tools=[MemoryTool(memory=memory, namespace="user:1")],
)
```

This expands into three tools:
- `remember(content, importance?, categories?)` — store a memory
- `recall(query, limit?)` — search memories
- `forget(memory_id)` — delete a memory

## Custom Extractors

Implement the `MemoryExtractor` protocol:

```python
from troopai.adk.memory import MemoryExtractor, ExtractionResult

class RuleBasedExtractor:
    async def extract(self, messages, *, namespace):
        results = []
        for msg in messages:
            if isinstance(msg, dict) and "prefer" in str(msg.get("content", "")):
                results.append(ExtractionResult(
                    content=msg["content"],
                    importance=4,
                    categories=("preference",),
                ))
        return results
```

## Metadata & Filtering

```python
from troopai.adk.memory import MemoryMetadata, MemorySource, MemorySearchFilter

# Add with metadata
await memory.add(
    "User prefers dark mode",
    namespace="user:1",
    metadata=MemoryMetadata(
        source=MemorySource.MANUAL,
        importance=5,
        categories=("preference", "ui"),
        agent_name="onboarding",
    ),
)

# Search with filters
results = await memory.search(
    "preferences",
    namespace="user:1",
    filter=MemorySearchFilter(
        importance=3,           # Minimum importance
        categories=["preference"],
        agent_name="onboarding",
    ),
)
```

## Injection Positions

Control where memories appear in the prompt:

```python
from troopai.adk.memory import MemoryConfig, MemoryInjectionPosition

# As a developer message (default)
config = MemoryConfig(
    memory=memory,
    namespace="user:1",
    inject=True,
    inject_position=MemoryInjectionPosition.DEVELOPER_MESSAGE,
)

# As a suffix to the system prompt
config = MemoryConfig(
    memory=memory,
    namespace="user:1",
    inject=True,
    inject_position=MemoryInjectionPosition.SYSTEM_SUFFIX,
)
```
