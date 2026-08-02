# Memory Module

Extracted, searchable knowledge that persists across sessions. Distinct from Session (chronological log): Memory stores semantic knowledge, not conversation history.

## Key Files

- `memory.py` — `Memory` ABC with `add`, `search`, `get`, `delete`, `clear`, `add_from_session`
- `memory_types.py` — `MemoryEntry`, `MemoryMetadata`, `MemorySource`, `MemorySearchResult`, `MemorySearchFilter`
- `memory_config.py` — `MemoryConfig` for Runner integration (injection + extraction)
- `MemoryTool`/`RememberMemoryTool`/`RecallMemoryTool`/`ForgetMemoryTool` — agent-facing tools (in `tools/builtin/memory_tool.py`); backend-agnostic over any `Memory`.
- `extractor.py` — `MemoryExtractor` ABC + `LLMExtractor` (opt-in, costs tokens)
- `in_memory.py` — `TemporaryMemory` (keyword-based, for prototyping)
- `sqlite_memory.py` — `SQLiteMemory` (FTS5, for production)
- `vector_store.py` — `VectorStore` Protocol + `VectorRecord`/`VectorQueryResult` (client-side; namespace = metadata filter)
- `vector_memory.py` — `VectorMemory(Memory)` — composes `Embedder` + `VectorStore`
- `stores/` — `InMemoryVectorStore` (baseline) + `PgVectorStore`/`PineconeVectorStore`/`ChromaVectorStore`/`QdrantVectorStore` (optional extras)
- `distill.py` — `distill_to_semantic` (explicit episodic→semantic distillation; developer-invoked, never background)

The `Embedder` ABC lives in `llms/embedder.py` (not on `Memory`) — embeddings are a distinct capability from memory storage and from chat completion.

## Architecture Decisions

| Decision | What | Why |
|----------|------|-----|
| **Session ≠ Memory** | Separate modules with separate storage | Session = chronological log, Memory = semantic knowledge |
| **Namespace always explicit** | No hidden default namespace | Developer controls scoping (e.g. `"user:123"`, `"agent:support"`) |
| **Injection is opt-in** | `MemoryConfig.inject=False` default | No hidden token costs |
| **Extraction is opt-in** | `MemoryConfig.auto_extract=False` default | LLM extraction costs tokens |
| **MemoryTool follows JITContextAwareTool pattern** | `BuiltinTool` → expands to `FunctionTool` list | Consistent tool expansion pattern |
| **Two keyword backends** | TemporaryMemory (keyword) + SQLiteMemory (FTS5) | Prototyping vs. production |
| **Semantic memory** | `VectorMemory` satisfies the `Memory` ABC via `Embedder` + `VectorStore` | Vector search is additive; Runner + tools unchanged |
| **Namespace = metadata filter** | `VectorStore.get(id)` is namespace-free | Record ids are global; avoids double-keying |
| **Distillation is explicit** | `distill_to_semantic` is developer-invoked only | Embedding + LLM extraction costs are always chosen explicitly |

## Flow

```
Session → (extraction) → Memory → (injection) → Context → LLM
```

- **Injection**: After session loading, before guardrails in Runner
- **Extraction**: After session saving in Runner

## Runner Integration

Pass `memory=MemoryConfig(...)` to `Runner.run()`/`Runner.arun()` or use
`Runner.configure().agent(agent).memory(...)`.

See `docs/memory/memory.md` for general usage.
See `docs/memory/vector_stores.md` for the vector layer (VectorMemory, backends, episodic/semantic).
See `examples/memory/` for examples.
