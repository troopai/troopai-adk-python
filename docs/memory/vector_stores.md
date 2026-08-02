# Vector Stores

The vector layer adds embedding-based semantic search to the memory system.
Conceptually it has three pieces: an `Embedder` (content/query → vector),
a `VectorStore` backend (stores and queries vectors), and `VectorMemory`
(a `Memory` implementation that composes both). Because `VectorMemory`
satisfies the existing `Memory` ABC, the Runner's injection/extraction
pipeline and the built-in `RememberMemoryTool`/`RecallMemoryTool`/`ForgetMemoryTool`
work over it unchanged.

## Layering

```
Embedder  →  embed text/query
               ↓
VectorStore  →  upsert / query / get / delete / clear
               ↓
VectorMemory(Memory)  →  add / search / get / delete / clear
               ↓
Runner + RecallMemoryTool / RememberMemoryTool (unchanged)
```

**`Embedder` ABC** (`llms/embedder.py`) — two async methods: `aembed_documents`
(batch, for storing) and `aembed_query` (single, for searching). The only
concrete implementation shipped is `LiteLLMEmbedder`.

**`VectorStore` Protocol** (`memory/vector_store.py`) — a `runtime_checkable`
Protocol with `upsert`, `query`, `get`, `delete`, `clear`, `close`. Namespace
is modeled as a metadata filter, not a physical partition: record ids are
global, so `get(memory_id)` requires no namespace (matching `Memory.get`).

**`VectorMemory`** (`memory/vector_memory.py`) — bridges `Embedder` and
`VectorStore` into the `Memory` ABC. `add` embeds the content as a document
and upserts; `search` embeds the query and queries the store.

## Choosing a Backend

| Backend | Install extra | Notes |
|---|---|---|
| `InMemoryVectorStore` | *(none)* | Zero-dep baseline; O(N) cosine; not persistent |
| `PgVectorStore` | `troopai-adk-python[memory-pgvector]` | Needs Postgres with the `pgvector` extension |
| `PineconeVectorStore` | `troopai-adk-python[memory-pinecone]` | Hosted (Pinecone cloud); no local option |
| `ChromaVectorStore` | `troopai-adk-python[memory-chroma]` | Can run embedded/in-process or as a server |
| `QdrantVectorStore` | `troopai-adk-python[memory-qdrant]` | Can run embedded/in-process or as a server |

For development and testing, `InMemoryVectorStore` requires no external
process. For production, pgvector and Qdrant support embedded modes;
Chroma also supports a local persistent mode; Pinecone is always hosted.

## Wiring VectorMemory

```python
from troopai.adk.llms.litellm.litellm_embedder import LiteLLMEmbedder
from troopai.adk.memory import MemoryConfig, VectorMemory
from troopai.adk.memory.stores.in_memory import InMemoryVectorStore
from troopai.adk.tools import RecallMemoryTool, RememberMemoryTool

memory = VectorMemory(
    store=InMemoryVectorStore(),
    embedder=LiteLLMEmbedder(model="text-embedding-3-small"),
)

# Use it directly:
entry = await memory.add("User prefers dark mode.", namespace="user:42")
results = await memory.search("UI preferences", namespace="user:42")

# Or give agents explicit tool access:
tools = [
    RememberMemoryTool(memory=memory, namespace="user:42"),
    RecallMemoryTool(memory=memory, namespace="user:42"),
]

# Or configure the Runner to auto-inject before the agent loop:
config = MemoryConfig(memory=memory, namespace="user:42", inject=True)
```

See `examples/memory/vector_memory.py` for a runnable example.

## Episodic and Semantic Memory

`MemoryKind` (on `MemoryMetadata`) distinguishes two kinds of entries:

- `MemoryKind.EPISODIC` — raw or lightly-processed interaction content.
- `MemoryKind.SEMANTIC` — distilled, durable knowledge (typically facts
  extracted from episodic entries).

Episodic entries are written with the default `MemoryKind.EPISODIC`. Semantic
entries are produced by calling `distill_to_semantic` explicitly — the
framework never calls it automatically.

### Distilling Episodic to Semantic

`distill_to_semantic` is intentionally developer-invoked: every call costs
embedding tokens (per fact) plus one LLM extraction call.

```python
from troopai.adk.memory import (
    MemoryKind,
    MemorySearchFilter,
    distill_to_semantic,
)
from troopai.adk.memory.extractor import LLMExtractor
from troopai.adk.llms.litellm.litellm_model import LiteLLM
from troopai.adk.llms.llm_config import LLMConfig

# Retrieve episodic entries to distill:
episodic = await memory.search("user", namespace="user:42", limit=20)

facts = await distill_to_semantic(
    [r.entry for r in episodic],
    into=memory,
    extractor=LLMExtractor(
        llm=LiteLLM(model="gpt-4o-mini"),
        llm_config=LLMConfig(temperature=0.0),
    ),
    namespace="user:42",
)

# Query semantic-only results:
semantic = await memory.search(
    "dietary preferences",
    namespace="user:42",
    filter=MemorySearchFilter(kind=MemoryKind.SEMANTIC),
)
```

See `examples/memory/episodic_semantic.py` for a runnable example.

## Cross-Session Memory

Pass a stable `namespace` (e.g. `"user:<id>"`) to scope memory per user
across sessions. All memory calls that share that key read and write the same
pool of records.

```python
user_ns = f"user:{user_id}"
config = MemoryConfig(memory=memory, namespace=user_ns, inject=True)
```

## Embedding Cache (Cost Lever)

Embedding is deterministic per (model, text). `EmbeddingLRUCache` caches
results in memory, bounded to `max_size` entries. Pass it when constructing
the embedder:

```python
from troopai.adk.llms.embedder import EmbeddingLRUCache
from troopai.adk.llms.litellm.litellm_embedder import LiteLLMEmbedder

embedder = LiteLLMEmbedder(
    model="text-embedding-3-small",
    cache=EmbeddingLRUCache(max_size=2048),
)
```

This is the only embedding cost-reduction lever. It is opt-in.

## Framework Memory vs. Provider-Hosted Vector Stores

`VectorMemory` + `VectorStore` is **client-side, framework-owned** memory:
vectors are stored in a backend of your choice and queried directly by the
framework. This is entirely separate from `FileSearchTool.vector_store_ids`,
which references **provider-hosted** vector stores that are searched
server-side by the LLM provider. The two systems are independent and do not
interact.
