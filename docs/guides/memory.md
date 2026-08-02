(guides/memory)=

# 🧠 Memory

Agents that matter across conversations need memory. This guide walks through
the full memory stack in this ADK: the four layers, when each applies,
how to choose and wire a backend, and how episodic turns become durable
semantic facts.

## Memory layers at a glance

Four words describe what "memory" means at different scopes. They are not
interchangeable.

| Layer | Lifetime | Shape | When to reach for it |
|-------|----------|-------|----------------------|
| **Context** | One run | Developer object passed through `RunContext` | Per-request state shared between tools and hooks |
| **Sessions** | Across runs, same identity | `Session` rows (SQLite-backed by default) | Continuity for a returning user: "pick up where we left off" |
| **Episodic memory** | Across runs, identity-scoped | `MemoryEntry` records (`MemoryKind.EPISODIC`) | Facts the agent *experienced* — user preferences, past decisions, recent turns |
| **Semantic memory** | Across runs, similarity-indexed | `MemoryEntry` records (`MemoryKind.SEMANTIC`) + embedding vectors | Facts the agent can *recall* by topic across users or domains |

Full compare-and-contrast lives in {ref}`concepts/index` (Memory layers section).

---

## Episodic memory

Episodic memory stores **raw or lightly-processed interaction content** scoped
to an identity. It is the agent's log of what happened.

### Core types

`MemoryEntry` is the atom.  Every stored fact is a `MemoryEntry`:

```python
from troopai.adk.memory import MemoryEntry, MemoryMetadata, MemoryKind, MemorySource
```

`MemoryMetadata` travels with every entry and controls how the entry is
found, filtered, and weighted:

```python
meta = MemoryMetadata(
    source=MemorySource.MANUAL,   # MANUAL | EXTRACTION | TOOL
    importance=4,                  # 1 (low) … 5 (critical)
    categories=("preferences",),
    kind=MemoryKind.EPISODIC,      # default; no need to set explicitly
    session_id="sess:abc",
    agent_name="support",
)
```

`MemoryKind` has exactly two values: `EPISODIC` (raw / lightly-processed)
and `SEMANTIC` (distilled, durable).  The kind field is what separates the
two layers — both live in the same `Memory` backend.

### Backends for episodic use

| Class | Module | Best for |
|-------|--------|----------|
| `TemporaryMemory` | `memory.in_memory` | Prototyping, unit tests; keyword search; no persistence |
| `SQLiteMemory` | `memory.sqlite_memory` | Production episodic storage; full-text FTS5 search; file-backed |

Both implement the same `Memory` ABC, so you swap them without touching
the rest of your code.

```python
from troopai.adk.memory import TemporaryMemory, SQLiteMemory

# Prototype
proto_memory = TemporaryMemory()

# Production
prod_memory = SQLiteMemory(path="agent_memory.db")
await prod_memory.setup()  # creates tables on first call
```

### Identity scoping via namespaces

The `namespace` parameter is always explicit — the ADK never injects a
default.  Use it to scope every operation to the right identity:

```python
await memory.add("User prefers Berlin time zone", namespace="user:42")
results = await memory.search("timezone preference", namespace="user:42")
```

A namespace is any string: `"user:42"`, `"org:acme"`, `"agent:billing"`.
The framework stores and filters by it; multi-tenant isolation follows
from consistent namespace choice.

---

## Semantic memory

Semantic memory stores **distilled, durable facts** indexed by similarity.
A user's city becomes a searchable vector rather than a keyword row.  The
ADK models this through two collaborating abstractions: `VectorStore` and
`VectorMemory`.

### VectorStore Protocol

`VectorStore` is a `typing.Protocol` — not an ABC.  Each backend satisfies
the contract structurally; no base-class import is required.  The protocol
is intentionally minimal:

```
upsert(records)  →  None
query(vector, namespace, k, filter)  →  list[VectorQueryResult]
get(record_id)  →  VectorRecord | None
delete(ids)  →  int
clear(namespace)  →  int
close()  →  None
```

Five vendor backends are supported, each with its own SDK, with zero coupling
required between them.

### VectorMemory bridge

`VectorMemory` is a `Memory` subclass that composes an `Embedder` and a
`VectorStore`.  Because it satisfies the `Memory` ABC, the Runner,
`MemoryConfig`, and the recall/remember tools use it unchanged — no new
wiring is needed.

```python
from troopai.adk.memory import VectorMemory
from troopai.adk.memory.stores.in_memory import InMemoryVectorStore
from troopai.adk.llms.litellm.litellm_embedder import LiteLLMEmbedder

memory = VectorMemory(
    store=InMemoryVectorStore(),
    embedder=LiteLLMEmbedder(model="text-embedding-3-small"),
)

await memory.add("User prefers dark mode", namespace="user:42")
results = await memory.search("UI preferences", namespace="user:42")
for r in results:
    print(r.score, r.entry.content)  # score 0.0–1.0
```

### VectorStore backends

Five backends ship out of the box:

| Class | Module path | Use case | Extra dependency |
|-------|-------------|----------|-----------------|
| `InMemoryVectorStore` | `memory.stores.in_memory` | Tests, prototyping; cosine similarity in Python | None |
| `PgVectorStore` | `memory.stores.pgvector` | Managed Postgres deployments; SQL filtering | `psycopg[binary]`, `pgvector` |
| `PineconeVectorStore` | `memory.stores.pinecone` | Serverless managed vector search | `pinecone-client` |
| `ChromaVectorStore` | `memory.stores.chroma` | Local or self-hosted open-source vector DB | `chromadb` |
| `QdrantVectorStore` | `memory.stores.qdrant` | High-performance self-hosted or cloud | `qdrant-client` |

All five implement the `VectorStore` Protocol.  Swap backends by replacing
the `store=` argument on `VectorMemory` — no other code changes.

```python
from troopai.adk.memory.stores.pgvector import PgVectorStore

store = PgVectorStore(dsn="postgresql://user:pass@localhost/db")
await store.setup()   # creates the pgvector table once

memory = VectorMemory(store=store, embedder=embedder)
```

---

## Embedders

An `Embedder` turns text into a vector.  The `Embedder` ABC lives in
`llms/embedder.py` — distinct from the `LLM` ABC because not every
provider offers embeddings (Anthropic ships none).

```
aembed_documents(texts: list[str])  →  list[Embedding]
aembed_query(text: str)  →  Embedding
dimensions  →  int | None
```

The `aembed_documents` / `aembed_query` split matters for asymmetric
models that encode documents and queries differently.  For symmetric models
`aembed_query` delegates to `aembed_documents`; override only when needed.

`LiteLLMEmbedder` is the default implementation, backed by litellm's
100+ provider routing:

```python
from troopai.adk.llms.litellm.litellm_embedder import LiteLLMEmbedder

embedder = LiteLLMEmbedder(model="text-embedding-3-small")
# or OpenAI large, Cohere, Bedrock Titan, etc. — any litellm embedding model
```

`Embedding` carries the vector as `tuple[float, ...]` and the producing
`model` string.  An optional `EmbeddingLRUCache` provides bounded, thread-safe
caching of deterministic lookups — it is opt-in and never wired by default.

---

## The distill step

`distill_to_semantic` is the only sanctioned bridge between the two layers.
It extracts semantic facts from episodic content and stores them with
`MemoryKind.SEMANTIC`.

```python
from troopai.adk.memory import (
    VectorMemory, MemoryKind, MemoryMetadata, MemorySource,
    MemorySearchFilter, distill_to_semantic,
)
from troopai.adk.memory.extractor import LLMExtractor
from troopai.adk.memory.stores.in_memory import InMemoryVectorStore
from troopai.adk.llms.litellm.litellm_model import LiteLLM
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.llms.litellm.litellm_embedder import LiteLLMEmbedder

memory = VectorMemory(
    store=InMemoryVectorStore(),
    embedder=LiteLLMEmbedder(model="text-embedding-3-small"),
)

# 1. Store raw episodic turns.
for turn in ["I just moved to Berlin.", "I'm vegetarian.", "I prefer trains."]:
    await memory.add(
        turn,
        namespace="user:7",
        metadata=MemoryMetadata(source=MemorySource.MANUAL, kind=MemoryKind.EPISODIC),
    )

# 2. Retrieve the episodic entries to distill.
episodic = await memory.search("user", namespace="user:7", limit=20)

# 3. Distill — explicit, opt-in, costs tokens.
facts = await distill_to_semantic(
    [r.entry for r in episodic],
    into=memory,
    extractor=LLMExtractor(
        llm=LiteLLM(model="gpt-4o-mini"),
        llm_config=LLMConfig(temperature=0.0),
    ),
    namespace="user:7",
)

# 4. Recall semantic facts by topic.
results = await memory.search(
    "dietary and travel preferences",
    namespace="user:7",
    filter=MemorySearchFilter(kind=MemoryKind.SEMANTIC),
)
```

The distill step:
- calls `MemoryExtractor.extract()` over the source content,
- deduplicates facts within the call by content hash and cross-call by
  similarity search,
- stores each new fact with `MemoryKind.SEMANTIC`.

**The framework never schedules distillation automatically.** Embedding +
LLM extraction cost is always explicitly chosen.

---

## Reading from memory

### Agent-side injection

Pass a `MemoryConfig` to `Runner.arun()` to inject retrieved memories
automatically before the agent loop:

```python
from troopai.adk.memory import MemoryConfig, MemoryInjectionPosition

config = MemoryConfig(
    memory=memory,
    namespace="user:42",
    inject=True,           # off by default
    inject_limit=5,        # cap token cost
    inject_position=MemoryInjectionPosition.DEVELOPER_MESSAGE,
)

result = await Runner.arun(agent, user_input, memory=config)
```

`inject=False` is the default — no hidden token cost.

`MemoryInjectionPosition` controls placement: `DEVELOPER_MESSAGE` (before
the user prompt, as a developer-role message) or `SYSTEM_SUFFIX` (appended
to the system prompt).

### Tool-side explicit recall

Give the agent a `RecallMemoryTool` so it can query memory on demand during
a turn:

```python
from troopai.adk.tools import RecallMemoryTool, RememberMemoryTool, ForgetMemoryTool

tools = [
    RecallMemoryTool(memory=memory, namespace="user:42"),
    RememberMemoryTool(memory=memory, namespace="user:42"),
    ForgetMemoryTool(memory=memory, namespace="user:42"),
]

agent = Agent(name="Assistant", system_prompt="...", tools=tools)
```

The three tools map to `memory.search`, `memory.add`, and `memory.delete`.
All three work over any `Memory` backend — `TemporaryMemory`, `SQLiteMemory`,
or `VectorMemory` — without change.

---

## Writing to memory

### Implicit: per-turn capture via Runner

Set `auto_extract=True` on `MemoryConfig` to capture knowledge after every
run.  An `extractor` is required:

```python
from troopai.adk.memory import MemoryConfig, LLMExtractor
from troopai.adk.llms.litellm.litellm_model import LiteLLM
from troopai.adk.llms.llm_config import LLMConfig

config = MemoryConfig(
    memory=memory,
    namespace="user:42",
    auto_extract=True,
    extractor=LLMExtractor(
        llm=LiteLLM(model="gpt-4o-mini"),
        llm_config=LLMConfig(temperature=0.0),
    ),
)
```

`auto_extract=False` is the default.  Opt in deliberately — extraction
costs one LLM call per run.

### Explicit: direct add

Write entries directly when your application logic, not an LLM, knows what
to persist:

```python
await memory.add(
    "User confirmed their account email is confirmed",
    namespace="user:42",
    metadata=MemoryMetadata(
        source=MemorySource.MANUAL,
        importance=5,
        categories=("account",),
    ),
)
```

`MemorySource.TOOL` is set automatically when the agent invokes
`RememberMemoryTool`; `MemorySource.EXTRACTION` when `add_from_session`
or `auto_extract` pipelines store facts.

---

## Common patterns

### Per-user episodic + shared-knowledge semantic

Run two separate memory objects in the same agent:

```python
user_memory = SQLiteMemory(path="users.db")    # episodic, per-user
domain_memory = VectorMemory(                   # semantic, shared
    store=PgVectorStore(dsn=PGVECTOR_DSN),
    embedder=LiteLLMEmbedder(model="text-embedding-3-small"),
)

# Inject from both before each run.
user_config   = MemoryConfig(memory=user_memory,   namespace=f"user:{uid}", inject=True)
domain_config = MemoryConfig(memory=domain_memory, namespace="domain:global", inject=True)
```

The `Runner.arun()` `memory=` parameter accepts a single `MemoryConfig`; for
multiple sources, inject manually before the run or wrap them in a composite
`Memory` subclass.

### Multi-tenant memory isolation

Namespaces are the isolation boundary.  Keep them consistent and never
search across namespace boundaries:

```python
# Tenant A
await memory.add("...", namespace="tenant:acme")

# Tenant B — completely separate at the query level
await memory.add("...", namespace="tenant:globex")

# Correct: scoped to one tenant
results = await memory.search("...", namespace="tenant:acme")
```

No cross-tenant bleed is possible as long as the namespace string is correct
— the filter is applied inside every backend before results are returned.

### Archive and forget

To archive a user's history before deletion, export it first:

```python
# Archive: search broadly, write elsewhere.
all_entries = await memory.search("", namespace="user:42", limit=1000)

# Forget: clear the entire namespace.
deleted_count = await memory.clear(namespace="user:42")

# Delete a single entry by id.
await memory.delete(entry_id)
```

`clear()` returns the number of deleted entries.  `delete()` returns `True`
if the entry existed.

---

## See also

- {ref}`concepts/index` — Memory layers section: context vs sessions vs episodic vs semantic
- `examples/memory/` — runnable examples: `basic_memory.py`, `vector_memory.py`, `episodic_semantic.py`, `persistent_memory.py`, `session_and_memory.py`
- `src/troopai/adk/memory/` — full source for the module
