(rag/document_search)=

# Document search

`DocumentSearchTool` gives an agent a `search(query)` capability over a corpus
of documents you curate. It loads each source, splits it into bounded chunks,
embeds them, and retrieves the most relevant passages — grounding the agent's
answers in your documents.

The whole pipeline is built on the framework's existing
{class}`~troopai.adk.llms.embedder.Embedder` and
{class}`~troopai.adk.memory.vector_store.VectorStore` abstractions, so any vector
backend (in-memory, pgvector, Chroma, Pinecone, Qdrant) works unchanged.

## Quick start

Bind your sources at construction; the agent only supplies the query.

```python
from troopai.adk import Agent, Runner
from troopai.adk.llms.litellm.litellm_embedder import LiteLLMEmbedder
from troopai.adk.tools import PDFSearchTool

search = PDFSearchTool(
    sources=["handbook.pdf", "policies.pdf"],
    embedder=LiteLLMEmbedder(model="text-embedding-3-small"),
)

agent = Agent(
    name="Docs Assistant",
    system_prompt="Answer using pdf_search and cite the source.",
    tools=[search],
)
result = await Runner.arun(agent, "What is the refund window?")
```

The corpus is indexed lazily on the first search (or call `await
search.index()` to pre-warm and pay the embedding cost up front). Indexing is
idempotent — it runs once.

## The tool family

One core tool does the work; the typed wrappers pin a loader and tailor the
description shown to the model. They add no pipeline of their own.

| Tool | Source type | Extra |
|---|---|---|
| `DocumentSearchTool` | any (loader auto-dispatched per source) | core |
| `TXTSearchTool` / `MarkdownSearchTool` | `.txt` / `.md` | core |
| `CSVSearchTool` / `JSONSearchTool` | `.csv` / `.json` | core |
| `DirectorySearchTool` | a directory tree | core |
| `PDFSearchTool` | `.pdf` | `rag-pdf` |
| `DOCXSearchTool` | `.docx` | `rag-docx` |
| `WebsiteSearchTool` | http(s) page | `rag-web` |
| `GithubSearchTool` | repository URL | `rag-github` |
| `YoutubeVideoSearchTool` / `YoutubeChannelSearchTool` | video / channel URL | `rag-youtube` |

Install a loader's optional dependency with its extra, e.g. `pip install
'troopai-adk-python[rag-pdf]'` (or `[rag]` for all loaders). A loader verifies its
package at construction and raises a clear `ImportError` with the extra to
install if it is missing.

Pass `DocumentSearchTool` a mixed `sources` list to let it route each source to
the right loader by extension or URL shape:

```python
from troopai.adk.tools import DocumentSearchTool

tool = DocumentSearchTool(
    sources=["guide.pdf", "notes.md", "https://example.com/faq"],
    embedder=LiteLLMEmbedder(model="text-embedding-3-small"),
)
```

## Configuration

| Field | Default | Purpose |
|---|---|---|
| `embedder` | *required* | Embeds chunks and queries. No default — embedding spends tokens. |
| `vector_store` | in-memory | Backend for chunk vectors. Use a persistent store to keep an index across runs. |
| `chunker` | `TextChunker()` | Bounded recursive splitter; tune `chunk_size` / `chunk_overlap`. |
| `namespace` | `"documents"` | Scopes this corpus's chunks; lets one store hold several corpora. |
| `default_limit` | `5` | Passages returned when the model omits `limit`. |

## Using the index directly

For retrieval without an agent, use {class}`~troopai.adk.rag.index.DocumentIndex`
with the loaders directly:

```python
from troopai.adk.rag import DocumentIndex
from troopai.adk.rag.loaders import resolve_loader

index = DocumentIndex(embedder=LiteLLMEmbedder(model="text-embedding-3-small"))
docs = await resolve_loader("guide.pdf").load("guide.pdf")
await index.add_documents(docs)
hits = await index.search("installation steps", limit=3)
```

## Cost and safety

- **Embedding is explicit.** The tool requires an `Embedder`; nothing is
  embedded at import or construction — only on the first search.
- **Ephemeral by default.** Without a `vector_store`, chunks live in memory and
  vanish with the process. Supply a persistent backend to retain them.
- **Bounded ingestion.** `GithubLoader.max_files`, `YoutubeChannelLoader.max_videos`,
  and the chunker bounds cap how much one source can add to the index.
- **Sources are yours.** Because the corpus is fixed at construction and the
  LLM only supplies the query, the tool never opens an attacker-named file or
  URL.

See `examples/rag/document_search.py` for a runnable end-to-end example.
