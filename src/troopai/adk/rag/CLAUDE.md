# RAG Module

Retrieval-augmented generation primitives: turn documents into a semantically
searchable index, built on the existing `Embedder` + `VectorStore`
abstractions. Powers the agent-facing `DocumentSearchTool` family (in
`tools/builtin/document_search_tool.py`).

## Key Files

- `document.py` — `LoadedDocument` (a loaded text span + provenance),
  `DocumentSearchHit` (a search result). Decoupled from `MemoryMetadata`.
- `chunking.py` — `TextChunker`, a bounded recursive character splitter
  (stdlib-only; sizes in characters, not tokens).
- `index.py` — `DocumentIndex`: chunk → embed (batched) → upsert → search over
  any `VectorStore`. The reusable retrieval core; usable without the tool.
- `loaders/` — `DocumentLoader` ABC + per-format loaders + `resolve_loader`
  dispatch. The only format-specific surface.

## loaders/

| Loader | Source | Dependency (extra) |
|---|---|---|
| `TextLoader` / `MarkdownLoader` | `.txt` / `.md` `.markdown` `.mdx` | stdlib |
| `CSVLoader` / `JSONLoader` | `.csv` / `.json` | stdlib |
| `DirectoryLoader` | a directory (fans out, skips unknown) | stdlib |
| `PDFLoader` | `.pdf` (one doc/page) | `rag-pdf` (pymupdf) |
| `DOCXLoader` | `.docx` | `rag-docx` (python-docx) |
| `WebsiteLoader` | http(s) page | `rag-web` (requests, bs4) |
| `GithubLoader` | repo URL | `rag-github` (PyGithub) |
| `YoutubeVideoLoader` / `YoutubeChannelLoader` | video / channel URL | `rag-youtube` (youtube-transcript-api, pytube) |

## Architecture Decisions

| Decision | What | Why |
|---|---|---|
| **Reuse `Embedder` + `VectorStore`** | `DocumentIndex` composes the same primitives as `VectorMemory` | All five vector backends work; no new storage code |
| **Loaders are the only format surface** | Everything downstream is format-agnostic | New formats = new loader, nothing else |
| **One pipeline, typed wrappers** | `*SearchTool` pin a loader + tailor the description; no copied pipeline | Avoids ~15 near-identical tool classes |
| **Optional deps verified at construction** | `DocumentLoader.ensure_dependencies` via `find_spec`; lazy import in `load` | Fails fast with an install hint; importing the package never needs the extra |
| **Provider imports stay lazy** | Third-party imports live inside `load` helpers, tagged `# pyright: ignore[reportMissingImports]` | `import troopai.adk.rag` works on a minimal install |
| **Own provenance type** | `LoadedDocument` / `DocumentSearchHit`, not `MemoryMetadata` | Document provenance ≠ conversation-memory semantics; bridged in `index.py` |
| **Char-based chunking** | `TextChunker` measures characters | No tokenizer coupling; defaults sit inside every embedder's context |

## Cost & Safety Posture

- **No implicit embedding cost.** `DocumentSearchTool` requires an explicit
  `Embedder`; indexing is lazy and idempotent (paid on first search or an
  explicit `index()`), never at import or construction.
- **Ephemeral by default.** Absent an explicit `vector_store`, an in-memory
  store is used — nothing touches disk without opt-in.
- **Bounded ingestion.** `GithubLoader.max_files`, `YoutubeChannelLoader.max_videos`,
  and `TextChunker` bounds cap how much a single source can grow the index.
- **Sources bound at construction.** The corpus is the developer's; the LLM
  supplies only the query, never a path or URL — so the tool is off the
  attacker-controlled file/SSRF surface.

See `docs/rag/document_search.md` and `examples/rag/`.
