"""Framework-provided builtin tools.

Tools the **framework** executes — not the LLM provider, not the
developer's local environment. These are tools the ADK ships
out-of-the-box and runs internally, plus the abstract bases that
define what "builtin" means.

Public surface:

- :class:`BuiltinTool` — abstract dataclass base for any tool the
  framework or developer environment runs (the ABC shared by
  ``MemoryTool``, ``JITContextAwareTool``, ``ShellTool``, and
  ``ApplyPatchTool``).
- :class:`ExecutableBuiltinTool` — extends ``BuiltinTool`` with
  ``description`` / ``schema`` / ``on_invoke`` so the LLM layer
  exposes the tool to the model directly.
- :class:`MemoryTool` (and decomposed variants
  :class:`RememberMemoryTool`, :class:`RecallMemoryTool`,
  :class:`ForgetMemoryTool`) — local memory store backed by the
  framework's :class:`MemoryStore` Protocol.
- :class:`JITContextAwareTool` — active context-management tool that
  expands into multiple framework-handled function tools at build
  time (``save_note``, ``recall_notes``, ``manage_context``, …).

Tools NOT in this folder:

- ``ShellTool`` / ``ApplyPatchTool`` — local-execution tools whose
  executor / editor is supplied by the **developer**. They live in
  ``tools/local/``. Both subclass the ``BuiltinTool`` ABC defined
  here.
- ``HostedTool`` subclasses (``WebSearchTool``, ``CodeExecutionTool``,
  …) — the LLM provider runs them server-side. They live in
  ``tools/hosted/``.
- ``FunctionTool`` — user-authored Python functions. Lives at the
  ``tools/`` root because it's the most common case.
"""

from troopai.adk.tools.builtin.builtin_tool import (
    BuiltinTool,
    ExecutableBuiltinTool,
)
from troopai.adk.tools.builtin.document_search_tool import (
    CSVSearchTool,
    DirectorySearchTool,
    DocumentSearchInput,
    DocumentSearchTool,
    DOCXSearchTool,
    GithubSearchTool,
    JSONSearchTool,
    MarkdownSearchTool,
    PDFSearchTool,
    TXTSearchTool,
    WebsiteSearchTool,
    YoutubeChannelSearchTool,
    YoutubeVideoSearchTool,
)
from troopai.adk.tools.builtin.jit_context_aware_tool import (
    InMemoryNoteStore,
    JITContextAwareTool,
    NoteEntry,
    NoteStore,
)
from troopai.adk.tools.builtin.memory_tool import (
    ForgetMemoryTool,
    MemoryTool,
    RecallMemoryTool,
    RememberMemoryTool,
)

__all__ = [
    "BuiltinTool",
    "CSVSearchTool",
    "DOCXSearchTool",
    "DirectorySearchTool",
    "DocumentSearchInput",
    "DocumentSearchTool",
    "ExecutableBuiltinTool",
    "ForgetMemoryTool",
    "GithubSearchTool",
    "InMemoryNoteStore",
    "JITContextAwareTool",
    "JSONSearchTool",
    "MarkdownSearchTool",
    "MemoryTool",
    "NoteEntry",
    "NoteStore",
    "PDFSearchTool",
    "RecallMemoryTool",
    "RememberMemoryTool",
    "TXTSearchTool",
    "WebsiteSearchTool",
    "YoutubeChannelSearchTool",
    "YoutubeVideoSearchTool",
]
