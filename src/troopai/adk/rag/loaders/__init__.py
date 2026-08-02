"""Document loaders — turn a source (path or URL) into loaded text spans.

Loaders are the only format-specific surface in the RAG layer. Stdlib loaders
(text, Markdown, CSV, JSON, directory) are always available; the remaining
loaders need an optional packaging extra (``rag-pdf``, ``rag-docx``,
``rag-web``, ``rag-github``, ``rag-youtube``) and verify it at construction.

:func:`resolve_loader` routes a source to a default-constructed loader by URL
shape or file extension; the ``*SearchTool`` family pins explicit, configured
loaders instead.
"""

from troopai.adk.rag.loaders.base import DocumentLoader
from troopai.adk.rag.loaders.directory import DirectoryLoader
from troopai.adk.rag.loaders.docx import DOCXLoader
from troopai.adk.rag.loaders.github import GithubLoader
from troopai.adk.rag.loaders.pdf import PDFLoader
from troopai.adk.rag.loaders.plaintext import MarkdownLoader, TextLoader
from troopai.adk.rag.loaders.registry import FILE_EXTENSIONS, is_url, resolve_loader
from troopai.adk.rag.loaders.structured import CSVLoader, JSONLoader
from troopai.adk.rag.loaders.website import WebsiteLoader
from troopai.adk.rag.loaders.youtube import YoutubeChannelLoader, YoutubeVideoLoader

__all__ = [
    "FILE_EXTENSIONS",
    "CSVLoader",
    "DOCXLoader",
    "DirectoryLoader",
    "DocumentLoader",
    "GithubLoader",
    "JSONLoader",
    "MarkdownLoader",
    "PDFLoader",
    "TextLoader",
    "WebsiteLoader",
    "YoutubeChannelLoader",
    "YoutubeVideoLoader",
    "is_url",
    "resolve_loader",
]
