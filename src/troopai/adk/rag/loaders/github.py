"""GitHub repository loader — indexes a repo's text files.

Walks a repository's default-branch tree and loads each text file whose
extension is in ``content_extensions`` (docs and code by default), one document
per file. Uses PyGithub; install the ``rag-github`` extra. A token (constructor
arg or ``GITHUB_TOKEN``) lifts the unauthenticated rate limit. ``max_files``
bounds how many blobs are fetched so a large monorepo cannot grow ingestion
without limit.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import ClassVar, override
from urllib.parse import urlparse

from troopai.adk.exceptions.exceptions import DocumentLoadError
from troopai.adk.rag.document import LoadedDocument
from troopai.adk.rag.loaders.base import DocumentLoader

logger = logging.getLogger(__name__)

DEFAULT_CONTENT_EXTENSIONS: tuple[str, ...] = (
    ".md",
    ".markdown",
    ".rst",
    ".txt",
    ".py",
    ".js",
    ".ts",
    ".java",
    ".go",
    ".rb",
    ".rs",
)
"""Text-file extensions indexed by default (documentation + common code)."""


def _parse_repo(source: str) -> str:
    """Extract ``owner/repo`` from a github.com URL."""
    parts = [segment for segment in urlparse(source).path.split("/") if len(segment) > 0]
    if len(parts) < 2:
        raise DocumentLoadError(source, f"Not a github.com repository URL: {source}")
    owner, repo = parts[0], parts[1].removesuffix(".git")
    return f"{owner}/{repo}"


def _fetch_repo_files(
    source: str, token: str | None, extensions: tuple[str, ...], max_files: int
) -> list[LoadedDocument]:
    """Fetch matching text blobs from a repo's default branch (worker thread)."""
    from github import Github, GithubException  # pyright: ignore[reportMissingImports]

    full_name = _parse_repo(source)
    client = Github(token) if token is not None and len(token) > 0 else Github()
    try:
        repo = client.get_repo(full_name)
        tree = repo.get_git_tree(repo.default_branch, recursive=True)
        documents: list[LoadedDocument] = []
        fetched = 0
        for element in tree.tree:
            if element.type != "blob" or not element.path.lower().endswith(extensions):
                continue
            if fetched >= max_files:
                logger.warning("GithubLoader: %s hit max_files=%d; remaining files skipped", full_name, max_files)
                break
            blob = repo.get_contents(element.path, ref=repo.default_branch)
            fetched += 1
            content = blob.decoded_content.decode("utf-8", errors="replace") if not isinstance(blob, list) else ""
            if len(content.strip()) > 0:
                meta = {"repo": full_name, "path": element.path}
                documents.append(LoadedDocument(content=content, source=source, metadata=meta))
    except GithubException as exc:
        raise DocumentLoadError(source, f"GitHub API error for {full_name}: {exc}") from exc
    return documents


class GithubLoader(DocumentLoader):
    """Loads a GitHub repository's text files (via PyGithub).

    Attributes:
        token: GitHub access token. Falls back to ``GITHUB_TOKEN`` when
            ``None``; unauthenticated (rate-limited) when neither is set.
        content_extensions: File extensions to index.
        max_files: Maximum number of files fetched per repository.
    """

    requires_packages: ClassVar[tuple[str, ...]] = ("github",)
    install_extra: ClassVar[str] = "rag-github"

    def __init__(
        self,
        *,
        token: str | None = None,
        content_extensions: tuple[str, ...] = DEFAULT_CONTENT_EXTENSIONS,
        max_files: int = 200,
    ) -> None:
        """
        Args:
            token: GitHub access token; falls back to ``GITHUB_TOKEN``.
            content_extensions: File extensions to index.
            max_files: Maximum number of files fetched per repository.

        Raises:
            ImportError: If PyGithub is not installed.
            ValueError: If ``max_files`` is not positive or no extensions given.
        """
        if max_files <= 0:
            raise ValueError(f"GithubLoader.max_files must be > 0, got {max_files}")
        if len(content_extensions) == 0:
            raise ValueError("GithubLoader.content_extensions must be non-empty")
        self.token = token if token is not None else os.environ.get("GITHUB_TOKEN")
        self.content_extensions = content_extensions
        self.max_files = max_files
        self.ensure_dependencies()

    @override
    async def load(self, source: str) -> list[LoadedDocument]:
        """Load matching text files from repository ``source``.

        Args:
            source: A ``https://github.com/owner/repo`` URL.

        Returns:
            One document per matching text file (up to ``max_files``).

        Raises:
            DocumentLoadError: If the URL is not a repo or the API errors.
        """
        documents = await asyncio.to_thread(
            _fetch_repo_files, source, self.token, self.content_extensions, self.max_files
        )
        logger.debug("GithubLoader: %s -> %d file document(s)", source, len(documents))
        return documents
