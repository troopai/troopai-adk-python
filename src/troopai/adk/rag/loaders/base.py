"""The :class:`DocumentLoader` abstract base.

A loader turns one *source* (a file path or URL) into a list of
:class:`LoadedDocument` spans. Loaders are the only format-specific surface in
the RAG layer — everything downstream (chunking, embedding, vector search) is
format-agnostic. A loader for a remote or heavyweight format declares the
third-party packages it needs in ``requires_packages`` and verifies them at
construction (see :meth:`ensure_dependencies`), so a missing optional
dependency surfaces as a clear, actionable error before any work begins rather
than deep inside an agent run.

Blocking I/O (file reads, network calls, third-party parsers) MUST be wrapped
in :func:`asyncio.to_thread` inside :meth:`load` so loaders never stall the
event loop.
"""

from __future__ import annotations

import importlib.util
import logging
from abc import ABC, abstractmethod
from typing import ClassVar

from troopai.adk.rag.document import LoadedDocument

logger = logging.getLogger(__name__)


class DocumentLoader(ABC):
    """Turns a single source (path or URL) into loaded document spans.

    Subclasses set :attr:`requires_packages` (import names of the
    third-party libraries they need) and implement :meth:`load`. Pure-stdlib
    loaders leave ``requires_packages`` empty.

    Attributes:
        requires_packages: Import names whose absence makes this loader
            unusable. Verified by :meth:`ensure_dependencies`, which
            subclasses call from ``__init__`` so the failure is raised at
            tool-construction time.
        install_extra: The packaging extra that provides
            ``requires_packages`` (e.g. ``"rag-pdf"``), surfaced in the
            missing-dependency error message. Empty when stdlib-only.
    """

    requires_packages: ClassVar[tuple[str, ...]] = ()
    """Import names whose absence makes this loader unusable."""

    install_extra: ClassVar[str] = ""
    """The packaging extra that provides :attr:`requires_packages`."""

    @abstractmethod
    async def load(self, source: str) -> list[LoadedDocument]:
        """Load ``source`` into one or more document spans.

        Args:
            source: A file path or URL this loader handles.

        Returns:
            The extracted spans (e.g. one per PDF page). May be empty when
            the source contains no extractable text.

        Raises:
            DocumentLoadError: If the source cannot be read or parsed.
        """

    def ensure_dependencies(self) -> None:
        """Verify every entry in :attr:`requires_packages` is importable.

        Subclasses call this from ``__init__`` so a missing optional
        dependency fails fast at construction, with guidance toward the
        packaging extra that supplies it.

        Raises:
            ImportError: If any required package is not importable.
        """
        missing = [name for name in self.requires_packages if importlib.util.find_spec(name) is None]
        if len(missing) == 0:
            return
        hint = f" Install the '{self.install_extra}' extra." if len(self.install_extra) > 0 else ""
        raise ImportError(f"{type(self).__name__} requires missing package(s): {', '.join(missing)}.{hint}")
