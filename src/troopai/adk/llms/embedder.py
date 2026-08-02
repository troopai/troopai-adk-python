"""Provider-agnostic embedding abstraction.

Mirrors the ``LLM`` ABC pattern: a framework-owned ``Embedder`` interface with
provider implementations under ``llms/<provider>/``.  Embeddings are a distinct
capability from chat completion (Anthropic, for instance, ships none), so they
live on their own ABC rather than on ``LLM``.
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Embedding:
    """A single embedding vector and the model that produced it.

    Attributes:
        vector: The embedding components.
        model: The model identifier that produced the vector.
    """

    vector: tuple[float, ...]
    """The embedding components."""

    model: str
    """The model identifier that produced the vector."""

    @property
    def dimensions(self) -> int:
        """Number of components in the vector."""
        return len(self.vector)


class Embedder(ABC):
    """Abstract embedding interface.

    ``aembed_documents`` (batch) and ``aembed_query`` (single) are split because
    asymmetric models encode documents and queries with different instructions.
    For symmetric models the default ``aembed_query`` simply delegates to
    ``aembed_documents``; override it only for asymmetric models.
    """

    @abstractmethod
    async def aembed_documents(self, texts: list[str]) -> list[Embedding]:
        """Embed a batch of documents, preserving input order.

        Args:
            texts: The strings to embed.

        Returns:
            A list of ``Embedding`` instances in the same order as
            ``texts``.
        """

    async def aembed_query(self, text: str) -> Embedding:
        """Embed a single query string.

        Args:
            text: The query string to embed.

        Returns:
            The ``Embedding`` for the query.

        Raises:
            RuntimeError: If the underlying ``aembed_documents`` returns
                no results for the single-element batch.
        """
        results = await self.aembed_documents([text])
        if len(results) == 0:
            raise RuntimeError("Embedder returned no embedding for query")
        return results[0]

    @property
    @abstractmethod
    def dimensions(self) -> int | None:
        """Fixed output dimension if known, else ``None``."""


class EmbeddingLRUCache:
    """Bounded, thread-safe LRU cache keyed by ``(model, text)``.

    Opt-in cost lever: embedding is deterministic per (model, text), so caching
    identical lookups is always safe.  Bounded to ``max_size`` entries.
    """

    def __init__(self, max_size: int) -> None:
        """
        Args:
            max_size: Maximum entries retained; must be > 0.
        """
        if max_size <= 0:
            raise ValueError(f"EmbeddingLRUCache max_size must be > 0, got {max_size}")
        self._max_size = max_size
        self._store: OrderedDict[tuple[str, str], Embedding] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, model: str, text: str) -> Embedding | None:
        """Return the cached embedding for ``(model, text)``, or ``None``.

        Args:
            model: Model identifier component of the cache key.
            text: Text component of the cache key.

        Returns:
            The cached ``Embedding``, or ``None`` on a miss.
        """
        key = (model, text)
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return self._store[key]

    def put(self, embedding: Embedding, *, text: str) -> None:
        """Cache ``embedding`` under ``(embedding.model, text)``.

        Args:
            embedding: The embedding to cache. Its ``model`` attribute
                forms part of the key.
            text: The text the embedding was produced from; forms the
                other half of the key.
        """
        key = (embedding.model, text)
        with self._lock:
            self._store[key] = embedding  # appends if new; updates in place if existing
            self._store.move_to_end(key)  # no-op for new key; promotes existing key to MRU
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)
