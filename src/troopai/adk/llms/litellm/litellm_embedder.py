"""litellm-backed embedder.

Uses ``litellm.aembedding`` so any litellm-supported embedding model works
(OpenAI, Cohere, Voyage, Gemini, etc.).  litellm is imported lazily inside the
method, mirroring ``litellm_model.py``.

Refs:
    - https://docs.litellm.ai/docs/embedding/supported_embedding
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import override

from troopai.adk.llms.embedder import Embedder, Embedding, EmbeddingLRUCache

logger = logging.getLogger(__name__)


@dataclass
class LiteLLMEmbedder(Embedder):
    """Embedder backed by ``litellm.aembedding``.

    Attributes:
        model: litellm model id (e.g. ``"text-embedding-3-small"``).
        cache: Optional opt-in LRU cache; ``None`` disables caching.
        dimensions_hint: Output dimension for models that accept one, also
            returned by :attr:`dimensions`.
        api_key: Optional explicit key; ``None`` uses litellm's env resolution.
        api_base: Optional explicit endpoint; ``None`` uses the provider default.
    """

    model: str
    cache: EmbeddingLRUCache | None = None
    dimensions_hint: int | None = None
    api_key: str | None = None
    api_base: str | None = None

    @override
    async def aembed_documents(self, texts: list[str]) -> list[Embedding]:
        """Embed a batch of texts via ``litellm.aembedding``, with optional cache.

        Args:
            texts: The strings to embed. Returns an empty list immediately
                when ``texts`` is empty.

        Returns:
            A list of ``Embedding`` instances in the same order as
            ``texts``.

        Raises:
            RuntimeError: If the provider returns fewer embeddings than
                requested, or if an index has no embedding after merging
                cached and fetched results.
        """
        if len(texts) == 0:
            return []

        cached: dict[int, Embedding] = {}
        misses: list[str] = []
        miss_indexes: list[int] = []
        for i, text in enumerate(texts):
            hit = self.cache.get(self.model, text) if self.cache is not None else None
            if hit is not None:
                cached[i] = hit
            else:
                misses.append(text)
                miss_indexes.append(i)

        fetched: list[Embedding] = []
        if len(misses) > 0:
            import litellm

            logger.debug("LiteLLMEmbedder: embedding %d texts (model=%s)", len(misses), self.model)
            # Optional params are passed by name. dimensions=None equals litellm's own
            # parameter default and is dropped before reaching the provider; set
            # dimensions_hint to a non-None value to forward it to models that accept it.
            response = await litellm.aembedding(
                model=self.model,
                input=misses,
                api_key=self.api_key,
                api_base=self.api_base,
                dimensions=self.dimensions_hint,
            )
            raw_data = response.get("data")
            if not raw_data:
                raise RuntimeError(f"LiteLLMEmbedder: model {self.model!r} returned a response with no 'data' field")
            data = sorted(raw_data, key=lambda item: item["index"])
            if len(data) != len(misses):
                raise RuntimeError(
                    f"LiteLLMEmbedder: expected {len(misses)} embeddings from model {self.model!r}, got {len(data)}"
                )
            for item in data:
                fetched.append(Embedding(vector=tuple(float(x) for x in item["embedding"]), model=self.model))
            if self.cache is not None:
                for text, emb in zip(misses, fetched, strict=True):
                    self.cache.put(emb, text=text)

        result: list[Embedding | None] = [None] * len(texts)
        for idx, emb in cached.items():
            result[idx] = emb
        for idx, emb in zip(miss_indexes, fetched, strict=True):
            result[idx] = emb

        out: list[Embedding] = []
        for i, item in enumerate(result):
            if item is None:
                raise RuntimeError(f"LiteLLMEmbedder: missing embedding at index {i}")
            out.append(item)
        return out

    @property
    @override
    def dimensions(self) -> int | None:
        return self.dimensions_hint
