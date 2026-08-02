"""Memory extraction from conversation history.

Defines the :class:`MemoryExtractor` protocol and a default
:class:`LLMExtractor` that uses an LLM to identify knowledge
worth persisting.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, override

from troopai.adk.exceptions import MemoryExtractionError

if TYPE_CHECKING:
    from troopai.adk.llms.llm import LLM
    from troopai.adk.llms.llm_config import LLMConfig

logger = logging.getLogger(__name__)

_DEFAULT_EXTRACTION_PROMPT = """\
You are a memory extraction assistant.  Analyze the conversation below and
extract important facts, preferences, decisions, and knowledge that would
be useful to remember in future conversations.

For each extracted memory, return a JSON array of objects with:
- "content": The knowledge to remember (1-2 sentences).
- "importance": Integer 1-5 (1=trivia, 5=critical).
- "categories": Array of short category tags.

Return ONLY the JSON array, no other text.  If nothing is worth
remembering, return an empty array [].

Conversation:
{messages}
"""

_MESSAGES_PLACEHOLDER = "{messages}"
"""Marker replaced by the serialized conversation in an extraction prompt."""


@dataclass(frozen=True)
class ExtractionResult:
    """A single piece of knowledge extracted from a conversation.

    Attributes:
        content: The extracted knowledge text.
        importance: Priority from 1 (low) to 5 (critical).
        categories: Semantic tags.
    """

    content: str
    importance: int = 3
    categories: tuple[str, ...] = ()


class MemoryExtractor(ABC):
    """Abstract interface for extracting memories from conversation messages.

    Implementations receive raw conversation messages and return
    structured extraction results that can be stored in memory.
    """

    @abstractmethod
    async def extract(
        self,
        messages: list[Any],
        *,
        namespace: str,
    ) -> list[ExtractionResult]:
        """Extract memories from conversation messages.

        Args:
            messages: Conversation messages to analyze.
            namespace: The namespace context for extraction.

        Returns:
            List of extracted knowledge items.
        """
        ...


@dataclass
class LLMExtractor(MemoryExtractor):
    """LLM-based memory extraction (opt-in, costs tokens).

    Uses an LLM call to analyze conversation history and identify
    knowledge worth persisting.  This is intentionally opt-in —
    every call costs tokens.

    Attributes:
        llm: The LLM instance to use for extraction.  The model is
            set at construction time on the LLM instance (e.g.,
            ``LiteLLM(model="gpt-4o-mini")``).
        llm_config: LLM configuration (temperature, etc.).
        system_prompt: Custom extraction prompt.  If ``None``, uses
            the built-in default prompt.  A literal ``{messages}`` marker is
            replaced by the serialized conversation; a prompt without the
            marker still receives the conversation appended.
        max_entries: Maximum number of entries to extract per call.

    Example::

        from troopai.adk.llms.litellm.litellm_model import LiteLLM
        from troopai.adk.llms.llm_config import LLMConfig

        extractor = LLMExtractor(
            llm=LiteLLM(model="gpt-4o-mini"),
            llm_config=LLMConfig(temperature=0.0),
        )
    """

    llm: LLM
    """The LLM instance to use for extraction."""

    llm_config: LLMConfig | None = None
    """LLM configuration (temperature, max_tokens, etc.)."""

    system_prompt: str | None = None
    """Custom extraction prompt. If ``None``, uses the built-in default. A
    literal ``{messages}`` marker is replaced by the serialized conversation;
    without the marker the conversation is appended instead."""

    max_entries: int = 10
    """Maximum number of entries to extract per call."""

    @override
    async def extract(
        self,
        messages: list[Any],
        *,
        namespace: str,
    ) -> list[ExtractionResult]:
        """Extract memories from conversation messages via LLM.

        Args:
            messages: Conversation messages to analyze.
            namespace: The namespace context for extraction.

        Returns:
            List of extracted knowledge items.
        """
        prompt = self.system_prompt or _DEFAULT_EXTRACTION_PROMPT
        messages_text = json.dumps(messages, default=str, separators=(",", ":"))
        # Substitute via literal placeholder replace, not str.format, so a
        # developer prompt may contain literal "{"/"}" (e.g. JSON examples)
        # without raising; a prompt lacking the placeholder still receives the
        # conversation, appended, so the messages are never silently dropped.
        if _MESSAGES_PLACEHOLDER in prompt:
            formatted = prompt.replace(_MESSAGES_PLACEHOLDER, messages_text)
        else:
            formatted = f"{prompt}\n\nConversation:\n{messages_text}"

        logger.info(
            "LLMExtractor: extracting memories (model=%s, namespace=%s)",
            getattr(self.llm, "model", "unknown"),
            namespace,
        )

        response = await self.llm.acomplete(
            messages=[{"role": "user", "content": formatted}],
            llm_config=self.llm_config,
        )

        raw_content = response.content or "[]"
        return self._parse_response(raw_content)

    def _parse_response(self, raw: str) -> list[ExtractionResult]:
        """Parse LLM response into ExtractionResult instances."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            items = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MemoryExtractionError(f"LLMExtractor: failed to parse response as JSON: {text[:200]}") from exc

        if not isinstance(items, list):
            raise MemoryExtractionError(f"LLMExtractor: expected JSON array, got {type(items).__name__}")

        results: list[ExtractionResult] = []
        for item in items[: self.max_entries]:
            if not isinstance(item, dict) or "content" not in item:
                logger.warning(
                    "LLMExtractor: skipping malformed item (missing content or not a dict): %s",
                    str(item)[:100],
                )
                continue
            try:
                importance = max(1, min(5, int(item.get("importance", 3))))
            except (ValueError, TypeError):
                importance = 3
            raw_categories = item.get("categories", [])
            categories = tuple(str(c) for c in raw_categories) if isinstance(raw_categories, list) else ()
            results.append(
                ExtractionResult(
                    content=str(item["content"]),
                    importance=importance,
                    categories=categories,
                )
            )

        logger.info("LLMExtractor: extracted %d memories", len(results))
        return results
