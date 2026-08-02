"""Context compaction via LLM summarization.

Compacts older conversation messages into a concise summary produced by
an LLM call, preserving the most-recent turns verbatim.

The summarization call routes through the framework's :class:`LLM` ABC
(``llm.acomplete(...)``) so compaction tokens land in
:attr:`RunContext.usage`, ``Agent.middleware.llms`` sees the call, and
the developer's configured ``LLM`` subclass (Anthropic native, OpenAI
Responses, etc.) is honoured rather than hardcoding litellm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .context_config import CompactionConfig
from .token_counter import TokenCounter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from troopai.adk.llms.llm import LLM
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.input.llm_input_easy_message import LLMInputEasyMessage
    from troopai.adk.types.tokens.llm_usage import LLMUsage


@dataclass
class CompactionResult:
    """Result of a compaction operation.

    Attributes:
        summary: The LLM-generated summary of older messages.
        original_token_count: Token count before compaction.
        compacted_token_count: Token count after compaction.
        items_compacted: Number of messages that were summarized.
        usage: Token usage from the summarization ``LLM.acomplete`` call,
            or ``None`` when no LLM call was needed (conversation too
            short to compact). Callers MUST accumulate this into
            ``RunContext.usage`` so framework-wide usage limits and
            cost reporting include compaction tokens.
    """

    summary: str
    original_token_count: int
    compacted_token_count: int
    items_compacted: int
    usage: LLMUsage | None = None


class ContextCompactor:
    """Compacts conversation history via LLM summarization."""

    DEFAULT_INSTRUCTIONS = (
        "Summarise the conversation so far in a way that preserves all "
        "information needed to continue the task.  Include: the overall "
        "task description, current state, important discoveries or "
        "decisions, and any next steps that were planned."
    )

    @classmethod
    async def compact(
        cls,
        messages: list[LLMInputContentItem],
        llm: LLM,
        model_name: str,
        config: CompactionConfig,
        run_config: Any | None = None,  # noqa: ARG003
    ) -> CompactionResult:
        """Summarize older messages, preserving recent ones.

        The ``preserve_recent_items`` most-recent **non-system** messages
        are kept verbatim; everything else is fed to the summarization
        model via :meth:`LLM.acomplete`.

        Args:
            messages: Full conversation (including system message).
            llm: The ``LLM`` instance to call for summarization. Use
                :func:`troopai.adk.run.llm_calls.resolve_compaction_llm`
                to obtain it (honours ``RunConfig.compaction_llm`` then
                falls back to the agent's primary LLM).
            model_name: Model identifier used for token counting via
                :class:`TokenCounter`. The token counter uses litellm's
                tokenizer interface which is name-based, so a string is
                still required even though the LLM call itself is
                provider-agnostic.
            config: Compaction settings.
            run_config: Optional ``RunConfig`` passed through to the
                compactor (currently unused by the compactor itself).

        Returns:
            A :class:`CompactionResult` with the summary and token stats.
        """
        instructions = config.instructions or cls.DEFAULT_INSTRUCTIONS
        preserve = config.preserve_recent_items

        # Separate system message from conversation body.
        system_msg: LLMInputContentItem | None = None
        body: list[LLMInputContentItem] = list(messages)
        if body and body[0].get("role") == "system":
            system_msg = body[0]
            body = body[1:]

        # Split into compactable / preserved. ``preserve == 0`` means
        # "preserve no recent turns" — compact the entire body. A plain
        # ``body[:-preserve]`` would mis-handle this (``-0`` slices to the
        # whole list / an empty list), so the zero case is explicit.
        if preserve == 0:
            to_compact = body
            preserved = []
        elif 0 < preserve < len(body):
            to_compact = body[:-preserve]
            preserved = body[-preserve:]
        else:
            # Nothing to compact (conversation too short).
            to_compact = []
            preserved = body

        if len(to_compact) == 0:
            original_tokens = TokenCounter.count_messages(messages, model_name)
            return CompactionResult(
                summary="",
                original_token_count=original_tokens,
                compacted_token_count=original_tokens,
                items_compacted=0,
            )

        original_tokens = TokenCounter.count_messages(messages, model_name)

        # Build summarization prompt as Layer 1 input items. The
        # ``LLMInputEasyMessage`` TypedDict accepts ``content: str``,
        # so these dict literals are valid Layer 1 inputs at runtime
        # and through the type checker.
        system_input: LLMInputEasyMessage = {"role": "system", "content": instructions}
        user_input: LLMInputEasyMessage = {
            "role": "user",
            "content": cls._format_for_summary(to_compact),
        }
        summary_input: list[LLMInputContentItem] = [system_input, user_input]

        response = await llm.acomplete(messages=summary_input)
        summary = response.content or ""

        # Guard: an empty summary with items_compacted > 0 would silently
        # drop the compacted messages with no replacement text. Return a
        # no-op result so the caller keeps the original messages intact.
        if len(summary) == 0:
            logger.warning(
                "Compaction LLM returned empty summary for %d messages; skipping compaction",
                len(to_compact),
            )
            return CompactionResult(
                summary="",
                original_token_count=original_tokens,
                compacted_token_count=original_tokens,
                items_compacted=0,
                usage=response.usage,
            )

        # Compute new token count.
        new_messages = cls.build_compacted_messages(summary, preserved, system_msg)
        compacted_tokens = TokenCounter.count_messages(new_messages, model_name)

        return CompactionResult(
            summary=summary,
            original_token_count=original_tokens,
            compacted_token_count=compacted_tokens,
            items_compacted=len(to_compact),
            usage=response.usage,
        )

    @staticmethod
    def build_compacted_messages(
        summary: str,
        preserved_messages: list[LLMInputContentItem],
        system_message: LLMInputContentItem | None = None,
    ) -> list[LLMInputContentItem]:
        """Reassemble messages from a compaction summary.

        The summary is stored as an assistant message with a
        ``_compaction`` metadata flag so downstream code can recognise it.

        Args:
            summary: The LLM-generated summary.
            preserved_messages: Recent messages kept verbatim.
            system_message: Optional system message to prepend.

        Returns:
            A new message list ready for the next LLM call.
        """
        result: list[LLMInputContentItem] = []

        if system_message is not None:
            result.append(system_message)

        if len(summary) > 0:
            # ``_compaction`` is an out-of-band marker outside every TypedDict in
            # the ``LLMInputContentItem`` union. Launder through ``Any`` so the
            # extra key is preserved at runtime without a targeted ignore; the
            # message is re-typed on the next LLM call boundary.
            compaction_msg: Any = {
                "role": "assistant",
                "content": summary,
                "_compaction": True,
            }
            result.append(compaction_msg)

        result.extend(preserved_messages)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_for_summary(messages: list[LLMInputContentItem]) -> str:
        """Convert messages into a readable transcript for the summarizer.

        Tool calls and tool results are Layer-1 items with no ``role`` /
        ``content`` (``function_call`` carries ``name`` / ``arguments``;
        ``function_call_output`` carries ``output``), so a plain
        ``[role]: content`` render collapses them to empty ``[unknown]:``
        lines and drops the tool exchange from the summary.  Each item kind
        is formatted explicitly instead.

        Args:
            messages: The messages to format.

        Returns:
            A plain-text transcript, one labelled line per item, joined by
            blank lines.
        """
        lines: list[str] = []
        for msg in messages:
            item_type = msg.get("type")
            if item_type == "function_call":
                name = msg.get("name", "")
                arguments = msg.get("arguments", "")
                lines.append(f"[tool call] {name}({arguments})")
            elif item_type == "function_call_output":
                lines.append(f"[tool result] {ContextCompactor._stringify_content(msg.get('output', ''))}")
            elif msg.get("role") is not None:
                role = msg.get("role")
                lines.append(f"[{role}]: {ContextCompactor._stringify_content(msg.get('content', ''))}")
            elif item_type is not None:
                # Other typed items (e.g. reasoning) — label by kind and
                # surface any embedded text rather than an empty line.
                lines.append(f"[{item_type}] {ContextCompactor._stringify_content(msg.get('content', ''))}")
            else:
                lines.append(f"[unknown]: {ContextCompactor._stringify_content(msg.get('content', ''))}")
        return "\n\n".join(lines)

    @staticmethod
    def _stringify_content(content: Any) -> str:
        """Flatten a message ``content`` / tool ``output`` value to text.

        A plain string passes through; a multimodal parts list has each
        part's ``text`` extracted (falling back to the block's ``repr`` for
        non-text parts such as images).

        Args:
            content: A string, a list of content-part dicts, or any value.

        Returns:
            The flattened text.
        """
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict):
                    parts.append(block.get("text", str(block)))
                else:
                    parts.append(str(block))
            return "\n".join(parts)
        return str(content)
