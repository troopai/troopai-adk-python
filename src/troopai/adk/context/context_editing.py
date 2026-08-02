"""Selective context editing strategies.

Provides utilities to clear old tool-call results and thinking blocks
from conversation history, freeing tokens without full compaction.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from troopai.adk.types.input import LLMInputContentItem

logger = logging.getLogger(__name__)

_CLEARED_TOOL_PLACEHOLDER = "[tool result cleared to save context]"
_CLEARED_THINKING_PLACEHOLDER = "[thinking cleared to save context]"


class ContextEditor:
    """Applies context editing strategies to a list of messages.

    All methods return a **new** list (the originals are not mutated).
    """

    @staticmethod
    def remove_orphaned_tool_results(
        messages: list[LLMInputContentItem],
    ) -> list[LLMInputContentItem]:
        """Remove tool-result messages with no matching tool_use in the conversation.

        Compaction (which replaces old messages with a summary),
        truncation (which drops oldest messages), and a ``DropDirective``
        recency-slice can all separate a tool_use block from its
        tool_result.  Anthropic and Gemini reject orphaned tool results,
        so this cleanup is essential before sending to the provider.

        Operates on Layer 1 framework messages (``LLMInputContentItem``):

        * ``{"type": "function_call", "call_id": <id>, ...}`` contributes
          ``call_id`` to the valid set.
        * Layer 2 ``{"role": "assistant", "tool_calls": [...]}`` is still
          accepted so that callers constructing hand-rolled Chat Completions
          messages (older examples, tests) stay compatible.
        * ``{"type": "function_call_output", "call_id": <id>, ...}`` is
          dropped when its ``call_id`` is not in the valid set.
        * Layer 2 ``{"role": "tool", "tool_call_id": <id>, ...}`` is
          dropped on the same criterion.

        Args:
            messages: Conversation messages (Layer-1 items).

        Returns:
            A new messages list with orphaned tool results removed.
        """
        valid_ids: set[str] = set()
        for msg in messages:
            msg_type = msg.get("type")
            if msg_type == "function_call":
                call_id = msg.get("call_id")
                if isinstance(call_id, str) and len(call_id) > 0:
                    valid_ids.add(call_id)
                continue
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls")
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                        if tc_id is not None:
                            valid_ids.add(tc_id)

        result: list[LLMInputContentItem] = []
        dropped = 0
        for msg in messages:
            msg_type = msg.get("type")
            if msg_type == "function_call_output":
                call_id = msg.get("call_id", "")
                if not isinstance(call_id, str) or call_id not in valid_ids:
                    dropped += 1
                    continue
            elif msg.get("role") == "tool":
                tool_call_id = msg.get("tool_call_id", "")
                if not isinstance(tool_call_id, str) or tool_call_id not in valid_ids:
                    dropped += 1
                    continue
            result.append(msg)

        if dropped > 0:
            logger.debug("Removed %d orphaned tool-result message(s)", dropped)

        return result

    @staticmethod
    def clear_tool_results(
        messages: list[LLMInputContentItem], keep: int = 3, exclude_tools: list[str] | None = None
    ) -> list[LLMInputContentItem]:
        """Replace old tool-result content with a placeholder.

        Keeps the *N* most-recent tool results intact. Operates on the
        Layer-1 message stream, where a tool result is a
        ``{"type": "function_call_output", "call_id": ..., "output": ...}``
        item with no ``role`` field; its matching call is a separate
        ``{"type": "function_call", "call_id": ..., "name": ...}`` item.

        Args:
            messages: Conversation messages (Layer-1 items).
            keep: Number of most-recent tool results to preserve.
            exclude_tools: Tool names whose results are never cleared.
                Resolved via the matching ``function_call`` item's
                ``name`` (the result item carries only ``call_id``).

        Returns:
            A new messages list with old tool results replaced.
        """
        exclude = set(exclude_tools) if exclude_tools is not None else set()

        # A tool result carries only ``call_id``; map it to the tool name
        # via the matching ``function_call`` item so ``exclude_tools`` works.
        call_id_to_name: dict[str, str] = {}
        for msg in messages:
            if msg.get("type") == "function_call":
                call_id = msg.get("call_id")
                name = msg.get("name")
                if isinstance(call_id, str) and isinstance(name, str):
                    call_id_to_name[call_id] = name

        # Collect indices of tool-result items (chronological order).
        tool_indices: list[int] = []
        for idx, msg in enumerate(messages):
            if msg.get("type") == "function_call_output":
                call_id = msg.get("call_id")
                tool_name = call_id_to_name.get(call_id, "") if isinstance(call_id, str) else ""
                if tool_name not in exclude:
                    tool_indices.append(idx)

        # Determine which to clear (all except the last *keep*).
        to_clear = set(tool_indices[: max(0, len(tool_indices) - keep)])

        if len(to_clear) == 0:
            return list(messages)

        result: list[LLMInputContentItem] = []
        for idx, msg in enumerate(messages):
            if idx in to_clear:
                # cast keeps the intent explicit: the copy is a plain dict
                # at runtime, and we mutate the "output" key; the item is
                # re-typed at the LLM boundary on the next call.
                cleared = cast(dict[str, Any], dict(msg))
                cleared["output"] = _CLEARED_TOOL_PLACEHOLDER
                result.append(cast("LLMInputContentItem", cleared))
            else:
                result.append(msg)

        logger.debug("Context editing cleared %d old tool result(s)", len(to_clear))
        return result

    @staticmethod
    def clear_thinking_blocks(messages: list[LLMInputContentItem], keep_turns: int = 1) -> list[LLMInputContentItem]:
        """Remove thinking blocks from older assistant turns.

        Reasoning replays in three shapes and every one is cleared here so
        older thinking actually frees tokens:

        * an assistant message whose ``content`` list holds
          ``{"type": "thinking"}`` blocks — the block is replaced with a
          placeholder text part;
        * a standalone ``{"type": "reasoning"}`` replay item
          (``LLMResponseReasoningParam``) — the whole item is dropped, since
          there is no surrounding message to keep;
        * any message carrying a top-level ``thinking_blocks`` list (HITL
          resumption / already-wire-format history) — the field is removed.

        Only the last *keep_turns* thinking-bearing items are preserved.

        Args:
            messages: Conversation messages.
            keep_turns: Number of most-recent thinking-bearing items whose
                thinking is kept.

        Returns:
            A new messages list with old thinking removed.
        """
        # Identify thinking-bearing items across all three replay shapes.
        thinking_indices: list[int] = []
        for idx, msg in enumerate(messages):
            if ContextEditor._carries_thinking(msg):
                thinking_indices.append(idx)

        to_clear = set(thinking_indices[: max(0, len(thinking_indices) - keep_turns)])

        if len(to_clear) == 0:
            return list(messages)

        result: list[LLMInputContentItem] = []
        for idx, msg in enumerate(messages):
            if idx not in to_clear:
                result.append(msg)
                continue
            # A standalone reasoning replay item is cleared by dropping it
            # entirely — there is no surrounding message to preserve.
            if msg.get("type") == "reasoning":
                continue
            # ``copy.deepcopy`` preserves TypedDict shape, but we mutate the
            # content blocks below and re-insert; widening through ``Any``
            # concentrates the variance on one local instead of cast + ignore.
            cleared: Any = copy.deepcopy(msg)
            if isinstance(cleared.get("thinking_blocks"), list):
                del cleared["thinking_blocks"]
            content = cleared.get("content")
            if isinstance(content, list):
                cleared["content"] = [
                    {"type": "text", "text": _CLEARED_THINKING_PLACEHOLDER}
                    if isinstance(block, dict) and block.get("type") == "thinking"
                    else block
                    for block in content
                ]
            result.append(cleared)

        return result

    @staticmethod
    def _carries_thinking(msg: LLMInputContentItem) -> bool:
        """Whether *msg* carries thinking in any of the three replay shapes."""
        if msg.get("type") == "reasoning":
            return True
        # ``thinking_blocks`` is an out-of-band assistant-message key (HITL
        # resumption / provider replay) outside every TypedDict in the union,
        # so it is read through a widened view.
        msg_any: Any = msg
        thinking_blocks = msg_any.get("thinking_blocks")
        if isinstance(thinking_blocks, list) and len(thinking_blocks) > 0:
            return True
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if isinstance(content, list):
                return any(isinstance(block, dict) and block.get("type") == "thinking" for block in content)
        return False
