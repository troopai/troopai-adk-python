"""Inject ``cache_control`` markers for Anthropic prompt caching.

Anthropic caches by **prefix**: marking a content block with
``cache_control={"type": "ephemeral"}`` tells the server to cache
everything up to and including that block. On the next request that
shares the same prefix, the cached portion is read instead of
recomputed.

Three canonical injection points cover the typical "cache the long
context" use case:

1. **Last system block** — caches the system prompt.
2. **Last tool definition** — caches the tool schemas.
3. **Last user-message text block** — caches the conversation up to
   that turn.

This module implements the same three insertions used inside the
litellm path's Anthropic shim, but operates directly on
``anthropic.types.*`` params instead of going through litellm's
``cache_control_injection_points`` indirection.

Caveat — marker drift: the cache prefix hash covers tools, then
system, then messages, in order. The tool marker is placed on the
*last* tool definition, so adding, removing, or reordering tools
between turns moves the marker and changes the hashed prefix — every
cached segment after the first changed byte is a miss and is re-billed
as a cache write. Agents whose tool list changes per turn (dynamic
toolsets, JIT tools) should keep the tool list stable across a
conversation to benefit from caching.

Anthropic prompt-caching docs:
https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

from anthropic.types import (
    CacheControlEphemeralParam,
    MessageParam,
    TextBlockParam,
)

from troopai.adk.llms.anthropic.anthropic_config import AnthropicConfig

if TYPE_CHECKING:
    from anthropic.types import ToolParam, ToolUnionParam

    from troopai.adk.llms.llm_config import LLMConfig

logger = logging.getLogger(__name__)


def _build_cache_marker(ttl: Literal["5m", "1h"] | None) -> CacheControlEphemeralParam:
    """Build an ephemeral cache_control marker.

    Args:
        ttl: TTL hint. ``None`` uses Anthropic's default (``"5m"``).

    Returns:
        An ``CacheControlEphemeralParam`` dict with ``type="ephemeral"``
        and an optional ``ttl`` key.
    """
    marker: CacheControlEphemeralParam = {"type": "ephemeral"}
    if ttl is not None:
        marker["ttl"] = ttl
    return marker


def _is_text_tool(tool: ToolUnionParam) -> bool:
    """Return ``True`` when *tool* is a regular ``ToolParam`` (custom function).

    ``ToolUnionParam`` is a TypedDict union — every variant is a dict
    at runtime, so no ``isinstance`` guard is needed. Server-tool
    variants (``"web_search_*"``, ``"computer_*"``) declare a
    discriminator value other than ``"custom"`` and would silently
    accept ``cache_control`` on a different schema; we skip them.

    Args:
        tool: A ``ToolUnionParam`` dict from the wire tools list.

    Returns:
        ``True`` when the tool's ``type`` is absent (default custom) or
        ``"custom"``; ``False`` for server-side hosted tools.
    """
    declared_type = tool.get("type")
    return declared_type is None or declared_type == "custom"


def apply_cache_control(
    system: str | list[TextBlockParam] | None,
    messages: list[MessageParam],
    tools: list[ToolUnionParam] | None,
    config: LLMConfig,
) -> tuple[str | list[TextBlockParam] | None, list[MessageParam], list[ToolUnionParam] | None]:
    """Inject ``cache_control`` markers when ``auto_cache_control`` is on.

    Args:
        system: System prompt — string, list of ``TextBlockParam``, or
            ``None``. When auto-caching is on and the value is a
            non-empty string, it is converted to a single
            ``TextBlockParam`` carrying the marker.
        messages: User / assistant message list. The last user
            message's last text block is marked when present.
        tools: Tool definitions. The last entry is marked when
            present.
        config: The LLM configuration. Only :class:`AnthropicConfig`
            instances trigger injection — plain :class:`LLMConfig`
            values pass through unchanged.

    Returns:
        ``(system, messages, tools)`` with markers applied. The input
        objects are NOT mutated — the function returns shallow copies
        when changes are made.
    """
    if not isinstance(config, AnthropicConfig) or config.auto_cache_control is not True:
        return system, messages, tools

    ttl = config.cache_control_ttl
    marker = _build_cache_marker(ttl)

    # ----- system -----
    new_system: str | list[TextBlockParam] | None
    if system is None:
        new_system = None
    elif isinstance(system, str):
        new_system = system if len(system) == 0 else [TextBlockParam(type="text", text=system, cache_control=marker)]
    elif len(system) == 0:
        new_system = system
    else:
        updated_system = list(system)
        last = dict(updated_system[-1])
        last["cache_control"] = marker
        # ``last`` is a plain dict[str, Any] after the dict() copy;
        # mypy cannot prove the keys still match TextBlockParam's
        # required shape, but the source list element is already a
        # TextBlockParam so the runtime invariant holds.
        updated_system[-1] = TextBlockParam(**last)  # type: ignore[typeddict-item]
        new_system = updated_system

    # ----- tools -----
    new_tools: list[ToolUnionParam] | None
    if tools is None or len(tools) == 0:
        new_tools = tools
    else:
        updated_tools = list(tools)
        # Mark the last "regular" tool — a server-tool union variant
        # may not accept cache_control on the same shape.
        for i in range(len(updated_tools) - 1, -1, -1):
            tool_entry = updated_tools[i]
            if _is_text_tool(tool_entry):
                # ``ToolUnionParam`` is a union of TypedDicts; mypy
                # widens the dict() copy to dict[str, Any], so the
                # narrowing back to ``ToolParam`` for the
                # ``cache_control`` assignment needs an isolated
                # ignore. ``_is_text_tool`` guarantees the runtime
                # variant.
                marked: ToolParam = dict(tool_entry)  # type: ignore[assignment]
                marked["cache_control"] = marker
                updated_tools[i] = marked
                break
        new_tools = updated_tools

    # ----- last user-message text block -----
    if len(messages) == 0:
        new_messages = messages
    else:
        new_messages = list(messages)
        for i in range(len(new_messages) - 1, -1, -1):
            msg_entry = new_messages[i]
            if msg_entry.get("role") != "user":
                continue
            content = msg_entry.get("content")
            if isinstance(content, str):
                if len(content) == 0:
                    break
                new_messages[i] = MessageParam(
                    role="user",
                    content=[TextBlockParam(type="text", text=content, cache_control=marker)],
                )
            elif isinstance(content, list) and len(content) > 0:
                updated_blocks = list(content)
                # Find the last text block — only TextBlockParam
                # accepts cache_control on the user-message side
                # without changing semantics. tool_result blocks do
                # accept it; mark them only if no text block exists.
                marked_idx = -1
                for j in range(len(updated_blocks) - 1, -1, -1):
                    block = updated_blocks[j]
                    if isinstance(block, dict) and block.get("type") == "text":
                        marked_block = dict(block)
                        marked_block["cache_control"] = marker
                        # Same ToolUnionParam-narrowing rationale as
                        # the tools loop above: ``dict(block)`` is a
                        # plain dict to mypy, but the loop's type
                        # filter pinned it to a TextBlockParam /
                        # ToolResultBlockParam variant at runtime.
                        updated_blocks[j] = marked_block  # type: ignore[call-overload]
                        marked_idx = j
                        break
                if marked_idx == -1:
                    for j in range(len(updated_blocks) - 1, -1, -1):
                        block = updated_blocks[j]
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            marked_block = dict(block)
                            marked_block["cache_control"] = marker
                            # Same ToolUnionParam-narrowing rationale
                            # as the tools loop above.
                            updated_blocks[j] = marked_block  # type: ignore[call-overload]
                            break
                new_messages[i] = MessageParam(role="user", content=updated_blocks)
            break

    logger.debug(
        "auto_cache_control applied: system=%s tools=%s last_user_marked=%s ttl=%s",
        "marked" if new_system is not system else "skipped",
        "marked" if new_tools is not tools else "skipped",
        "marked" if new_messages is not messages else "skipped",
        ttl or "default",
    )

    return new_system, new_messages, new_tools
