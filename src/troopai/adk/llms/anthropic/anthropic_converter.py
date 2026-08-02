"""Stateless converter between Layer 1 items and Anthropic Messages API format.

Bridges provider-agnostic ``LLMInputContentItem`` types to/from
``anthropic.types.*`` wire types.  All methods are ``@classmethod``
— no instance state, pure functions.

Anthropic SDK types are used directly (``MessageParam``, ``ToolParam``,
etc.) and NEVER leak outside this module.

Key Anthropic differences from Chat Completions:

- System prompt is a separate ``system=`` parameter (not a message).
- Tool calls and results are content blocks (not separate arrays).
- Thinking blocks carry ``signature`` for multi-turn replay.
- Structured output uses a synthetic-tool pattern (no native
  ``response_format``): a single ``ToolParam(name="structured_output")``
  is registered, ``tool_choice`` forces the model to call it, and the
  resulting ``ToolUseBlock.input`` is the validated JSON.

Refs:
    - Anthropic Messages API: https://docs.anthropic.com/en/api/messages
    - Anthropic tool use: https://docs.anthropic.com/en/docs/build-with-claude/tool-use
    - Anthropic extended thinking: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Union

from anthropic.types import (
    Base64ImageSourceParam,
    ImageBlockParam,
    Message,
    MessageParam,
    RedactedThinkingBlock as AnthropicRedactedThinkingBlock,
    RedactedThinkingBlockParam,
    TextBlock,
    TextBlockParam,
    ThinkingBlock as AnthropicThinkingBlock,
    ThinkingBlockParam,
    ToolChoiceAnyParam,
    ToolChoiceAutoParam,
    ToolChoiceNoneParam,
    ToolChoiceParam,
    ToolChoiceToolParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUnionParam,
    ToolUseBlock,
    ToolUseBlockParam,
    URLImageSourceParam,
    Usage,
    WebSearchTool20250305Param,
)
from anthropic.types.web_search_tool_20250305_param import (
    UserLocation as AnthropicWebSearchUserLocation,
)

from troopai.adk.types.input import LLMInputContentItem
from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
    LLMResponsePart,
    LLMResponseProviderItem,
    LLMResponseReasoning,
    LLMResponseText,
)

if TYPE_CHECKING:
    from troopai.adk.llms.llm_usage import LLMUsage
    from troopai.adk.schemas import AgentOutputSchemaBase
    from troopai.adk.tools import Tool
    from troopai.adk.tools.hosted import WebSearchTool

# Name of the synthetic tool used to enforce structured output.
# Public so callers (and tests) can pin against it without re-deriving.
STRUCTURED_OUTPUT_TOOL_NAME = "structured_output"

# Defence-in-depth byte cap for serialised structured-output
# payloads. Anthropic's tool_use ``input`` is a Python dict with no
# server-side size bound, so we cap at twice
# ``AgentOutputSchema.MAX_SCHEMA_BYTES`` (256 KiB) before letting
# ``validate_json`` see the payload.
MAX_STRUCTURED_OUTPUT_BYTES: int = 512 * 1024

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type alias for content blocks we build.
# Every producer in this module returns one of these concrete params; the
# ``_convert_content_part`` fallback coerces unknown shapes into a
# ``TextBlockParam`` rather than emitting a raw dict, so the union matches
# Anthropic's ``MessageParam.content`` element type directly.
# ---------------------------------------------------------------------------
AnthropicContentBlock = Union[
    TextBlockParam,
    ToolUseBlockParam,
    ToolResultBlockParam,
    ImageBlockParam,
    ThinkingBlockParam,
    RedactedThinkingBlockParam,
]


class AnthropicConverter:
    """Stateless converter: Layer 1 ↔ Anthropic Messages API wire format."""

    # ------------------------------------------------------------------
    # Layer 1 → Anthropic messages
    # ------------------------------------------------------------------

    @staticmethod
    def _route_system_texts(
        texts: list[str],
        *,
        system_parts: list[str],
        messages: list[MessageParam],
        pending_content: list[AnthropicContentBlock],
        flush: Callable[[], None],
        preserve_mid_system: bool,
    ) -> None:
        """Send system texts to the top-level prompt or keep them in place.

        In-place placement only applies once at least one user/assistant
        turn exists (emitted or pending) — the API rejects a
        ``role:"system"`` entry at ``messages[0]``, so a leading system
        run always hoists.
        """
        non_empty = [t for t in texts if len(t) > 0]
        if len(non_empty) == 0:
            return
        has_turns = len(messages) > 0 or len(pending_content) > 0
        if preserve_mid_system and has_turns:
            flush()
            messages.append(
                MessageParam(
                    role="system",
                    content=[TextBlockParam(type="text", text=t) for t in non_empty],
                )
            )
        else:
            system_parts.extend(non_empty)

    @classmethod
    def items_to_messages(
        cls,
        items: str | list[LLMInputContentItem],
        *,
        preserve_mid_system: bool = False,
    ) -> tuple[str | None, list[MessageParam]]:
        """Convert Layer 1 items to Anthropic message format.

        Anthropic requires the system prompt as a separate ``system=``
        parameter, not as a message in the array.  This method extracts
        system/developer messages and returns them separately.

        Args:
            items: A plain string or list of Layer 1 content items.
            preserve_mid_system: When ``True``, system/developer items
                that appear after the first user/assistant turn stay in
                the messages array as ``role:"system"`` entries (the
                mid-conversation-system beta shape, which preserves the
                cached prefix). Leading system items are still extracted
                — the API rejects ``role:"system"`` at ``messages[0]``.
                ``False`` extracts every system item (the default).

        Returns:
            A ``(system_prompt, messages)`` tuple.
        """
        if isinstance(items, str):
            return None, [MessageParam(role="user", content=items)]

        system_parts: list[str] = []
        messages: list[MessageParam] = []
        # Accumulate content blocks for the current message. ``current_role``
        # is narrowed to Anthropic's ``Literal["user", "assistant"]`` so the
        # ``MessageParam`` construction below does not need an arg-type
        # suppression — every branch that assigns ``current_role`` restricts
        # the value to one of those two strings.
        current_role: Literal["user", "assistant"] | None = None
        current_content: list[AnthropicContentBlock] = []

        def flush() -> None:
            nonlocal current_role, current_content
            if current_role is not None and len(current_content) > 0:
                messages.append(MessageParam(role=current_role, content=current_content))
            current_role = None
            current_content = []

        for item in items:
            item_type = item.get("type") if isinstance(item, dict) else None

            # --- System / developer messages → extract to system_prompt,
            #     or keep in place after the first turn when opted in ---
            if isinstance(item, dict) and item_type == "message":
                role = item.get("role", "")
                if role in ("system", "developer"):
                    content = item.get("content", "")
                    texts: list[str] = []
                    if isinstance(content, str):
                        texts.append(content)
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "input_text":
                                texts.append(str(part.get("text", "")))
                    cls._route_system_texts(
                        texts,
                        system_parts=system_parts,
                        messages=messages,
                        pending_content=current_content,
                        flush=flush,
                        preserve_mid_system=preserve_mid_system,
                    )
                    continue

            # --- Determine target role and content block ---
            block: AnthropicContentBlock | None = None

            # User message
            if isinstance(item, dict) and item_type == "message":
                role = item.get("role", "user")
                # Convert content parts
                flush()
                msg_role: Literal["user", "assistant"] = "user" if role == "user" else "assistant"
                current_role = msg_role
                content = item.get("content", "")
                if isinstance(content, str):
                    current_content.append(TextBlockParam(type="text", text=content))
                elif isinstance(content, list):
                    for part in content:
                        converted = cls._convert_content_part(part)
                        if converted is not None:
                            current_content.append(converted)
                continue

            # Easy message (no type field, has content + optional role)
            if isinstance(item, dict) and "content" in item and item_type is None:
                role = item.get("role", "user")
                if role in ("system", "developer"):
                    content = item.get("content", "")
                    texts = []
                    if isinstance(content, str):
                        texts.append(content)
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "input_text":
                                texts.append(str(part.get("text", "")))
                    cls._route_system_texts(
                        texts,
                        system_parts=system_parts,
                        messages=messages,
                        pending_content=current_content,
                        flush=flush,
                        preserve_mid_system=preserve_mid_system,
                    )
                    continue
                target_role: Literal["user", "assistant"] = "user" if role == "user" else "assistant"
                flush()
                current_role = target_role
                content = item.get("content", "")
                if isinstance(content, str):
                    current_content.append(TextBlockParam(type="text", text=content))
                elif isinstance(content, list):
                    for part in content:
                        converted = cls._convert_content_part(part)
                        if converted is not None:
                            current_content.append(converted)
                continue

            # Text content part
            if isinstance(item, dict) and item_type == "input_text":
                if current_role is None:
                    current_role = "user"
                block = TextBlockParam(type="text", text=str(item.get("text", "")))

            # Image content part
            elif isinstance(item, dict) and item_type == "input_image":
                if current_role is None:
                    current_role = "user"
                # ``_convert_image`` is typed against plain ``dict[str, Any]``
                # because it inspects optional ``source`` / ``image_url`` keys
                # that no single Layer 1 TypedDict variant declares uniformly.
                # The ``isinstance(item, dict)`` guard above makes the runtime
                # shape a plain dict; launder via ``Any`` so pyright accepts
                # the call without a targeted ignore.
                image_item: dict[str, Any] = dict(item)
                block = cls._convert_image(image_item)

            # Tool call replay (assistant message with tool_use block)
            elif isinstance(item, dict) and item_type == "function_call":
                if current_role != "assistant":
                    flush()
                    current_role = "assistant"
                call_id = str(item.get("call_id", ""))
                name = str(item.get("name", ""))
                arguments = item.get("arguments", "{}")
                try:
                    raw_input = json.loads(arguments) if isinstance(arguments, str) else arguments
                except (json.JSONDecodeError, TypeError):
                    raw_input = {"raw": arguments}
                # Anthropic requires ``dict[str, object]`` for tool-use input.
                # Wrap non-dict values so the upstream JSON parse returning a
                # bare list/string still produces a well-formed param.
                input_data: dict[str, object] = dict(raw_input) if isinstance(raw_input, dict) else {"raw": raw_input}
                block = ToolUseBlockParam(
                    type="tool_use",
                    id=call_id,
                    name=name,
                    input=input_data,
                )

            # Tool result replay (user message with tool_result block)
            elif isinstance(item, dict) and item_type == "function_call_output":
                if current_role != "user":
                    flush()
                    current_role = "user"
                # Map framework status → Anthropic is_error. The framework
                # carries no dedicated boolean, but ``status == "incomplete"``
                # is the established signal that tool execution failed
                # (see ``FunctionToolCallResult.status``); surface that to
                # the model so it can decide whether to retry or recover.
                tool_result_param = ToolResultBlockParam(
                    type="tool_result",
                    tool_use_id=str(item.get("call_id", "")),
                    content=cls._tool_result_content(item.get("output", "")),
                )
                if item.get("status") == "incomplete":
                    tool_result_param["is_error"] = True
                block = tool_result_param

            # Assistant message replay
            elif isinstance(item, dict) and item_type == "message" and item.get("role") == "assistant":
                flush()
                current_role = "assistant"
                content = item.get("content", [])
                if isinstance(content, str):
                    current_content.append(TextBlockParam(type="text", text=content))
                elif isinstance(content, list):
                    for part in content:
                        converted = cls._convert_content_part(part)
                        if converted is not None:
                            current_content.append(converted)
                continue

            # Reasoning replay (thinking blocks for multi-turn)
            elif isinstance(item, dict) and item_type == "reasoning":
                if current_role != "assistant":
                    flush()
                    current_role = "assistant"
                # Replay thinking blocks
                content = item.get("content", [])
                encrypted = item.get("encrypted_content")
                encrypted_str = encrypted if isinstance(encrypted, str) else ""
                # Record what the current message already holds so the
                # signature-only fallback below keys on whether THIS
                # reasoning item produced a block — not on whether the
                # whole assistant turn is empty.
                blocks_before = len(current_content)
                if isinstance(content, list):
                    for tb in content:
                        if isinstance(tb, dict):
                            tb_type = tb.get("type", "")
                            if tb_type == "thinking":
                                current_content.append(
                                    ThinkingBlockParam(
                                        type="thinking",
                                        thinking=str(tb.get("thinking", "")),
                                        signature=str(tb.get("signature", "")),
                                    )
                                )
                            elif tb_type == "reasoning_text":
                                # Layer-1 reasoning content from
                                # ``LLMResponseReasoning.to_param``: the thinking text is
                                # under ``text`` and its signature is carried at the item
                                # level in ``encrypted_content``. Without this branch the
                                # loop matches nothing and the fallback below replays an
                                # EMPTY thinking block + the signature, which Anthropic
                                # rejects as a signature mismatch on multi-turn tool use.
                                current_content.append(
                                    ThinkingBlockParam(
                                        type="thinking",
                                        thinking=str(tb.get("text", "")),
                                        signature=encrypted_str,
                                    )
                                )
                            elif tb_type == "redacted_thinking":
                                current_content.append(
                                    RedactedThinkingBlockParam(
                                        type="redacted_thinking",
                                        data=str(tb.get("data", "")),
                                    )
                                )
                if encrypted is not None and isinstance(encrypted, str) and len(current_content) == blocks_before:
                    # Fallback: this reasoning item produced no thinking
                    # block of its own, so replay a single thinking block
                    # carrying its signature. Firing on the per-item count
                    # (not whole-message emptiness) preserves a
                    # signature-only reasoning item even when the assistant
                    # turn already holds preceding content — dropping it
                    # breaks Anthropic's multi-turn signature validation.
                    current_content.append(
                        ThinkingBlockParam(
                            type="thinking",
                            thinking="",
                            signature=encrypted,
                        )
                    )
                continue

            if block is not None:
                current_content.append(block)

        flush()

        system_prompt = "\n\n".join(system_parts) if len(system_parts) > 0 else None

        # Anthropic requires the first message to be a user turn. The
        # framework's normal flow always builds a user prompt first, so
        # only a developer-supplied history that opens with an
        # assistant/tool turn reaches this branch. Fabricating a filler
        # user message would inject a token the developer never wrote;
        # raise a clear, actionable error at the framework boundary
        # instead of an opaque provider 400.
        if len(messages) > 0 and messages[0]["role"] != "user":
            raise ValueError(
                "Anthropic requires the first message to have role 'user', but the "
                f"converted history starts with role '{messages[0]['role']}'. Supply a "
                "conversation that opens with a user turn — an assistant-first history "
                "is not a valid Anthropic request."
            )

        return system_prompt, messages

    @classmethod
    def _convert_content_part(cls, part: Any) -> AnthropicContentBlock | None:
        """Convert a single content part to an Anthropic content block."""
        if isinstance(part, str):
            return TextBlockParam(type="text", text=part)
        if not isinstance(part, dict):
            return None
        part_type = part.get("type", "")
        if part_type == "input_text":
            return TextBlockParam(type="text", text=str(part.get("text", "")))
        if part_type == "input_image":
            return cls._convert_image(part)
        if part_type == "output_text":
            return TextBlockParam(type="text", text=str(part.get("text", "")))
        if part_type == "text":
            return TextBlockParam(type="text", text=str(part.get("text", "")))
        if part_type == "thinking":
            return ThinkingBlockParam(
                type="thinking",
                thinking=str(part.get("thinking", "")),
                signature=str(part.get("signature", "")),
            )
        if part_type == "redacted_thinking":
            return RedactedThinkingBlockParam(
                type="redacted_thinking",
                data=str(part.get("data", "")),
            )
        logger.debug("Unknown content part type: %s", part_type)
        return TextBlockParam(type="text", text=str(part))

    @classmethod
    def _tool_result_content(
        cls,
        output: object,
    ) -> str | list[TextBlockParam | ImageBlockParam]:
        """Convert a tool-result ``output`` to Anthropic tool_result content.

        A plain string passes through unchanged. A list of multimodal
        content parts (``LLMInputText`` / ``LLMInputImage``) is converted
        to typed Anthropic content blocks so an image or structured tool
        result is NOT collapsed to a Python ``repr`` via ``str(list)``.
        Any other type is stringified. An empty conversion falls back to
        the stringified output — Anthropic rejects ``content=[]``.
        """
        if isinstance(output, str):
            return output
        if isinstance(output, list):
            blocks: list[TextBlockParam | ImageBlockParam] = []
            for part in output:
                converted = cls._convert_tool_result_part(part)
                if converted is not None:
                    blocks.append(converted)
            if len(blocks) == 0:
                return str(output)
            return blocks
        return str(output)

    @classmethod
    def _convert_tool_result_part(cls, part: object) -> TextBlockParam | ImageBlockParam | None:
        """Convert one tool-result content part to a tool_result block.

        Only text and image blocks are valid inside a ``tool_result``;
        text-shaped parts become :class:`TextBlockParam`, image parts
        become :class:`ImageBlockParam`. An unknown dict shape is
        preserved as text rather than dropped.
        """
        if isinstance(part, str):
            return TextBlockParam(type="text", text=part)
        if not isinstance(part, dict):
            return None
        part_type = part.get("type", "")
        if part_type in ("input_text", "output_text", "text"):
            return TextBlockParam(type="text", text=str(part.get("text", "")))
        if part_type in ("input_image", "output_image"):
            # ``_convert_image`` inspects optional ``source`` / ``image_url``
            # keys; launder to a plain dict so pyright accepts the call.
            image_item: dict[str, Any] = dict(part)
            return cls._convert_image(image_item)
        logger.debug("Unknown tool_result content part type: %s", part_type)
        return TextBlockParam(type="text", text=str(part))

    # Anthropic's base64 image source only accepts these media types.
    _BASE64_MEDIA_TYPES: tuple[
        Literal["image/jpeg", "image/png", "image/gif", "image/webp"],
        ...,
    ] = ("image/jpeg", "image/png", "image/gif", "image/webp")

    @classmethod
    def _narrow_image_media_type(
        cls,
        value: str,
    ) -> Literal["image/jpeg", "image/png", "image/gif", "image/webp"]:
        """Narrow a raw media-type string to Anthropic's base64 literal set.

        Falls back to ``"image/jpeg"`` for an unrecognised value so the
        wire param stays well-typed (Anthropic rejects other strings).
        """
        for allowed in cls._BASE64_MEDIA_TYPES:
            if value == allowed:
                return allowed
        return "image/jpeg"

    @classmethod
    def _convert_image(cls, item: dict[str, Any]) -> ImageBlockParam:
        """Convert an LLMInputImage to Anthropic ImageBlockParam."""
        source = item.get("source", item.get("image_url", ""))
        if isinstance(source, str):
            if source.startswith("data:"):
                # Base64 data URI — Anthropic requires a Base64ImageSourceParam
                # with split ``media_type`` / ``data``; a ``data:`` string in a
                # URL source is rejected by the API.
                return cls._data_uri_to_image_block(source)
            # URL-based image
            url_source: URLImageSourceParam = {"type": "url", "url": source}
            return ImageBlockParam(type="image", source=url_source)
        if isinstance(source, dict):
            # Caller-provided dict — already a URL or base64 source shape.
            # Route by ``type`` so the resulting ImageBlockParam is typed.
            src_type = source.get("type")
            if src_type == "url":
                url_src: URLImageSourceParam = {
                    "type": "url",
                    "url": str(source.get("url", "")),
                }
                return ImageBlockParam(type="image", source=url_src)
            if src_type == "base64":
                media_type = source.get("media_type")
                if media_type is None:
                    raise ValueError(
                        "base64 image source is missing required field 'media_type' (e.g. 'image/jpeg', 'image/png')"
                    )
                b64_src: Base64ImageSourceParam = {
                    "type": "base64",
                    "media_type": cls._narrow_image_media_type(str(media_type)),
                    "data": str(source.get("data", "")),
                }
                return ImageBlockParam(type="image", source=b64_src)
            # Unknown source shape — fall through to stringified URL below.
        fallback_source: URLImageSourceParam = {"type": "url", "url": str(source)}
        return ImageBlockParam(type="image", source=fallback_source)

    @classmethod
    def _data_uri_to_image_block(cls, source: str) -> ImageBlockParam:
        """Split a ``data:<mime>[;base64],<body>`` URI into a base64 image block."""
        header, _, body = source.partition(",")
        # ``header`` is ``data:<mime>[;param...]`` — the mime is the first
        # token after ``data:`` up to the first ``;``.
        descriptor = header[len("data:") :]
        media_type = descriptor.split(";", 1)[0]
        b64_src: Base64ImageSourceParam = {
            "type": "base64",
            "media_type": cls._narrow_image_media_type(media_type),
            "data": body,
        }
        return ImageBlockParam(type="image", source=b64_src)

    # ------------------------------------------------------------------
    # Tool conversion
    # ------------------------------------------------------------------

    @classmethod
    def convert_tools(cls, tools: list[Tool]) -> list[ToolUnionParam]:
        """Convert Tool instances to Anthropic tool definitions.

        Handles three framework-owned tool categories that reach the
        LLM wire layer:

        - ``FunctionTool`` → ``ToolParam`` (custom function)
        - ``ExecutableBuiltinTool`` → ``ToolParam`` (function format)
        - ``HostedTool`` subclass → provider-native tool param.
          Anthropic supports :class:`WebSearchTool`. Other variants
          raise :class:`UnsupportedHostedToolError`.

        ``ShellTool`` / ``ApplyPatchTool`` with a local executor/editor
        are wrapped as ``FunctionTool`` upstream (in
        ``run.llm_calls.build_tools``) and reach this method as plain
        function tools.

        Args:
            tools: List of Tool instances (pre-filtered by Runner).

        Returns:
            List of Anthropic tool definitions.

        Raises:
            UnsupportedHostedToolError: If a hosted-tool variant is
                supplied that Anthropic does not support
                (``CodeExecutionTool``, ``FileSearchTool``,
                ``ImageGenerationTool``, ``URLContextTool``).
        """
        from troopai.adk.tools.builtin.builtin_tool import ExecutableBuiltinTool
        from troopai.adk.tools.function_tool import FunctionTool
        from troopai.adk.tools.hosted import (
            HostedTool,
            UnsupportedHostedToolError,
            WebSearchTool,
        )

        wire_tools: list[ToolUnionParam] = []
        for tool in tools:
            if isinstance(tool, FunctionTool):
                tool_param = ToolParam(
                    name=tool.name,
                    input_schema=tool.get_json_schema() or {"type": "object", "properties": {}},
                )
                if tool.description is not None:
                    tool_param["description"] = tool.description
                wire_tools.append(tool_param)

            elif isinstance(tool, ExecutableBuiltinTool):
                # Executable builtins (JITContextAwareTool, MemoryTool) → function format
                from pydantic import BaseModel as PydanticBaseModel

                from troopai.adk.schemas.utils import normalize_schema

                raw_schema = (
                    tool.schema.model_json_schema()
                    if isinstance(tool.schema, type) and issubclass(tool.schema, PydanticBaseModel)
                    else tool.schema
                )
                schema = normalize_schema(raw_schema) or {"type": "object", "properties": {}}
                exec_tool_param = ToolParam(
                    name=tool.name,
                    input_schema=schema,
                )
                if tool.description is not None:
                    exec_tool_param["description"] = tool.description
                wire_tools.append(exec_tool_param)

            elif isinstance(tool, WebSearchTool):
                wire_tools.append(cls._convert_web_search(tool))

            elif isinstance(tool, HostedTool):
                # Hosted tool Anthropic does not ship — surface a
                # typed error rather than silently dropping. The
                # class's own ``SUPPORTED_PROVIDERS`` tuple drives the
                # error message so the developer learns where to use
                # the tool instead.
                raise UnsupportedHostedToolError(
                    tool,
                    "anthropic",
                    supported_providers=tool.SUPPORTED_PROVIDERS,
                )

            else:
                logger.warning("Unknown tool type for Anthropic: %s", type(tool))

        return wire_tools

    @classmethod
    def _convert_web_search(cls, tool: WebSearchTool) -> WebSearchTool20250305Param:
        """Translate a framework :class:`WebSearchTool` to Anthropic's wire shape.

        Reads the Anthropic-honoured attributes (``max_uses``,
        ``allowed_domains``, ``blocked_domains``, ``user_location``)
        and ignores OpenAI-only attributes (``search_context_size``)
        with a debug log.
        """
        param: WebSearchTool20250305Param = {
            "type": "web_search_20250305",
            "name": "web_search",
        }
        if tool.max_uses is not None:
            param["max_uses"] = tool.max_uses
        if tool.allowed_domains is not None:
            param["allowed_domains"] = list(tool.allowed_domains)
        if tool.blocked_domains is not None:
            param["blocked_domains"] = list(tool.blocked_domains)
        if tool.user_location is not None:
            # Anthropic's ``UserLocation`` requires ``type: "approximate"``.
            location: AnthropicWebSearchUserLocation = {"type": "approximate"}
            for key in ("city", "country", "region", "timezone"):
                value = tool.user_location.get(key)
                if value is not None:
                    location[key] = value  # type: ignore[literal-required]
            param["user_location"] = location
        if tool.search_context_size is not None:
            logger.debug(
                "Anthropic: ignoring WebSearchTool.search_context_size=%s (OpenAI-only attribute).",
                tool.search_context_size,
            )
        return param

    @classmethod
    def convert_tool_choice(
        cls,
        tool_choice: str | None,
        tools_present: bool,
    ) -> ToolChoiceParam | None:
        """Convert ToolChoice string to Anthropic format.

        Mapping:
        - ``"auto"`` → ``{"type": "auto"}``
        - ``"required"`` → ``{"type": "any"}``
        - ``"none"`` → ``None`` (omit tools)
        - ``tool_name`` → ``{"type": "tool", "name": tool_name}``

        Args:
            tool_choice: The ToolChoice string or None.
            tools_present: Whether tools are being sent.

        Returns:
            Anthropic tool_choice dict, or None.
        """
        if tool_choice is None or not tools_present:
            return None
        if tool_choice == "auto":
            return ToolChoiceAutoParam(type="auto")
        if tool_choice == "required":
            return ToolChoiceAnyParam(type="any")
        if tool_choice == "none":
            return ToolChoiceNoneParam(type="none")
        # Named tool
        return ToolChoiceToolParam(type="tool", name=tool_choice)

    # ------------------------------------------------------------------
    # Anthropic response → LLMResponse
    # ------------------------------------------------------------------

    @classmethod
    def response_to_llm_response(
        cls,
        response: Message,
    ) -> LLMResponse:
        """Convert Anthropic ``Message`` to provider-agnostic ``LLMResponse``.

        Extracts text, tool calls, thinking blocks, and usage. Structured
        output (``output_schema``) is handled upstream in
        ``AnthropicLLM.acomplete`` via the synthetic-tool pattern before
        this method is called, so it has no effect here. Only the
        ``stream=True`` + ``output_schema`` combination is unsupported —
        that raises ``NotImplementedError`` there, since validation needs
        the complete tool_use block.

        Args:
            response: The Anthropic API response.

        Returns:
            A provider-agnostic ``LLMResponse``.
        """
        parts: list[LLMResponsePart] = []

        for block in response.content:
            if isinstance(block, TextBlock):
                parts.append(LLMResponseText(text=block.text))

            elif isinstance(block, ToolUseBlock):
                arguments = json.dumps(block.input) if isinstance(block.input, dict) else str(block.input)
                parts.append(
                    LLMResponseFunctionToolCall(
                        call_id=block.id,
                        name=block.name,
                        arguments=arguments,
                    )
                )

            elif isinstance(block, AnthropicThinkingBlock):
                parts.append(
                    LLMResponseReasoning(
                        thinking=block.thinking,
                        signature=block.signature,
                    )
                )

            elif isinstance(block, AnthropicRedactedThinkingBlock):
                parts.append(
                    LLMResponseReasoning(
                        thinking="",
                        encrypted_content=block.data,
                        is_redacted=True,
                    )
                )

            else:
                # Server-executed tool blocks (``server_tool_use``,
                # ``web_search_tool_result``, ``code_execution_tool_result``,
                # …) don't fit the text / reasoning / function-call taxonomy.
                # Surface them verbatim through the provider-item channel so
                # the search results / citations they carry are not silently
                # dropped — the loop already round-trips
                # ``LLMResponseProviderItem`` via the ``ProviderItem``
                # RunItem. The raw payload is kept JSON-safe for tracing and
                # durable checkpointing.
                block_type = getattr(block, "type", "")
                if len(block_type) > 0 and hasattr(block, "model_dump"):
                    parts.append(
                        LLMResponseProviderItem(
                            item_type=block_type,
                            raw=block.model_dump(mode="json"),
                        )
                    )
                else:
                    logger.debug("Unhandled content block type: %s", type(block).__name__)

        # Parse usage
        usage = cls._parse_usage(response.usage)

        # Encode structured refusal details into finish_reason so callers can
        # distinguish safety categories without a separate API round-trip.
        # When stop_reason == "refusal" and stop_details carries a category,
        # emit "refusal:<category>" (e.g. "refusal:cyber", "refusal:bio").
        finish_reason: str | None = response.stop_reason
        if finish_reason == "refusal":
            stop_details = getattr(response, "stop_details", None)
            if stop_details is not None:
                category = getattr(stop_details, "category", None)
                if category:
                    finish_reason = f"refusal:{category}"

        return LLMResponse(
            response_id=response.id,
            model=response.model,
            response=parts,
            usage=usage,
            finish_reason=finish_reason,
        )

    # ------------------------------------------------------------------
    # Structured output (synthetic-tool pattern)
    # ------------------------------------------------------------------

    @classmethod
    def build_structured_output_tool(
        cls,
        output_schema: AgentOutputSchemaBase,
    ) -> ToolParam:
        """Build the synthetic tool used to enforce structured output.

        Anthropic's Messages API has no native ``response_format``;
        the documented pattern is to register a single tool whose
        ``input_schema`` matches the desired JSON shape and force
        ``tool_choice`` onto it. The model emits a ``ToolUseBlock``
        whose ``.input`` is the validated structured payload.

        Args:
            output_schema: The framework schema wrapper. Must NOT be
                ``is_plain_text()`` — callers handle the plain-text
                short-circuit before invoking this method.

        Returns:
            A single :class:`ToolParam` named ``structured_output``.
        """
        schema_json = output_schema.json_schema()
        return ToolParam(
            name=STRUCTURED_OUTPUT_TOOL_NAME,
            description=(
                "Return the answer as a JSON object matching this schema. "
                "You MUST use this tool — it is the only way to deliver the "
                "final response."
            ),
            input_schema=schema_json,
        )

    @classmethod
    def build_structured_output_tool_choice(cls) -> ToolChoiceToolParam:
        """Build the ``tool_choice`` payload that forces the synthetic tool.

        Returned shape: ``{"type": "tool", "name": "structured_output"}``.
        """
        return ToolChoiceToolParam(type="tool", name=STRUCTURED_OUTPUT_TOOL_NAME)

    @classmethod
    def parse_structured_output(
        cls,
        message: Message,
        output_schema: AgentOutputSchemaBase,
    ) -> object:
        """Locate and validate the synthetic tool's output.

        Walks ``message.content`` for the first ``ToolUseBlock`` named
        ``structured_output``, JSON-encodes its ``input`` dict, and runs
        it through :meth:`AgentOutputSchemaBase.validate_json` so the
        same wrapping / unwrapping / strict-mode rules apply as on the
        litellm and OpenAI paths.

        Args:
            message: The Anthropic response.
            output_schema: The schema wrapper that produced the tool.

        Returns:
            The validated Python object.

        Raises:
            ValueError: If no matching ``ToolUseBlock`` is present, or
                if its input fails schema validation, or if the
                serialised payload exceeds
                :data:`MAX_STRUCTURED_OUTPUT_BYTES` (defence-in-depth
                cap before ``validate_json`` parses).
        """
        for block in message.content:
            if isinstance(block, ToolUseBlock) and block.name == STRUCTURED_OUTPUT_TOOL_NAME:
                payload = block.input if isinstance(block.input, dict) else {"value": block.input}
                serialized = json.dumps(payload)
                # Hard cap before handing to ``validate_json``.
                # Anthropic's tool_use ``input`` is a Python dict
                # delivered by the SDK with no server-side size
                # bound; under high concurrency a pathological
                # response could pressure memory before the
                # downstream cap fires. 512 KiB is generous (twice
                # ``AgentOutputSchema.MAX_SCHEMA_BYTES``) but bounded.
                if len(serialized.encode("utf-8")) > MAX_STRUCTURED_OUTPUT_BYTES:
                    raise ValueError(
                        f"Anthropic structured-output payload exceeds {MAX_STRUCTURED_OUTPUT_BYTES} "
                        "bytes; refusing to validate. Reduce the schema or split the tool output."
                    )
                return output_schema.validate_json(serialized)
        raise ValueError(
            "Anthropic structured-output response did not contain the synthetic "
            f"'{STRUCTURED_OUTPUT_TOOL_NAME}' tool call. The model may have "
            "produced a refusal or text reply instead — inspect "
            "``LLMResponse.response`` for diagnostics."
        )

    @classmethod
    def _parse_usage(cls, usage: Usage) -> LLMUsage:
        """Convert Anthropic ``Usage`` to ``LLMUsage``."""
        from troopai.adk.types.tokens import (
            InputTokensDetails,
            LLMSingleRequestUsage,
            LLMUsage,
            OutputTokensDetails,
        )

        cached = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0

        # Extended-thinking token count — available when thinking is enabled.
        # The SDK exposes ``usage.output_tokens_details.thinking_tokens``; fall
        # back to 0 when the field is absent (non-thinking models).
        _out_details = getattr(usage, "output_tokens_details", None)
        reasoning_tokens: int = getattr(_out_details, "thinking_tokens", 0) or 0 if _out_details is not None else 0

        # Anthropic reports ``input_tokens`` EXCLUSIVE of cache-read and
        # cache-creation tokens; both are billed prompt input the token
        # limits and cost tracking must see. Report the inclusive total so
        # the native path matches the litellm path (which sums the same
        # three counts into ``prompt_tokens``) — otherwise limits never
        # trip when prompt caching is active. ``cached_tokens`` keeps the
        # cache-read subset for the per-request breakdown.
        input_tokens = usage.input_tokens + cached + cache_creation
        total_tokens = input_tokens + usage.output_tokens

        input_details = InputTokensDetails(
            cached_tokens=cached,
            cache_creation_input_tokens=cache_creation,
        )
        output_details = OutputTokensDetails(reasoning_tokens=reasoning_tokens)

        single = LLMSingleRequestUsage(
            input_tokens=input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=total_tokens,
            input_tokens_details=input_details,
            output_tokens_details=output_details,
        )

        return LLMUsage(
            requests=1,
            input_tokens=input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=total_tokens,
            input_tokens_details=input_details,
            output_tokens_details=output_details,
            usage=[single],
        )
