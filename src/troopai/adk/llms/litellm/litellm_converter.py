"""Bidirectional converter between provider-agnostic items and Chat Completions messages.

Centralizes ALL conversion between the runner's provider-agnostic item layer
and the Chat Completions wire format.  All ``LLM`` implementations that use
the Chat Completions API share this converter.

Conversion methods:

- **Items → messages** (``items_to_messages``): ``UserPrompt``
  → ``list[ChatCompletionMessageParam]``
- **Tools → wire format** (``convert_tool``, ``convert_tools``):
  ``FunctionTool`` → ``ChatCompletionFunctionToolParam``
- **Tool choice** (``convert_tool_choice``): ``ToolChoice`` string
  → ``ChatCompletionToolChoiceOptionParam``
- **Response format** (``convert_response_format``): ``AgentOutputSchemaBase``
  → ``ChatCompletionResponseFormatParam``
- **Content extraction** (``extract_text_content``, ``extract_all_content``):
  Content normalization for Chat Completions

Design:
- Stateless (all ``@classmethod`` methods)
- Handles assistant message buffering (tool calls must be attached to an
  assistant message in Chat Completions format)
- Preserves thinking blocks / reasoning for multi-turn tool use
- ``maybe_*()`` type discriminators for clean dispatch
- Provider-specific handling for Anthropic (thinking blocks), DeepSeek
  (reasoning_content), and Gemini (thought_signature)

See also:
- OpenAI agents SDK ``chatcmpl_converter.py`` — similar pattern
  https://github.com/openai/openai-agents-python/blob/main/src/agents/models/chatcmpl_converter.py
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Literal, Union

# litellm SDK types — input (TypedDict, sent TO litellm)
# Aliased to preserve existing names used throughout the converter
from litellm.types.llms.openai import (  # pyright: ignore[reportPrivateImportUsage]
    AllMessageValues as ChatCompletionMessageParam,
    ChatCompletionAssistantMessage as ChatCompletionAssistantMessageParam,
    ChatCompletionAssistantToolCall as ChatCompletionToolCallParam,
    ChatCompletionContentPartInputAudioParam,  # pyright: ignore[reportPrivateImportUsage]
    ChatCompletionDeveloperMessage as ChatCompletionDeveloperMessageParam,
    ChatCompletionImageObject as ChatCompletionContentPartImageParam,
    ChatCompletionImageUrlObject as ImageURL,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionSystemMessage as ChatCompletionSystemMessageParam,
    ChatCompletionTextObject as ChatCompletionContentPartTextParam,
    ChatCompletionToolCallFunctionChunk as ChatCompletionFunctionCallParam,
    ChatCompletionToolChoiceValues as ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolMessage as ChatCompletionToolMessageParam,
    ChatCompletionUserMessage as ChatCompletionUserMessageParam,
)

from troopai.adk.types.input import (
    LLMInputAudio,
    LLMInputContentItem,
    LLMInputContentPart,
    LLMInputEasyMessage,
    LLMInputImage,
    LLMInputMessage,
    LLMInputText,
)
from troopai.adk.types.output import (
    FunctionToolCallResultParam,
    LLMResponseFunctionToolCallParam,
    LLMResponseMessageParam,
    LLMResponseReasoningParam,
)

if TYPE_CHECKING:
    from litellm.types.llms.openai import ChatCompletionToolParam as ChatCompletionFunctionToolParam
    from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio

    from troopai.adk.schemas import AgentOutputSchemaBase
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.types.items import RunItem

logger = logging.getLogger(__name__)

# Content part union (litellm doesn't define a combined union type)
ChatCompletionContentPartParam = Union[
    ChatCompletionContentPartTextParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
]


class ChatCompletionConverter:
    """Stateless converter between provider-agnostic items and Chat Completions messages.

    All methods are ``@classmethod`` — no instance state.

    Load-bearing marker rationale (collective):

    - ``# type: ignore[return-value]`` on ``maybe_*`` discriminators —
      pyright cannot narrow a TypedDict union by ``.get("type") == "..."``.
      The runtime equality check is the load-bearing invariant.
    - ``# type: ignore[typeddict-item]`` / ``[typeddict-unknown-key]`` on
      ``msg["content"] = ...`` and ``msg["refusal"] = ...`` — the
      Chat-Completions message TypedDicts overconstrain ``content`` and
      omit ``refusal`` entirely, but litellm accepts both at runtime; the
      marker is scoped to the one assignment where shape alignment is
      known.
    - ``# type: ignore[arg-type]`` on ``ItemHelpers.message(s)_to_run_items``
      calls — the helpers are typed against Layer 1 messages while this
      converter crosses the Layer 2 ↔ Layer 3 boundary; the narrow ignore
      isolates the one boundary hop.
    """

    # ==================================================================
    # Item type discriminators (maybe_* methods)
    # ==================================================================

    # --- Discriminators ---
    # Each maybe_* method checks the discriminator field(s) and returns
    # the item narrowed to the target type. Pyright cannot narrow a
    # Union[TypedDict, ...] via field equality, so the ``return item``
    # line carries ``# type: ignore[return-value]`` — the runtime check
    # proves safety.

    @staticmethod
    def maybe_easy_input_message(item: LLMInputContentItem) -> LLMInputEasyMessage | None:
        """Check if item is an easy input message (has role, no type)."""
        if item.get("type") is None and "role" in item and ("content" in item or "tool_calls" in item):
            return item  # type: ignore[return-value]
        return None

    @staticmethod
    def maybe_input_message(item: LLMInputContentItem) -> LLMInputMessage | None:
        """Check if item is a strict input message (type="message", non-assistant role)."""
        if item.get("type") == "message" and item.get("role") in ("user", "system", "developer"):
            return item  # type: ignore[return-value]
        return None

    @staticmethod
    def maybe_output_message(item: LLMInputContentItem) -> LLMResponseMessageParam | None:
        """Check if item is an output message replay (type="message", role="assistant")."""
        if item.get("type") == "message" and item.get("role") == "assistant":
            return item  # type: ignore[return-value]
        return None

    @staticmethod
    def maybe_function_tool_call(item: LLMInputContentItem) -> LLMResponseFunctionToolCallParam | None:
        """Check if item is a function tool call (type="function_call")."""
        if item.get("type") == "function_call":
            return item  # type: ignore[return-value]
        return None

    @staticmethod
    def maybe_function_tool_call_output(item: LLMInputContentItem) -> FunctionToolCallResultParam | None:
        """Check if item is a function tool call output (type="function_call_output")."""
        if item.get("type") == "function_call_output":
            return item  # type: ignore[return-value]
        return None

    @staticmethod
    def maybe_reasoning(item: LLMInputContentItem) -> LLMResponseReasoningParam | None:
        """Check if item is a reasoning item (type="reasoning")."""
        if item.get("type") == "reasoning":
            return item  # type: ignore[return-value]
        return None

    @staticmethod
    def maybe_input_text(item: LLMInputContentItem) -> LLMInputText | None:
        """Check if item is an input text content part (type="input_text")."""
        if item.get("type") == "input_text":
            return item  # type: ignore[return-value]
        return None

    @staticmethod
    def maybe_input_image(item: LLMInputContentItem) -> LLMInputImage | None:
        """Check if item is an input image content part (type="input_image")."""
        if item.get("type") == "input_image":
            return item  # type: ignore[return-value]
        return None

    @staticmethod
    def maybe_input_audio(item: LLMInputContentItem) -> LLMInputAudio | None:
        """Check if item is an input audio content part (type="input_audio")."""
        if item.get("type") == "input_audio":
            return item  # type: ignore[return-value]
        return None

    @staticmethod
    def maybe_item_reference(item: LLMInputContentItem) -> dict[str, object] | None:
        """Check if item is an item reference (type="item_reference")."""
        if item.get("type") == "item_reference":
            return item  # type: ignore[return-value]
        return None

    @staticmethod
    def maybe_compaction(item: LLMInputContentItem) -> dict[str, object] | None:
        """Check if item is a compaction item (type="compaction")."""
        if item.get("type") == "compaction":
            return item  # type: ignore[return-value]
        return None

    # ==================================================================
    # Request direction: items → messages
    # ==================================================================

    @classmethod
    def items_to_messages(
        cls,
        items: str | Iterable[LLMInputContentItem],
        model: str | None = None,
        preserve_thinking_blocks: bool = False,
        preserve_tool_output_all_content: bool = False,
    ) -> list[ChatCompletionMessageParam]:
        """Convert provider-agnostic input items to Chat Completions messages.

        Handles the full ``LLMInputContentItem`` union:
        - Content parts (text, image, audio) → wrapped in a user message
        - Messages (strict and easy) → mapped by role
        - Tool call replay → attached to assistant message buffer
        - Tool call output replay → tool message
        - Assistant message replay → assistant message
        - Reasoning replay → pending thinking blocks for next assistant message

        Maintains an assistant message buffer to accumulate tool calls before
        flushing, matching the Chat Completions API requirement that tool calls
        are part of an assistant message.

        Args:
            items: A plain string or an iterable of ``LLMInputContentItem``.
            model: Model identifier — used for provider-specific handling
                (e.g., thinking block format varies by provider). Pass the
                model string to enable Anthropic thinking block restoration
                and DeepSeek reasoning_content.
            preserve_thinking_blocks: If ``True``, include thinking blocks
                in assistant message content (required for multi-turn tool
                use with extended thinking on Anthropic).
            preserve_tool_output_all_content: If ``True``, preserve multimodal
                tool output content as-is. If ``False``, extract text only.

        Returns:
            A list of ``ChatCompletionMessageParam`` ready for
            ``litellm.acompletion()``.
        """
        if isinstance(items, str):
            return [ChatCompletionUserMessageParam(role="user", content=items)]

        messages: list[ChatCompletionMessageParam] = []
        current_assistant: ChatCompletionAssistantMessageParam | None = None
        pending_content_parts: list[ChatCompletionContentPartParam] = []
        pending_thinking_blocks: list[dict[str, str]] = []
        pending_reasoning_content: str | None = None

        model_lower = model.lower() if model is not None else ""
        is_anthropic = "claude" in model_lower or "anthropic" in model_lower
        is_deepseek = "deepseek" in model_lower

        def flush_assistant_message() -> None:
            nonlocal current_assistant, pending_reasoning_content
            if current_assistant is not None:
                # Clean up empty tool_calls — API rejects empty arrays
                if "tool_calls" in current_assistant and not current_assistant["tool_calls"]:
                    del current_assistant["tool_calls"]
                messages.append(current_assistant)
                current_assistant = None
            pending_reasoning_content = None

        for item in items:
            # --- Content parts → accumulate for user message ---
            narr_text = cls.maybe_input_text(item)
            if narr_text is not None:
                pending_content_parts.append(
                    ChatCompletionContentPartTextParam(type="text", text=str(narr_text["text"]))
                )
                continue

            narr_img = cls.maybe_input_image(item)
            if narr_img is not None:
                image_url = str(narr_img.get("image_url", ""))
                raw_detail = str(narr_img.get("detail", "auto"))
                # Normalize "original" → "high" (Chat Completions only supports auto/low/high)
                detail: Literal["auto", "low", "high"]
                if raw_detail == "low":
                    detail = "low"
                elif raw_detail in ("high", "original"):
                    detail = "high"
                else:
                    detail = "auto"
                pending_content_parts.append(
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url=ImageURL(url=image_url, detail=detail),
                    )
                )
                continue

            narr_audio = cls.maybe_input_audio(item)
            if narr_audio is not None:
                raw_format = str(narr_audio["format"])
                audio_format: Literal["wav", "mp3"] = "mp3" if raw_format == "mp3" else "wav"
                audio_data: InputAudio = {"data": str(narr_audio["data"]), "format": audio_format}
                pending_content_parts.append(
                    ChatCompletionContentPartInputAudioParam(
                        type="input_audio",
                        input_audio=audio_data,
                    )
                )
                continue

            # Flush pending content parts as a user message. Emit any open
            # assistant buffer first so a user turn that arrived after a tool
            # call is not reordered ahead of the assistant turn it follows.
            if len(pending_content_parts) > 0:
                flush_assistant_message()
                messages.append(
                    ChatCompletionUserMessageParam(
                        role="user",
                        content=pending_content_parts,
                    )
                )
                pending_content_parts = []

            # --- Item reference → not supported ---
            if cls.maybe_item_reference(item):
                logger.warning("item_reference items are not supported in Chat Completions — skipping")
                continue

            # --- Compaction → not supported ---
            if cls.maybe_compaction(item):
                logger.warning("compaction items are not supported in Chat Completions — skipping")
                continue

            # --- Reasoning → pending thinking blocks ---
            narr_reasoning = cls.maybe_reasoning(item)
            if narr_reasoning is not None:
                item_provider_data = narr_reasoning.get("provider_data", {})
                item_model = str(item_provider_data.get("model", "")) if isinstance(item_provider_data, dict) else ""

                if is_deepseek:
                    # The chain-of-thought lives in ``content`` (DeepSeek emits it
                    # as ``thinking`` → ``reasoning_text``); ``summary`` is only set
                    # when the model exposed an explicit summary. Prefer ``content``
                    # and fall back to ``summary`` so neither shape loses the text.
                    reasoning_blocks: list[Any] | None = narr_reasoning.get("content")
                    if not (isinstance(reasoning_blocks, list) and len(reasoning_blocks) > 0):
                        reasoning_blocks = narr_reasoning.get("summary", [])
                    if isinstance(reasoning_blocks, list) and len(reasoning_blocks) > 0:
                        texts = [str(s.get("text", "")) for s in reasoning_blocks if isinstance(s, dict)]
                        pending_reasoning_content = "\n".join(texts) if len(texts) > 0 else None
                elif preserve_thinking_blocks and is_anthropic:
                    item_model_lower = item_model.lower()
                    if len(item_model) == 0 or "claude" in item_model_lower or "anthropic" in item_model_lower:
                        thinking = cls._extract_thinking_blocks(narr_reasoning)
                        pending_thinking_blocks.extend(thinking)
                continue

            # --- Output message (assistant) replay ---
            narr_out_msg = cls.maybe_output_message(item)
            if narr_out_msg is not None:
                flush_assistant_message()
                msg = ChatCompletionAssistantMessageParam(role="assistant")

                content_list: list[dict[str, str] | ChatCompletionContentPartTextParam] = []
                if len(pending_thinking_blocks) > 0:
                    content_list.extend(pending_thinking_blocks)
                    pending_thinking_blocks = []

                raw_content = narr_out_msg.get("content", [])
                text_parts: list[str] = []
                refusal_text: str | None = None
                for c in raw_content if isinstance(raw_content, list) else []:
                    if isinstance(c, dict):
                        ctype = c.get("type", "")
                        if ctype in ("output_text", "text"):
                            text_parts.append(str(c.get("text", "")))
                        elif ctype == "refusal":
                            refusal_text = str(c.get("refusal", ""))

                joined_text = "\n".join(text_parts)
                if len(content_list) > 0:
                    if len(joined_text) > 0:
                        content_list.append(ChatCompletionContentPartTextParam(type="text", text=joined_text))
                    msg["content"] = content_list  # type: ignore[typeddict-item]
                elif len(joined_text) > 0:
                    msg["content"] = joined_text

                if refusal_text is not None:
                    msg["refusal"] = refusal_text  # type: ignore[typeddict-unknown-key]

                current_assistant = msg
                continue

            # --- Strict input message (user/system/developer) ---
            narr_in_msg = cls.maybe_input_message(item)
            if narr_in_msg is not None:
                flush_assistant_message()
                role = str(narr_in_msg["role"])
                content = narr_in_msg.get("content")
                if role == "system":
                    messages.append(
                        ChatCompletionSystemMessageParam(
                            role="system",
                            content=cls._normalize_content_to_str(content),
                        )
                    )
                elif role == "developer":
                    messages.append(
                        ChatCompletionDeveloperMessageParam(
                            role="developer",
                            content=cls._normalize_content_to_str(content),
                        )
                    )
                else:
                    messages.append(
                        ChatCompletionUserMessageParam(
                            role="user",
                            content=cls._normalize_content(content),
                        )
                    )
                continue

            # --- Easy input message (role + content/tool_calls, no type) ---
            narr_easy = cls.maybe_easy_input_message(item)
            if narr_easy is not None:
                flush_assistant_message()
                # narr_easy is a TypedDict (dict at runtime) — shallow-copy
                # into a plain dict so undeclared keys (tool_calls,
                # thinking_blocks, tool_call_id) injected by HITL
                # resumption history can be read without marker churn.
                easy_dict: dict[str, Any] = dict(narr_easy)
                role = str(narr_easy["role"])
                easy_content = narr_easy.get("content")

                if role == "assistant":
                    msg = ChatCompletionAssistantMessageParam(role="assistant")
                    if len(pending_thinking_blocks) > 0:
                        parts: list[dict[str, str] | ChatCompletionContentPartParam] = list(pending_thinking_blocks)
                        pending_thinking_blocks = []
                        if isinstance(easy_content, str):
                            parts.append(ChatCompletionContentPartTextParam(type="text", text=easy_content))
                        elif easy_content is not None:
                            all_content = cls.extract_all_content(easy_content)
                            if isinstance(all_content, list):
                                parts.extend(all_content)
                        msg["content"] = parts  # type: ignore[typeddict-item]
                    elif isinstance(easy_content, str):
                        msg["content"] = easy_content
                    elif easy_content is not None:
                        text_content = cls.extract_text_content(easy_content)
                        if isinstance(text_content, str):
                            msg["content"] = text_content
                        else:
                            msg["content"] = text_content
                    # Preserve tool_calls and thinking_blocks from
                    # already-converted messages (e.g. HITL resumption
                    # history that is already in wire format).
                    if "tool_calls" in easy_dict:
                        msg["tool_calls"] = easy_dict["tool_calls"]
                    if "thinking_blocks" in easy_dict:
                        msg["thinking_blocks"] = easy_dict["thinking_blocks"]
                    current_assistant = msg
                elif role == "system":
                    messages.append(
                        ChatCompletionSystemMessageParam(
                            role="system",
                            content=easy_content
                            if isinstance(easy_content, str)
                            else cls._normalize_content_to_str(easy_content),
                        )
                    )
                elif role == "developer":
                    messages.append(
                        ChatCompletionDeveloperMessageParam(
                            role="developer",
                            content=easy_content
                            if isinstance(easy_content, str)
                            else cls._normalize_content_to_str(easy_content),
                        )
                    )
                elif role == "user":
                    messages.append(
                        ChatCompletionUserMessageParam(
                            role="user",
                            # Normalize so multimodal parts (input_image /
                            # input_audio) are converted to their wire shape
                            # (image_url / input_audio). Passing the raw list
                            # through sends provider-agnostic part types the
                            # API rejects with a 400.
                            content=cls._normalize_content(easy_content),
                        )
                    )
                elif role == "tool":
                    # Tool result messages from HITL resumption history —
                    # already in wire format, pass through as-is.
                    tool_call_id = easy_dict.get("tool_call_id", "")
                    messages.append(
                        ChatCompletionToolMessageParam(
                            role="tool",
                            tool_call_id=str(tool_call_id),
                            content=easy_content if isinstance(easy_content, str) else str(easy_content),
                        )
                    )
                else:
                    logger.warning("Unknown role %r in easy input message — skipping", role)
                continue

            # --- Function tool call → attach to assistant message ---
            narr_fc = cls.maybe_function_tool_call(item)
            if narr_fc is not None:
                if current_assistant is None:
                    current_assistant = ChatCompletionAssistantMessageParam(role="assistant")
                    if len(pending_thinking_blocks) > 0:
                        current_assistant["content"] = list(pending_thinking_blocks)  # type: ignore[arg-type]
                        pending_thinking_blocks = []

                if pending_reasoning_content is not None and is_deepseek:
                    current_assistant["reasoning_content"] = pending_reasoning_content

                existing_tool_calls = current_assistant.get("tool_calls")
                if existing_tool_calls is None:
                    existing_tool_calls = []
                    current_assistant["tool_calls"] = existing_tool_calls

                existing_tool_calls.append(
                    ChatCompletionToolCallParam(
                        id=str(narr_fc["call_id"]),
                        type="function",
                        function=ChatCompletionFunctionCallParam(
                            name=str(narr_fc["name"]),
                            arguments=str(narr_fc.get("arguments") or "{}"),
                        ),
                    )
                )
                continue

            # --- Function tool call output → tool message ---
            narr_fc_out = cls.maybe_function_tool_call_output(item)
            if narr_fc_out is not None:
                flush_assistant_message()
                output = narr_fc_out.get("output", "")
                tool_content: str | list[ChatCompletionContentPartParam]
                if isinstance(output, str):
                    tool_content = output
                elif preserve_tool_output_all_content:
                    # Convert multimodal parts to their wire shape. ``str(output)``
                    # would serialize the raw part dicts as a Python repr and
                    # corrupt the tool result. litellm's provider adapters accept
                    # image parts inside a tool message at runtime (e.g. Anthropic
                    # tool_result image blocks), even though the Chat-Completions
                    # tool-message TypedDict only declares text objects.
                    tool_content = cls.extract_all_content(output)
                else:
                    tool_content = cls._extract_text_from_output(output)
                messages.append(
                    ChatCompletionToolMessageParam(
                        role="tool",
                        tool_call_id=str(narr_fc_out["call_id"]),
                        content=tool_content,  # type: ignore[typeddict-item]
                    )
                )
                continue

            logger.warning("Unknown item type %r — skipping", item.get("type"))

        # Flush remaining content parts. Emit any open assistant buffer first
        # so trailing user content is not reordered ahead of the assistant
        # turn it chronologically followed.
        if len(pending_content_parts) > 0:
            flush_assistant_message()
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user",
                    content=pending_content_parts,
                )
            )

        flush_assistant_message()

        return messages

    # ==================================================================
    # Tool conversion
    # ==================================================================

    @classmethod
    def convert_tool(
        cls,
        tool: FunctionTool,
    ) -> ChatCompletionFunctionToolParam:
        """Convert a single ``FunctionTool`` to Chat Completions wire format.

        Maps fields to the OpenAI/litellm API format:
        ``get_json_schema()`` → ``parameters``, ``schema_enforcement`` → ``strict``.

        Args:
            tool: A ``FunctionTool`` dataclass.

        Returns:
            A ``ChatCompletionFunctionToolParam``-shaped dict.
        """
        from litellm.types.llms.openai import (
            ChatCompletionToolParam as ChatCompletionFunctionToolParam,
            ChatCompletionToolParamFunctionChunk as ChatCompletionFunctionToolDefinitionParam,
        )

        from troopai.adk.schemas.utils import SchemaEnforcement

        function_def = ChatCompletionFunctionToolDefinitionParam(
            name=tool.name,
            description=tool.description or "",
        )
        json_schema = tool.get_json_schema()
        if json_schema is not None:
            function_def["parameters"] = json_schema
            function_def["strict"] = tool.schema_enforcement == SchemaEnforcement.STRICT

        return ChatCompletionFunctionToolParam(type="function", function=function_def)

    @classmethod
    def convert_tools(
        cls,
        tools: list[FunctionTool],
    ) -> list[ChatCompletionFunctionToolParam]:
        """Convert a list of ``FunctionTool`` to Chat Completions wire format.

        Only handles function tools. Builtin tools require provider-specific
        capability detection and are handled by the concrete ``LLM`` implementation.

        Args:
            tools: List of ``FunctionTool`` dataclass instances.

        Returns:
            A list of ``ChatCompletionFunctionToolParam``-shaped dicts.
        """
        return [cls.convert_tool(t) for t in tools]

    @classmethod
    def fix_tool_message_ordering(
        cls,
        messages: list[ChatCompletionMessageParam],
    ) -> list[ChatCompletionMessageParam]:
        """Reorder messages so each tool result immediately follows its tool call.

        Required by Anthropic and Gemini APIs, which reject tool responses
        that aren't adjacent to their corresponding tool calls.

        When an assistant message contains multiple tool calls, this method
        splits it into individual assistant messages, each followed by its
        corresponding tool result.

        Args:
            messages: Conversation messages (may have non-adjacent tool
                call/result pairs).

        Returns:
            Reordered messages with tool adjacency guaranteed.
        """
        result: list[ChatCompletionMessageParam] = []
        i = 0

        while i < len(messages):
            msg = messages[i]

            # Only process assistant messages with tool_calls. Using
            # direct ``msg["role"]`` subscript (rather than ``.get``) lets
            # pyright narrow the TypedDict union by discriminator — so
            # ``msg`` is seen as ``ChatCompletionAssistantMessage`` in the
            # branch below and ``tool_calls`` / ``content`` typecheck.
            if msg["role"] != "assistant":
                result.append(msg)
                i += 1
                continue

            raw_tool_calls = msg.get("tool_calls")
            if not isinstance(raw_tool_calls, list) or len(raw_tool_calls) == 0:
                result.append(msg)
                i += 1
                continue
            tool_calls: list[ChatCompletionToolCallParam] = raw_tool_calls

            # Collect all following tool results, preserving order and every
            # result. Bucketing into a plain ``dict[id, msg]`` collapses distinct
            # results that share a tool_call_id (e.g. an empty "" id from
            # malformed history) — the later one silently overwrites the earlier.
            # Keep an ordered list plus a per-id position queue so none are lost.
            collected_results: list[ChatCompletionMessageParam] = []
            positions_by_id: dict[str, list[int]] = {}
            j = i + 1
            while j < len(messages):
                next_msg = messages[j]
                if isinstance(next_msg, dict) and next_msg.get("role") == "tool":
                    positions_by_id.setdefault(str(next_msg.get("tool_call_id", "")), []).append(len(collected_results))
                    collected_results.append(next_msg)
                    j += 1
                else:
                    break

            if len(tool_calls) <= 1 or len(collected_results) == 0:
                # Single tool call or no results to reorder — keep as-is
                result.append(msg)
                i += 1
                continue

            # Split multi-tool assistant message into individual ones. The shared
            # narration content and any structured thinking_blocks belong to the
            # original combined turn, so attach them only to the first split —
            # copying onto every split would duplicate the text and inflate cost.
            # Dropping thinking_blocks entirely would break Anthropic replay,
            # which rejects a follow-up missing the prior turn's signed thinking.
            content_raw = msg.get("content")
            thinking_raw = msg.get("thinking_blocks")
            consumed_positions: set[int] = set()
            for idx, tc in enumerate(tool_calls):
                tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
                single_assistant = ChatCompletionAssistantMessageParam(role="assistant")
                if idx == 0:
                    if content_raw is not None:
                        single_assistant["content"] = content_raw
                    if thinking_raw is not None:
                        single_assistant["thinking_blocks"] = thinking_raw
                single_assistant["tool_calls"] = [tc]
                result.append(single_assistant)

                # Attach the first not-yet-consumed result matching this call.
                queue = positions_by_id.get(str(tc_id))
                if queue is not None and len(queue) > 0:
                    position = queue.pop(0)
                    consumed_positions.add(position)
                    result.append(collected_results[position])

            # Preserve any results not paired to a call in this split (orphans
            # from an earlier turn, or extras sharing an id). Silently dropping
            # them would corrupt conversation history.
            for position, orphan_msg in enumerate(collected_results):
                if position not in consumed_positions:
                    logger.warning(
                        "fix_tool_message_ordering: tool result id=%r has no matching tool call; appending as-is",
                        str(orphan_msg.get("tool_call_id", "")),
                    )
                    result.append(orphan_msg)

            # Skip past the tool result messages we already consumed
            i = j
            continue

        return result

    @classmethod
    def convert_tool_choice(
        cls,
        tool_choice: str | ChatCompletionToolChoiceOptionParam | None,
    ) -> ChatCompletionToolChoiceOptionParam | None:
        """Convert a tool choice value to Chat Completions ``tool_choice`` format.

        Handles:
        - ``None`` → ``None`` (omitted from API call)
        - ``"auto"`` / ``"required"`` / ``"none"`` → pass through
        - Arbitrary string → named tool choice ``{"type": "function", "function": {"name": ...}}``
        - Dict → pass through (already formatted)

        Args:
            tool_choice: A string, a ``ChatCompletionToolChoiceOptionParam``,
                or ``None``.

        Returns:
            A ``ChatCompletionToolChoiceOptionParam``-compatible value,
            or ``None`` if no override.
        """
        if tool_choice is None:
            return None

        if isinstance(tool_choice, dict):
            return tool_choice

        if tool_choice == "auto":
            return "auto"
        if tool_choice == "required":
            return "required"
        if tool_choice == "none":
            return "none"
        return ChatCompletionNamedToolChoiceParam(
            type="function",
            function={"name": tool_choice},
        )

    @classmethod
    def convert_response_format(
        cls,
        output_schema: AgentOutputSchemaBase | None,
    ) -> dict[str, Any] | None:
        """Convert an ``AgentOutputSchemaBase`` to a ``response_format`` dict.

        Builds the ``json_schema`` response format for structured output.
        Returns ``None`` if the schema is ``None`` or plain text.
        litellm accepts ``dict | BaseModel | None`` for response_format.

        Note: The concrete ``LLM`` implementation may apply additional
        capability detection (e.g., three-tier fallback in ``LiteLLM``)
        on top of this base conversion.

        Args:
            output_schema: An ``AgentOutputSchemaBase`` instance, or ``None``.

        Returns:
            A response_format dict, or ``None``.
        """
        if output_schema is None:
            return None
        if output_schema.is_plain_text():
            return None

        return {
            "type": "json_schema",
            "json_schema": {
                "name": output_schema.name(),
                "schema": output_schema.json_schema(),
                "strict": output_schema.is_strict_json_schema(),
            },
        }

    # ==================================================================
    # Content extraction (public API)
    # ==================================================================

    @classmethod
    def extract_text_content(
        cls,
        content: str | list[LLMInputContentPart] | None,
    ) -> str | list[ChatCompletionContentPartTextParam]:
        """Extract text-only content from a content list or string.

        Returns a plain string for simple text, or a list of
        ``ChatCompletionContentPartTextParam`` for multi-part content.

        Raises a warning for unsupported content types (video_url, output_audio).

        Args:
            content: A string, list of content parts, or ``None``.

        Returns:
            A string or list of text content part TypedDicts.
        """
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content) if content is not None else ""

        parts: list[ChatCompletionContentPartTextParam] = []
        for part in content:
            ptype = part.get("type", "")
            if ptype in ("output_text", "input_text", "text"):
                parts.append(ChatCompletionContentPartTextParam(type="text", text=str(part.get("text", ""))))

        if len(parts) == 1:
            return parts[0]["text"]
        return parts if len(parts) > 0 else ""

    @classmethod
    def extract_all_content(
        cls,
        content: str | Sequence[LLMInputContentPart] | None,
    ) -> str | list[ChatCompletionContentPartParam]:
        """Convert all content types to Chat Completions content parts.

        Handles ``input_text``, ``input_image``, ``input_audio``,
        ``output_text``, ``refusal``, and ``video_url`` content part types.

        Args:
            content: A string or sequence of content part dicts. Accepts a
                covariant ``Sequence`` so narrower part lists (e.g. a tool
                output's ``list[LLMInputText | LLMInputImage]``) convert
                without an invariant-``list`` mismatch.

        Returns:
            A string (passthrough) or list of Chat Completions content part TypedDicts.
        """
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return str(content) if content is not None else ""

        result: list[ChatCompletionContentPartParam] = []
        for part in content:
            ptype = part.get("type", "")

            if ptype in ("input_text", "output_text", "text"):
                result.append(ChatCompletionContentPartTextParam(type="text", text=str(part.get("text", ""))))
            elif ptype in ("input_image", "image_url"):
                image_url = part.get("image_url", part.get("url", ""))
                if isinstance(image_url, dict):
                    url = str(image_url.get("url", ""))
                    raw_detail = str(image_url.get("detail", "auto"))
                else:
                    url = str(image_url)
                    raw_detail = str(part.get("detail", "auto"))
                detail: Literal["auto", "low", "high"]
                if raw_detail == "low":
                    detail = "low"
                elif raw_detail in ("high", "original"):
                    detail = "high"
                else:
                    detail = "auto"
                result.append(
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url=ImageURL(url=url, detail=detail),
                    )
                )
            elif ptype == "input_audio":
                data = str(part.get("data", ""))
                raw_fmt = str(part.get("format", "wav"))
                if len(data) == 0:
                    logger.warning("input_audio content missing data — skipping")
                    continue
                audio_format: Literal["wav", "mp3"] = "mp3" if raw_fmt == "mp3" else "wav"
                audio_data: InputAudio = {"data": data, "format": audio_format}
                result.append(
                    ChatCompletionContentPartInputAudioParam(
                        type="input_audio",
                        input_audio=audio_data,
                    )
                )
            else:
                result.append(part)  # type: ignore[arg-type]
        return result if len(result) > 0 else ""

    # ==================================================================
    # Internal helpers
    # ==================================================================

    @classmethod
    def _normalize_content(
        cls,
        content: str | list[LLMInputContentPart] | None,
    ) -> str | list[ChatCompletionContentPartParam]:
        """Normalize content for a user message (str or list of parts)."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return cls.extract_all_content(content)

    @classmethod
    def _normalize_content_to_str(cls, content: str | list[LLMInputContentPart] | None) -> str:
        """Normalize content to a plain string (for system/developer messages)."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return " ".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in content)

    @classmethod
    def _extract_text_from_output(cls, output: str | list[LLMInputText | LLMInputImage]) -> str:
        """Extract text from multimodal tool output."""
        if isinstance(output, str):
            return output
        texts = [str(part.get("text", str(part))) for part in output]
        return " ".join(texts)

    @classmethod
    def _extract_thinking_blocks(cls, item: LLMResponseReasoningParam) -> list[dict[str, str]]:
        """Extract thinking block dicts from a reasoning item for content injection."""
        blocks: list[dict[str, str]] = []
        content = item.get("content", [])
        encrypted = item.get("encrypted_content")

        signatures: list[str] = []
        if encrypted is not None:
            signatures.append(str(encrypted))

        if isinstance(content, list) and content:
            for part in content:
                if isinstance(part, dict):
                    text = str(part.get("text", ""))
                    block: dict[str, str] = {"type": "thinking", "thinking": text}
                    if len(signatures) > 0:
                        block["signature"] = signatures[0]
                    blocks.append(block)
        elif len(signatures) > 0:
            blocks.append({"type": "thinking", "thinking": "", "signature": signatures[0]})

        return blocks

    # ==================================================================
    # RunItem conversion (Layer 2 ↔ items)
    # ==================================================================

    @classmethod
    def message_to_run_items(
        cls,
        msg: ChatCompletionMessageParam,
    ) -> list[RunItem]:
        """Convert a single Layer 2 message to one or more RunItems.

        Delegates to ``ItemHelpers.message_to_run_items()``.

        Args:
            msg: A Chat Completions message dict.

        Returns:
            One or more RunItem instances.
        """
        from troopai.adk.types.items import ItemHelpers

        return ItemHelpers.message_to_run_items(msg)  # type: ignore[arg-type]

    @classmethod
    def messages_to_run_items(
        cls,
        msgs: Sequence[ChatCompletionMessageParam],
    ) -> tuple[RunItem, ...]:
        """Convert a sequence of Layer 2 messages to a flat tuple of RunItems.

        Delegates to ``ItemHelpers.messages_to_run_items()``.

        Args:
            msgs: A sequence of Chat Completions message dicts.

        Returns:
            A flat tuple of RunItem instances.
        """
        from troopai.adk.types.items import ItemHelpers

        return tuple(
            ItemHelpers.messages_to_run_items(
                msgs,  # type: ignore[arg-type]
            )
        )
