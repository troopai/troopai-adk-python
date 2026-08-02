"""Stateless converter between Layer 1 items and OpenAI Chat Completions format.

Bridges framework-owned ``LLMInputContentItem`` / ``LLMResponsePart``
types to/from ``openai.types.chat.*`` wire types. All methods are
``@classmethod`` — no instance state, pure functions.

The Chat Completions API expects ``list[ChatCompletionMessageParam]``
as ``messages=``. Unlike the Responses API (which mirrors the
framework's Layer 1 TypedDicts almost verbatim), the Chat Completions
format is message-centric and has three load-bearing shape contrasts
against the Responses API:

- **Named tool choice is NESTED**:
  ``{"type": "function", "function": {"name": "x"}}`` — the inner
  ``function`` dict is required. The Responses API uses a flat
  ``{"type": "function", "name": "x"}``.
- **Tool calls live on the assistant message**: ``assistant.tool_calls``
  is a list of ``{id, type, function: {name, arguments}}`` — not
  separate ``function_call`` items in the message stream.
- **Tool results are ``role: "tool"`` messages**: keyed by ``tool_call_id``,
  not ``function_call_output`` items.

Reasoning is NOT replayable via Chat Completions — OpenAI surfaces
reasoning tokens in ``usage.completion_tokens_details.reasoning_tokens``
but does not include them in the assistant message. Reasoning replay
items dropped with a warning.

Refs:
    - Chat Completions: https://platform.openai.com/docs/api-reference/chat
    - Function calling: https://platform.openai.com/docs/guides/function-calling
    - Structured output: https://platform.openai.com/docs/guides/structured-outputs
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageParam,
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolUnionParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import ImageURL
from openai.types.chat.chat_completion_content_part_input_audio_param import InputAudio
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.shared_params import ResponseFormatJSONSchema
from openai.types.shared_params.function_definition import FunctionDefinition
from openai.types.shared_params.response_format_json_schema import JSONSchema

from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
    LLMResponsePart,
    LLMResponseRefusal,
    LLMResponseText,
)
from troopai.adk.types.tokens.llm_usage import LLMUsage
from troopai.adk.types.tokens.tokens import InputTokensDetails, OutputTokensDetails

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion
    from openai.types.completion_usage import CompletionUsage

    from troopai.adk.schemas import AgentOutputSchemaBase
    from troopai.adk.tools.builtin.builtin_tool import BuiltinTool
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.tools.hosted import HostedTool
    from troopai.adk.types.input import LLMInputContentItem


logger = logging.getLogger(__name__)


class OpenAIChatCompletionsConverter:
    """Convert between Layer 1 framework types and OpenAI Chat Completions wire types.

    All methods are ``@classmethod`` — stateless pure functions. Wire
    types are imported from ``openai.types.chat.*`` and never leak
    outside this module's call sites.
    """

    # ------------------------------------------------------------------
    # Inputs: Layer 1 → Chat-Completions messages list
    # ------------------------------------------------------------------

    @classmethod
    def items_to_messages(
        cls,
        items: list[LLMInputContentItem],
    ) -> list[ChatCompletionMessageParam]:
        """Normalise Layer 1 input items into ``ChatCompletionMessageParam`` entries.

        Adjacent ``function_call`` items are accumulated into a single
        assistant message so the output stream matches the Chat
        Completions wire protocol (one assistant message carrying the
        full ``tool_calls`` list, followed by the matching
        ``role="tool"`` results).
        """
        out: list[ChatCompletionMessageParam] = []
        pending_calls: list[ChatCompletionMessageFunctionToolCallParam] = []

        def flush_pending() -> None:
            if len(pending_calls) == 0:
                return
            assistant: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": None,
                "tool_calls": list(pending_calls),
            }
            out.append(assistant)
            pending_calls.clear()

        for item in items:
            item_type = item.get("type") if isinstance(item, dict) else None

            if item_type == "function_call":
                pending_calls.append(cls._build_tool_call(item))
                continue

            if item_type == "function_call_output":
                # Chat Completions requires every ``role="tool"`` message
                # to be preceded by an assistant message that carries the
                # matching ``tool_calls[i].id``. If we emit one without a
                # companion assistant message, the API returns an opaque
                # 400 ("invalid_request_error: tool message not after
                # assistant message"). Warn loudly at conversion time so
                # the source of the malformed history shows up in logs
                # before the API rejects the call. Two valid precursors:
                # (a) pending function_calls about to be flushed into an
                # assistant message on this same iteration, or (b) the
                # last emitted message was already a tool_calls-bearing
                # assistant (e.g. a multi-output replay batch).
                has_pending = len(pending_calls) > 0
                # Scan backward past already-emitted ``role="tool"`` messages
                # to the nearest non-tool message: multiple outputs after one
                # multi-call assistant message are valid, so the precursor may
                # sit behind sibling tool results rather than at ``out[-1]``.
                preceded_by_tool_calls = False
                for prior in reversed(out):
                    if prior.get("role") == "tool":
                        continue
                    preceded_by_tool_calls = prior.get("role") == "assistant" and "tool_calls" in prior
                    break
                if not has_pending and not preceded_by_tool_calls:
                    call_id = item.get("call_id") if isinstance(item, dict) else None
                    logger.warning(
                        "function_call_output (call_id=%s) has no preceding "
                        "function_call — the Chat Completions API will reject "
                        "this history with a 400.",
                        call_id,
                    )

            # Every non-function_call item flushes the accumulator.
            flush_pending()
            converted = cls._convert_input_item(item, item_type)
            if converted is not None:
                out.append(converted)

        flush_pending()
        return out

    @classmethod
    def _build_tool_call(
        cls,
        item: Any,
    ) -> ChatCompletionMessageFunctionToolCallParam:
        """Build one ``tool_calls`` entry from a ``function_call`` replay item."""
        call_id = str(item.get("call_id") or item.get("id") or "")
        name = str(item.get("name", ""))
        arguments = str(item.get("arguments", "") or "")
        tool_call: ChatCompletionMessageFunctionToolCallParam = {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
        return tool_call

    @classmethod
    def _convert_input_item(
        cls,
        item: LLMInputContentItem,
        item_type: Any,
    ) -> Optional[ChatCompletionMessageParam]:
        """Convert one non-function_call Layer 1 input item to a Chat message."""
        if item_type == "reasoning":
            logger.warning("Chat Completions API cannot replay reasoning items; dropping.")
            return None

        if item_type == "provider_item":
            logger.warning(
                "Chat Completions API does not support provider_item replay; dropping %s",
                item.get("item_type", "") if isinstance(item, dict) else "",
            )
            return None

        if item_type == "function_call_output":
            call_id = str(item.get("call_id") or "") if isinstance(item, dict) else ""
            raw_output = item.get("output", "") if isinstance(item, dict) else ""
            content_str = raw_output if isinstance(raw_output, str) else str(raw_output)
            tool_message: ChatCompletionToolMessageParam = {
                "role": "tool",
                "tool_call_id": call_id,
                "content": content_str,
            }
            return tool_message

        if item_type == "message":
            role = item.get("role") if isinstance(item, dict) else None
            content = item.get("content") if isinstance(item, dict) else None
            if role == "assistant":
                return cls._convert_assistant_message(item)
            return cls._build_simple_message(role, content)

        if item_type in ("input_text", "input_image", "input_audio"):
            return cls._build_simple_message("user", [item])

        if isinstance(item, dict) and "role" in item:
            return cls._build_simple_message(item.get("role"), item.get("content"))

        logger.warning(
            "Unrecognised LLMInputContentItem; dropping (got type=%s, keys=%s).",
            type(item).__name__,
            sorted(item.keys()) if isinstance(item, dict) else "n/a",
        )
        return None

    @classmethod
    def _convert_assistant_message(
        cls,
        item: Any,
    ) -> ChatCompletionAssistantMessageParam:
        """Build an assistant message from an ``LLMResponseMessageParam``-shaped dict.

        The content list may contain ``output_text`` and/or ``refusal``
        parts. Chat Completions expects ``content`` as a string OR a list
        of text/refusal content parts — we flatten text parts to a
        single string when there is a single text and no refusal, and
        emit the full list otherwise.
        """
        content_parts = item.get("content", []) or []
        if not isinstance(content_parts, list):
            msg: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": str(content_parts),
            }
            return msg

        text_chunks: list[str] = []
        refusal: Optional[str] = None
        for part in content_parts:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "output_text":
                text_chunks.append(str(part.get("text", "")))
            elif ptype == "refusal":
                refusal = str(part.get("refusal", ""))

        assistant: ChatCompletionAssistantMessageParam = {
            "role": "assistant",
            "content": "\n".join(text_chunks) if len(text_chunks) > 0 else None,
        }
        if refusal is not None:
            assistant["refusal"] = refusal
        return assistant

    @classmethod
    def _build_simple_message(
        cls,
        role: Any,
        content: Any,
    ) -> ChatCompletionMessageParam:
        """Build a user/system/developer message with string or list content."""
        if role == "system":
            text = content if isinstance(content, str) else cls._join_text_parts(content)
            system_msg: ChatCompletionSystemMessageParam = {"role": "system", "content": text}
            return system_msg

        # Developer messages map to system (current SDK accepts both, but
        # framing the default at "user" is safer for downstream providers).
        if role == "developer":
            text = content if isinstance(content, str) else cls._join_text_parts(content)
            dev_msg: ChatCompletionSystemMessageParam = {"role": "system", "content": text}
            return dev_msg

        # Default / unknown role → user.
        if isinstance(content, str):
            user_str: ChatCompletionUserMessageParam = {"role": "user", "content": content}
            return user_str
        if isinstance(content, list):
            content_parts = cls._user_content_parts(content)
            if len(content_parts) > 0:
                user_parts: ChatCompletionUserMessageParam = {
                    "role": "user",
                    "content": content_parts,
                }
                return user_parts
            user_empty: ChatCompletionUserMessageParam = {"role": "user", "content": ""}
            return user_empty
        user_fallback: ChatCompletionUserMessageParam = {"role": "user", "content": ""}
        return user_fallback

    @classmethod
    def _user_content_parts(cls, content: list[Any]) -> list[ChatCompletionContentPartParam]:
        """Convert a Layer 1 content list into Chat Completions user content parts.

        Maps ``input_text`` / ``output_text`` / ``text`` to ``text`` parts,
        ``input_image`` to ``image_url`` parts, and ``input_audio`` to
        ``input_audio`` parts. Image and audio parts were previously dropped,
        silently reducing a multimodal user turn to its text alone.
        """
        parts: list[ChatCompletionContentPartParam] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype in ("input_text", "output_text", "text"):
                text_part: ChatCompletionContentPartTextParam = {"type": "text", "text": str(part.get("text", ""))}
                parts.append(text_part)
            elif ptype == "input_image":
                image_part = cls._image_content_part(part)
                if image_part is not None:
                    parts.append(image_part)
            elif ptype == "input_audio":
                audio_part = cls._audio_content_part(part)
                if audio_part is not None:
                    parts.append(audio_part)
        return parts

    @classmethod
    def _image_content_part(cls, part: dict[str, Any]) -> ChatCompletionContentPartImageParam | None:
        """Convert an ``input_image`` Layer 1 part to a Chat Completions ``image_url`` part.

        Returns ``None`` when the URL is absent so a malformed part is
        dropped rather than sent as an empty image reference.
        """
        url = part.get("image_url")
        if not isinstance(url, str) or len(url) == 0:
            return None
        image_url: ImageURL = {"url": url}
        detail = part.get("detail")
        if detail in ("auto", "low", "high"):
            image_url["detail"] = detail
        return {"type": "image_url", "image_url": image_url}

    @classmethod
    def _audio_content_part(cls, part: dict[str, Any]) -> ChatCompletionContentPartInputAudioParam | None:
        """Convert an ``input_audio`` Layer 1 part to a Chat Completions ``input_audio`` part.

        Returns ``None`` when the base64 ``data`` or a supported ``format``
        (``wav`` / ``mp3``) is missing so malformed audio is dropped.
        """
        data = part.get("data")
        fmt = part.get("format")
        if not isinstance(data, str) or len(data) == 0 or fmt not in ("wav", "mp3"):
            return None
        input_audio: InputAudio = {"data": data, "format": fmt}
        return {"type": "input_audio", "input_audio": input_audio}

    @classmethod
    def _join_text_parts(cls, content: Any) -> str:
        """Concatenate the ``text`` field of any text-like parts in a content list."""
        if content is None:
            return ""
        if not isinstance(content, list):
            return str(content)
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in (
                "input_text",
                "output_text",
                "text",
            ):
                chunks.append(str(part.get("text", "")))
        return "\n".join(chunks)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @classmethod
    def convert_tools(
        cls,
        tools: Sequence[Union[FunctionTool, BuiltinTool, HostedTool]],
    ) -> list[ChatCompletionToolUnionParam]:
        """Convert framework tools to Chat Completions tool entries.

        Translates ``FunctionTool`` and ``ExecutableBuiltinTool`` to
        the Chat Completions function-tool shape. The Chat Completions
        ``tools=`` array does NOT accept hosted tool params — for web
        search and similar capabilities, use the ``web_search_options``
        config field on :class:`OpenAIChatCompletionsConfig` instead.
        Hosted-tool variants raise :class:`UnsupportedHostedToolError`
        with a message pointing at the alternative path.
        """
        from pydantic import BaseModel

        from troopai.adk.schemas.utils import SchemaEnforcement, normalize_schema
        from troopai.adk.tools.builtin.builtin_tool import ExecutableBuiltinTool
        from troopai.adk.tools.function_tool import FunctionTool
        from troopai.adk.tools.hosted import (
            HostedTool,
            UnsupportedHostedToolError,
        )

        wire_tools: list[ChatCompletionToolUnionParam] = []
        for tool in tools:
            if isinstance(tool, FunctionTool):
                schema = tool.get_json_schema() or {"type": "object", "properties": {}}
                fn_def: FunctionDefinition = {
                    "name": tool.name,
                    "parameters": dict(schema),
                    "strict": tool.schema_enforcement == SchemaEnforcement.STRICT,
                }
                if tool.description is not None:
                    fn_def["description"] = tool.description
                wire_tools.append({"type": "function", "function": fn_def})
                continue

            if isinstance(tool, ExecutableBuiltinTool):
                raw_schema = (
                    tool.schema.model_json_schema()
                    if isinstance(tool.schema, type) and issubclass(tool.schema, BaseModel)
                    else tool.schema
                )
                norm = normalize_schema(raw_schema) or {"type": "object", "properties": {}}
                exec_fn: FunctionDefinition = {
                    "name": tool.name,
                    "parameters": dict(norm),
                    "strict": True,
                }
                if tool.description is not None:
                    exec_fn["description"] = tool.description
                wire_tools.append({"type": "function", "function": exec_fn})
                continue

            if isinstance(tool, HostedTool):
                # Chat Completions does not expose hosted tools via
                # the tools array. Web search lives on the
                # ``web_search_options`` config field; other hosted
                # tools belong on the Responses API or another
                # provider. Raise rather than silently dropping.
                raise UnsupportedHostedToolError(
                    tool,
                    "openai-chatcompletions",
                    supported_providers=tool.SUPPORTED_PROVIDERS,
                )

            logger.warning("Unknown tool type for OpenAI Chat Completions: %s", type(tool))

        return wire_tools

    @classmethod
    def convert_tool_choice(
        cls,
        tool_choice: Optional[str],
        tools_present: bool,
    ) -> Optional[ChatCompletionToolChoiceOptionParam]:
        """Convert framework ``ToolChoice`` to the Chat Completions shape.

        Chat Completions' named-tool shape is NESTED:
        ``{"type": "function", "function": {"name": "<tool_name>"}}`` —
        the inner ``function`` wrapper is load-bearing (contrast with
        the Responses API's flat shape).

        Mapping:
        - ``None`` / tools absent → ``None`` (omit).
        - ``"auto"`` / ``"required"`` / ``"none"`` → the string literal.
        - Any other string → nested ``ChatCompletionNamedToolChoiceParam``.
        """
        if tool_choice is None or not tools_present:
            return None
        if tool_choice == "auto":
            return "auto"
        if tool_choice == "required":
            return "required"
        if tool_choice == "none":
            return "none"
        named: ChatCompletionNamedToolChoiceParam = {
            "type": "function",
            "function": {"name": tool_choice},
        }
        return named

    # ------------------------------------------------------------------
    # Structured output → response_format dict
    # ------------------------------------------------------------------

    @classmethod
    def resolve_response_format(
        cls,
        output_schema: Optional[AgentOutputSchemaBase],
    ) -> Optional[ResponseFormatJSONSchema]:
        """Build a Chat Completions ``response_format`` value.

        Returns a :class:`ResponseFormatJSONSchema` TypedDict (the
        ``type: "json_schema"`` variant of the SDK's
        ``completion_create_params.ResponseFormat`` union). Capability
        downgrade to ``json_object`` / ``text`` is the caller's
        responsibility — the framework trusts the model to honour the
        schema.

        Returns ``None`` when no schema is requested so the caller can
        ``omit`` the parameter entirely.
        """
        if output_schema is None:
            return None
        schema: JSONSchema = {
            "name": output_schema.name(),
            "schema": output_schema.json_schema(),
            "strict": output_schema.is_strict_json_schema(),
        }
        return {
            "type": "json_schema",
            "json_schema": schema,
        }

    # ------------------------------------------------------------------
    # Response → LLMResponse
    # ------------------------------------------------------------------

    @classmethod
    def chat_completion_to_llm_response(cls, completion: ChatCompletion) -> LLMResponse:
        """Convert a ``ChatCompletion`` to a framework ``LLMResponse``.

        Reads ``choices[0]`` only; ``n > 1`` is unsupported by the
        framework (the LLM ABC returns a single ``LLMResponse``).
        """
        if len(completion.choices) == 0:
            return LLMResponse(
                response_id=completion.id,
                model=completion.model,
                response=[],
                usage=cls.parse_usage(completion.usage) if completion.usage is not None else None,
                finish_reason="stop",
            )

        choice = completion.choices[0]
        message = choice.message
        parts: list[LLMResponsePart] = []

        if message.refusal is not None and len(message.refusal) > 0:
            parts.append(LLMResponseRefusal(refusal=message.refusal))
        elif message.content is not None and len(message.content) > 0:
            parts.append(LLMResponseText(text=message.content))

        if message.tool_calls is not None:
            for call in message.tool_calls:
                # ``tool_calls`` is a discriminated union of function + custom
                # tool calls. Only function calls carry ``.function`` — custom
                # tool calls (type="custom") have a different payload and are
                # skipped here.
                if not isinstance(call, ChatCompletionMessageFunctionToolCall):
                    continue
                parts.append(
                    LLMResponseFunctionToolCall(
                        call_id=call.id,
                        name=call.function.name,
                        arguments=call.function.arguments,
                        id=call.id,
                        status="completed",
                    )
                )

        # ``choice.finish_reason`` is a closed Literal; widen to ``str`` so
        # we can annotate the refusal short-circuit without the Literal
        # mypy error (we store ``str | None`` on ``LLMResponse``).
        finish_reason: str = choice.finish_reason
        if message.refusal is not None and len(message.refusal) > 0:
            finish_reason = "refusal"

        return LLMResponse(
            response_id=completion.id,
            model=completion.model,
            response=parts,
            usage=cls.parse_usage(completion.usage) if completion.usage is not None else None,
            finish_reason=finish_reason,
        )

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------

    @classmethod
    def parse_usage(cls, usage: CompletionUsage) -> LLMUsage:
        """Convert ``CompletionUsage`` to framework ``LLMUsage``.

        Populates ``cached_tokens`` from ``prompt_tokens_details`` and
        ``reasoning_tokens`` from ``completion_tokens_details``.
        """
        cached = 0
        reasoning = 0
        if usage.prompt_tokens_details is not None and usage.prompt_tokens_details.cached_tokens is not None:
            cached = usage.prompt_tokens_details.cached_tokens
        if usage.completion_tokens_details is not None and usage.completion_tokens_details.reasoning_tokens is not None:
            reasoning = usage.completion_tokens_details.reasoning_tokens

        input_details = InputTokensDetails(
            cached_tokens=cached,
            cache_creation_input_tokens=0,
        )
        output_details = OutputTokensDetails(reasoning_tokens=reasoning)
        return LLMUsage(
            requests=1,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            input_tokens_details=input_details,
            output_tokens_details=output_details,
        )
