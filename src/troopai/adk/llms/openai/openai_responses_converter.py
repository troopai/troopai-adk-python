"""Stateless converter between Layer 1 items and OpenAI Responses API format.

Bridges framework-owned ``LLMInputContentItem`` / ``LLMResponsePart``
types to/from ``openai.types.responses.*`` wire types. All methods are
``@classmethod`` — no instance state, pure functions.

The Responses API input shapes already match the framework's Layer 1
TypedDicts almost verbatim (``type: "input_text"`` / ``"input_image"`` /
``"message"`` / ``"function_call"`` / ``"function_call_output"`` /
``"reasoning"``), so conversion is largely a shape normalisation rather
than a rewrite. Provider-native hosted tool items that do not fit the
4 core part types (text, reasoning, function call, refusal) land in
:class:`LLMResponseProviderItem` with ``raw`` carrying the verbatim
provider payload — replayed unchanged on the next turn.

Refs:
    - OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses
    - Function tools: https://platform.openai.com/docs/guides/function-calling
    - Structured output: https://platform.openai.com/docs/guides/structured-outputs
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Optional, Union

from openai.types.responses import (
    EasyInputMessageParam,
    FileSearchToolParam,
    FunctionToolParam,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseTextConfigParam,
    ToolChoiceFunctionParam,
    ToolChoiceOptions,
    ToolParam,
    WebSearchToolParam,
)
from openai.types.responses.computer_use_preview_tool_param import ComputerUsePreviewToolParam
from openai.types.responses.response_output_text import AnnotationURLCitation
from openai.types.responses.tool_param import (
    CodeInterpreter as CodeInterpreterToolParam,
    ImageGeneration as ImageGenerationToolParam,
    Mcp as McpToolParam,
)
from openai.types.responses.web_search_tool_param import (
    UserLocation as OpenAIWebSearchUserLocation,
)

from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseAnnotation,
    LLMResponseFunctionToolCall,
    LLMResponsePart,
    LLMResponseProviderItem,
    LLMResponseReasoning,
    LLMResponseRefusal,
    LLMResponseText,
)
from troopai.adk.types.tokens.llm_usage import LLMUsage
from troopai.adk.types.tokens.tokens import InputTokensDetails, OutputTokensDetails

if TYPE_CHECKING:
    from openai.types.responses import Response
    from openai.types.responses.response_input_item_param import ResponseInputItemParam
    from openai.types.responses.response_usage import ResponseUsage

    from troopai.adk.schemas import AgentOutputSchemaBase
    from troopai.adk.tools.builtin.builtin_tool import BuiltinTool
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.tools.hosted import (
        CodeExecutionTool,
        ComputerTool,
        FileSearchTool,
        HostedMCPTool,
        HostedTool,
        ImageGenerationTool,
        WebSearchTool,
    )
    from troopai.adk.types.input import LLMInputContentItem


logger = logging.getLogger(__name__)


class OpenAIResponsesConverter:
    """Convert between Layer 1 framework types and OpenAI Responses API wire types.

    All methods are ``@classmethod`` — stateless pure functions. Wire
    types are imported from ``openai.types.responses.*`` and never leak
    outside this module's call sites.
    """

    # ------------------------------------------------------------------
    # Inputs: Layer 1 → Responses-API input list
    # ------------------------------------------------------------------

    @classmethod
    def items_to_input(
        cls,
        items: list[LLMInputContentItem],
    ) -> list[ResponseInputItemParam]:
        """Normalise Layer 1 input items into ``ResponseInputParam`` elements.

        The Responses API accepts ``list[ResponseInputItemParam]`` as
        ``input=``. The framework's ``LLMInputContentItem`` union was
        modelled on the same shapes, so most variants pass through with
        only light normalisation:

        - ``LLMInputEasyMessage`` → ``EasyInputMessageParam`` (role +
          content, where ``str`` content is wrapped into a single
          ``{"type": "input_text"}`` part).
        - ``LLMInputMessage`` → ``EasyInputMessageParam`` (drop the
          ``"message"`` discriminator that the Responses API infers).
        - ``LLMResponseFunctionToolCallParam`` → pass through (the
          TypedDict already matches ``ResponseFunctionToolCallParam``).
        - ``FunctionToolCallResultParam`` → pass through (matches the
          ``FunctionCallOutput`` TypedDict).
        - ``LLMResponseMessageParam`` → pass through (matches
          ``ResponseOutputMessageParam``).
        - ``LLMResponseReasoningParam`` → pass through (matches
          ``ResponseReasoningItemParam``).
        - ``LLMResponseProviderItemParam`` → unwrap ``raw`` and forward
          verbatim (replay of hosted-tool output items).
        - Bare content parts (``LLMInputText`` / ``LLMInputImage`` /
          ``LLMInputAudio``) are wrapped in a single ``user``-role
          message so the caller can mix bare parts with messages.
        """
        out: list[ResponseInputItemParam] = []
        for item in items:
            converted = cls._convert_input_item(item)
            if converted is not None:
                out.append(converted)
        return out

    @classmethod
    def _convert_input_item(
        cls,
        item: LLMInputContentItem,
    ) -> Optional[ResponseInputItemParam]:
        """Convert a single Layer 1 input item to a Responses-API item.

        Load-bearing ``# type: ignore[return-value]`` rationale: the
        framework's Layer 1 TypedDicts in ``types/input/`` and
        ``types/responses/`` were deliberately shaped to mirror the
        Responses-API wire types (``type`` discriminator, field names,
        optionality). Pyright cannot narrow a TypedDict-union return type
        from a ``.get("type") == "x"`` branch — it reports each
        ``return item`` as unrelated to ``ResponseInputItemParam``. The
        runtime equality check on the ``type`` discriminator and the
        framework's shape alignment are the load-bearing invariants; each
        marker below corresponds to one specific branch where the shape
        match is guaranteed by design.
        """
        item_type = item.get("type") if isinstance(item, dict) else None

        if item_type == "provider_item":
            raw = item.get("raw", {})
            if not isinstance(raw, dict):
                logger.warning(
                    "provider_item.raw is not a dict; dropping (got type=%s).",
                    type(raw).__name__,
                )
                return None
            # raw is the verbatim Responses-API item captured on a prior turn.
            return raw  # type: ignore[return-value]

        if item_type == "function_call":
            # LLMResponseFunctionToolCallParam ≡ ResponseFunctionToolCallParam.
            return item  # type: ignore[return-value]

        if item_type == "function_call_output":
            # FunctionToolCallResultParam ≡ Responses-API FunctionCallOutput.
            return item  # type: ignore[return-value]

        if item_type == "reasoning":
            # LLMResponseReasoningParam ≡ ResponseReasoningItemParam.
            return item  # type: ignore[return-value]

        if item_type == "message":
            role = item.get("role")
            if role == "assistant":
                return cls._normalize_assistant_message(item)
            return cls._build_easy_message(role, item.get("content"))

        if item_type in ("input_text", "input_image", "input_audio"):
            # Bare content part — wrap in a user-role message. The narrow
            # ignore covers the single-element list whose element type is
            # one of the input content TypedDicts; each is a valid
            # ResponseInputContentParam by shape.
            return cls._build_easy_message("user", [item])  # type: ignore[list-item]

        # LLMInputEasyMessage has no ``type`` discriminator; detect by role.
        if isinstance(item, dict) and "role" in item:
            return cls._build_easy_message(item.get("role"), item.get("content"))

        logger.warning(
            "Unrecognised LLMInputContentItem; dropping (got type=%s, keys=%s).",
            type(item).__name__,
            sorted(item.keys()) if isinstance(item, dict) else "n/a",
        )
        return None

    @classmethod
    def _build_easy_message(
        cls,
        role: Any,
        content: Any,
    ) -> EasyInputMessageParam:
        """Build an ``EasyInputMessageParam`` with string or list content."""
        if role not in ("user", "assistant", "system", "developer"):
            # Coerce unknown roles to "user" — the Responses API would 400 otherwise.
            role = "user"
        if isinstance(content, str):
            msg: EasyInputMessageParam = {"role": role, "content": content}
            return msg
        if isinstance(content, list):
            parts: list[Any] = list(content)
            return {"role": role, "content": parts}
        # Empty content — rare, but defensively send an empty string.
        return {"role": role, "content": ""}

    @classmethod
    def _normalize_assistant_message(
        cls,
        item: Any,
    ) -> ResponseInputItemParam:
        """Normalise an assistant ``LLMResponseMessageParam`` for Responses input.

        ``LLMResponseMessageParam`` mirrors ``ResponseOutputMessageParam``
        but the framework's replay params keep ``id`` / ``status`` /
        ``annotations`` optional (Layer 1 stays provider-agnostic), whereas
        the Responses wire shape types all three as ``Required``. Replaying
        the dict verbatim therefore risks a 400 on the next turn:

        - ``output_text`` parts produced from a plain assistant turn carry
          no ``annotations`` (only web-search citations set them), so each
          part gets a defaulted ``"annotations": []``.
        - When ``id`` is absent (e.g. a cross-provider history replayed
          onto the Responses API) the item cannot be a valid
          ``ResponseOutputMessageParam``; fall back to an
          ``EasyInputMessageParam`` (assistant role, joined text), which the
          Responses API accepts and has no ``id`` / ``status`` requirement.
        """
        parts = cls._assistant_content(item)
        if "id" not in item:
            text = " ".join(str(p.get("text", "")) for p in parts if p.get("type") == "output_text")
            return cls._build_easy_message("assistant", text)
        normalised = dict(item)
        if isinstance(item.get("content"), list):
            normalised["content"] = [cls._normalize_output_part(p) for p in parts]
        if "status" not in normalised:
            normalised["status"] = "completed"
        return normalised  # type: ignore[return-value]

    @staticmethod
    def _normalize_output_part(part: dict[str, Any]) -> dict[str, Any]:
        """Ensure an ``output_text`` content part carries ``annotations``.

        The Responses wire shape types ``annotations`` as required on
        ``output_text``; the framework omits it when there are no web-search
        citations. Default to an empty list so verbatim replay stays valid.
        """
        if part.get("type") != "output_text":
            return part
        out = dict(part)
        if out.get("annotations") is None:
            out["annotations"] = []
        return out

    @staticmethod
    def _assistant_content(item: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the assistant message's content parts as a list of dicts."""
        content = item.get("content")
        if isinstance(content, list):
            return [part for part in content if isinstance(part, dict)]
        return []

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    @classmethod
    def convert_tools(
        cls,
        tools: Sequence[Union[FunctionTool, BuiltinTool, HostedTool]],
    ) -> list[ToolParam]:
        """Convert framework tools to ``ToolParam`` entries for the Responses API.

        Translates three categories:

        - ``FunctionTool`` → ``FunctionToolParam``
        - ``ExecutableBuiltinTool`` → ``FunctionToolParam`` (function format)
        - ``HostedTool`` subclass → matching Responses-API
          hosted-tool param (``WebSearchToolParam``,
          ``CodeInterpreter``, ``FileSearchToolParam``,
          ``ImageGeneration``). Variants the Responses API does not
          ship raise :class:`UnsupportedHostedToolError`.

        Raises:
            UnsupportedHostedToolError: If a hosted-tool variant is
                supplied that the Responses API does not support
                (currently :class:`URLContextTool`, which is
                Gemini-only).
        """
        from pydantic import BaseModel

        from troopai.adk.schemas.utils import SchemaEnforcement, normalize_schema
        from troopai.adk.tools.builtin.builtin_tool import ExecutableBuiltinTool
        from troopai.adk.tools.function_tool import FunctionTool
        from troopai.adk.tools.hosted import (
            CodeExecutionTool,
            ComputerTool,
            FileSearchTool,
            HostedMCPTool,
            HostedTool,
            ImageGenerationTool,
            UnsupportedHostedToolError,
            WebSearchTool,
        )

        wire_tools: list[ToolParam] = []
        for tool in tools:
            if isinstance(tool, FunctionTool):
                schema = tool.get_json_schema() or {"type": "object", "properties": {}}
                param: FunctionToolParam = {
                    "type": "function",
                    "name": tool.name,
                    "parameters": dict(schema),
                    "strict": tool.schema_enforcement == SchemaEnforcement.STRICT,
                }
                if tool.description is not None:
                    param["description"] = tool.description
                wire_tools.append(param)
                continue

            if isinstance(tool, ExecutableBuiltinTool):
                raw_schema = (
                    tool.schema.model_json_schema()
                    if isinstance(tool.schema, type) and issubclass(tool.schema, BaseModel)
                    else tool.schema
                )
                norm = normalize_schema(raw_schema) or {"type": "object", "properties": {}}
                exec_param: FunctionToolParam = {
                    "type": "function",
                    "name": tool.name,
                    "parameters": dict(norm),
                    "strict": True,
                }
                if tool.description is not None:
                    exec_param["description"] = tool.description
                wire_tools.append(exec_param)
                continue

            if isinstance(tool, WebSearchTool):
                wire_tools.append(cls._convert_web_search(tool))
                continue

            if isinstance(tool, CodeExecutionTool):
                wire_tools.append(cls._convert_code_execution(tool))
                continue

            if isinstance(tool, FileSearchTool):
                wire_tools.append(cls._convert_file_search(tool))
                continue

            if isinstance(tool, ImageGenerationTool):
                wire_tools.append(cls._convert_image_generation(tool))
                continue

            if isinstance(tool, HostedMCPTool):
                wire_tools.append(cls._convert_hosted_mcp(tool))
                continue

            if isinstance(tool, ComputerTool):
                wire_tools.append(cls._convert_computer(tool))
                continue

            if isinstance(tool, HostedTool):
                # Hosted tool the Responses API does not ship.
                raise UnsupportedHostedToolError(
                    tool,
                    "openai-responses",
                    supported_providers=tool.SUPPORTED_PROVIDERS,
                )

            logger.warning("Unknown tool type for OpenAI Responses: %s", type(tool))

        return wire_tools

    # ------------------------------------------------------------------
    # Hosted-tool translators
    # ------------------------------------------------------------------

    @classmethod
    def _convert_web_search(cls, tool: WebSearchTool) -> WebSearchToolParam:
        """Translate :class:`WebSearchTool` to ``WebSearchToolParam``.

        Reads OpenAI-honoured attributes (``search_context_size``,
        ``user_location``) and silently ignores Anthropic-only knobs
        (``max_uses``, ``allowed_domains``, ``blocked_domains``) with
        a debug log.
        """
        param: WebSearchToolParam = {"type": "web_search"}
        if tool.search_context_size is not None:
            param["search_context_size"] = tool.search_context_size
        if tool.user_location is not None:
            location: OpenAIWebSearchUserLocation = {"type": "approximate"}
            for key in ("city", "country", "region", "timezone"):
                value = tool.user_location.get(key)
                if value is not None:
                    location[key] = value  # type: ignore[literal-required]
            param["user_location"] = location
        if any(v is not None for v in (tool.max_uses, tool.allowed_domains, tool.blocked_domains)):
            logger.debug(
                "OpenAI Responses: ignoring WebSearchTool Anthropic-only attrs "
                "(max_uses, allowed_domains, blocked_domains)."
            )
        return param

    @classmethod
    def _convert_computer(cls, tool: ComputerTool[Any]) -> ComputerUsePreviewToolParam:
        """Translate :class:`ComputerTool` to ``ComputerUsePreviewToolParam``.

        Emits the Responses-API ``computer_use_preview`` tool with the
        reported display geometry and environment hint. The local
        :class:`Computer` executor and the safety / approval gates are
        runtime concerns handled by the Runner — they are not part of the
        wire tool definition.
        """
        return {
            "type": "computer_use_preview",
            "display_width": tool.display_width,
            "display_height": tool.display_height,
            "environment": tool.environment,
        }

    @classmethod
    def _convert_code_execution(cls, tool: CodeExecutionTool) -> CodeInterpreterToolParam:
        """Translate :class:`CodeExecutionTool` to the ``code_interpreter`` param.

        Honours ``container``: when set, binds to a specific container
        resource; ``None`` lets OpenAI auto-provision.
        """
        param: CodeInterpreterToolParam = {
            "type": "code_interpreter",
            "container": tool.container if tool.container is not None else "auto",
        }
        return param

    @classmethod
    def _convert_file_search(cls, tool: FileSearchTool) -> FileSearchToolParam:
        """Translate :class:`FileSearchTool` to ``FileSearchToolParam``.

        ``vector_store_ids`` is required by the API; an empty list is
        forwarded as-is so the API surfaces the validation error to
        the caller.
        """
        param: FileSearchToolParam = {
            "type": "file_search",
            "vector_store_ids": list(tool.vector_store_ids),
        }
        if tool.max_num_results is not None:
            param["max_num_results"] = tool.max_num_results
        if tool.ranking_options is not None:
            # Trust the caller's dict matches RankingOptions; the SDK
            # validates at the network boundary.
            param["ranking_options"] = tool.ranking_options  # type: ignore[typeddict-item]
        return param

    @classmethod
    def _convert_image_generation(
        cls,
        tool: ImageGenerationTool,
    ) -> ImageGenerationToolParam:
        """Translate :class:`ImageGenerationTool` to the ``image_generation`` param."""
        param: ImageGenerationToolParam = {"type": "image_generation"}
        if tool.model is not None:
            param["model"] = tool.model  # type: ignore[typeddict-unknown-key]
        if tool.quality is not None:
            param["quality"] = tool.quality
        if tool.size is not None:
            param["size"] = tool.size
        if tool.output_format is not None:
            param["output_format"] = tool.output_format
        return param

    @classmethod
    def _convert_hosted_mcp(cls, tool: HostedMCPTool) -> McpToolParam:
        """Translate :class:`HostedMCPTool` to the Responses-API ``mcp`` param.

        Validates the ``server_url`` / ``connector_id`` mutual-exclusion
        invariant the wire protocol enforces. Optional fields are
        omitted from the param when ``None`` so the SDK applies its
        own defaults.
        """
        if (tool.server_url is None) == (tool.connector_id is None):
            raise ValueError(f"HostedMCPTool '{tool.server_label}' must set exactly one of server_url or connector_id.")
        param: McpToolParam = {"type": "mcp", "server_label": tool.server_label}
        if tool.server_url is not None:
            param["server_url"] = tool.server_url
        if tool.connector_id is not None:
            # The OpenAI SDK's Mcp TypedDict types ``connector_id`` as a
            # ``Literal[...]`` of fixed connector identifiers; the framework
            # accepts any string so callers can target connectors the
            # currently-installed SDK doesn't yet enumerate. The wire SDK
            # validates against the live API at request time.
            param["connector_id"] = tool.connector_id  # type: ignore[typeddict-item]
        if tool.server_description is not None:
            param["server_description"] = tool.server_description
        if tool.headers is not None:
            param["headers"] = dict(tool.headers)
        if tool.authorization is not None:
            param["authorization"] = tool.authorization
        if tool.require_approval is not None:
            # ``Mcp.require_approval`` is typed as a TypedDict union of
            # specific shapes; the framework accepts the simpler
            # ``Literal["always", "never"] | Mapping[str, list[str]]``
            # form and lets the SDK validate the structure.
            param["require_approval"] = tool.require_approval  # type: ignore[typeddict-item]
        if tool.allowed_tools is not None:
            # ``allowed_tools`` is typed as a TypedDict shape with optional
            # ``tool_names``/``always``/``never`` keys; the framework
            # accepts a flat list of names (the most common case) and
            # forwards verbatim — SDK validates at request time.
            param["allowed_tools"] = list(tool.allowed_tools)  # type: ignore[typeddict-item]
        if tool.defer_loading:
            param["defer_loading"] = True
        return param

    @classmethod
    def convert_tool_choice(
        cls,
        tool_choice: Optional[str],
        tools_present: bool,
    ) -> Union[ToolChoiceOptions, ToolChoiceFunctionParam, None]:
        """Convert framework ``ToolChoice`` to the Responses-API shape.

        The Responses API's named-tool shape is FLAT:
        ``{"type": "function", "name": "<tool_name>"}`` — no
        ``function: {name: ...}`` nesting (that's the Chat Completions
        shape).

        Mapping:
        - ``None`` / tools absent → ``None`` (omit).
        - ``"auto"`` / ``"required"`` / ``"none"`` → the string literal
          (``ToolChoiceOptions``).
        - Any other string → ``{"type": "function", "name": value}``.
        """
        if tool_choice is None or not tools_present:
            return None
        if tool_choice == "auto":
            return "auto"
        if tool_choice == "required":
            return "required"
        if tool_choice == "none":
            return "none"
        return {"type": "function", "name": tool_choice}

    # ------------------------------------------------------------------
    # Structured output → ResponseTextConfigParam
    # ------------------------------------------------------------------

    @classmethod
    def resolve_response_format(
        cls,
        output_schema: Optional[AgentOutputSchemaBase],
    ) -> Optional[ResponseTextConfigParam]:
        """Build a ``ResponseTextConfigParam`` with a JSON-schema format.

        The Responses API carries structured-output configuration inside
        ``text.format`` (unlike Chat Completions' ``response_format``
        field). Returns ``None`` when no schema is requested so the
        caller can omit ``text=`` entirely.
        """
        if output_schema is None:
            return None
        config: ResponseTextConfigParam = {
            "format": {
                "type": "json_schema",
                "name": output_schema.name(),
                "schema": output_schema.json_schema(),
                "strict": output_schema.is_strict_json_schema(),
            },
        }
        return config

    # ------------------------------------------------------------------
    # Response → LLMResponse
    # ------------------------------------------------------------------

    @classmethod
    def response_to_llm_response(cls, response: Response) -> LLMResponse:
        """Convert an OpenAI ``Response`` to a framework ``LLMResponse``.

        Core output items (``message`` / ``function_call`` /
        ``reasoning``) are mapped to the matching typed
        :class:`LLMResponsePart`. Every other item type (hosted-tool
        calls, hosted-tool outputs, MCP traffic, image generation,
        custom tool calls, compaction markers, …) lands in an
        :class:`LLMResponseProviderItem` with the verbatim provider
        payload under ``raw`` so that replay round-trips losslessly.
        """
        parts: list[LLMResponsePart] = []
        refusal_text: Optional[str] = None

        for item in response.output:
            if isinstance(item, ResponseOutputMessage):
                for block in item.content:
                    if isinstance(block, ResponseOutputText):
                        annotations = cls._annotations_from_output_text(block)
                        parts.append(LLMResponseText(text=block.text, annotations=annotations))
                    elif isinstance(block, ResponseOutputRefusal):
                        refusal_text = block.refusal
                        parts.append(LLMResponseRefusal(refusal=block.refusal))
                continue

            if isinstance(item, ResponseFunctionToolCall):
                parts.append(
                    LLMResponseFunctionToolCall(
                        call_id=item.call_id,
                        name=item.name,
                        arguments=item.arguments,
                        id=item.id,
                        status=item.status,
                    )
                )
                continue

            if isinstance(item, ResponseReasoningItem):
                summary_text = "\n".join(s.text for s in item.summary) if len(item.summary) > 0 else ""
                # Reasoning ``content`` entries may be
                # ``ResponseReasoningText`` or ``ResponseReasoningEncryptedContent``;
                # the latter has no ``.text`` field. Guard with ``hasattr``
                # so encrypted variants don't raise AttributeError here.
                thinking_text = (
                    "\n".join(c.text for c in item.content if hasattr(c, "text"))
                    if item.content is not None and len(item.content) > 0
                    else ""
                )
                parts.append(
                    LLMResponseReasoning(
                        thinking=thinking_text,
                        summary=summary_text if len(summary_text) > 0 else None,
                        id=item.id,
                        encrypted_content=item.encrypted_content,
                        status=item.status,
                    )
                )
                continue

            # Catch-all: every other output item becomes a provider_item.
            # ``mode="json"`` guarantees every nested value is a JSON
            # primitive — no aliased Pydantic instances in ``raw`` — so
            # mutations by downstream consumers (history processors,
            # compaction) cannot corrupt the source ``Response``.
            raw = item.model_dump(mode="json", exclude_unset=False)
            item_type = raw.get("type", "")
            parts.append(LLMResponseProviderItem(item_type=item_type, raw=raw))

        finish_reason = "stop"
        if refusal_text is not None:
            finish_reason = "refusal"
        elif response.status == "incomplete":
            # A truncated response is checked BEFORE the tool-call case: an
            # ``incomplete`` response can still carry a function_call whose
            # JSON ``arguments`` were cut off mid-generation. Reporting
            # "tool_calls" there would let the Runner execute a tool with
            # malformed arguments; "length"/"incomplete" makes the caller
            # treat the turn as truncated instead. ``incomplete_details``
            # holds the reason; surface "length" for the common max-tokens
            # case and fall back to the raw status.
            details = response.incomplete_details
            if details is not None and getattr(details, "reason", None) == "max_output_tokens":
                finish_reason = "length"
            else:
                finish_reason = "incomplete"
        elif response.status == "failed":
            # A failed terminal response (surfaced on the streaming path via
            # ResponseFailedEvent) must not read as a natural "stop"; mark it
            # so callers can distinguish a provider-side failure.
            finish_reason = "error"
        elif any(isinstance(p, LLMResponseFunctionToolCall) for p in parts):
            finish_reason = "tool_calls"

        return LLMResponse(
            response_id=response.id,
            model=response.model,
            response=parts,
            usage=cls.parse_usage(response.usage) if response.usage is not None else None,
            finish_reason=finish_reason,
        )

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    @classmethod
    def _annotations_from_output_text(
        cls,
        block: ResponseOutputText,
    ) -> Optional[list[LLMResponseAnnotation]]:
        """Map ``ResponseOutputText.annotations`` URL citations to framework annotations.

        ``ResponseOutputText.annotations`` is a discriminated union of
        four variants — ``url_citation``, ``file_citation``,
        ``container_file_citation``, and ``file_path``. The framework's
        :class:`LLMResponseAnnotation` models only URL citations
        (the primary web_search output); other variants are dropped
        here because the framework has no faithful representation for
        them. When that changes, extend this helper rather than adding
        a new field on ``LLMResponseText``.
        """
        raw_annotations = block.annotations if block.annotations is not None else []
        if len(raw_annotations) == 0:
            return None
        result: list[LLMResponseAnnotation] = []
        for ann in raw_annotations:
            if isinstance(ann, AnnotationURLCitation):
                result.append(
                    LLMResponseAnnotation(
                        url=ann.url,
                        title=ann.title,
                        start_index=ann.start_index,
                        end_index=ann.end_index,
                    )
                )
        return result if len(result) > 0 else None

    # ------------------------------------------------------------------
    # Usage
    # ------------------------------------------------------------------

    @classmethod
    def parse_usage(cls, usage: ResponseUsage) -> LLMUsage:
        """Convert ``ResponseUsage`` to framework ``LLMUsage``.

        Populates ``cached_tokens`` from ``input_tokens_details`` and
        ``reasoning_tokens`` from ``output_tokens_details``. Callers
        accumulate across turns via :meth:`LLMUsage.__add__`.
        """
        input_details = InputTokensDetails(
            cached_tokens=usage.input_tokens_details.cached_tokens,
            cache_creation_input_tokens=0,
        )
        output_details = OutputTokensDetails(
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
        )
        return LLMUsage(
            requests=1,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            input_tokens_details=input_details,
            output_tokens_details=output_details,
        )
