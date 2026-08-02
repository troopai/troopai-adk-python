"""Stateless converter between Layer 1 items and Gemini's native format.

Bridges provider-agnostic ``LLMInputContentItem`` types to/from
``google.genai.types.*`` wire types. All methods are ``@classmethod``
— no instance state, pure functions.

Key Gemini differences from Anthropic / OpenAI:

- **System instruction is a separate top-level parameter** on
  ``GenerateContentConfig.system_instruction``, not a message in the
  ``contents`` array.
- **Conversation messages use the ``Content`` shape**: a list of
  ``Content(parts=[Part], role="user"|"model")``. Roles are
  ``"user"`` and ``"model"`` (not ``"assistant"``).
- **Function calls and responses are ``Part`` variants**, not separate
  array entries. Multiple function calls in one turn share a single
  assistant ``Content`` with multiple ``Part(function_call=...)``.
- **Structured output is native** via
  ``GenerateContentConfig.response_mime_type="application/json"``
  paired with ``response_schema`` — no synthetic-tool pattern.
- **Hosted tools live as fields on a single ``Tool`` instance**
  (``Tool(function_declarations=..., google_search=..., code_execution=...,
  url_context=...)``). The converter folds every framework tool into
  one ``Tool`` instance per request.
- **Thinking parts** are regular ``Part(thought=True, text=...,
  thought_signature=<bytes>)`` items. The signature is opaque bytes
  that must be replayed verbatim across turns to preserve thinking
  context.

Refs:
    - Gemini API: https://ai.google.dev/api
    - Function calling: https://ai.google.dev/gemini-api/docs/function-calling
    - Structured output: https://ai.google.dev/gemini-api/docs/structured-output
    - Thinking: https://ai.google.dev/gemini-api/docs/thinking
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from typing import TYPE_CHECKING, Any

from google.genai.types import (
    Content,
    FunctionCall,
    FunctionCallingConfig,
    FunctionCallingConfigMode,
    FunctionDeclaration,
    FunctionResponse,
    GoogleSearch,
    Part,
    Tool,
    ToolCodeExecution,
    ToolConfig,
    UrlContext,
)

from troopai.adk.types.responses.llm_response import (
    LLMResponse,
    LLMResponseFunctionToolCall,
    LLMResponsePart,
    LLMResponseReasoning,
    LLMResponseText,
)

if TYPE_CHECKING:
    from google.genai.types import GenerateContentResponse, GenerateContentResponseUsageMetadata

    from troopai.adk.tools import Tool as FrameworkTool
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.tokens import LLMUsage

logger = logging.getLogger(__name__)


class GeminiConverter:
    """Stateless converter: Layer 1 ↔ Gemini wire format."""

    # ------------------------------------------------------------------
    # Layer 1 → Gemini contents + system instruction
    # ------------------------------------------------------------------

    @classmethod
    def items_to_contents(
        cls,
        items: str | list[LLMInputContentItem],
    ) -> tuple[str | None, list[Content]]:
        """Convert Layer 1 items to (system_instruction, contents) pair.

        Gemini requires the system prompt as a separate
        ``system_instruction`` parameter on ``GenerateContentConfig``.
        This method extracts ``role: "system"`` and ``role: "developer"``
        messages and joins them into a single string returned as the
        first element. Everything else is converted into the
        ``contents`` list using the ``Content(parts=..., role=...)``
        shape.

        Returns:
            ``(system_instruction, contents)`` where
            ``system_instruction`` is ``None`` when no system message
            was present.
        """
        if isinstance(items, str):
            return None, [Content(role="user", parts=[Part.from_text(text=items)])]

        system_parts: list[str] = []
        contents: list[Content] = []
        # Gemini correlates a FunctionResponse to its FunctionCall by
        # ``name`` (matching FunctionDeclaration.name), not just by
        # ``id``. The Layer 1 tool-result item only carries ``call_id``
        # (which equals the original FunctionCall.id on models that
        # return one), so recover the real tool name from the matching
        # ``function_call`` item before emitting the response Part.
        call_id_to_name: dict[str, str] = {}
        for item in items:
            if isinstance(item, dict) and item.get("type") == "function_call":
                cid = str(item.get("call_id", ""))
                if len(cid) > 0:
                    call_id_to_name[cid] = str(item.get("name", ""))
        # Accumulate parts for the current message so we can flush
        # when role changes.
        current_role: str | None = None
        current_parts: list[Part] = []

        def flush() -> None:
            nonlocal current_role, current_parts
            if current_role is not None and len(current_parts) > 0:
                contents.append(Content(role=current_role, parts=current_parts))
            current_role = None
            current_parts = []

        for item in items:
            item_type = item.get("type") if isinstance(item, dict) else None

            # System / developer messages → system_instruction.
            if isinstance(item, dict) and item_type == "message":
                role = item.get("role", "")
                if role in ("system", "developer"):
                    system_parts.extend(cls._system_text_from_content(item.get("content", "")))
                    continue

            # Strict message (LLMInputMessage / LLMResponseMessageParam).
            if isinstance(item, dict) and item_type == "message":
                role = item.get("role", "user")
                target_role = cls._map_role(str(role))
                if current_role != target_role:
                    flush()
                    current_role = target_role
                content = item.get("content", "")
                if isinstance(content, str):
                    current_parts.append(Part.from_text(text=content))
                elif isinstance(content, list):
                    for part in content:
                        converted = cls._convert_content_part(part)
                        if converted is not None:
                            current_parts.append(converted)
                continue

            # Easy message (no type field, has content + role).
            if isinstance(item, dict) and "content" in item and item_type is None:
                role = item.get("role", "user")
                if role in ("system", "developer"):
                    # A list content here previously collapsed to "" — drop the
                    # whole system prompt. Extract its text parts instead.
                    system_parts.extend(cls._system_text_from_content(item.get("content", "")))
                    continue
                target_role = cls._map_role(str(role))
                if current_role != target_role:
                    flush()
                    current_role = target_role
                content = item.get("content", "")
                if isinstance(content, str):
                    current_parts.append(Part.from_text(text=content))
                elif isinstance(content, list):
                    for part in content:
                        converted = cls._convert_content_part(part)
                        if converted is not None:
                            current_parts.append(converted)
                continue

            # Tool-call replay: function_call → assistant Part.
            if isinstance(item, dict) and item_type == "function_call":
                if current_role != "model":
                    flush()
                    current_role = "model"
                call_id = str(item.get("call_id", ""))
                name = str(item.get("name", ""))
                arguments = item.get("arguments", "{}")
                try:
                    args = json.loads(arguments) if isinstance(arguments, str) else arguments
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": arguments}
                args_dict: dict[str, Any] = dict(args) if isinstance(args, dict) else {"raw": args}
                # Re-attach the thinking model's ``thought_signature`` (stored
                # as a base64 str in ``signature``) so the model's thinking
                # context replays verbatim. Invalid / absent base64 is omitted
                # rather than forwarded as corrupted bytes.
                call_sig_str = item.get("signature")
                call_sig_bytes: bytes | None = None
                if isinstance(call_sig_str, str) and len(call_sig_str) > 0:
                    try:
                        call_sig_bytes = base64.b64decode(call_sig_str, validate=True)
                    except (binascii.Error, ValueError):
                        # Reachable only via externally corrupted/tampered
                        # history — our encode path always emits valid base64.
                        # Drop the opaque bytes rather than forward corruption,
                        # but log it: a silent drop resets the thinking context
                        # for this call with no breadcrumb for the operator.
                        logger.warning(
                            "Dropping malformed base64 thought signature for "
                            "tool call %r; its thinking context will not replay.",
                            name,
                        )
                        call_sig_bytes = None
                current_parts.append(
                    Part(
                        function_call=FunctionCall(
                            id=call_id or None,
                            name=name,
                            args=args_dict,
                        ),
                        thought_signature=call_sig_bytes,
                    )
                )
                continue

            # Tool result replay: function_call_output → user Part.
            if isinstance(item, dict) and item_type == "function_call_output":
                if current_role != "user":
                    flush()
                    current_role = "user"
                call_id = str(item.get("call_id", ""))
                # Gemini's FunctionResponse.name MUST match the
                # originating FunctionCall.name. Recover it from the
                # paired ``function_call`` item; fall back to ``call_id``
                # only when no match is present (malformed history).
                response_name = call_id_to_name.get(call_id) or str(item.get("name", call_id))
                response_str = item.get("output", "")
                if isinstance(response_str, str):
                    try:
                        response_obj = json.loads(response_str)
                        if not isinstance(response_obj, dict):
                            response_obj = {"result": response_obj}
                    except (json.JSONDecodeError, TypeError):
                        response_obj = {"result": response_str}
                else:
                    response_obj = (
                        dict(response_str) if isinstance(response_str, dict) else {"result": str(response_str)}
                    )
                current_parts.append(
                    Part(
                        function_response=FunctionResponse(
                            id=call_id or None,
                            name=response_name,
                            response=response_obj,
                        )
                    )
                )
                continue

            # Reasoning replay: emit a thought Part. Gemini's
            # ``thought_signature`` is opaque bytes; Layer 1 stores it as a
            # base64 string in ``encrypted_content`` (see
            # ``_convert_response_part``), which round-trips those bytes
            # losslessly — a utf-8 decode/encode would mangle any non-utf-8
            # signature and break Gemini's thinking-context validation.
            if isinstance(item, dict) and item_type == "reasoning":
                if current_role != "model":
                    flush()
                    current_role = "model"
                content = item.get("content", [])
                thinking_text = ""
                if isinstance(content, list):
                    for tb in content:
                        if isinstance(tb, dict) and tb.get("type") == "reasoning_text":
                            thinking_text = str(tb.get("text", ""))
                            break
                signature_str = item.get("encrypted_content")
                signature_bytes: bytes | None = None
                if isinstance(signature_str, str) and len(signature_str) > 0:
                    try:
                        signature_bytes = base64.b64decode(signature_str, validate=True)
                    except (binascii.Error, ValueError):
                        # Not valid base64 — cannot be replayed verbatim, so
                        # drop it rather than forwarding corrupted bytes.
                        signature_bytes = None
                current_parts.append(
                    Part(
                        thought=True,
                        text=thinking_text,
                        thought_signature=signature_bytes,
                    )
                )
                continue

            # Bare content parts (input_text, input_image) — wrap into
            # a user message.
            if isinstance(item, dict) and item_type in ("input_text", "input_image"):
                if current_role != "user":
                    flush()
                    current_role = "user"
                converted = cls._convert_content_part(item)
                if converted is not None:
                    current_parts.append(converted)
                continue

        flush()

        system_instruction = "\n\n".join(system_parts) if len(system_parts) > 0 else None
        return system_instruction, contents

    @staticmethod
    def _map_role(role: str) -> str:
        """Map framework roles to Gemini roles.

        Gemini uses ``"user"`` and ``"model"`` (not ``"assistant"``).
        """
        if role in ("assistant", "model"):
            return "model"
        return "user"

    @staticmethod
    def _system_text_from_content(content: Any) -> list[str]:
        """Extract system-instruction text fragments from a message ``content``.

        Accepts a plain string or a list of content parts, pulling the text
        out of ``input_text`` / ``output_text`` / ``text`` parts. A list
        content on the easy-message path previously collapsed to an empty
        string, silently dropping the system prompt.
        """
        if isinstance(content, str):
            return [content]
        fragments: list[str] = []
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("input_text", "output_text", "text"):
                    fragments.append(str(part.get("text", "")))
        return fragments

    @classmethod
    def _convert_content_part(cls, part: Any) -> Part | None:
        """Convert a single content part to a Gemini ``Part``."""
        if isinstance(part, str):
            return Part.from_text(text=part)
        if not isinstance(part, dict):
            return None
        part_type = part.get("type", "")
        if part_type in ("input_text", "output_text", "text"):
            return Part.from_text(text=str(part.get("text", "")))
        if part_type == "input_image":
            source = part.get("source", part.get("image_url", ""))
            if isinstance(source, str):
                return cls._image_part_from_str(source)
            if isinstance(source, dict):
                src_type = source.get("type")
                if src_type == "url":
                    return Part.from_uri(
                        file_uri=str(source.get("url", "")),
                        mime_type=str(source.get("media_type", "image/jpeg")),
                    )
                if src_type == "base64":
                    raw = base64.b64decode(str(source.get("data", "")))
                    return Part.from_bytes(
                        data=raw,
                        mime_type=str(source.get("media_type", "image/jpeg")),
                    )
        logger.debug("Gemini: dropping unsupported content part type=%s", part_type)
        return None

    @classmethod
    def _image_part_from_str(cls, source: str) -> Part:
        """Convert a string image source (URL or data URI) to a ``Part``.

        A ``data:`` URI carries the bytes inline, so it must become an
        inline ``Blob`` via ``Part.from_bytes`` — Gemini treats a
        ``file_uri`` as a fetchable GCS / Files-API reference and cannot
        load a ``data:`` URI. A remote URL becomes a ``file_uri`` with the
        MIME type inferred from the path (the documented inputs include
        PNG / GIF / WebP, not only JPEG).
        """
        if source.startswith("data:"):
            header, _, payload = source.partition(",")
            # header looks like ``data:image/png;base64``.
            media_type = header[len("data:") :].split(";")[0] or "image/jpeg"
            raw = base64.b64decode(payload)
            return Part.from_bytes(data=raw, mime_type=media_type)
        import mimetypes

        inferred, _ = mimetypes.guess_type(source)
        return Part.from_uri(file_uri=source, mime_type=inferred or "image/jpeg")

    # ------------------------------------------------------------------
    # Tool conversion
    # ------------------------------------------------------------------

    @classmethod
    def convert_tools(cls, tools: list[FrameworkTool]) -> list[Tool] | None:
        """Convert framework tools to Gemini's single-``Tool`` shape.

        Gemini expects a list of ``Tool`` instances where each instance
        bundles function declarations alongside hosted-tool fields
        (``google_search``, ``code_execution``, ``url_context``). For
        simplicity we collect every framework tool into a SINGLE
        ``Tool`` instance — the SDK accepts that and it matches the
        canonical wire shape.

        Raises:
            UnsupportedHostedToolError: If a hosted-tool variant is
                supplied that Gemini does not support
                (``FileSearchTool``, ``ImageGenerationTool``).
        """
        if len(tools) == 0:
            return None

        from pydantic import BaseModel

        from troopai.adk.schemas.utils import normalize_schema
        from troopai.adk.tools.builtin.builtin_tool import ExecutableBuiltinTool
        from troopai.adk.tools.function_tool import FunctionTool
        from troopai.adk.tools.hosted import (
            CodeExecutionTool,
            HostedTool,
            UnsupportedHostedToolError,
            URLContextTool,
            WebSearchTool,
        )

        function_decls: list[FunctionDeclaration] = []
        google_search: GoogleSearch | None = None
        code_execution: ToolCodeExecution | None = None
        url_context: UrlContext | None = None

        for tool in tools:
            if isinstance(tool, FunctionTool):
                schema = tool.get_json_schema() or {"type": "object", "properties": {}}
                function_decls.append(
                    FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters_json_schema=dict(schema),
                    )
                )
                continue

            if isinstance(tool, ExecutableBuiltinTool):
                raw_schema = (
                    tool.schema.model_json_schema()
                    if isinstance(tool.schema, type) and issubclass(tool.schema, BaseModel)
                    else tool.schema
                )
                norm = normalize_schema(raw_schema) or {"type": "object", "properties": {}}
                function_decls.append(
                    FunctionDeclaration(
                        name=tool.name,
                        description=tool.description,
                        parameters_json_schema=dict(norm),
                    )
                )
                continue

            if isinstance(tool, WebSearchTool):
                google_search = GoogleSearch()
                if any(
                    v is not None
                    for v in (
                        tool.max_uses,
                        tool.allowed_domains,
                        tool.blocked_domains,
                        tool.user_location,
                        tool.search_context_size,
                    )
                ):
                    logger.debug(
                        "Gemini: WebSearchTool tunable knobs are silently dropped — "
                        "google_search has no parameters at this time."
                    )
                continue

            if isinstance(tool, CodeExecutionTool):
                code_execution = ToolCodeExecution()
                if tool.container is not None:
                    logger.debug("Gemini: CodeExecutionTool.container is OpenAI-only; ignoring.")
                continue

            if isinstance(tool, URLContextTool):
                url_context = UrlContext()
                continue

            if isinstance(tool, HostedTool):
                # Hosted tool Gemini does not ship.
                raise UnsupportedHostedToolError(
                    tool,
                    "gemini",
                    supported_providers=tool.SUPPORTED_PROVIDERS,
                )

            logger.warning("Unknown tool type for Gemini: %s", type(tool))

        # Fold everything into a single Tool instance. If the loop
        # left every field unset (e.g. caller passed only unrecognised
        # tool types), return None so we don't ship an empty Tool that
        # the SDK validation would reject.
        if len(function_decls) == 0 and google_search is None and code_execution is None and url_context is None:
            return None
        wire_tool = Tool(
            function_declarations=function_decls if len(function_decls) > 0 else None,
            google_search=google_search,
            code_execution=code_execution,
            url_context=url_context,
        )
        return [wire_tool]

    @classmethod
    def convert_tool_choice(
        cls,
        tool_choice: str | None,
        *,
        tools_present: bool,
    ) -> ToolConfig | None:
        """Convert framework ``ToolChoice`` to Gemini's ``ToolConfig``.

        Mapping:
        - ``None`` / tools absent → ``None`` (omit; Gemini defaults to AUTO).
        - ``"auto"`` → ``mode=AUTO``.
        - ``"required"`` → ``mode=ANY`` (force the model to call SOME tool).
        - ``"none"`` → ``mode=NONE``.
        - Any other string → ``mode=ANY`` with ``allowed_function_names=[name]``.
        """
        if tool_choice is None or not tools_present:
            return None
        if tool_choice == "auto":
            mode = FunctionCallingConfigMode.AUTO
            allowed = None
        elif tool_choice == "required":
            mode = FunctionCallingConfigMode.ANY
            allowed = None
        elif tool_choice == "none":
            mode = FunctionCallingConfigMode.NONE
            allowed = None
        else:
            mode = FunctionCallingConfigMode.ANY
            allowed = [tool_choice]
        return ToolConfig(
            function_calling_config=FunctionCallingConfig(
                mode=mode,
                allowed_function_names=allowed,
            )
        )

    # ------------------------------------------------------------------
    # Gemini response → LLMResponse
    # ------------------------------------------------------------------

    @classmethod
    def response_to_llm_response(
        cls,
        response: GenerateContentResponse,
    ) -> LLMResponse:
        """Convert Gemini ``GenerateContentResponse`` to ``LLMResponse``.

        Walks ``candidates[0].content.parts``: text → ``LLMResponseText``,
        thought-flagged text → ``LLMResponseReasoning`` with
        ``encrypted_content`` carrying the ``thought_signature`` bytes
        as a base64 string, ``function_call`` → ``LLMResponseFunctionToolCall``
        with JSON-serialised args and any ``thought_signature`` carried
        (base64) in ``signature``.
        """
        parts: list[LLMResponsePart] = []
        finish_reason: str | None = None
        candidates = response.candidates or []
        if len(candidates) > 0:
            cand = candidates[0]
            if cand.finish_reason is not None:
                finish_reason = (
                    cand.finish_reason.value if hasattr(cand.finish_reason, "value") else str(cand.finish_reason)
                )
            content = cand.content
            if content is not None and content.parts is not None:
                for part in content.parts:
                    converted = cls._convert_response_part(part)
                    if converted is not None:
                        parts.append(converted)

        usage = cls._parse_usage(response.usage_metadata) if response.usage_metadata is not None else None

        return LLMResponse(
            response_id=response.response_id or "",
            model=response.model_version or "",
            response=parts,
            usage=usage,
            finish_reason=finish_reason,
        )

    @classmethod
    def _convert_response_part(cls, part: Part) -> LLMResponsePart | None:
        """Convert a single response :class:`Part` to a framework part."""
        # Thought first: Gemini sets thought=True on a Part whose text
        # is the reasoning summary; thought_signature carries the opaque
        # bytes for replay. Base64-encode them (lossless for arbitrary
        # bytes) so the round-trip through ``encrypted_content`` and back
        # to ``thought_signature`` preserves the exact signature — a plain
        # utf-8 decode would corrupt any non-utf-8 bytes.
        if part.thought is True:
            sig_bytes = part.thought_signature
            sig_str = base64.b64encode(sig_bytes).decode("ascii") if sig_bytes is not None else None
            return LLMResponseReasoning(
                thinking=part.text or "",
                signature=None,
                encrypted_content=sig_str,
            )
        if part.text is not None and len(part.text) > 0:
            return LLMResponseText(text=part.text)
        if part.function_call is not None:
            fc = part.function_call
            args_dict = fc.args if fc.args is not None else {}
            # A thinking model attaches a ``thought_signature`` to the
            # function_call part; it is opaque bytes that MUST replay verbatim
            # to preserve thinking context. Base64-encode (lossless for
            # arbitrary bytes) so it survives JSON serialize → reload → replay —
            # a utf-8 decode would corrupt any non-utf-8 signature.
            fc_sig_bytes = part.thought_signature
            fc_sig_str = base64.b64encode(fc_sig_bytes).decode("ascii") if fc_sig_bytes is not None else None
            return LLMResponseFunctionToolCall(
                call_id=fc.id or fc.name or "",
                name=fc.name or "",
                arguments=json.dumps(dict(args_dict)),
                signature=fc_sig_str,
            )
        # Other variants (inline_data, file_data, executable_code,
        # code_execution_result) are not yet surfaced as framework types.
        # Warn rather than silently drop so developers using CodeExecutionTool
        # with GeminiLLM are not surprised to find their execution results
        # missing from the LLMResponse.
        if part.executable_code is not None:
            logger.warning(
                "Gemini: executable_code response part is not yet wired to a "
                "framework LLMResponsePart — the code snippet is dropped. "
                "Execution output will not appear in LLMResponse.response."
            )
        elif part.code_execution_result is not None:
            logger.warning(
                "Gemini: code_execution_result response part is not yet wired to a "
                "framework LLMResponsePart — the execution output is dropped. "
                "Results will not appear in LLMResponse.response."
            )
        return None

    @classmethod
    def _parse_usage(cls, usage: GenerateContentResponseUsageMetadata) -> LLMUsage:
        """Convert Gemini usage metadata to framework ``LLMUsage``."""
        from troopai.adk.types.tokens import (
            InputTokensDetails,
            LLMSingleRequestUsage,
            LLMUsage,
            OutputTokensDetails,
        )

        prompt = usage.prompt_token_count or 0
        cand = usage.candidates_token_count or 0
        cached = usage.cached_content_token_count or 0
        thoughts = usage.thoughts_token_count or 0
        total = usage.total_token_count or (prompt + cand)

        input_details = InputTokensDetails(
            cached_tokens=cached,
            cache_creation_input_tokens=0,
        )
        output_details = OutputTokensDetails(reasoning_tokens=thoughts)
        single = LLMSingleRequestUsage(
            input_tokens=prompt,
            output_tokens=cand,
            total_tokens=total,
            input_tokens_details=input_details,
            output_tokens_details=output_details,
        )
        return LLMUsage(
            requests=1,
            input_tokens=prompt,
            output_tokens=cand,
            total_tokens=total,
            input_tokens_details=input_details,
            output_tokens_details=output_details,
            usage=[single],
        )
