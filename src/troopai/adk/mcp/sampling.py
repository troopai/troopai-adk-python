"""MCP sampling — let an MCP server call back into the host LLM.

Some MCP servers want to consult the host's LLM during a tool call
(e.g. a "summarise" tool that asks the host model to compress its
own working memory). The MCP protocol exposes this via the
``sampling/createMessage`` request. The handler in this module
bridges that request to the framework's ``LLM`` ABC: any provider
the developer already configured for the agent can serve sampling
requests with no extra setup.

The bridge is opt-in. To enable, pass ``llm`` to the MCP server
(``llm`` flows through to the underlying ``ClientSession``):

    server = MCPServerStdio(name="x", params=..., llm=my_llm)

Without ``llm``, the underlying ``ClientSession`` advertises no
sampling capability and servers fall back to alternative paths.

When the server's request includes tool definitions, they are forwarded
to the LLM so it can respond with tool-use content. The server drives
the tool-execution loop; this callback handles exactly one LLM turn.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp import types as mcp_types

    from troopai.adk.llms import LLM, LLMConfig
    from troopai.adk.types.input import LLMInputContentItem

logger = logging.getLogger(__name__)


def make_sampling_callback(llm: LLM) -> Any:
    """Build a ``ClientSession.sampling_callback`` bound to ``llm``.

    The callback receives the MCP server's
    ``CreateMessageRequestParams`` and returns a
    ``CreateMessageResult`` or ``CreateMessageResultWithTools``.
    Internally it converts the MCP messages to ``LLMInputContentItem``
    (Layer 1), calls ``llm.acomplete``, and converts the response
    back to the MCP wire shape.

    When the request carries tool definitions, they are forwarded to
    the LLM. The LLM may respond with tool-call content; in that case
    the callback returns ``CreateMessageResultWithTools`` with
    ``stopReason="toolUse"`` so the server can issue follow-up
    ``createMessage`` calls carrying the tool results.

    Failures are caught and converted to ``ErrorData`` so a buggy
    agent LLM cannot crash the MCP session — the server sees an
    error response and decides what to do next.

    Args:
        llm: The framework ``LLM`` instance to call for sampling
            requests.

    Returns:
        An async coroutine matching the MCP SDK's
        ``sampling_callback`` signature, ready to pass to
        ``ClientSession``.
    """

    async def _callback(
        ctx: Any, params: mcp_types.CreateMessageRequestParams
    ) -> mcp_types.CreateMessageResult | mcp_types.CreateMessageResultWithTools | mcp_types.ErrorData:
        del ctx  # The MCP SDK passes a session context we do not need.
        from mcp import types as mcp_types_

        try:
            tools = _build_sampling_tools(params) if params.tools is not None and len(params.tools) > 0 else None
            messages = _mcp_messages_to_layer1(params)
            config = _build_sampling_config(params)
            response = await llm.acomplete(messages=messages, tools=tools, llm_config=config)

            if len(response.tool_calls) > 0:
                # Detect tool calls by their PRESENCE, not by finish_reason:
                # Gemini reports finish_reason="STOP" even when it emits
                # function calls, so gating on finish_reason would silently
                # drop those calls and answer as plain text.
                # Annotate with the wire ``content`` union (not the narrower
                # ToolUseContent) so the list type matches
                # ``CreateMessageResultWithTools.content`` — a
                # ``list[ToolUseContent]`` is not assignable to the union list
                # under list invariance.
                content_blocks: list[mcp_types_.SamplingMessageContentBlock] = [
                    _tool_call_to_mcp_content(tc) for tc in response.tool_calls
                ]
                return mcp_types_.CreateMessageResultWithTools(
                    role="assistant",
                    content=content_blocks,
                    model=response.model,
                    stopReason="toolUse",
                )

            if response.finish_reason in ("tool_calls", "tool_use"):
                # Degenerate response: the LLM signalled tool intent but
                # produced no calls. Answer as a plain end-of-turn text
                # completion — a toolUse stop reason with a text body
                # would contradict the protocol and confuse the server.
                logger.warning(
                    "MCP sampling: finish_reason=%r with no tool calls; answering as endTurn text",
                    response.finish_reason,
                )

            text = response.content
            if text is None:
                logger.warning("MCP sampling produced no text part; returning a blank completion to the server.")
                text = ""
            return mcp_types_.CreateMessageResult(
                role="assistant",
                content=mcp_types_.TextContent(type="text", text=text),
                model=response.model,
                stopReason=_finish_reason_to_stop_reason(response.finish_reason),
            )
        except Exception as exc:
            # Log full exception locally; do NOT include the message
            # text in the wire response. A malicious MCP server could
            # otherwise harvest internal stack details / connection
            # strings / paths from exception messages.
            logger.warning("MCP sampling callback failed: %s", exc, exc_info=True)
            return mcp_types_.ErrorData(
                code=mcp_types_.INTERNAL_ERROR,
                message="sampling failed",
            )

    return _callback


def _build_sampling_config(params: mcp_types.CreateMessageRequestParams) -> LLMConfig:
    """Map the MCP sampling request's generation controls onto an ``LLMConfig``.

    The ``sampling/createMessage`` request carries ``maxTokens`` (required by
    the MCP spec), ``temperature``, and ``stopSequences`` that bound the host
    LLM's generation. Ignoring them would let a server's ``maxTokens`` contract
    be violated and its stop sequences and sampling temperature be discarded.
    Only fields the server actually set are forwarded; the rest stay unset.

    Args:
        params: The ``CreateMessageRequestParams`` from the MCP server.

    Returns:
        An ``LLMConfig`` carrying the request's generation controls.
    """
    from troopai.adk.llms.llm_config import LLMConfig

    stop_sequences = (
        list(params.stopSequences) if params.stopSequences is not None and len(params.stopSequences) > 0 else None
    )
    return LLMConfig(
        max_output_tokens=params.maxTokens,
        temperature=params.temperature,
        stop_sequences=stop_sequences,
    )


def _build_sampling_tools(params: mcp_types.CreateMessageRequestParams) -> list[Any]:
    """Convert MCP ``Tool`` definitions from the request to ``FunctionTool`` objects.

    Creates lightweight ``FunctionTool`` wrappers that carry the tool
    schema to the LLM. The ``on_invoke`` is a no-op placeholder because
    the server drives tool execution — this callback handles one LLM
    turn only.

    Args:
        params: The ``CreateMessageRequestParams`` carrying tool definitions.

    Returns:
        A list of ``FunctionTool`` objects ready for ``llm.acomplete``.
    """
    from troopai.adk.schemas import SchemaEnforcement
    from troopai.adk.tools.function_tool import FunctionTool

    tools: list[Any] = []
    if params.tools is None:
        return tools
    for mcp_tool in params.tools:
        raw_schema: dict[str, Any] = dict(mcp_tool.inputSchema or {})
        if "properties" not in raw_schema:
            raw_schema["properties"] = {}

        async def _noop_invoke(ctx: Any, raw_args: str) -> str:
            del ctx, raw_args
            return ""

        tools.append(
            FunctionTool(
                name=mcp_tool.name,
                description=mcp_tool.description or "",
                schema=raw_schema,
                schema_enforcement=SchemaEnforcement.NONE,
                on_invoke=_noop_invoke,
            )
        )
    return tools


def _tool_call_to_mcp_content(tool_call: Any) -> Any:
    """Convert a ``LLMResponseFunctionToolCall`` to an MCP ``ToolUseContent``.

    Args:
        tool_call: A ``LLMResponseFunctionToolCall`` from the LLM response.

    Returns:
        An MCP ``ToolUseContent`` block suitable for inclusion in
        ``CreateMessageResultWithTools.content``.
    """
    from mcp import types as mcp_types_

    try:
        input_dict = json.loads(tool_call.arguments) if len(tool_call.arguments) > 0 else {}
    except (json.JSONDecodeError, ValueError):
        logger.debug(
            "MCP sampling: tool call %r has non-JSON arguments %r; wrapping as raw string",
            tool_call.name,
            tool_call.arguments,
        )
        input_dict = {"raw_arguments": tool_call.arguments}
    return mcp_types_.ToolUseContent(
        type="tool_use",
        id=tool_call.call_id,
        name=tool_call.name,
        input=input_dict,
    )


def _mcp_messages_to_layer1(
    params: mcp_types.CreateMessageRequestParams,
) -> list[LLMInputContentItem]:
    """Convert MCP ``SamplingMessage`` list to Layer 1 input items.

    Text and image content surface as ``LLMInputEasyMessage`` items.
    ``ToolUseContent`` blocks convert to ``LLMResponseFunctionToolCallParam``
    (tool-call replay). ``ToolResultContent`` blocks convert to
    ``FunctionToolCallResultParam`` (tool-result replay). Audio content
    surfaces as a text placeholder.

    Security: a remote MCP server's ``systemPrompt`` is NOT promoted
    to a ``role="system"`` message because doing so would let a
    compromised server override the developer's agent-level system
    prompt. Instead we wrap it with a visible ``[MCP server hint]:``
    prefix and inject as a ``user`` role so the host model treats it
    as user-supplied context, not as authoritative instruction.

    Args:
        params: The ``CreateMessageRequestParams`` from the MCP server,
            carrying the message list and an optional system prompt.

    Returns:
        A list of ``LLMInputContentItem`` (Layer 1) ready for
        ``llm.acomplete``.
    """
    from troopai.adk.types.input.llm_input_easy_message import LLMInputEasyMessage

    items: list[LLMInputContentItem] = []
    if params.systemPrompt is not None:
        items.append(
            LLMInputEasyMessage(
                role="user",
                content=f"[MCP server hint]: {params.systemPrompt}",
            )
        )

    for msg in params.messages:
        for content_block in msg.content_as_list:
            item = _content_block_to_layer1(content_block, role=msg.role)
            if item is not None:
                items.append(item)

    return items


def _content_block_to_layer1(content: Any, role: str) -> LLMInputContentItem | None:
    """Convert a single MCP ``SamplingMessageContentBlock`` to a Layer 1 input item.

    Args:
        content: A content block from a ``SamplingMessage``; may be
            ``TextContent``, ``ImageContent``, ``AudioContent``,
            ``ToolUseContent``, or ``ToolResultContent``.
        role: The message role (``"user"`` or ``"assistant"``).

    Returns:
        A ``LLMInputContentItem`` (Layer 1), or ``None`` when the
        content type cannot be mapped.
    """
    from typing import Literal, cast

    from troopai.adk.types.input.llm_input_easy_message import LLMInputEasyMessage
    from troopai.adk.types.output.function_tool_call_result_param import FunctionToolCallResultParam
    from troopai.adk.types.output.llm_response_function_tool_call_param import LLMResponseFunctionToolCallParam

    # MCP Role is Literal["user", "assistant"]; both are valid LLMInputEasyMessage roles.
    typed_role = cast(Literal["user", "system", "developer", "assistant"], role)
    content_type = getattr(content, "type", None)

    if content_type == "text":
        return LLMInputEasyMessage(role=typed_role, content=str(content.text))

    if content_type == "tool_use":
        # Assistant is requesting a tool call — replay as function_call param
        result: LLMResponseFunctionToolCallParam = {
            "type": "function_call",
            "call_id": content.id,
            "name": content.name,
            "arguments": json.dumps(content.input),
        }
        return result

    if content_type == "tool_result":
        # User/assistant providing tool results — replay as function_call_output param
        if getattr(content, "structuredContent", None) is not None:
            logger.debug(
                "MCP sampling: tool_result %s carries structuredContent that is not "
                "forwarded to the LLM; only its text parts are replayed",
                content.toolUseId,
            )
        result_texts: list[str] = []
        for part in content.content:
            if hasattr(part, "text"):
                result_texts.append(str(part.text))
        output_str = "\n".join(result_texts)
        output_param: FunctionToolCallResultParam = {
            "type": "function_call_output",
            "call_id": content.toolUseId,
            "output": output_str,
        }
        return output_param

    if content_type == "image":
        return LLMInputEasyMessage(role=typed_role, content="[image content]")

    if content_type == "audio":
        return LLMInputEasyMessage(role=typed_role, content="[audio content]")

    logger.debug("MCP sampling: unhandled content type %r; skipping", content_type)
    return None


def _finish_reason_to_stop_reason(finish_reason: str | None) -> str:
    """Map an LLM ``finish_reason`` to an MCP ``stopReason`` literal.

    The MCP spec defines stop-reason literals: ``"endTurn"``,
    ``"maxTokens"``, ``"stopSequence"``, and ``"toolUse"``.
    Provider finish-reason strings are normalised to the closest MCP
    equivalent; unknown values fall back to ``"endTurn"``.

    Args:
        finish_reason: The ``finish_reason`` from ``LLMResponse``.

    Returns:
        An MCP-spec ``stopReason`` literal string.
    """
    if finish_reason in ("length", "max_tokens"):
        return "maxTokens"
    if finish_reason == "stop_sequence":
        return "stopSequence"
    # tool_calls/tool_use never reaches this mapper: the caller returns a
    # tool-use result before falling through to the text path, and the
    # no-calls degenerate case is deliberately answered as endTurn.
    return "endTurn"
