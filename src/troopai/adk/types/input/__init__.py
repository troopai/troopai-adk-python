"""LLM input type definitions — types sent TO the LLM.

All input types are TypedDict (``total=False`` + ``Required``).

The input union includes both native input types (text, image, audio, messages)
and replay versions of output types (tool calls, tool results, assistant messages,
reasoning). Every output eventually becomes input for the next turn.

Naming convention:
- ``LLMInput*`` (TypedDict): native input types.
- ``LLMInputEasyMessage``: convenience type accepting ``content: str | list``.
- Replay params (``LLMOutput*Param``, ``FunctionToolCallResultParam``) are
  imported from ``types.output`` and included in the input union.
"""

from troopai.adk.types.input.llm_input_audio import LLMInputAudio
from troopai.adk.types.input.llm_input_easy_message import LLMInputEasyMessage
from troopai.adk.types.input.llm_input_image import LLMInputImage
from troopai.adk.types.input.llm_input_message import LLMInputContents, LLMInputMessage

# Native input types
from troopai.adk.types.input.llm_input_text import LLMInputText

type LLMInputContentPart = LLMInputText | LLMInputImage | LLMInputAudio
"""Union of bare content part types (text, image, audio).

Used for typing content lists inside messages and tool outputs where a
single-level content part is expected (no message wrapper).
"""

# Replay params (TypedDict versions of output types)
from troopai.adk.types.output.function_tool_call_result_param import FunctionToolCallResultParam
from troopai.adk.types.output.llm_response_function_tool_call_param import LLMResponseFunctionToolCallParam
from troopai.adk.types.output.llm_response_message_param import LLMResponseMessageParam
from troopai.adk.types.output.llm_response_provider_item_param import LLMResponseProviderItemParam
from troopai.adk.types.output.llm_response_reasoning_param import LLMResponseReasoningParam

type LLMInputContentItem = (
    LLMInputText
    | LLMInputImage
    | LLMInputAudio
    | LLMInputMessage
    | LLMInputEasyMessage
    | LLMResponseFunctionToolCallParam
    | FunctionToolCallResultParam
    | LLMResponseMessageParam
    | LLMResponseReasoningParam
    | LLMResponseProviderItemParam
)
"""Union of all input content items.

Includes native input types AND replay versions of output types.
The full conversation is: user message (input) → assistant tool call (output replay)
→ tool result (output replay) → assistant response (output replay).

Reasoning replay uses ``LLMResponseReasoningParam`` — one TypedDict for both
the initial reasoning output and its replay form on subsequent turns.

``LLMResponseProviderItemParam`` replays provider-hosted-tool output items
(OpenAI Responses: ``file_search_call``, ``web_search_call``,
``image_generation_call``, ``code_interpreter_call``, ``computer_call``,
``mcp_call``). Providers that do not understand these items MUST ignore
them (the litellm and anthropic paths do).
"""

__all__ = [
    "FunctionToolCallResultParam",
    "LLMInputAudio",
    # Union
    "LLMInputContentItem",
    # Content part union (used by converters for inner message content lists)
    "LLMInputContentPart",
    "LLMInputContents",
    "LLMInputEasyMessage",
    "LLMInputImage",
    "LLMInputMessage",
    # Native input types
    "LLMInputText",
    # Replay params (re-exported from types.output)
    "LLMResponseFunctionToolCallParam",
    "LLMResponseMessageParam",
    "LLMResponseProviderItemParam",
    "LLMResponseReasoningParam",
]
