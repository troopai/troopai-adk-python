"""LLM output type definitions — Param TypedDicts for conversation replay.

Param types (TypedDict) are used for replaying conversation history —
sending output items back to the LLM on subsequent turns.

The canonical output types (dataclasses) live in ``types/responses/llm_response.py``.
This module provides the TypedDict replay versions and ``FunctionToolCallResult``
(a framework type for tool execution results, not a provider response).
"""

# Framework type (tool execution result — stays here, not in responses/)
from troopai.adk.types.output.function_tool_call_result import FunctionToolCallResult
from troopai.adk.types.output.function_tool_call_result_param import FunctionToolCallResultParam
from troopai.adk.types.output.llm_response_function_tool_call_param import LLMResponseFunctionToolCallParam
from troopai.adk.types.output.llm_response_message_param import LLMResponseMessageParam
from troopai.adk.types.output.llm_response_provider_item_param import LLMResponseProviderItemParam
from troopai.adk.types.output.llm_response_reasoning_param import LLMResponseReasoningParam
from troopai.adk.types.output.llm_response_refusal_param import LLMResponseRefusalParam

# TypedDict Param types (sent — replay versions)
from troopai.adk.types.output.llm_response_text_param import LLMResponseTextParam
from troopai.adk.types.output.reasoning_content_text_param import ReasoningContentTextParam
from troopai.adk.types.output.reasoning_summary_text_param import ReasoningSummaryTextParam

__all__ = [
    # Framework type
    "FunctionToolCallResult",
    "FunctionToolCallResultParam",
    "LLMResponseFunctionToolCallParam",
    "LLMResponseMessageParam",
    "LLMResponseProviderItemParam",
    "LLMResponseReasoningParam",
    "LLMResponseRefusalParam",
    # TypedDict Param types (sent — replay)
    "LLMResponseTextParam",
    "ReasoningContentTextParam",
    "ReasoningSummaryTextParam",
]
