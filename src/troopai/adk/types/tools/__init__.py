from enum import StrEnum
from typing import Literal

from .approval_policy import ApprovalPolicy
from .builtin_tool_types import (
    ApplyPatchToolCall,
    ApplyPatchToolCallResult,
    CodeInterpreterOutput,
    CodeInterpreterToolCall,
    CodeInterpreterToolCallResult,
    ComputerAction,
    ComputerToolCall,
    ComputerToolCallResult,
    FileSearchResult,
    FileSearchToolCall,
    FileSearchToolCallResult,
    ImageGenerationToolCall,
    ImageGenerationToolCallResult,
    MCPApprovalRequest,
    MCPApprovalResponse,
    MCPCall,
    MCPCallResult,
    MCPListTools,
    MCPListToolsTool,
    ShellToolCall,
    ShellToolCallResult,
    ToolSearchResultEntry,
    ToolSearchToolCall,
    ToolSearchToolCallResult,
    WebSearchResult,
    WebSearchToolCall,
    WebSearchToolCallResult,
)
from .tool_rate_limit import ToolRateLimit, ToolRateLimitBehavior
from .tool_stream_event import ToolStreamEvent
from .tool_use_behavior import (
    FunctionToolResult,
    StopAtTools,
    ToolsToFinalOutputFunction,
    ToolsToFinalOutputResult,
    ToolUseBehavior,
)

type ToolChoice = Literal["auto", "required", "none"] | str
"""Tool selection strategy for LLM requests.

- ``"auto"`` — LLM decides when to call tools.
- ``"required"`` — LLM must call a tool every turn.
- ``"none"`` — LLM cannot call tools (text only).
- Any other string — force a specific tool by name.
"""


class ToolExecutionMode(StrEnum):
    """Controls whether the LLM may invoke multiple tools in a single turn.

    Maps to the ``parallel_tool_calls`` parameter originating from the
    OpenAI Chat Completions API.  Supported by multiple providers via
    litellm.

    Args:
        SEQUENTIAL: One tool call per LLM turn (default). The LLM must
            complete one tool call before requesting another.
        PARALLEL: The LLM may request multiple tool calls in a single
            response. All requested tools execute concurrently.
    """

    SEQUENTIAL = "sequential"
    """One tool call per LLM turn. Safer, easier to debug."""

    PARALLEL = "parallel"
    """Multiple tool calls per LLM turn. Faster, but tools must be independent.

    Supported by:

    - OpenAI (not all models)
    - Anthropic
    - Groq
    """


__all__ = [
    "ApplyPatchToolCall",
    "ApplyPatchToolCallResult",
    "ApprovalPolicy",
    "CodeInterpreterOutput",
    "CodeInterpreterToolCall",
    "CodeInterpreterToolCallResult",
    "ComputerAction",
    "ComputerToolCall",
    "ComputerToolCallResult",
    "FileSearchResult",
    "FileSearchToolCall",
    "FileSearchToolCallResult",
    "FunctionToolResult",
    "ImageGenerationToolCall",
    "ImageGenerationToolCallResult",
    "MCPApprovalRequest",
    "MCPApprovalResponse",
    "MCPCall",
    "MCPCallResult",
    "MCPListTools",
    "MCPListToolsTool",
    "ShellToolCall",
    "ShellToolCallResult",
    "StopAtTools",
    "ToolChoice",
    "ToolExecutionMode",
    "ToolRateLimit",
    "ToolRateLimitBehavior",
    "ToolSearchResultEntry",
    "ToolSearchToolCall",
    "ToolSearchToolCallResult",
    "ToolStreamEvent",
    "ToolUseBehavior",
    "ToolsToFinalOutputFunction",
    "ToolsToFinalOutputResult",
    "WebSearchResult",
    "WebSearchToolCall",
    "WebSearchToolCallResult",
]
