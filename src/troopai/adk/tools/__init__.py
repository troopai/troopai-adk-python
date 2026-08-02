from troopai.adk.tools.deferred_tool import (
    DeferredToolCall,
    DeferredToolCallMetadata,
    DeferredToolRequests,
    DeferredToolResults,
    ExternalToolCallResult,
    NestedDeferredToolRequests,
    ToolApprovalResult,
)
from troopai.adk.types.tools.tool_rate_limit import ToolRateLimit, ToolRateLimitBehavior

from .builtin import (
    BuiltinTool,
    CSVSearchTool,
    DirectorySearchTool,
    DocumentSearchInput,
    DocumentSearchTool,
    DOCXSearchTool,
    ExecutableBuiltinTool,
    ForgetMemoryTool,
    GithubSearchTool,
    InMemoryNoteStore,
    JITContextAwareTool,
    JSONSearchTool,
    MarkdownSearchTool,
    MemoryTool,
    NoteEntry,
    NoteStore,
    PDFSearchTool,
    RecallMemoryTool,
    RememberMemoryTool,
    TXTSearchTool,
    WebsiteSearchTool,
    YoutubeChannelSearchTool,
    YoutubeVideoSearchTool,
)
from .function_tool import FunctionTool, ToolCachePolicy, function_tool
from .hosted import (
    CodeExecutionTool,
    Computer,
    ComputerTool,
    FileSearchTool,
    HostedMCPTool,
    HostedTool,
    ImageGenerationTool,
    SafetyCheck,
    UnsupportedHostedToolError,
    URLContextTool,
    WebSearchTool,
)
from .local import (
    ApplyPatchEditor,
    ApplyPatchTool,
    ShellExecutor,
    ShellTool,
)
from .tool_context import ExecutionAwareToolContext, HistoryAwareToolContext, ToolContext
from .tool_guardrails import (
    AllowBehavior,
    RaiseExceptionBehavior,
    RejectContentBehavior,
    ToolGuardrailFunctionOutput,
    ToolGuardrails,
    ToolInputGuardrail,
    ToolInputGuardrailData,
    ToolInputGuardrailResult,
    ToolOutputGuardrail,
    ToolOutputGuardrailData,
    ToolOutputGuardrailResult,
    tool_input_guardrail,
    tool_output_guardrail,
)
from .tool_middleware import (
    ToolLoggingMiddleware,
    ToolMetricsMiddleware,
    ToolMetricsRecorder,
    ToolMiddleware,
    ToolMiddlewareNext,
    ToolMiddlewareTermination,
    compose_tool_middleware,
    wrap_tool_with_middleware,
)
from .tool_output_trimmer import DEFAULT_TRUNCATION_MARKER, trim_tool_output
from .tool_search import ToolSearchState, build_tool_search
from .toolsets import (
    CombinedToolset,
    FilteredToolset,
    FunctionToolset,
    PrefixedToolset,
    RenamedToolset,
    Toolset,
    ToolsetFilter,
    WrapperToolset,
)

type Tool = (
    FunctionTool
    | ShellTool
    | ApplyPatchTool
    | JITContextAwareTool
    | MemoryTool
    | RememberMemoryTool
    | RecallMemoryTool
    | ForgetMemoryTool
    | DocumentSearchTool
    | WebSearchTool
    | CodeExecutionTool
    | FileSearchTool
    | ImageGenerationTool
    | URLContextTool
    | HostedMCPTool
    | ComputerTool
)
"""Union of all tool types that can be passed to ``Agent.tools``.

The Runner partitions tools by type at runtime:

- ``FunctionTool``: Executed by the tool executor (custom Python functions).
- ``ExecutableBuiltinTool`` subclasses (``JITContextAwareTool``,
  ``MemoryTool``): Converted to function-call format by the LLM layer
  and executed by the framework via ``on_invoke``.
- Local-execution tools (``ShellTool``, ``ApplyPatchTool``): Expanded
  into ``FunctionTool`` instances at runtime using their user-supplied
  executor/editor.
- Provider-hosted tools (``HostedTool`` subclasses): typed config
  dataclasses that each provider's converter translates into the
  matching wire-format tool param. The framework does NOT execute
  these — the LLM provider does, server-side. Each provider's
  ``convert_tools`` either translates the variant or raises
  :class:`UnsupportedHostedToolError`.
"""

__all__ = [
    # Output trimming
    "DEFAULT_TRUNCATION_MARKER",
    # Guardrails - Behaviors
    "AllowBehavior",
    # Built-in Tools (framework-local)
    "ApplyPatchEditor",
    "ApplyPatchTool",
    "BuiltinTool",
    "CSVSearchTool",
    # Provider-hosted tools (HostedTool subclasses)
    "CodeExecutionTool",
    # Toolset composition
    "CombinedToolset",
    "Computer",
    "ComputerTool",
    "DOCXSearchTool",
    # Deferred execution (HITL)
    "DeferredToolCall",
    "DeferredToolCallMetadata",
    "DeferredToolRequests",
    "DeferredToolResults",
    "DirectorySearchTool",
    "DocumentSearchInput",
    "DocumentSearchTool",
    "ExecutableBuiltinTool",
    # Context
    "ExecutionAwareToolContext",
    "ExternalToolCallResult",
    "FileSearchTool",
    "FilteredToolset",
    # Function Tool
    "FunctionTool",
    "FunctionToolset",
    "GithubSearchTool",
    "HistoryAwareToolContext",
    "HostedMCPTool",
    "HostedTool",
    "ImageGenerationTool",
    "InMemoryNoteStore",
    "JITContextAwareTool",
    "JSONSearchTool",
    "MarkdownSearchTool",
    "NestedDeferredToolRequests",
    "NoteEntry",
    "NoteStore",
    "PDFSearchTool",
    "PrefixedToolset",
    "RaiseExceptionBehavior",
    "RejectContentBehavior",
    "RenamedToolset",
    "SafetyCheck",
    "ShellExecutor",
    "ShellTool",
    "TXTSearchTool",
    "Tool",
    "ToolApprovalResult",
    "ToolCachePolicy",
    "ToolContext",
    # Guardrails - Output
    "ToolGuardrailFunctionOutput",
    # Per-Tool guardrail config
    "ToolGuardrails",
    # Guardrails - Input
    "ToolInputGuardrail",
    "ToolInputGuardrailData",
    "ToolInputGuardrailResult",
    # Tool middleware
    "ToolLoggingMiddleware",
    "ToolMetricsMiddleware",
    "ToolMetricsRecorder",
    "ToolMiddleware",
    "ToolMiddlewareNext",
    "ToolMiddlewareTermination",
    # Guardrails - Output
    "ToolOutputGuardrail",
    "ToolOutputGuardrailData",
    "ToolOutputGuardrailResult",
    # Rate limiting
    "ToolRateLimit",
    "ToolRateLimitBehavior",
    # Deferred tool loading / search
    "ToolSearchState",
    "Toolset",
    "ToolsetFilter",
    "URLContextTool",
    "UnsupportedHostedToolError",
    "WebSearchTool",
    "WebsiteSearchTool",
    "WrapperToolset",
    "YoutubeChannelSearchTool",
    "YoutubeVideoSearchTool",
    "build_tool_search",
    "compose_tool_middleware",
    "function_tool",
    # Guardrails - Decorators
    "tool_input_guardrail",
    "tool_output_guardrail",
    "trim_tool_output",
    "wrap_tool_with_middleware",
]
