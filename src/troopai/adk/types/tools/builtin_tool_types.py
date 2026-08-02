"""Built-in tool call and result types.

Provider-agnostic types for structured access to built-in tool
calls (from LLM responses) and their results.  These complement
``BuiltinTool`` subclasses (which define tools sent TO the LLM) by
typing what comes BACK from the LLM.

All types have a ``type`` discriminator field for serialization.

Naming conventions:
- ``*ToolCall`` — a tool invocation from the LLM
- ``*ToolCallResult`` — the result of executing a tool call
- ``MCP*`` — Model Context Protocol types
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WebSearchResult:
    """A single web search result entry."""

    url: str
    """The URL of the search result."""

    title: str
    """The title of the search result page."""

    snippet: str | None = None
    """Optional text snippet from the result."""


@dataclass(frozen=True)
class WebSearchToolCall:
    """A web search call from the LLM response."""

    type: Literal["web_search_call"] = "web_search_call"
    """Discriminator."""

    id: str = ""
    """Unique identifier for this tool call."""

    query: str = ""
    """The search query issued by the LLM."""

    status: str | None = None
    """Optional status string (e.g. ``"completed"``, ``"failed"``)."""


@dataclass(frozen=True)
class WebSearchToolCallResult:
    """Result from a web search tool call."""

    type: Literal["web_search_call_output"] = "web_search_call_output"
    """Discriminator."""

    call_id: str = ""
    """ID of the tool call this result corresponds to."""

    results: list[WebSearchResult] = field(default_factory=list)
    """List of web search result entries."""


# ---------------------------------------------------------------------------
# File search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileSearchResult:
    """A single file search result entry."""

    file_id: str | None = None
    """Optional file identifier in the vector store."""

    filename: str | None = None
    """Optional filename of the matched document."""

    score: float | None = None
    """Optional relevance score (0.0 to 1.0)."""

    text: str | None = None
    """Optional matched text content."""


@dataclass(frozen=True)
class FileSearchToolCall:
    """A file search call from the LLM response."""

    type: Literal["file_search_call"] = "file_search_call"
    """Discriminator."""

    id: str = ""
    """Unique identifier for this tool call."""

    queries: list[str] = field(default_factory=list)
    """The search queries issued by the LLM."""

    status: str | None = None
    """Optional status string (e.g. ``"completed"``, ``"failed"``)."""


@dataclass(frozen=True)
class FileSearchToolCallResult:
    """Result from a file search tool call."""

    type: Literal["file_search_call_output"] = "file_search_call_output"
    """Discriminator."""

    call_id: str = ""
    """ID of the tool call this result corresponds to."""

    results: list[FileSearchResult] = field(default_factory=list)
    """List of file search result entries."""


# ---------------------------------------------------------------------------
# Computer use
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputerAction:
    """An action requested by the computer tool.

    Flat structure covering all providers' action types.
    The ``type`` field determines the action kind.
    """

    type: str = ""
    """Action kind (``"click"``, ``"type"``, ``"screenshot"``,
    ``"scroll"``, ``"keypress"``, ``"move"``, ``"drag"``,
    ``"double_click"``, ``"wait"``, etc.)."""

    x: int | None = None
    """Optional X coordinate for pointer actions."""

    y: int | None = None
    """Optional Y coordinate for pointer actions."""

    text: str | None = None
    """Optional text for type actions."""

    keys: list[str] | None = None
    """Optional key names for keypress actions."""

    button: str | None = None
    """Optional mouse button for click actions."""

    scroll_x: int | None = None
    """Optional horizontal scroll amount."""

    scroll_y: int | None = None
    """Optional vertical scroll amount."""


@dataclass(frozen=True)
class ComputerToolCall:
    """A computer use call from the LLM response."""

    type: Literal["computer_call"] = "computer_call"
    """Discriminator."""

    id: str = ""
    """Unique identifier for this tool call."""

    action: ComputerAction | None = None
    """The action requested by the LLM."""

    call_id: str = ""
    """ID used for matching results to calls."""

    status: str | None = None
    """Optional status string (e.g. ``"completed"``, ``"failed"``)."""


@dataclass(frozen=True)
class ComputerToolCallResult:
    """Result from a computer tool call."""

    type: Literal["computer_call_output"] = "computer_call_output"
    """Discriminator."""

    call_id: str = ""
    """ID of the tool call this result corresponds to."""

    output: str | None = None
    """Optional text output from the action."""

    screenshot: str | None = None
    """Optional base64-encoded screenshot image."""


# ---------------------------------------------------------------------------
# Code interpreter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CodeInterpreterOutput:
    """A single output from code interpreter execution."""

    type: str = ""
    """Output kind (``"logs"``, ``"image"``, ``"file"``)."""

    content: str | None = None
    """Text content (for logs/errors)."""

    image: str | None = None
    """Optional base64-encoded image output or URL."""

    file_id: str | None = None
    """Optional file identifier for generated files."""

    filename: str | None = None
    """Optional filename for generated files."""


@dataclass(frozen=True)
class CodeInterpreterToolCall:
    """A code interpreter call from the LLM response."""

    type: Literal["code_interpreter_call"] = "code_interpreter_call"
    """Discriminator."""

    id: str = ""
    """Unique identifier for this tool call."""

    code: str = ""
    """The code the LLM wants to execute."""

    language: str | None = None
    """Programming language of the code (e.g. ``"python"``)."""

    status: str | None = None
    """Optional status string (e.g. ``"completed"``, ``"failed"``)."""


@dataclass(frozen=True)
class CodeInterpreterToolCallResult:
    """Result from a code interpreter tool call."""

    type: Literal["code_interpreter_call_output"] = "code_interpreter_call_output"
    """Discriminator."""

    call_id: str = ""
    """ID of the tool call this result corresponds to."""

    outputs: list[CodeInterpreterOutput] = field(default_factory=list)
    """List of outputs from code execution (logs, images, files)."""


# ---------------------------------------------------------------------------
# Image generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageGenerationToolCall:
    """An image generation call from the LLM response."""

    type: Literal["image_generation_call"] = "image_generation_call"
    """Discriminator."""

    id: str = ""
    """Unique identifier for this tool call."""

    prompt: str = ""
    """The text prompt describing the image to generate."""

    quality: str | None = None
    """Requested quality level (e.g. ``"low"``, ``"medium"``, ``"high"``)."""

    size: str | None = None
    """Requested image dimensions (e.g. ``"1024x1024"``)."""

    status: str | None = None
    """Optional status string (e.g. ``"completed"``, ``"failed"``)."""


@dataclass(frozen=True)
class ImageGenerationToolCallResult:
    """Result from an image generation tool call."""

    type: Literal["image_generation_call_output"] = "image_generation_call_output"
    """Discriminator."""

    call_id: str = ""
    """ID of the tool call this result corresponds to."""

    image: str | None = None
    """Base64-encoded generated image."""

    format: str | None = None
    """Image format (e.g. ``"png"``, ``"jpeg"``)."""

    revised_prompt: str | None = None
    """Optional revised prompt used by the model for generation."""


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShellToolCall:
    """A shell command call from the LLM response."""

    type: Literal["shell_call"] = "shell_call"
    """Discriminator."""

    id: str = ""
    """Unique identifier for this tool call."""

    call_id: str = ""
    """ID used for matching results to calls."""

    command: str = ""
    """The shell command to execute."""

    status: str | None = None
    """Optional status string (e.g. ``"completed"``, ``"failed"``)."""


@dataclass(frozen=True)
class ShellToolCallResult:
    """Result from a shell tool call."""

    type: Literal["shell_call_output"] = "shell_call_output"
    """Discriminator."""

    call_id: str = ""
    """ID of the tool call this result corresponds to."""

    output: str | None = None
    """Standard output from the command."""

    exit_code: int | None = None
    """Process exit code (0 = success)."""

    error: str | None = None
    """Optional standard error output."""


# ---------------------------------------------------------------------------
# Apply patch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ApplyPatchToolCall:
    """An apply-patch call from the LLM response."""

    type: Literal["apply_patch_call"] = "apply_patch_call"
    """Discriminator."""

    id: str = ""
    """Unique identifier for this tool call."""

    call_id: str = ""
    """ID used for matching results to calls."""

    patch: str = ""
    """The unified diff patch to apply."""

    status: str | None = None
    """Optional status string (e.g. ``"completed"``, ``"failed"``)."""


@dataclass(frozen=True)
class ApplyPatchToolCallResult:
    """Result from an apply-patch tool call."""

    type: Literal["apply_patch_call_output"] = "apply_patch_call_output"
    """Discriminator."""

    call_id: str = ""
    """ID of the tool call this result corresponds to."""

    output: str | None = None
    """Summary of applied changes (files modified, etc.)."""

    success: bool = True
    """Whether the patch was applied successfully."""


# ---------------------------------------------------------------------------
# Tool search
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSearchResultEntry:
    """A single tool discovered by a tool search."""

    name: str = ""
    """The name of the discovered tool."""

    description: str | None = None
    """Description of the tool."""


@dataclass(frozen=True)
class ToolSearchToolCall:
    """A tool search call from the LLM response."""

    type: Literal["tool_search_call"] = "tool_search_call"
    """Discriminator."""

    id: str = ""
    """Unique identifier for this tool call."""

    query: str = ""
    """The search query to find relevant tools."""

    status: str | None = None
    """Optional status string (e.g. ``"completed"``, ``"failed"``)."""


@dataclass(frozen=True)
class ToolSearchToolCallResult:
    """Result from a tool search tool call."""

    type: Literal["tool_search_call_output"] = "tool_search_call_output"
    """Discriminator."""

    call_id: str = ""
    """ID of the tool call this result corresponds to."""

    tools: list[ToolSearchResultEntry] = field(default_factory=list)
    """List of discovered tools."""


# ---------------------------------------------------------------------------
# MCP (Model Context Protocol)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCPListToolsTool:
    """A single tool available on an MCP server.

    Represents a tool entry in an ``MCPListTools`` listing.
    """

    name: str = ""
    """The name of the tool."""

    input_schema: dict[str, Any] | None = None
    """The JSON schema describing the tool's input.

    ``dict[str, Any]`` because JSON schemas are inherently dynamic.
    """

    description: str | None = None
    """Description of the tool."""

    annotations: dict[str, Any] | None = None
    """Additional annotations about the tool.

    ``dict[str, Any]`` because annotation shapes are provider-defined.
    """


@dataclass(frozen=True)
class MCPListTools:
    """Snapshot of tools discovered from an MCP server.

    Raw type for ``MCPListToolsItem``.
    """

    type: Literal["mcp_list_tools"] = "mcp_list_tools"
    """Discriminator."""

    id: str = ""
    """Unique identifier for this listing."""

    server: str = ""
    """MCP server name that was queried."""

    tools: list[MCPListToolsTool] = field(default_factory=list)
    """Tools discovered from the server."""

    error: str | None = None
    """Error message if the server could not list tools."""


@dataclass(frozen=True)
class MCPCall:
    """An MCP tool invocation from the LLM response.

    Raw type for the MCP tool call.  Represents the actual
    invocation of a tool on an MCP server.
    """

    type: Literal["mcp_call"] = "mcp_call"
    """Discriminator."""

    id: str = ""
    """Unique identifier for this tool call."""

    server: str = ""
    """MCP server name that handles this call."""

    name: str = ""
    """Tool name on the MCP server."""

    arguments: str = ""
    """JSON-encoded arguments for the tool."""

    approval_request_id: str | None = None
    """Unique ID for the approval request, if approval was required."""

    output: str | None = None
    """The tool's output (populated after execution)."""

    error: str | None = None
    """Error message if the call failed."""

    status: str | None = None
    """Optional status string (e.g. ``"completed"``, ``"failed"``,
    ``"calling"``, ``"in_progress"``)."""


@dataclass(frozen=True)
class MCPCallResult:
    """Result from an MCP tool call."""

    type: Literal["mcp_call_output"] = "mcp_call_output"
    """Discriminator."""

    call_id: str = ""
    """ID of the tool call this result corresponds to."""

    output: str | None = None
    """The tool's output."""

    error: str | None = None
    """Optional error message if the call failed."""


@dataclass(frozen=True)
class MCPApprovalRequest:
    """A request for human approval of an MCP tool invocation.

    Raw type for ``MCPApprovalRequestItem``.  Distinct from ``MCPCall``
    which represents the actual invocation — this represents the
    *request for permission* to invoke.
    """

    type: Literal["mcp_approval_request"] = "mcp_approval_request"
    """Discriminator."""

    id: str = ""
    """Unique identifier for this approval request."""

    server: str = ""
    """MCP server name making the request."""

    name: str = ""
    """The name of the tool to run."""

    arguments: str = ""
    """JSON-encoded arguments for the tool."""


@dataclass(frozen=True)
class MCPApprovalResponse:
    """A response to an MCP approval request.

    Raw type for ``MCPApprovalResponseItem``.
    """

    type: Literal["mcp_approval_response"] = "mcp_approval_response"
    """Discriminator."""

    approval_request_id: str = ""
    """ID of the approval request being answered."""

    approved: bool = False
    """Whether the request was approved."""

    id: str | None = None
    """Optional unique ID for this response."""

    reason: str | None = None
    """Optional reason for the decision."""
