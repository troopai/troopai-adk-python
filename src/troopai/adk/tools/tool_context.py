from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar, override

if TYPE_CHECKING:
    from troopai.adk.llms.llm_usage import LLMUsage
    from troopai.adk.run.config import RunConfig
    from troopai.adk.types.items.items import RunItem

TContext = TypeVar("TContext")


@dataclass
class ToolContext[TContext]:
    """
    Context passed to tool functions and guardrails during execution.

    This provides tools with information about the current invocation
    and access to user-defined context data.

    Attributes:
        tool_name: The name of the tool being invoked.
        tool_call_id: Unique identifier for this specific tool call (from the LLM).
        tool_arguments: The parsed arguments being passed to the tool.
        raw_arguments: The raw JSON string of arguments from the LLM.
        context: User-defined context data passed through the execution pipeline.
        metadata: Additional metadata about the tool call (provider-specific info, etc.).

    Type Parameters:
        TContext: The type of user-defined context data.
    """

    tool_name: str
    """The name of the tool being invoked."""

    tool_call_id: str
    """Unique identifier for this specific tool call (from the LLM)."""

    tool_arguments: dict[str, Any]
    """The parsed arguments being passed to the tool."""

    raw_arguments: str
    """The raw JSON string of arguments from the LLM."""

    context: TContext | None = None
    """User-defined context data passed through the execution pipeline."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata about the tool call (provider-specific info, etc.)."""

    _run_config: RunConfig | None = None
    """Framework-internal parent ``RunConfig``, reachable only via
    :meth:`get_run_config`.

    Threaded by the runner so the agent-as-tool machinery can let a
    sub-agent inherit the parent's execution config. It is intentionally
    NOT a public field: user tool functions receive a ``ToolContext`` and
    must not reach run-wide execution state (the ``ToolContext`` vs.
    ``RunContext`` isolation boundary).
    """

    def get_run_config(self) -> RunConfig | None:
        """Return the parent ``RunConfig`` threaded by the runner, if any.

        Framework-internal accessor for the agent-as-tool path. User tool
        functions should not depend on run-wide execution state.

        Returns:
            The parent ``RunConfig``, or ``None`` when not running under one.
        """
        return self._run_config

    def with_context(self, context: TContext) -> ToolContext[TContext]:
        """Create a new ToolContext with the specified context data.

        Args:
            context: The user-defined context data to attach.

        Returns:
            A new ``ToolContext`` with all fields copied from ``self``
            and ``context`` replaced by the supplied value.
        """
        return ToolContext(
            tool_name=self.tool_name,
            tool_call_id=self.tool_call_id,
            tool_arguments=self.tool_arguments,
            raw_arguments=self.raw_arguments,
            context=context,
            metadata=self.metadata,
            _run_config=self._run_config,
        )


@dataclass
class ExecutionAwareToolContext(ToolContext[TContext]):
    """Extended tool context with read-only execution state snapshots.

    Tools that need visibility into execution state (e.g., monitoring,
    cost dashboards, budget-aware result generation) can opt in by
    annotating their first parameter as ``ExecutionAwareToolContext``
    instead of ``ToolContext``.

    All execution fields are **copies** (not references) — tools can
    observe but never mutate the runner's state.  This preserves the
    architectural guarantee that the runner is the sole gatekeeper
    between tool execution and the LLM's message history.

    Attributes:
        usage: Snapshot of cumulative token usage at time of tool call.
        turns: Number of agent loop turns completed so far.
        messages: Number of messages in the conversation history.
        tokens: Estimated token count of the current message history.
        tool_name: The name of the tool being invoked (inherited).
        tool_call_id: Unique identifier for this specific tool call (inherited).
        tool_arguments: The parsed arguments being passed to the tool (inherited).
        raw_arguments: The raw JSON string of arguments from the LLM (inherited).
        context: User-defined context data (inherited).
        metadata: Additional metadata about the tool call (inherited).

    Type Parameters:
        TContext: The type of user-defined context data.
    """

    usage: LLMUsage | None = None
    """Snapshot of cumulative token usage (copy, not reference).

    Reflects total input/output tokens across all LLM calls in the
    current run up to this tool invocation.  Tools can use this for
    budget-aware behavior (e.g., shorter results when tokens run low).
    """

    turns: int = 0
    """Number of agent loop turns completed so far.

    A turn is one LLM call → tool execution cycle.  Starts at 0
    for the first turn.
    """

    messages: int = 0
    """Number of messages in the current conversation history.

    Includes system, user, assistant, and tool messages.
    """

    tokens: int = 0
    """Estimated token count of the current message history.

    Computed by :class:`TokenCounter` before tool execution.
    Useful for tools that want to adjust output verbosity based
    on how much context window is already consumed.
    """

    @override
    def with_context(self, context: TContext) -> ExecutionAwareToolContext[TContext]:
        """Create a new ExecutionAwareToolContext preserving execution state.

        Args:
            context: The user-defined context data to attach.

        Returns:
            A new ``ExecutionAwareToolContext`` with all fields copied
            from ``self`` and ``context`` replaced by the supplied value.
        """
        return ExecutionAwareToolContext(
            tool_name=self.tool_name,
            tool_call_id=self.tool_call_id,
            tool_arguments=self.tool_arguments,
            raw_arguments=self.raw_arguments,
            context=context,
            metadata=self.metadata,
            _run_config=self._run_config,
            usage=self.usage,
            turns=self.turns,
            messages=self.messages,
            tokens=self.tokens,
        )


@dataclass
class HistoryAwareToolContext(ExecutionAwareToolContext[TContext]):
    """Extended tool context with read-only conversation history snapshot.

    Tools that need to search or inspect conversation history (e.g.,
    context management, note retrieval, history search) can opt in by
    annotating their first parameter as ``HistoryAwareToolContext``
    instead of ``ExecutionAwareToolContext`` or ``ToolContext``.

    The history is a **tuple of Layer 3 RunItems** — not raw wire types.
    The tools_executor converts messages (Layer 1) to ``RunItem``
    (Layer 3) via ``ItemHelpers`` before constructing this context,
    preserving the three-layer type boundary.

    All fields are **copies** (not references) — tools can observe but
    never mutate the runner's state.

    Attributes:
        history: Frozen snapshot of conversation history as Layer 3 RunItems.
        usage: Snapshot of cumulative token usage at time of tool call (inherited).
        turns: Number of agent loop turns completed so far (inherited).
        messages: Number of messages in the conversation history (inherited).
        tokens: Estimated token count of the current message history (inherited).
        tool_name: The name of the tool being invoked (inherited).
        tool_call_id: Unique identifier for this specific tool call (inherited).
        tool_arguments: The parsed arguments being passed to the tool (inherited).
        raw_arguments: The raw JSON string of arguments from the LLM (inherited).
        context: User-defined context data (inherited).
        metadata: Additional metadata about the tool call (inherited).

    Type Parameters:
        TContext: The type of user-defined context data.
    """

    history: tuple[RunItem, ...] = ()
    """Read-only snapshot of the conversation history.

    Contains Layer 3 RunItems (SystemItem, UserItem, MessageOutputItem,
    ToolCallItem, ToolCallOutputItem, etc.) — the same types used in
    ``HandoffInputData.context`` and ``HandoffInputData.output``.

    The tuple is immutable by design.  Tools can iterate, filter, and
    search but cannot modify the runner's message list.
    """

    @override
    def with_context(self, context: TContext) -> HistoryAwareToolContext[TContext]:
        """Create a new HistoryAwareToolContext preserving all state.

        Args:
            context: The user-defined context data to attach.

        Returns:
            A new ``HistoryAwareToolContext`` with all fields copied
            from ``self`` and ``context`` replaced by the supplied value.
        """
        return HistoryAwareToolContext(
            tool_name=self.tool_name,
            tool_call_id=self.tool_call_id,
            tool_arguments=self.tool_arguments,
            raw_arguments=self.raw_arguments,
            context=context,
            metadata=self.metadata,
            _run_config=self._run_config,
            usage=self.usage,
            turns=self.turns,
            messages=self.messages,
            tokens=self.tokens,
            history=self.history,
        )
