"""JIT Context Aware Tool — active context management for AI agents.

Provides tools that give agents active control over their context window.
Instead of passive threshold-based management (compaction, editing), the
LLM itself decides when to save notes, search history, and monitor its
context budget.

Inspired by Anthropic's "Just-in-Time context" pattern:
https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

Individual tools:

- ``SaveNoteTool``: Store findings and intermediate results
- ``RecallNotesTool``: Retrieve saved notes (with optional filter)
- ``ManageContextTool``: Emit compact/drop directives
- ``SearchHistoryTool``: Search conversation history by keyword
- ``ContextStatsTool``: Check token usage and context budget

Each is an :class:`ExecutableBuiltinTool` with its own schema and callback.

Example::

    from troopai.adk.tools import JITContextAwareTool

    agent = Agent(
        name="Research Assistant",
        system_prompt="Use save_note to preserve key findings.",
        tools=[SaveNoteContextAwareTool(), RecallNotesContextAwareTool()],
    )
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from troopai.adk.context.directives import (
    CompactDirective,
    DirectiveStore,
    DropDirective,
)
from troopai.adk.tools.builtin.builtin_tool import ExecutableBuiltinTool

logger = logging.getLogger(__name__)


# =====================================================================
# NoteEntry — immutable note record
# =====================================================================


@dataclass(frozen=True)
class NoteEntry:
    """A single note in the working scratchpad.

    Attributes:
        key: User-assigned identifier (unique within the store).
        content: The note text.
        importance: Priority from 1 (low) to 5 (critical).
        created_at: Monotonic timestamp for relative ordering.
        turn: Agent loop turn when the note was saved.
    """

    key: str
    content: str
    importance: int
    created_at: float
    turn: int


# =====================================================================
# NoteStore — pluggable storage protocol
# =====================================================================


@runtime_checkable
class NoteStore(Protocol):
    """Protocol for note storage backends.

    Default implementation: :class:`InMemoryNoteStore`.
    """

    def save(self, key: str, content: str, importance: int, turn: int) -> NoteEntry:
        """Store or overwrite a note under ``key``."""
        ...

    def recall(self, query: str | None = None) -> list[NoteEntry]:
        """Return notes matching ``query``, or all notes when ``None``."""
        ...

    def delete(self, key: str) -> bool:
        """Delete a note by key. Returns ``True`` if it existed."""
        ...

    def count(self) -> int:
        """Return the current number of stored notes."""
        ...

    def keys(self) -> list[str]:
        """Return all note keys."""
        ...


# =====================================================================
# InMemoryNoteStore — default implementation
# =====================================================================


class InMemoryNoteStore:
    """In-memory note store for the duration of a single run.

    Notes are evicted when the store exceeds ``capacity``, with the
    lowest-importance / oldest note removed first.
    """

    def __init__(self, capacity: int = 50) -> None:
        """Initialize the store.

        Args:
            capacity: Maximum number of notes to hold before evicting.
                Defaults to 50.
        """
        self._notes: dict[str, NoteEntry] = {}
        self._capacity = capacity

    def save(self, key: str, content: str, importance: int, turn: int) -> NoteEntry:
        """Store or overwrite a note.

        Args:
            key: Unique identifier for the note.
            content: The note text.
            importance: Priority from 1 (low) to 5 (critical). Values
                outside [1, 5] are clamped.
            turn: Agent loop turn number at save time.

        Returns:
            The newly created :class:`NoteEntry`.
        """
        entry = NoteEntry(
            key=key,
            content=content,
            importance=max(1, min(5, importance)),
            created_at=time.monotonic(),
            turn=turn,
        )
        self._notes[key] = entry
        if len(self._notes) > self._capacity:
            self._evict()
        logger.debug("Saved note '%s' (importance=%d, turn=%d)", key, importance, turn)
        return entry

    def recall(self, query: str | None = None) -> list[NoteEntry]:
        """Return notes sorted by descending importance, then creation time.

        Args:
            query: Optional substring filter applied to both key and
                content (case-insensitive). ``None`` returns all notes.

        Returns:
            Matching notes ordered by descending importance, then
            ascending creation time.
        """
        notes = list(self._notes.values())
        if query is not None:
            lower_query = query.lower()
            notes = [n for n in notes if lower_query in n.key.lower() or lower_query in n.content.lower()]
        notes.sort(key=lambda n: (-n.importance, n.created_at))
        return notes

    def delete(self, key: str) -> bool:
        """Delete the note with the given key.

        Args:
            key: Identifier of the note to remove.

        Returns:
            ``True`` if the note existed and was deleted, ``False``
            otherwise.
        """
        if key in self._notes:
            del self._notes[key]
            return True
        return False

    def count(self) -> int:
        """Return the current number of stored notes.

        Returns:
            The number of notes currently in the store.
        """
        return len(self._notes)

    def keys(self) -> list[str]:
        """Return a list of all note keys in the store.

        Returns:
            A snapshot list of key strings in insertion order.
        """
        return list(self._notes.keys())

    def _evict(self) -> None:
        if len(self._notes) == 0:
            return
        victim = min(self._notes.values(), key=lambda n: (n.importance, n.created_at))
        del self._notes[victim.key]


# =====================================================================
# Tool input schemas
# =====================================================================


class SaveNoteInput(BaseModel):
    """Input for the ``save_note`` tool."""

    key: str = Field(description="Short identifier for this note (e.g. 'user_preferences').")
    content: str = Field(description="The content to save.")
    importance: int | None = Field(default=None, ge=1, le=5, description="Priority 1-5.")


class RecallNotesInput(BaseModel):
    """Input for the ``recall_notes`` tool."""

    query: str | None = Field(default=None, description="Optional filter by key/content.")


class ManageContextInput(BaseModel):
    """Input for the ``manage_context`` tool."""

    action: Literal["compact", "drop"] = Field(description="'compact' summarizes, 'drop' removes.")
    preserve: int = Field(ge=1, le=100, description="Number of recent messages to keep.")


class SearchHistoryInput(BaseModel):
    """Input for the ``search_history`` tool."""

    pattern: str = Field(description="Text to search for in conversation history.")
    role: Literal["user", "assistant", "tool", "system"] | None = Field(default=None)
    max_results: int = Field(default=5, ge=1, le=20)


class ContextStatsInput(BaseModel):
    """Input for the ``context_stats`` tool (empty)."""

    pass


# =====================================================================
# on_invoke factories (module-level, same pattern as memory_tool.py)
# =====================================================================


def _make_save_note_invoke(store: NoteStore, importance: int):
    """Create the on_invoke callable for SaveNoteContextAwareTool.

    Args:
        store: The note store to write to.
        importance: Default importance level used when the LLM does not
            supply one (1–5).

    Returns:
        An async callable matching the ``ToolInvokeFunction`` signature.
    """

    async def _invoke(ctx: Any, raw_args: str) -> str:
        try:
            args = json.loads(raw_args) if len(raw_args) > 0 else {}
        except json.JSONDecodeError as e:
            return f"Invalid tool arguments (JSON parse error): {e}"
        key = args.get("key", "")
        content = args.get("content", "")
        imp = importance if args.get("importance") is None else args["importance"]
        turn = getattr(ctx, "turns", 0)
        entry = store.save(key, content, imp, turn)
        return f"Saved note '{entry.key}' (importance: {entry.importance}). You now have {store.count()} note(s)."

    return _invoke


def _make_recall_notes_invoke(store: NoteStore):
    """Create the on_invoke callable for RecallNotesContextAwareTool.

    Args:
        store: The note store to read from.

    Returns:
        An async callable matching the ``ToolInvokeFunction`` signature.
    """

    async def _invoke(ctx: Any, raw_args: str) -> str:  # noqa: ARG001
        try:
            args = json.loads(raw_args) if len(raw_args) > 0 else {}
        except json.JSONDecodeError as e:
            return f"Invalid tool arguments (JSON parse error): {e}"
        query = args.get("query")
        notes = store.recall(query)

        if len(notes) == 0:
            if query is not None:
                return f"No notes matching '{query}'. You have {store.count()} total note(s)."
            return "No notes saved yet. Use save_note to store information."

        header = f"## Notes ({store.count()} total"
        if query is not None:
            header += f", showing {len(notes)} matching '{query}'"
        header += ")\n"

        lines = [header]
        for note in notes:
            lines.append(f"### {note.key} [importance: {note.importance}, turn {note.turn}]")
            lines.append(note.content)
            lines.append("")
        return "\n".join(lines).strip()

    return _invoke


def _make_manage_context_invoke(tool: ManageContextAwareTool):
    """Create the on_invoke callable for ManageContextAwareTool.

    Uses lazy ``tool.directives`` access so shared directives work.

    Args:
        tool: The :class:`ManageContextAwareTool` instance whose
            ``directives`` store receives the compact/drop directive.

    Returns:
        An async callable matching the ``ToolInvokeFunction`` signature.
    """

    async def _invoke(ctx: Any, raw_args: str) -> str:
        try:
            args = json.loads(raw_args) if len(raw_args) > 0 else {}
        except json.JSONDecodeError as e:
            return f"Invalid tool arguments (JSON parse error): {e}"
        action = args.get("action", "")
        preserve = args.get("preserve", 5)
        msg_count = getattr(ctx, "messages", 0)

        if msg_count <= 1:
            return "Not enough messages to compact. At least 2 messages are required."

        if preserve > msg_count - 1:
            preserve = msg_count - 1

        if action == "compact":
            tool.directives.add(CompactDirective(preserve=preserve))
            return (
                f"Scheduled: compact conversation, keeping last {preserve} of "
                f"{msg_count} messages. Takes effect next turn."
            )
        elif action == "drop":
            tool.directives.add(DropDirective(preserve=preserve))
            return (
                f"Scheduled: drop old messages, keeping last {preserve} of "
                f"{msg_count} messages. Takes effect next turn."
            )
        else:
            return f"Error: unknown action '{action}'. Use 'compact' or 'drop'."

    return _invoke


def _make_search_history_invoke():
    """Create the on_invoke callable for SearchHistoryContextAwareTool.

    Returns:
        An async callable matching the ``ToolInvokeFunction`` signature.
        The callable expects a :class:`HistoryAwareToolContext` so it
        can access ``ctx.history``.
    """

    async def _invoke(ctx: Any, raw_args: str) -> str:
        try:
            args = json.loads(raw_args) if len(raw_args) > 0 else {}
        except json.JSONDecodeError as e:
            return f"Invalid tool arguments (JSON parse error): {e}"
        pattern = args.get("pattern", "")
        role_filter = args.get("role")
        max_results = args.get("max_results", 5)

        if len(pattern) == 0:
            return "Error: 'pattern' is required."

        history = getattr(ctx, "history", ())
        if len(history) == 0:
            return "No conversation history available."

        lower_pattern = pattern.lower()
        matches: list[tuple[int, str, str]] = []

        for idx, item in enumerate(history):
            item_type = getattr(item, "type", "")
            content = _extract_text_from_item(item)
            if len(content) == 0:
                continue

            role = _item_type_to_role(item_type)
            if role_filter and role != role_filter:
                continue

            if lower_pattern in content.lower():
                matches.append((idx, role, content))
                if len(matches) >= max_results:
                    break

        if len(matches) == 0:
            return f"No messages matching '{pattern}' in conversation history."

        lines = [f"## History search: '{pattern}' ({len(matches)} match(es))\n"]
        for msg_idx, role, content in matches:
            display = content[:500] + "..." if len(content) > 500 else content
            lines.append(f"### Message {msg_idx} [{role}]")
            lines.append(display)
            lines.append("")
        return "\n".join(lines).strip()

    return _invoke


def _make_context_stats_invoke(store: NoteStore):
    """Create the on_invoke callable for ContextStatsContextAwareTool.

    Args:
        store: The note store whose :meth:`count` is included in the
            stats report.

    Returns:
        An async callable matching the ``ToolInvokeFunction`` signature.
        The callable expects an :class:`ExecutionAwareToolContext` so it
        can read ``ctx.tokens``, ``ctx.turns``, ``ctx.messages``, and
        ``ctx.usage``.
    """

    async def _invoke(ctx: Any, raw_args: str) -> str:  # noqa: ARG001  # raw_args unused (empty schema)
        tokens = getattr(ctx, "tokens", 0)
        turns = getattr(ctx, "turns", 0)
        messages = getattr(ctx, "messages", 0)
        usage = getattr(ctx, "usage", None)

        lines = ["## Context Stats"]
        lines.append(f"- Tokens in context: {tokens:,}")
        lines.append(f"- Turns completed: {turns}")
        lines.append(f"- Messages in history: {messages}")
        lines.append(f"- Notes saved: {store.count()}")

        if usage is not None:
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            total = input_tokens + output_tokens
            lines.append(f"- Cumulative LLM tokens: {total:,} (input: {input_tokens:,}, output: {output_tokens:,})")

        return "\n".join(lines)

    return _invoke


# =====================================================================
# JITContextAwareTool — base class
# =====================================================================


@dataclass(kw_only=True)
class JITContextAwareTool(ExecutableBuiltinTool):
    """Base class for JIT context management tools.

    Holds the shared ``store`` (NoteStore) and ``directives`` (DirectiveStore).
    Not used directly — use the subclasses (``SaveNoteContextAwareTool``,
    ``RecallNotesContextAwareTool``, ``ManageContextAwareTool``,
    ``SearchHistoryContextAwareTool``, ``ContextStatsContextAwareTool``).

    To share state across tools, pass the same ``store`` and set
    ``directives`` after construction via ``object.__setattr__``.

    Attributes:
        name: The name of this tool.
        capacity: Maximum number of notes the store can hold.
        importance: Default importance level for ``save_note`` (1-5).
        store: Pluggable storage backend. Defaults to
            :class:`InMemoryNoteStore`.
        execution_aware: Whether this tool needs execution state
            (usage, turns, messages, tokens).
        history_aware: Whether this tool needs conversation history
            access.
        directives: Pending context directives consumed by the Runner
            before the next LLM call.
    """

    name: str = "jit_context"
    """The name of this tool."""

    capacity: int = 50
    """Maximum number of notes the store can hold."""

    importance: int = 3
    """Default importance level for ``save_note`` (1-5)."""

    store: NoteStore | None = None
    """Pluggable storage backend. Defaults to InMemoryNoteStore."""

    execution_aware: bool = False
    """Whether this tool needs execution state (usage, turns, messages, tokens)."""

    history_aware: bool = False
    """Whether this tool needs conversation history access."""

    directives: DirectiveStore = field(init=False, repr=False)
    """Pending context directives. The Runner reads this."""

    def __post_init__(self) -> None:
        if self.store is None:
            object.__setattr__(self, "store", InMemoryNoteStore(capacity=self.capacity))
        object.__setattr__(self, "directives", DirectiveStore())


# =====================================================================
# Individual JIT tools
# =====================================================================


@dataclass(kw_only=True)
class SaveNoteContextAwareTool(JITContextAwareTool):
    """Save a note to the working scratchpad.

    Attributes:
        name: Tool name shown to the LLM (``"save_note"``).
        description: Tool description shown to the LLM.
        schema: Input schema (:class:`SaveNoteInput`).
        capacity: Maximum number of notes the store can hold (inherited).
        importance: Default importance level for saved notes (inherited).
        store: Pluggable storage backend (inherited).
        execution_aware: Whether this tool needs execution state
            (inherited).
        history_aware: Whether this tool needs conversation history
            (inherited).
        directives: Pending context directives (inherited).
    """

    name: str = "save_note"
    description: str = (
        "Save a note to your working scratchpad. Use this to preserve "
        "important information (key findings, intermediate results, task "
        "progress, user preferences) that you may need later."
    )
    schema: type[BaseModel] | dict[str, Any] = SaveNoteInput

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.store is None:
            raise ValueError("SaveNoteContextAwareTool requires a NoteStore (store= must be set)")
        if self.on_invoke is None:
            self.on_invoke = _make_save_note_invoke(self.store, self.importance)


@dataclass(kw_only=True)
class RecallNotesContextAwareTool(JITContextAwareTool):
    """Retrieve saved notes from the scratchpad.

    Attributes:
        name: Tool name shown to the LLM (``"recall_notes"``).
        description: Tool description shown to the LLM.
        schema: Input schema (:class:`RecallNotesInput`).
        capacity: Maximum number of notes the store can hold (inherited).
        importance: Default importance level for saved notes (inherited).
        store: Pluggable storage backend (inherited).
        execution_aware: Whether this tool needs execution state
            (inherited).
        history_aware: Whether this tool needs conversation history
            (inherited).
        directives: Pending context directives (inherited).
    """

    name: str = "recall_notes"
    description: str = (
        "Retrieve your saved notes. Call without arguments to see all, or with a query to filter by key/content."
    )
    schema: type[BaseModel] | dict[str, Any] = RecallNotesInput

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.store is None:
            raise ValueError("RecallNotesContextAwareTool requires a NoteStore (store= must be set)")
        if self.on_invoke is None:
            self.on_invoke = _make_recall_notes_invoke(self.store)


@dataclass(kw_only=True)
class ManageContextAwareTool(JITContextAwareTool):
    """Manage context window by compacting or dropping old messages.

    Attributes:
        name: Tool name shown to the LLM (``"manage_context"``).
        description: Tool description shown to the LLM.
        schema: Input schema (:class:`ManageContextInput`).
        execution_aware: Always ``True`` — this tool reads ``ctx.messages``
            to bound the ``preserve`` parameter.
        capacity: Maximum number of notes the store can hold (inherited).
        importance: Default importance level for saved notes (inherited).
        store: Pluggable storage backend (inherited).
        history_aware: Whether this tool needs conversation history
            (inherited).
        directives: Pending context directives (inherited).
    """

    name: str = "manage_context"
    description: str = (
        "Manage your context window by dropping or compacting old messages. "
        "Use 'drop' to discard (fast). Use 'compact' to summarize (preserves info). "
        "Changes take effect next turn."
    )
    schema: type[BaseModel] | dict[str, Any] = ManageContextInput
    execution_aware: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.on_invoke is None:
            self.on_invoke = _make_manage_context_invoke(self)


@dataclass(kw_only=True)
class SearchHistoryContextAwareTool(JITContextAwareTool):
    """Search conversation history for messages matching a pattern.

    Attributes:
        name: Tool name shown to the LLM (``"search_history"``).
        description: Tool description shown to the LLM.
        schema: Input schema (:class:`SearchHistoryInput`).
        history_aware: Always ``True`` — this tool reads ``ctx.history``
            to search the conversation.
        capacity: Maximum number of notes the store can hold (inherited).
        importance: Default importance level for saved notes (inherited).
        store: Pluggable storage backend (inherited).
        execution_aware: Whether this tool needs execution state
            (inherited).
        directives: Pending context directives (inherited).
    """

    name: str = "search_history"
    description: str = (
        "Search conversation history for messages containing a text pattern. "
        "Useful for recalling earlier information that may have been compacted."
    )
    schema: type[BaseModel] | dict[str, Any] = SearchHistoryInput
    history_aware: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.on_invoke is None:
            self.on_invoke = _make_search_history_invoke()


@dataclass(kw_only=True)
class ContextStatsContextAwareTool(JITContextAwareTool):
    """Check current context window usage.

    Attributes:
        name: Tool name shown to the LLM (``"context_stats"``).
        description: Tool description shown to the LLM.
        schema: Input schema (:class:`ContextStatsInput`).
        execution_aware: Always ``True`` — this tool reads token, turn,
            and message counts from ``ctx``.
        capacity: Maximum number of notes the store can hold (inherited).
        importance: Default importance level for saved notes (inherited).
        store: Pluggable storage backend (inherited).
        history_aware: Whether this tool needs conversation history
            (inherited).
        directives: Pending context directives (inherited).
    """

    name: str = "context_stats"
    description: str = (
        "Check your current context window usage — tokens, turns, notes count. "
        "Use to decide whether to save notes or adjust response length."
    )
    schema: type[BaseModel] | dict[str, Any] = ContextStatsInput
    execution_aware: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.store is None:
            raise ValueError("ContextStatsContextAwareTool requires a NoteStore (store= must be set)")
        if self.on_invoke is None:
            self.on_invoke = _make_context_stats_invoke(self.store)


# =====================================================================
# Helpers for RunItem text extraction
# =====================================================================


def _extract_text_from_item(item: Any) -> str:
    """Extract text content from a RunItem for search purposes.

    Args:
        item: A Layer 3 RunItem (any concrete subtype).

    Returns:
        The plain-text content string, or an empty string when none
        can be found.
    """
    raw = getattr(item, "raw", None)
    if raw is None:
        return ""

    content = getattr(raw, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(raw, dict):
        c = raw.get("content", "")
        if isinstance(c, str):
            return c

    output = getattr(raw, "output", None)
    if isinstance(output, str):
        return output

    return ""


def _item_type_to_role(item_type: str) -> str:
    """Map RunItem type discriminator to a user-friendly role string.

    Args:
        item_type: The ``type`` discriminator from a RunItem (e.g.
            ``"message_output"``, ``"tool_call"``).

    Returns:
        A human-readable role string (``"user"``, ``"assistant"``,
        ``"tool"``, ``"system"``, or ``"unknown"``).
    """
    role_map = {
        "system": "system",
        "user": "user",
        "message_output": "assistant",
        "tool_call": "assistant",
        "tool_call_output": "tool",
        "reasoning": "assistant",
        "handoff_call": "assistant",
        "handoff_output": "tool",
        "compaction": "assistant",
    }
    return role_map.get(item_type, "unknown")
