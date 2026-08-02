"""MemoryTool — agent-facing memory tools.

Provides built-in tools that give agents explicit access to
long-term memory: ``RememberMemoryTool``, ``RecallMemoryTool``,
and ``ForgetMemoryTool``.

Each is an :class:`ExecutableBuiltinTool` with its own schema and
``on_invoke`` callback.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from troopai.adk.memory.memory import Memory
from troopai.adk.memory.memory_types import MemoryMetadata, MemorySource
from troopai.adk.tools.builtin.builtin_tool import ExecutableBuiltinTool

logger = logging.getLogger(__name__)


# =====================================================================
# Input schemas
# =====================================================================


class RememberInput(BaseModel):
    """Input schema for the ``remember`` tool."""

    content: str = Field(description="The knowledge to remember.")
    importance: int | None = Field(default=None, ge=1, le=5, description="Priority 1-5. Default: 3.")
    categories: list[str] | None = Field(default=None, description="Semantic tags.")


class RecallInput(BaseModel):
    """Input schema for the ``recall`` tool."""

    query: str = Field(description="Search query to find relevant memories.")
    limit: int = Field(default=5, ge=1, le=20, description="Max results. Default: 5.")


class ForgetInput(BaseModel):
    """Input schema for the ``forget`` tool."""

    memory_id: str = Field(description="The ID of the memory to delete.")


# =====================================================================
# on_invoke callables
# =====================================================================


def _make_remember_invoke(memory: Memory, namespace: str):
    """Create the on_invoke callable for RememberMemoryTool.

    Args:
        memory: The memory backend to write to.
        namespace: The namespace under which memories are stored.

    Returns:
        An async callable matching the ``ToolInvokeFunction`` signature.
    """

    async def _invoke(ctx: Any, raw_args: str) -> str:  # noqa: ARG001
        try:
            args = json.loads(raw_args) if len(raw_args) > 0 else {}
        except json.JSONDecodeError as e:
            return f"Invalid tool arguments (JSON parse error): {e}"
        content = args.get("content", "")
        # ``importance`` is ``int | None`` on the schema, but the executor calls
        # ``on_invoke`` directly without Pydantic validation, so an LLM sending
        # ``{"importance": null}`` reaches here as ``None`` — ``get`` returns the
        # stored ``None``, not the ``3`` default. Coerce it to the documented
        # default so ``max(1, min(5, ...))`` cannot raise ``TypeError``.
        raw_importance = args.get("importance")
        importance = 3 if raw_importance is None else raw_importance
        categories = tuple(args.get("categories") or [])
        metadata = MemoryMetadata(
            source=MemorySource.TOOL,
            importance=max(1, min(5, importance)),
            categories=categories,
        )
        entry = await memory.add(content=content, namespace=namespace, metadata=metadata)
        return f"Remembered (id: {entry.id}). Importance: {entry.metadata.importance}."

    return _invoke


def _make_recall_invoke(memory: Memory, namespace: str):
    """Create the on_invoke callable for RecallMemoryTool.

    Args:
        memory: The memory backend to search.
        namespace: The namespace to scope the search to.

    Returns:
        An async callable matching the ``ToolInvokeFunction`` signature.
    """

    async def _invoke(ctx: Any, raw_args: str) -> str:  # noqa: ARG001
        try:
            args = json.loads(raw_args) if len(raw_args) > 0 else {}
        except json.JSONDecodeError as exc:
            return f"Invalid tool arguments (JSON parse error): {exc}"
        query = args.get("query", "")
        limit = args.get("limit", 5)
        results = await memory.search(query, namespace=namespace, limit=limit)
        if len(results) == 0:
            return f"No memories found matching '{query}'."
        lines = [f"## Memories ({len(results)} result(s))\n"]
        for r in results:
            e = r.entry
            cats = ", ".join(e.metadata.categories) if e.metadata.categories else "none"
            lines.append(f"### [{e.metadata.importance}/5] {e.id}\n{e.content}\nCategories: {cats}\n")
        return "\n".join(lines).strip()

    return _invoke


def _make_forget_invoke(memory: Memory):
    """Create the on_invoke callable for ForgetMemoryTool.

    Args:
        memory: The memory backend to delete from.

    Returns:
        An async callable matching the ``ToolInvokeFunction`` signature.
    """

    async def _invoke(ctx: Any, raw_args: str) -> str:  # noqa: ARG001
        try:
            args = json.loads(raw_args) if len(raw_args) > 0 else {}
        except json.JSONDecodeError as e:
            return f"Invalid tool arguments (JSON parse error): {e}"
        memory_id = args.get("memory_id", "")
        deleted = await memory.delete(memory_id)
        return f"Memory {memory_id} deleted." if deleted else f"Memory {memory_id} not found."

    return _invoke


# =====================================================================
# MemoryTool base — shared config
# =====================================================================


@dataclass(kw_only=True)
class MemoryTool(ExecutableBuiltinTool):
    """Base class for memory tools.

    Holds the shared ``memory`` backend and ``namespace``.
    Not used directly — use the subclasses
    (``RememberMemoryTool``, ``RecallMemoryTool``, ``ForgetMemoryTool``).

    Attributes:
        name: The name of this tool.
        memory: The memory backend to use.
        namespace: Default namespace for memory operations.
    """

    name: str = "memory"
    """The name of this tool."""

    memory: Memory
    """The memory backend to use."""

    namespace: str
    """Default namespace for memory operations."""


# =====================================================================
# Individual memory tools
# =====================================================================


@dataclass(kw_only=True)
class RememberMemoryTool(MemoryTool):
    """Store knowledge in long-term memory.

    Attributes:
        name: Tool name shown to the LLM (``"remember"``).
        description: Tool description shown to the LLM.
        schema: Input schema (:class:`RememberInput`).
        memory: The memory backend to use (inherited).
        namespace: Default namespace for memory operations (inherited).
    """

    name: str = "remember"
    description: str = (
        "Store a piece of knowledge in long-term memory. Use this to "
        "remember important facts, user preferences, decisions, or "
        "anything that should persist across conversations."
    )
    schema: type[BaseModel] | dict[str, Any] = RememberInput

    def __post_init__(self) -> None:
        if self.on_invoke is None:
            self.on_invoke = _make_remember_invoke(self.memory, self.namespace)


@dataclass(kw_only=True)
class RecallMemoryTool(MemoryTool):
    """Search long-term memory for relevant knowledge.

    Attributes:
        name: Tool name shown to the LLM (``"recall"``).
        description: Tool description shown to the LLM.
        schema: Input schema (:class:`RecallInput`).
        memory: The memory backend to use (inherited).
        namespace: Default namespace for memory operations (inherited).
    """

    name: str = "recall"
    description: str = (
        "Search long-term memory for relevant knowledge. Use this to "
        "look up previously stored facts, preferences, or decisions."
    )
    schema: type[BaseModel] | dict[str, Any] = RecallInput

    def __post_init__(self) -> None:
        if self.on_invoke is None:
            self.on_invoke = _make_recall_invoke(self.memory, self.namespace)


@dataclass(kw_only=True)
class ForgetMemoryTool(MemoryTool):
    """Delete a specific memory by ID.

    Attributes:
        name: Tool name shown to the LLM (``"forget"``).
        description: Tool description shown to the LLM.
        schema: Input schema (:class:`ForgetInput`).
        memory: The memory backend to use (inherited).
        namespace: Default namespace for memory operations (inherited).
    """

    name: str = "forget"
    """The name of this tool."""

    description: str = (
        "Delete a specific memory by its ID. Use this to remove "
        "outdated or incorrect information from long-term memory."
    )

    schema: type[BaseModel] | dict[str, Any] = ForgetInput

    def __post_init__(self) -> None:
        if self.on_invoke is None:
            self.on_invoke = _make_forget_invoke(self.memory)
