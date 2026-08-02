"""Tool search — discover and reveal deferred tools at LLM-step time.

Companion to ``FunctionTool.defer_loading``. ``build_tool_search()``
returns a plain ``FunctionTool`` (no provider-specific wrapper) that
the LLM can call to search across the deferred-tool registry. Matched
tool names are added to the search tool's per-run ``revealed`` set;
``build_tools()`` consults that set when filtering deferred tools out
of the LLM's view, so a revealed tool becomes visible on the next
turn.

Why this shape:

- The LLM-side surface is just another function tool — provider-
  agnostic, no special wire shape.
- The deferred-tool registry is captured by closure at construction
  time. ``build_tool_search()`` returns a ``FunctionTool`` whose
  ``_search_state`` slot owns the registry and the ``revealed`` set.
- ``build_tools()`` consults the search tool via
  ``FunctionTool.get_search_state()`` to find the active reveal
  set on each step.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.tools.tool_context import ToolContext

logger = logging.getLogger(__name__)


def find_revealed_deferred_tools(tools: Iterable[Any]) -> frozenset[str]:
    """Return the set of revealed deferred-tool names for an agent.

    Walks ``tools`` looking for the first ``FunctionTool`` with a
    non-None ``_search_state`` (set by ``build_tool_search()``) and
    returns its ``revealed`` frozenset. If no search tool is present,
    returns an empty frozenset.

    Toolset entries are walked one level deep: a ``FunctionToolset``'s
    static ``.tools`` attribute is scanned. Live wrappers (filtered /
    prefixed / etc.) are not materialised here — putting a search tool
    inside a context-dependent wrapper is not supported, since the
    reveal set must exist before tools are filtered. The build-time
    convention is "search tool sits at the agent-tools top level or
    inside a plain ``FunctionToolset``."

    Used by ``build_tools()`` to filter the LLM-facing tool list AND by
    ``_execute_single_tool_call`` to refuse execution of unrevealed
    deferred tools — visibility filtering alone does not gate
    execution, since a misbehaving or prompt-injected LLM can emit a
    function-call to any tool name it has seen in context.
    """
    # Local import to avoid a circular at module-load time
    # (toolsets imports tool types defined in this package).
    from troopai.adk.tools.toolsets import FunctionToolset

    found: list[ToolSearchState] = []
    for tool in tools:
        if isinstance(tool, FunctionTool):
            state = tool.get_search_state()
            if state is not None:
                found.append(state)
        elif isinstance(tool, FunctionToolset):
            for inner_tool in tool.tools:
                state = inner_tool.get_search_state()
                if state is not None:
                    found.append(state)
    if len(found) > 1:
        logger.warning(
            "agent has %d tool_search instances; only the first one's "
            "revealed set is consulted. Use a single build_tool_search() per agent.",
            len(found),
        )
    if len(found) == 0:
        return frozenset()
    return found[0].revealed


def reset_revealed_sets(tools: Iterable[Any]) -> None:
    """Reset the revealed set of every search tool found in *tools*.

    Called at the start of each ``Runner.arun()`` execution to ensure
    that sequential ``await Runner.arun()`` calls made from the same
    coroutine (which share the same asyncio context) each start with an
    empty revealed set.

    Concurrent asyncio tasks already receive isolation automatically via
    the context copy that ``asyncio.create_task`` makes at scheduling
    time; this function covers the sequential-await pattern::

        while True:
            result = await Runner.arun(agent, msg)  # run 1 reveals X
            # Without this reset, run 2 would still see X revealed.
            result = await Runner.arun(agent, msg)

    Walks *tools* the same way ``find_revealed_deferred_tools`` does
    (top-level ``FunctionTool`` entries and one level inside
    ``FunctionToolset``). No-op when no search tool is present.
    """
    from troopai.adk.tools.toolsets import FunctionToolset

    for tool in tools:
        if isinstance(tool, FunctionTool):
            state = tool.get_search_state()
            if state is not None:
                state.reset()
        elif isinstance(tool, FunctionToolset):
            for inner_tool in tool.tools:
                state = inner_tool.get_search_state()
                if state is not None:
                    state.reset()


@dataclass
class ToolSearchState:
    """Closure state attached to a ``tool_search`` FunctionTool.

    Created by ``build_tool_search()`` and stored on the returned
    FunctionTool's ``_search_state`` slot. Other modules access it via
    :meth:`FunctionTool.get_search_state`.

    Attributes:
        deferred: Tools the search tool can match against, keyed by
            their ``name``.
        _revealed_var: ``ContextVar`` holding the per-run revealed
            ``set``.  The ContextVar stores a *mutable* set that
            :meth:`reset` binds fresh at the start of each run; reveals
            then mutate that set in place (``set.add``) rather than
            rebinding the ContextVar.  In-place mutation is what makes
            reveals survive a parallel tool batch: ``asyncio.gather``
            copies the context for each spawned task, so a
            ``ContextVar.set`` inside a gather task would be lost to the
            parent — but mutating the shared set object the parent
            already bound is visible to the parent once the batch
            completes.  Per-run isolation is still exact: :meth:`reset`
            binds a brand-new set into the current context at the start
            of every run, and concurrent ``Runner.arun()`` calls each
            run in their own task whose copied context carries its own
            binding.  Sequential ``await Runner.arun()`` calls share one
            context, so the ``Runner`` calls :meth:`reset` to rebind a
            fresh set before each run.

    Use :meth:`revealed` to read the current run's set,
    :meth:`reveal` to add a name to it, and :meth:`reset` to clear
    the set at the start of a new run.
    """

    deferred: dict[str, FunctionTool]
    # Not a constructor argument — initialised in __post_init__.
    _revealed_var: ContextVar[set[str] | None] = field(init=False, repr=False, default=None)  # type: ignore[arg-type, assignment]

    def __post_init__(self) -> None:
        # Each ToolSearchState instance gets its own ContextVar so that
        # multiple search tools in the same process are fully independent.
        # The default is ``None`` (not a mutable ``set()``) because a
        # mutable ContextVar default is a single shared object across
        # every context that never rebinds it — reveals would leak
        # process-wide. ``reset()`` binds a fresh set per run; ``reveal()``
        # lazily binds one if reached before any reset.
        object.__setattr__(
            self,
            "_revealed_var",
            ContextVar(f"_tool_search_revealed_{id(self)}", default=None),
        )

    @property
    def revealed(self) -> frozenset[str]:
        """Return the revealed names for the current execution context.

        Returns a ``frozenset`` snapshot so callers cannot mutate the
        backing set through the read API.
        """
        current = self._revealed_var.get()
        if current is None:
            return frozenset()
        return frozenset(current)

    def reveal(self, name: str) -> None:
        """Add *name* to the revealed set for the current context.

        Mutates the set the current context already holds, in place.
        When the run's ``reset()`` bound that set before a parallel tool
        batch was spawned, the gather-copied task contexts share the
        same set object, so a reveal here is visible to the parent after
        the batch completes.  If no set is bound yet (a reveal reached
        before any reset), one is created and bound in the current
        context — this rebind is context-local, so concurrent runs stay
        isolated.
        """
        current = self._revealed_var.get()
        if current is None:
            current = set()
            self._revealed_var.set(current)
        current.add(name)

    def reset(self) -> None:
        """Reset the revealed set to empty for the current execution context.

        Binds a brand-new set into the current context.  Called at the
        start of each ``Runner.arun()`` so sequential awaits in the same
        coroutine start with a clean slate, and so the bound set exists
        before any parallel tool batch copies the context.
        """
        self._revealed_var.set(set())


class ToolSearchInput(BaseModel):
    """Input schema for the tool_search FunctionTool."""

    query: str = Field(
        ...,
        description="Natural-language query describing the tool you need.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of matching tools to return.",
    )


def build_tool_search(
    deferred_tools: Sequence[FunctionTool],
    *,
    name: str = "tool_search",
    description: str | None = None,
) -> FunctionTool:
    """Construct a ``FunctionTool`` that reveals matching deferred tools.

    Args:
        deferred_tools: Tools to match against. Typically these are the
            ``FunctionTool`` instances declared with
            ``defer_loading=True``; passing tools that are not deferred
            is harmless (they will appear in the search results but
            were never hidden in the first place).
        name: Tool name shown to the LLM. Defaults to ``"tool_search"``.
        description: Override the default description.

    Returns:
        A ``FunctionTool``. The framework's ``build_tools()`` reads the
        attached :class:`ToolSearchState` via ``get_search_state()`` to
        find the per-run ``revealed`` set.

    Example::

        from troopai.adk.tools import function_tool, build_tool_search


        @function_tool(name="rare_api", defer_loading=True)
        def rare_api(payload: str) -> str: ...


        @function_tool(name="another_rare", defer_loading=True)
        def another_rare(x: str) -> str: ...


        search = build_tool_search([rare_api, another_rare])
        agent = Agent(
            name="Worker",
            system_prompt="...",
            tools=[core_tool, rare_api, another_rare, search],
        )
        # The LLM initially sees: core_tool, search.
        # After it calls search("rare api"), rare_api is revealed and
        # appears on the next turn.
    """
    state = ToolSearchState(
        deferred={t.name: t for t in deferred_tools},
    )

    default_description = (
        "Search the deferred tool registry by natural-language query. "
        "Returns matched tool names and descriptions; the matched tools "
        "become available on the next turn. Use this when the task "
        "requires a capability you don't currently have a tool for."
    )

    async def on_invoke(ctx: ToolContext[Any], raw_args: str) -> str:  # noqa: ARG001
        # ``ctx`` is part of the ToolInvokeFunction contract; the search
        # tool reads everything it needs from ``raw_args`` and the
        # closure-captured ``state``.
        try:
            args = json.loads(raw_args) if len(raw_args) > 0 else {}
        except json.JSONDecodeError as e:
            return f"Invalid JSON input: {e}"
        query = args.get("query", "")
        # ``top_k`` is declared with ``ge=1, le=50`` on the schema, but
        # the executor calls ``on_invoke`` directly without re-running
        # Pydantic validation, so a misbehaving LLM could submit
        # ``top_k=99999`` and reveal the whole catalogue. Clamp to the
        # documented bounds defensively.
        try:
            top_k = max(1, min(int(args.get("top_k", 5)), 50))
        except (TypeError, ValueError):
            return "Invalid 'top_k' — must be an integer between 1 and 50."

        matches = _rank_matches(state.deferred, query, top_k)
        for tool in matches:
            state.reveal(tool.name)

        logger.info(
            "tool_search query=%r revealed %d tool(s): %s",
            query,
            len(matches),
            [t.name for t in matches],
        )

        return json.dumps(
            [{"name": tool.name, "description": tool.description or ""} for tool in matches],
            ensure_ascii=False,
        )

    tool = FunctionTool(
        name=name,
        description=description or default_description,
        schema=ToolSearchInput,
        on_invoke=on_invoke,
    )
    tool.set_search_state(state)
    return tool


def _rank_matches(
    deferred: dict[str, FunctionTool],
    query: str,
    top_k: int,
) -> list[FunctionTool]:
    """Return up to ``top_k`` deferred tools ranked by query relevance.

    Module-private. Uses simple substring-match scoring against tool
    name and description — name hits weighted higher.

    Empty query returns an empty list. This is deliberate: combined
    with an unclamped ``top_k`` an empty query would otherwise serve
    as a catalogue enumeration primitive, which undermines the
    capability-gating intent of deferred loading.
    """
    if len(query.strip()) == 0:
        return []

    q = query.lower()
    scored: list[tuple[int, FunctionTool]] = []
    for tool in deferred.values():
        score = 0
        if q in tool.name.lower():
            score += 10
        if tool.description is not None and q in tool.description.lower():
            score += 5
        # Token-level partial matches
        for word in q.split():
            if len(word) == 0:
                continue
            if word in tool.name.lower():
                score += 3
            if tool.description is not None and word in tool.description.lower():
                score += 1
        if score > 0:
            scored.append((score, tool))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [tool for _, tool in scored[:top_k]]
