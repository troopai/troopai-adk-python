"""Merge strategies — how a node folds upstream ``NodeResult`` values
into the single ``ExecutableInput`` it receives.

A :class:`~troopai.adk.graphs.graph.Graph` does not use a global shared
state dict. Instead, each node declares how it wants its inputs
composed when multiple upstream edges feed it. This keeps the model
readable (the merge strategy is local to the receiving node) without
the ``Annotated[list, add_messages]`` magic LangGraph uses.

Five built-in strategies cover the common cases. A :class:`CustomMerge`
escape hatch lets the developer plug in any pure function.

Example::

    from troopai.adk.graphs import Graph, Merge

    g = (
        Graph.new("review")
        .node("researcher", researcher_agent)
        .node("critic", critic_agent)
        .node(
            "synthesizer",
            synthesizer_agent,
            merge=Merge.concat_text,  # joins upstream final_text
        )
        .pipe("researcher", "synthesizer")
        .pipe("critic", "synthesizer")  # AND-join by default
        .entry("researcher")
        .terminal("synthesizer")
        .compile()
    )

Design note. Merge functions receive the list of upstream
:class:`NodeResult` values (already ordered deterministically by the
graph loop — see ``graph_loop.py::_ordered_results``) plus the
original upstream node ids (same order). The function returns the
payload that becomes :attr:`ExecutableInput.content`. Returning a
``str`` wraps it in a single user message for adapter convenience;
returning a ``list[LLMInputContentItem]`` is used verbatim.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.orchestration.executable import NodeResult
    from troopai.adk.types.input import LLMInputContentItem


# A merge function: (ordered upstream results, ordered upstream node ids) ->
# either a ready-to-send Layer 1 item list, or a plain string (which the
# graph loop wraps into a single user message).
MergeFn = Callable[
    [list["NodeResult"], list[str]],
    str | list["LLMInputContentItem"],
]
"""Signature of a merge strategy. Returns Layer 1 content for the downstream node."""


def _extract_text(result: NodeResult) -> str:
    """Best-effort text view of a :class:`NodeResult`.

    Preference order: ``final_text`` > ``output`` if it is a ``str``
    > ``str(output)`` as a last resort. Returns an empty string when
    the result carries no text-shaped payload (e.g. a ``SwarmRunResult``
    with ``final_output=None``).
    """
    if result.final_text is not None and len(result.final_text) > 0:
        return result.final_text
    if isinstance(result.output, str) and len(result.output) > 0:
        return result.output
    if result.output is None:
        return ""
    return str(result.output)


class Merge:
    """Namespace for built-in merge strategies.

    All strategies are stateless functions exposed as class attributes so
    they can be referenced by identity::

        .node("target", agent, merge=Merge.concat_text)

    The class is never instantiated. ``Merge.custom(fn)`` wraps a
    user-supplied callable with the same signature so custom merges
    fit the declared :data:`MergeFn` type.
    """

    @staticmethod
    def last_wins(
        results: list[NodeResult],
        sources: list[str],
    ) -> str | list[LLMInputContentItem]:
        """Take the last upstream result's text, discard the rest.

        Deterministic because the graph loop sorts the upstream
        results by source node id before calling the merge function.
        "Last" here means lexicographically last source id — NOT
        completion order — so the output is reproducible across runs.

        Useful when a downstream node receives multiple signals but
        only cares about the most recent routing decision.
        """
        del sources
        if len(results) == 0:
            return ""
        return _extract_text(results[-1])

    @staticmethod
    def concat_text(
        results: list[NodeResult],
        sources: list[str],
    ) -> str | list[LLMInputContentItem]:
        """Join upstream texts with double newlines, labelled by source.

        Produces a single string of the form::

            [researcher]
            <researcher text>

            [critic]
            <critic text>

        Downstream adapters that wrap this into a user message see a
        readable fan-in document. The label comes from the source node
        id so the downstream LLM can reason about who said what.

        Empty texts are omitted so an upstream no-op doesn't produce
        an empty section.
        """
        chunks: list[str] = []
        for result, source in zip(results, sources, strict=False):
            text = _extract_text(result)
            if len(text) == 0:
                continue
            chunks.append(f"[{source}]\n{text}")
        return "\n\n".join(chunks)

    @staticmethod
    def extend_items(
        results: list[NodeResult],
        sources: list[str],
    ) -> str | list[LLMInputContentItem]:
        """Concatenate Layer 1 replay params from every upstream node.

        Each upstream :class:`NodeResult.new_items` is converted via
        :meth:`ItemHelpers.run_items_to_params` and the resulting
        Layer 1 items are flattened in source order. Use this when the
        downstream node is an ``AgentExecutable`` that wants to see
        the full turn history of every upstream contributor.

        Heavier on tokens than :meth:`concat_text` but preserves
        turn structure (tool calls, tool results, reasoning).
        """
        from troopai.adk.types.items.items import ItemHelpers

        del sources
        items: list[LLMInputContentItem] = []
        for result in results:
            items.extend(ItemHelpers.run_items_to_params(result.new_items))
        return items

    @staticmethod
    def first_wins(
        results: list[NodeResult],
        sources: list[str],
    ) -> str | list[LLMInputContentItem]:
        """Inverse of :meth:`last_wins` — take the first (source-sorted) result.

        Useful for "priority fan-in" where one upstream branch has
        precedence and the others are ignored.
        """
        del sources
        if len(results) == 0:
            return ""
        return _extract_text(results[0])

    @staticmethod
    def custom(fn: MergeFn) -> MergeFn:
        """Escape hatch — wrap a user-supplied merge function.

        The function MUST be pure (no side effects), deterministic in
        the order of its inputs (the graph loop already sorts by
        source id), and MUST return either a ``str`` or a
        ``list[LLMInputContentItem]``. Returning anything else raises
        at merge time.

        Args:
            fn: User-supplied merge function.

        Returns:
            ``fn`` unchanged. This wrapper exists to give a single
            reference point for documentation and future validation
            (e.g. checking the signature at registration time).
        """
        return fn


DEFAULT_MERGE: MergeFn = Merge.concat_text
"""Default merge when a node declares none. Concat-with-labels is the
most-readable baseline — upstream contributions are visible and
attributed."""


__all__ = [
    "DEFAULT_MERGE",
    "Merge",
    "MergeFn",
]
