"""``prepare_node_input`` — build the :class:`ExecutableInput` a node
receives at firing time.

This is the graph-world equivalent of
:func:`troopai.adk.swarms.shared_context.prepare_turn_input`. The graph
loop calls it once per node per superstep, right before invoking
:meth:`Executable.invoke`.

Given:

- The current :class:`GraphState`.
- The target :class:`GraphNode` (carries the merge strategy).
- The list of arrived upstream :class:`NodeResult`\\ s and the parallel
  list of their source ids (both already sorted by source id by
  :meth:`JoinBarrier.consume`).
- The :class:`NodeInputStrategy` from :class:`GraphConfig`.
- The original user prompt (used by the entry node on its first
  firing).

This function returns the :class:`ExecutableInput` the node consumes.
It encapsulates three independent concerns:

1. **Merge** — fold multiple upstream values via the node's
   ``merge_fn``. Output is either a ``str`` (wrap into a user message)
   or a ``list[LLMInputContentItem]`` (use verbatim). This is LOCAL
   scope — only the direct upstream contributions.

2. **Scope** — per :class:`NodeInputStrategy`:
   - ``LAST_OUTPUT`` (default) passes only the merged LOCAL payload.
   - ``MERGED_OUTPUTS`` prepends every completed node's ``final_text``
     as a running-document so the target sees "everything said so far".
   - ``FULL_HISTORY`` prepends every Layer 3 :class:`RunItem` produced
     anywhere in the run, converted to Layer 1 params.

3. **Entry bootstrap** — the entry node has no upstreams on its first
   firing, so the caller passes the original ``user_prompt`` through
   ``initial_prompt`` and we normalise it via
   :meth:`ItemHelpers.input_to_new_input_list`.

Keeping all three knobs inside one function (not sprinkled across the
loop) makes it easy to unit-test in isolation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from troopai.adk.graphs.config import NodeInputStrategy
from troopai.adk.orchestration.executable import ExecutableInput
from troopai.adk.types.items.items import ItemHelpers

if TYPE_CHECKING:
    from troopai.adk.graphs.node import GraphNode
    from troopai.adk.graphs.state import GraphState
    from troopai.adk.orchestration.executable import NodeResult
    from troopai.adk.run.types import UserPrompt
    from troopai.adk.types.input import LLMInputContentItem


logger = logging.getLogger(__name__)


def prepare_node_input(
    *,
    state: GraphState,
    node: GraphNode,
    arrived_results: list[NodeResult],
    arrived_sources: list[str],
    strategy: NodeInputStrategy,
    edge_label: str | None,
    initial_prompt: UserPrompt | None,
) -> ExecutableInput:
    """Build the :class:`ExecutableInput` a node should receive.

    The function is pure — all mutation happens through the returned
    value. Errors bubble up to the graph loop.

    Args:
        state: Current :class:`GraphState` (read-only).
        node: The target :class:`GraphNode`. Its ``merge`` callable
            folds ``arrived_results``.
        arrived_results: Upstream :class:`NodeResult`\\ s in
            source-id sorted order. Empty on the entry node's first
            firing.
        arrived_sources: Parallel list of source node ids, same order
            as ``arrived_results``.
        strategy: Graph-wide :class:`NodeInputStrategy`. A per-node
            override via ``node.metadata["input"]`` is not yet
            implemented.
        edge_label: Optional label of the triggering edge.
        initial_prompt: Original user prompt passed to
            :meth:`Runner.arun_graph`. Used only on the entry node's
            first firing (when ``arrived_results`` is empty).

    Returns:
        A freshly constructed :class:`ExecutableInput`.
    """
    # Case A: entry node, first firing. No upstreams yet.
    if len(arrived_results) == 0:
        if initial_prompt is None:
            logger.debug(
                "prepare_node_input: entry node %s has no initial prompt; defaulting to empty content.",
                node.id,
            )
            content: list[LLMInputContentItem] = []
        else:
            content = ItemHelpers.input_to_new_input_list(initial_prompt)
        return ExecutableInput(
            content=content,
            from_node=None,
            from_nodes=(),
            edge_label=edge_label,
        )

    # Case B: node has arrivals — run merge strategy.
    merged = node.merge(arrived_results, arrived_sources)
    if isinstance(merged, str):
        # Wrap a plain string into a single user message so every
        # adapter sees a uniform Layer 1 shape.
        from troopai.adk.types.input import LLMInputEasyMessage

        user_msg: LLMInputContentItem = LLMInputEasyMessage(
            role="user",
            content=merged,
        )
        local_content: list[LLMInputContentItem] = [user_msg]
    else:
        # merge_fn is declared to return str | list[LLMInputContentItem].
        local_content = list(merged)

    # Apply the scoping strategy.
    if strategy == NodeInputStrategy.LAST_OUTPUT:
        final_content = local_content
    elif strategy == NodeInputStrategy.MERGED_OUTPUTS:
        final_content = _apply_merged_outputs(state, local_content)
    elif strategy == NodeInputStrategy.FULL_HISTORY:
        final_content = _apply_full_history(state, local_content)
    else:
        # Defensive — Enum values are closed; this branch should be
        # unreachable. Fail loud rather than silently dropping scope.
        raise ValueError(f"Unknown NodeInputStrategy: {strategy!r}. Add a branch to prepare_node_input().")

    from_node = arrived_sources[0] if len(arrived_sources) == 1 else None
    return ExecutableInput(
        content=final_content,
        from_node=from_node,
        from_nodes=tuple(arrived_sources),
        edge_label=edge_label,
    )


def _apply_merged_outputs(
    state: GraphState,
    local_content: list[LLMInputContentItem],
) -> list[LLMInputContentItem]:
    """Prepend every completed node's ``final_text`` as a running doc.

    Produces one user-role message per completed node id, sorted by
    id for determinism. Silent on nodes with empty ``final_text`` so
    an LLM-free callable node doesn't dump ``None`` into the context.
    """
    from troopai.adk.types.input import LLMInputEasyMessage

    chunks: list[LLMInputContentItem] = []
    for node_id in sorted(state.node_results.keys()):
        result = state.node_results[node_id]
        text = result.final_text
        if text is None or len(text) == 0:
            continue
        prepended: LLMInputContentItem = LLMInputEasyMessage(
            role="user",
            content=f"[{node_id}]\n{text}",
        )
        chunks.append(prepended)
    return chunks + list(local_content)


def _apply_full_history(
    state: GraphState,
    local_content: list[LLMInputContentItem],
) -> list[LLMInputContentItem]:
    """Prepend every Layer 3 item in ``state.all_items`` as Layer 1 params.

    Expensive: every agent turn, tool call, and reasoning block from
    the whole run replays into the target. Use for auditing and
    compliance pipelines that genuinely need the full trace.
    """
    replay = list(ItemHelpers.run_items_to_params(state.all_items))
    return replay + list(local_content)


__all__ = ["prepare_node_input"]
