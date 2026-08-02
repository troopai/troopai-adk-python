"""Shared hook bridge for :class:`Checkpointer` implementations.

Every checkpointer persists the same way: on ``on_node_end`` and on
``on_graph_end``, snapshot the current :class:`GraphState` into a
:class:`GraphCheckpoint` and ``save`` it (skipping when the run did
not opt into checkpointing, i.e. ``state.thread_id is None``). This
bridge factors that identical logic out so each concrete checkpointer
only implements storage.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, override

from troopai.adk.graphs.checkpointer import Checkpointer, GraphCheckpoint
from troopai.adk.graphs.hooks import GraphHooks

if TYPE_CHECKING:
    from troopai.adk.graphs.result import GraphRunStatus
    from troopai.adk.graphs.state import GraphState
    from troopai.adk.orchestration.executable import NodeResult
    from troopai.adk.run.context import RunContext


logger = logging.getLogger(__name__)


class CheckpointerHooks(GraphHooks[Any]):
    """Forwards ``on_node_end`` / ``on_graph_end`` into any
    :class:`Checkpointer`. Constructed by a checkpointer's
    :meth:`Checkpointer.register`.

    Setting ``propagate_errors = True`` ensures that a failed
    :meth:`Checkpointer.save` is not silently swallowed by the hook
    fan-out — the caller receives the error and can decide how to handle it.
    """

    propagate_errors = True

    def __init__(self, owner: Checkpointer) -> None:
        self._owner = owner

    @override
    async def on_node_end(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        node_id: str,
        result: NodeResult,
    ) -> None:
        """Persist the current :class:`GraphState` after each node fires.

        Skipped when ``state.thread_id`` is ``None`` — the caller
        didn't opt in to checkpointing on this run even though a
        checkpointer was attached (common for "attach checkpointer
        globally, opt in per run").
        """
        del context, node_id, result

        if state.thread_id is None:
            return

        await self._owner.save(
            GraphCheckpoint(
                thread_id=state.thread_id,
                graph_id=state.graph.id,
                state=state.to_dict(),
                superstep=state.superstep,
            )
        )

    @override
    async def on_graph_end(
        self,
        context: RunContext[Any],
        state: GraphState[Any],
        status: GraphRunStatus,
        final_output: Any,
    ) -> None:
        """Final flush so the terminal state is persisted even if no
        node fired in the last superstep (unusual but possible with
        conditional-edge no-ops).
        """
        del context, status, final_output

        if state.thread_id is None:
            return

        await self._owner.save(
            GraphCheckpoint(
                thread_id=state.thread_id,
                graph_id=state.graph.id,
                state=state.to_dict(),
                superstep=state.superstep,
            )
        )


__all__ = ["CheckpointerHooks"]
