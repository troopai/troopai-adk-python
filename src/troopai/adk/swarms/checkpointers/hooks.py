"""Shared hook bridge for :class:`~troopai.adk.swarms.checkpointer.SwarmCheckpointer`
implementations.

Every swarm checkpointer persists the same way: on ``on_swarm_turn_end``
and on ``on_swarm_turn_interrupt``, snapshot the current
:class:`~troopai.adk.swarms.state.SwarmState` into a
:class:`~troopai.adk.swarms.checkpointer.SwarmCheckpoint` and ``save`` it.
This bridge factors that identical logic out so each concrete checkpointer
only implements storage.

The interrupt override is essential: the swarm loop returns before
``on_swarm_turn_end`` fires when a turn parks on a cooperative interrupt
(HITL ``request_human_input`` or a nested-agent tool deferral). Without
the ``on_swarm_turn_interrupt`` hook the parked
``pending_interrupts`` / ``nested_agent_snapshots`` would never reach the
checkpoint store.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast, override

from troopai.adk.swarms.checkpointer import SwarmCheckpoint
from troopai.adk.swarms.hooks import SwarmHooks

if TYPE_CHECKING:
    from troopai.adk.graphs.interrupt import Interrupt
    from troopai.adk.run.context import RunContext
    from troopai.adk.swarms.checkpointer import SwarmCheckpointer
    from troopai.adk.swarms.state import SwarmState
    from troopai.adk.types.items.items import RunItem


logger = logging.getLogger(__name__)


class SwarmCheckpointerHooks(SwarmHooks[Any]):
    """Forwards ``on_swarm_turn_end`` / ``on_swarm_turn_interrupt`` into any
    :class:`~troopai.adk.swarms.checkpointer.SwarmCheckpointer`.

    Constructed by a checkpointer's
    :meth:`~troopai.adk.swarms.checkpointer.SwarmCheckpointer.register` method.

    Setting ``propagate_errors = True`` ensures that a failed
    :meth:`~troopai.adk.swarms.checkpointer.SwarmCheckpointer.save` is not
    silently swallowed by the hook fan-out — the caller receives the error
    and can decide how to handle it.
    """

    propagate_errors = True

    def __init__(self, owner: SwarmCheckpointer, thread_id: str) -> None:
        """Attach the hook bridge to ``owner``.

        Args:
            owner: The checkpointer that owns this hook instance. Receives
                ``save`` calls with the current state snapshot.
            thread_id: The logical run identifier under which checkpoints
                are stored.
        """
        self._owner = owner
        self._thread_id = thread_id

    @override
    async def on_swarm_turn_end(
        self,
        context: RunContext[Any],
        state: SwarmState[Any],
        items: list[RunItem],
    ) -> None:
        """Persist the current state after a normal turn completes."""
        del context, items
        checkpoint = SwarmCheckpoint(
            thread_id=self._thread_id,
            state=cast(dict[str, Any], state.to_dict()),
            turn=state.total_turns,
        )
        try:
            await self._owner.save(checkpoint)
        except Exception:
            logger.exception(
                "SwarmCheckpointerHooks: save failed for thread_id=%s turn=%d",
                self._thread_id,
                checkpoint.turn,
            )
            raise
        else:
            logger.debug(
                "SwarmCheckpointerHooks: saved thread_id=%s turn=%d",
                checkpoint.thread_id,
                checkpoint.turn,
            )

    @override
    async def on_swarm_turn_interrupt(
        self,
        context: RunContext[Any],
        state: SwarmState[Any],
        member_name: str,
        interrupt: Interrupt,
    ) -> None:
        """Persist the current state when a member turn parks on an interrupt.

        Without this save the parked ``pending_interrupts`` and
        ``nested_agent_snapshots`` would never reach the store, because the
        swarm loop returns before ``on_swarm_turn_end`` fires when a turn
        suspends.
        """
        del context, member_name, interrupt
        checkpoint = SwarmCheckpoint(
            thread_id=self._thread_id,
            state=cast(dict[str, Any], state.to_dict()),
            turn=state.total_turns,
        )
        try:
            await self._owner.save(checkpoint)
        except Exception:
            logger.exception(
                "SwarmCheckpointerHooks: save failed for thread_id=%s turn=%d",
                self._thread_id,
                checkpoint.turn,
            )
            raise
        else:
            logger.debug(
                "SwarmCheckpointerHooks: saved thread_id=%s turn=%d",
                checkpoint.thread_id,
                checkpoint.turn,
            )


__all__ = ["SwarmCheckpointerHooks"]
