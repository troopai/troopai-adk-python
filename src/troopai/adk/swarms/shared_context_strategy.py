"""Shared-context strategy — how much of the swarm history each
member agent sees on its turn.

Default is ``SCOPED``: each agent sees only its own prior messages
plus anything explicitly passed in a
``SwarmHandoff.message``. This stance is the most explicit: no silent
cross-agent broadcast, no unbounded context growth, and no
summarization cost by default.

The other strategies exist because the pattern is genuinely useful
in different workloads:

- ``FULL_BROADCAST`` matches AutoGen Swarm semantics — every agent
  sees every message. Good for small, short swarms.
- ``LAST_N`` bounds the broadcast to the last N items. A sane middle
  ground when cross-pollination helps but context budget matters.
- ``SUMMARIZED`` runs older history through ``ContextCompactor``
  (same path used by ``HandoffConfig.budget`` + ``summary`` strategy)
  so long swarms stay within a token budget without losing signal.
"""

from __future__ import annotations

from enum import StrEnum


class SharedContextStrategy(StrEnum):
    """Controls the messages a member agent sees on its turn.

    Attributes:
        SCOPED: Each agent sees only its own prior messages + the
            explicit ``SwarmHandoff.message`` (if any). Default.
        LAST_N: Each agent sees the last N items of the shared
            history. Requires ``SharedContextConfig.window``.
        SUMMARIZED: Older history is compacted via
            ``ContextCompactor`` to fit within
            ``SharedContextConfig.budget`` tokens. Recent items are
            kept verbatim.
        FULL_BROADCAST: Each agent sees the complete shared history.
            AutoGen Swarm parity. Unbounded — use only for small
            swarms.
    """

    SCOPED = "scoped"
    """Per-agent scratch + explicit handoff message only. Default."""

    LAST_N = "last_n"
    """Last N items of the shared history (requires ``window``)."""

    SUMMARIZED = "summarized"
    """Compacted via ``ContextCompactor`` to fit a token budget."""

    FULL_BROADCAST = "full_broadcast"
    """Every message visible to every agent. AutoGen parity."""
