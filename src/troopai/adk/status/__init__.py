"""Agent status tracking and cumulative quota enforcement.

This module provides persistent tracking of agent run metrics
(tokens, requests, errors, duration) and time-windowed quota
enforcement.  It integrates with the Runner via
:class:`StatusTrackingHooks` — no Runner changes needed.

Usage::

    from troopai.adk.status import (
        AgentStatusStore,
        StatusTrackingHooks,
        AgentQuota,
    )

    store = AgentStatusStore(path="agent_status.db")
    hooks = StatusTrackingHooks(
        store=store,
        quotas=[
            AgentQuota(agent_name="*", window_seconds=86400, max_total_tokens=500_000),
        ],
    )
    result = await Runner.arun(agent, "Hello!", hooks=hooks)
"""

from troopai.adk.status.hooks import StatusTrackingHooks
from troopai.adk.status.store import AgentStatusStore
from troopai.adk.status.types import AgentQuota, AgentRunRecord, AgentStatus

__all__ = [
    "AgentQuota",
    "AgentRunRecord",
    "AgentStatus",
    "AgentStatusStore",
    "StatusTrackingHooks",
]
