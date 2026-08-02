"""Data types for agent status tracking and cumulative quotas.

Three types:

- :class:`AgentRunRecord` — a single run record (one row per agent execution)
- :class:`AgentStatus` — aggregated stats over a time window (computed, not stored)
- :class:`AgentQuota` — a time-windowed usage quota definition
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class AgentRunRecord:
    """A single run record for one agent execution.

    Created at the end of each ``Runner.arun()`` call by
    :class:`~troopai.adk.status.hooks.StatusTrackingHooks` and
    persisted to an :class:`~troopai.adk.status.store.AgentStatusStore`.

    Attributes:
        id: Unique run identifier (UUID4 hex string).
        agent_name: The agent's name (``Agent.name``).
        status: Run outcome — ``"success"`` or ``"error"``.
        started_at: UTC timestamp when the run began (``time.time()``).
        ended_at: UTC timestamp when the run ended.
        duration_ms: Wall-clock duration in milliseconds.
        requests: Number of LLM requests made.
        input_tokens: Total input tokens consumed.
        output_tokens: Total output tokens consumed.
        total_tokens: Total tokens consumed (input + output).
        error: Error message if status is ``"error"``, else ``None``.
        tenant_id: Tenant identifier for this run, or ``None`` if not tenant-scoped.
        cost_usd: Best-effort USD cost of this run, or ``None`` if not tracked.
    """

    id: str
    """Unique run identifier (UUID4 hex string)."""

    agent_name: str
    """The agent's name (``Agent.name``)."""

    status: str
    """Run outcome — ``"success"`` or ``"error"``."""

    started_at: float
    """UTC timestamp when the run began."""

    ended_at: float
    """UTC timestamp when the run ended."""

    duration_ms: float
    """Wall-clock duration in milliseconds."""

    requests: int
    """Number of LLM requests made."""

    input_tokens: int
    """Total input tokens consumed."""

    output_tokens: int
    """Total output tokens consumed."""

    total_tokens: int
    """Total tokens consumed (input + output)."""

    error: str | None
    """Error message if status is ``"error"``, else ``None``."""

    tenant_id: str | None = None
    """Tenant identifier for this run, or ``None`` if not tenant-scoped."""

    cost_usd: float | None = None
    """Best-effort USD cost of this run, or ``None`` if not tracked."""


@dataclasses.dataclass(frozen=True)
class AgentStatus:
    """Aggregated status for an agent over a time window.

    Computed from :class:`AgentRunRecord` rows via SQL aggregation.
    Not stored directly — returned by
    :meth:`~troopai.adk.status.store.AgentStatusStore.get_status`.

    Attributes:
        agent_name: The agent's name.
        total_runs: Number of runs in the window.
        successful_runs: Runs with status ``"success"``.
        failed_runs: Runs with status ``"error"``.
        total_requests: Sum of LLM requests across all runs.
        total_input_tokens: Sum of input tokens across all runs.
        total_output_tokens: Sum of output tokens across all runs.
        total_tokens: Sum of all tokens across all runs.
        total_duration_ms: Sum of wall-clock duration across all runs.
        error_rate: Fraction of runs with errors (0.0 to 1.0).
        avg_duration_ms: Average run duration in milliseconds.
        first_run_at: Timestamp of the earliest run, or ``None``.
        last_run_at: Timestamp of the most recent run, or ``None``.
        total_cost_usd: Sum of ``cost_usd`` across the window; ``0.0`` when
            cost was never tracked (all records have ``cost_usd = NULL``).
    """

    agent_name: str
    """The agent's name."""

    total_runs: int
    """Number of runs in the window."""

    successful_runs: int
    """Runs with status ``"success"``."""

    failed_runs: int
    """Runs with status ``"error"``."""

    total_requests: int
    """Sum of LLM requests across all runs."""

    total_input_tokens: int
    """Sum of input tokens across all runs."""

    total_output_tokens: int
    """Sum of output tokens across all runs."""

    total_tokens: int
    """Sum of all tokens across all runs."""

    total_duration_ms: float
    """Sum of wall-clock duration across all runs."""

    error_rate: float
    """Fraction of runs with errors (0.0 to 1.0)."""

    avg_duration_ms: float
    """Average run duration in milliseconds."""

    first_run_at: float | None
    """Timestamp of the earliest run, or ``None``."""

    last_run_at: float | None
    """Timestamp of the most recent run, or ``None``."""

    total_cost_usd: float = 0.0
    """Sum of cost_usd across the window; ``0.0`` when cost was never tracked
    (all records have ``cost_usd = NULL``)."""


@dataclasses.dataclass
class AgentQuota:
    """A time-windowed usage quota for an agent.

    Defines limits that are checked against accumulated
    :class:`AgentRunRecord` history before each run.  Use
    ``agent_name="*"`` to apply the quota to all agents.

    Attributes:
        agent_name: Agent to apply the quota to. ``"*"`` = all agents.
        window_seconds: Sliding window length in seconds.
            Common values: ``3600`` (1 hour), ``86400`` (1 day).
        max_requests: Maximum LLM requests in the window, or ``None`` for no limit.
        max_total_tokens: Maximum total tokens in the window, or ``None`` for no limit.
        max_runs: Maximum number of runs in the window, or ``None`` for no limit.
    """

    agent_name: str
    """Agent to apply the quota to. ``"*"`` = all agents."""

    window_seconds: int
    """Sliding window length in seconds."""

    max_requests: int | None = None
    """Maximum LLM requests in the window, or ``None`` for no limit."""

    max_total_tokens: int | None = None
    """Maximum total tokens in the window, or ``None`` for no limit."""

    max_runs: int | None = None
    """Maximum number of runs in the window, or ``None`` for no limit."""
