"""RunHooks subclass for agent status recording and quota enforcement.

:class:`StatusTrackingHooks` integrates with ``Runner.arun()`` via the
existing hooks mechanism — no Runner changes needed.  It records per-run
metrics in an :class:`~troopai.adk.status.store.AgentStatusStore` and
optionally enforces :class:`~troopai.adk.status.types.AgentQuota`
limits before each run.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, TypeVar, override

from troopai.adk.hooks import RunHooks
from troopai.adk.status.store import AgentStatusStore
from troopai.adk.status.types import AgentQuota, AgentRunRecord

if TYPE_CHECKING:
    from troopai.adk.agents import Agent
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.stream import RunResultStreaming
    from troopai.adk.types.run.run_result import RunResult

logger = logging.getLogger(__name__)

TContext = TypeVar("TContext")


class StatusTrackingHooks[TContext](RunHooks[TContext]):
    """RunHooks subclass that records agent run status and enforces quotas.

    Tracks per-run timing and usage via ``on_agent_start`` /
    ``on_agent_end``, persists records to an
    :class:`~troopai.adk.status.store.AgentStatusStore`, and
    optionally enforces cumulative quotas before each run.

    Args:
        store: The :class:`AgentStatusStore` to record to and query from.
        quotas: Optional list of :class:`AgentQuota` to enforce before
            each run.  Quotas matching the agent name (or ``"*"``) are
            checked in ``on_agent_start``.
        stale_run_seconds: Age after which an unclosed run-start entry is
            evicted (default one day). Entries are normally removed by
            ``on_agent_end`` or :meth:`record_error`; a run that raised
            without the caller invoking :meth:`record_error` would
            otherwise pin its entry forever on a long-lived hooks
            instance. Evicted entries are logged at warning level.

    Example::

        store = AgentStatusStore(path="status.db")
        hooks = StatusTrackingHooks(
            store=store,
            quotas=[
                AgentQuota(
                    agent_name="*",
                    window_seconds=86400,
                    max_total_tokens=500_000,
                ),
            ],
        )
        result = await Runner.arun(agent, "Hello!", hooks=hooks)
    """

    def __init__(
        self,
        store: AgentStatusStore,
        quotas: list[AgentQuota] | None = None,
        stale_run_seconds: float = 86_400.0,
    ) -> None:
        if stale_run_seconds <= 0:
            raise ValueError(f"stale_run_seconds must be positive, got {stale_run_seconds}")
        self._store = store
        self._quotas = quotas or []
        self._stale_run_seconds = stale_run_seconds
        self._run_starts: dict[tuple[str | None, str, int], float] = {}

    def _evict_stale_starts(self, now: float) -> None:
        """Drop run-start entries older than ``stale_run_seconds``.

        These belong to runs that raised without the caller invoking
        :meth:`record_error` — the only path that would otherwise free
        them. Keeps ``_run_starts`` bounded on long-lived instances.
        """
        cutoff = now - self._stale_run_seconds
        stale = [key for key, started in self._run_starts.items() if started < cutoff]
        for key in stale:
            del self._run_starts[key]
        if len(stale) > 0:
            logger.warning(
                "Evicted %d run-start entr%s older than %.0fs (run raised without record_error?)",
                len(stale),
                "y" if len(stale) == 1 else "ies",
                self._stale_run_seconds,
            )

    @override
    async def on_agent_start(
        self,
        context: RunContext[TContext],
        agent: Agent,
    ) -> None:
        """Check quotas and record run start time.

        Raises :class:`~troopai.adk.exceptions.QuotaExceeded`
        before any LLM call if the agent has exceeded its cumulative
        limits.

        Args:
            context: The active run context, used to read ``tenant_id``
                and scope quota checks per tenant.
            agent: The agent about to run; its ``name`` is used as the
                quota lookup key and the timing record key.

        Raises:
            QuotaExceeded: If any matching quota limit is exceeded
                before the run starts.
        """
        # Check quotas BEFORE the run proceeds (scoped to this run's tenant)
        if len(self._quotas) > 0:
            logger.debug(
                "Checking %d quota(s) for agent '%s'",
                len(self._quotas),
                agent.name,
            )
            await self._store.check_quotas(agent.name, self._quotas, tenant_id=context.tenant_id)

        # Record start time keyed by (tenant_id, agent_name, context_id) to
        # avoid collisions when the same agent runs for the same tenant
        # concurrently (e.g. two asyncio.gather calls on one hooks instance).
        now = time.time()
        self._evict_stale_starts(now)
        self._run_starts[(context.tenant_id, agent.name, id(context))] = now

    @override
    async def on_agent_end(
        self,
        context: RunContext[TContext],
        agent: Agent,
        result: RunResult | RunResultStreaming,
    ) -> None:
        """Record the completed run to the status store.

        Reads timing from the ``on_agent_start`` entry, computes
        ``duration_ms``, and persists an :class:`AgentRunRecord` with
        ``status="success"``.

        Args:
            context: The active run context; provides ``tenant_id``,
                ``usage``, and ``cost_usd`` for the record.
            agent: The agent that finished running.
            result: The run result (streaming or non-streaming); not
                inspected directly — token usage is read from ``context``.
        """
        ended_at = time.time()
        started_at = self._run_starts.pop((context.tenant_id, agent.name, id(context)), None)
        if started_at is None:
            logger.warning(
                "on_agent_end: no start time for agent=%r tenant_id=%r; recording duration_ms=0 "
                "(on_agent_start was not called or tenant_id mismatched)",
                agent.name,
                context.tenant_id,
            )
            started_at = ended_at
        duration_ms = (ended_at - started_at) * 1000

        usage = context.usage

        record = AgentRunRecord(
            id=uuid.uuid4().hex,
            agent_name=agent.name,
            status="success",
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            requests=usage.requests,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            error=None,
            tenant_id=context.tenant_id,
            cost_usd=context.cost_usd,
        )
        await self._store.record(record)

    async def record_error(
        self,
        agent_name: str,
        error: str,
        *,
        context: RunContext[TContext] | None = None,
        tenant_id: str | None = None,
        cost_usd: float | None = None,
    ) -> None:
        """Manually record a failed run.

        Call from your error handler when ``Runner.arun()`` raises
        an exception — ``on_agent_end`` is not called in that case.

        Example::

            try:
                result = await Runner.arun(agent, prompt, hooks=hooks)
            except Exception as e:
                await hooks.record_error(
                    agent.name,
                    str(e),
                    context=context,
                    cost_usd=context.cost_usd,
                )
                raise

        Args:
            agent_name: Name of the agent that failed.
            error: Error message or traceback string.
            context: The run's :class:`RunContext`, if available.  When
                provided, its identity is used to pop the exact timing entry
                recorded by ``on_agent_start`` — required to attribute the
                correct duration when several runs of the same agent and
                tenant are in flight concurrently.  ``tenant_id`` is then
                read from the context (the explicit ``tenant_id`` argument is
                ignored).
            tenant_id: Tenant to attribute this error record to, used only
                when ``context`` is ``None``.  Must match the ``tenant_id``
                passed to ``on_agent_start`` for the same run so that a
                timing entry can be retrieved.  Without ``context`` the
                lookup cannot disambiguate concurrent same-agent/same-tenant
                runs and falls back to the oldest matching entry.
            cost_usd: Partial USD cost accrued before the failure, or
                ``None`` if not tracked.  Pass ``context.cost_usd`` to
                preserve any cost that accumulated before the exception so
                it is included in ``AgentStatus.total_cost_usd``.
        """
        ended_at = time.time()
        started_at_opt: float | None = None
        if context is not None:
            # Exact lookup by run identity — never steals a concurrent
            # sibling's entry. tenant_id comes from the context itself.
            tenant_id = context.tenant_id
            started_at_opt = self._run_starts.pop((tenant_id, agent_name, id(context)), None)
        else:
            # No context: scan for starts matching (tenant_id, agent_name).
            # Pop the OLDEST matching entry (lowest timestamp) — among
            # ambiguous in-flight siblings it is the least likely to belong
            # to a run that started after this failure began.
            matching_keys = [k for k in self._run_starts if k[0] == tenant_id and k[1] == agent_name]
            if len(matching_keys) > 0:
                best_key = min(matching_keys, key=lambda k: self._run_starts[k])
                started_at_opt = self._run_starts.pop(best_key)
        if started_at_opt is None:
            logger.warning(
                "record_error: no start time for agent=%r tenant_id=%r; recording duration_ms=0 "
                "(on_agent_start was not called or tenant_id mismatched)",
                agent_name,
                tenant_id,
            )
            started_at_opt = ended_at
        started_at = started_at_opt
        duration_ms = (ended_at - started_at) * 1000

        record = AgentRunRecord(
            id=uuid.uuid4().hex,
            agent_name=agent_name,
            status="error",
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            requests=0,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            error=error,
            tenant_id=tenant_id,
            cost_usd=cost_usd,
        )
        await self._store.record(record)
