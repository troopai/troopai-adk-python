"""Cost-accounting ledger Protocol + in-memory backend.

Mirrors the checkpointer Protocol idiom: ``@runtime_checkable``, async
I/O. ``record()`` is an atomic increment; concrete backends keep one
counter per (tenant_id, period_key).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class CostLedger(Protocol):
    """Cross-run, time-windowed spend accounting per tenant."""

    async def spend(self, tenant_id: str, period_key: str) -> float:
        """Return accumulated USD for ``tenant_id`` in this window (0 if none).

        Args:
            tenant_id: Tenant identifier.
            period_key: Time-window key produced by ``period_key()`` for the
                desired calendar bucket.

        Returns:
            Accumulated spend in USD, or ``0.0`` when no record exists.
        """
        ...

    async def record(self, tenant_id: str, period_key: str, cost_usd: float) -> None:
        """Atomically add ``cost_usd`` to ``tenant_id``'s window total.

        ``cost_usd`` must be non-negative; the framework only records
        ``LLM.cost()`` output, which is ≥ 0. Implementations must raise
        ``ValueError`` when ``cost_usd < 0`` to prevent silent corruption of
        the running total.

        Args:
            tenant_id: Tenant identifier.
            period_key: Time-window key produced by ``period_key()`` for the
                desired calendar bucket.
            cost_usd: Non-negative USD amount to add to the running total.

        Raises:
            ValueError: If ``cost_usd`` is negative.
        """
        ...


class InMemoryCostLedger:
    """Process-local ledger. Default for single-process use and tests."""

    def __init__(self) -> None:
        self._spend: dict[tuple[str, str], float] = {}
        self._lock = asyncio.Lock()

    async def spend(self, tenant_id: str, period_key: str) -> float:
        """Return accumulated USD for ``tenant_id`` in ``period_key`` (0 if absent).

        Args:
            tenant_id: Tenant identifier.
            period_key: Time-window key produced by ``period_key()`` for the
                desired calendar bucket.

        Returns:
            Accumulated spend in USD, or ``0.0`` when no record exists.
        """
        async with self._lock:
            return self._spend.get((tenant_id, period_key), 0.0)

    async def record(self, tenant_id: str, period_key: str, cost_usd: float) -> None:
        """Atomically add ``cost_usd`` to ``tenant_id``'s window total.

        Args:
            tenant_id: Tenant identifier.
            period_key: Time-window key produced by ``period_key()`` for the
                desired calendar bucket.
            cost_usd: Non-negative USD amount to add to the running total.

        Raises:
            ValueError: If ``cost_usd`` is negative.
        """
        if cost_usd < 0:
            raise ValueError(f"cost_usd must be non-negative; got {cost_usd}")
        async with self._lock:
            key = (tenant_id, period_key)
            self._spend[key] = self._spend.get(key, 0.0) + cost_usd
            logger.debug("ledger record tenant=%s key=%s +%.6f", tenant_id, period_key, cost_usd)
