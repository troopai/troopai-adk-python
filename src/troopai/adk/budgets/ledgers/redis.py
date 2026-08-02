"""Redis-backed CostLedger. Atomic INCRBYFLOAT per window, optional TTL.

Redis docs: https://redis.io/commands/incrbyfloat/
Redis EXPIRE (NX flag, requires server 7.0+): https://redis.io/commands/expire/
"""

from __future__ import annotations

import logging
from urllib.parse import quote

try:
    from redis.asyncio import Redis
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "RedisCostLedger requires redis>=5.0: pip install 'troopai-adk-python[cost-ledger-redis]'"
    ) from exc

logger = logging.getLogger(__name__)

_KEY_PREFIX = "cost:ledger:"


class RedisCostLedger:
    """Fast ephemeral ledger. Stale windows self-evict when ``ttl_seconds`` set.

    Each ``(tenant_id, period_key)`` pair maps to one Redis key. ``record``
    performs an atomic ``INCRBYFLOAT`` so no optimistic-locking token is
    needed.

    **TTL semantics.** The TTL is a self-eviction hygiene mechanism — accounting
    correctness comes from the distinct ``period_key`` per calendar bucket, not
    from the TTL. Set ``ttl_seconds`` to a value at least as long as your budget
    period so the key outlives the window it guards. The TTL is stamped once on
    bucket creation (``EXPIRE … NX``) and never slid forward on subsequent
    writes, which requires Redis server 7.0+ (the ``redis>=5.0`` client supports
    the NX flag). ``None`` (the default) keeps keys until explicitly deleted —
    the cost-conservative default; the developer opts in to eviction.

    Supply either a configured ``client`` or a ``url``. When ``url`` is
    supplied this instance creates and owns the underlying client; call
    :meth:`close` to release it. A caller-supplied ``client`` is left open
    — the caller owns its lifecycle.
    """

    def __init__(
        self,
        *,
        client: Redis | None = None,
        url: str | None = None,
        ttl_seconds: int | None = None,
    ) -> None:
        """Create a ledger backed by a Redis connection.

        Args:
            client: A pre-configured ``redis.asyncio.Redis`` (e.g. for tests
                or a shared pool). Mutually exclusive with ``url``.
            url: A Redis URL (``redis://host:port/db``) used to build a client
                when ``client`` is not supplied.
            ttl_seconds: Optional per-key expiry in seconds. ``None`` keeps
                keys until explicitly deleted.

        Raises:
            ValueError: If both ``client`` and ``url`` are supplied, if neither
                is supplied, or if ``ttl_seconds`` is not positive.
        """
        if client is not None and url is not None:
            raise ValueError("RedisCostLedger accepts client= or url=, not both.")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive or None; got {ttl_seconds!r}.")
        if client is not None:
            self._client: Redis = client
            self._owns_client = False
        elif url is not None:
            self._client = Redis.from_url(url)
            self._owns_client = True
        else:
            raise ValueError("RedisCostLedger requires either client= or url=.")
        self._ttl = ttl_seconds
        logger.debug("RedisCostLedger initialised (ttl_seconds=%r).", self._ttl)

    def _key(self, tenant_id: str, period_key: str) -> str:
        # Encode tenant_id so a ':' in it cannot collide across the period
        # separator: f"...{tenant_id}:{period_key}" is not injective — tenant
        # 'a:b' + period 'c' would otherwise share a bucket with tenant 'a' +
        # period 'b:c' (cross-tenant budget bleed). quote(safe="") leaves
        # ordinary ids unchanged, so existing keys are preserved.
        return f"{_KEY_PREFIX}{quote(tenant_id, safe='')}:{period_key}"

    async def spend(self, tenant_id: str, period_key: str) -> float:
        """Return accumulated USD for ``tenant_id`` in ``period_key`` (0 if absent).

        Args:
            tenant_id: Tenant identifier.
            period_key: Time-window key (e.g. ``"2026-05-01"`` for a DAY bucket).

        Returns:
            Accumulated spend in USD, or ``0.0`` when no record exists.
        """
        raw = await self._client.get(self._key(tenant_id, period_key))
        return float(raw) if raw is not None else 0.0

    async def record(self, tenant_id: str, period_key: str, cost_usd: float) -> None:
        """Atomically add ``cost_usd`` to ``tenant_id``'s window total.

        Args:
            tenant_id: Tenant identifier.
            period_key: Time-window key (e.g. ``"2026-05-01"`` for a DAY bucket).
            cost_usd: Non-negative USD amount to add.

        Raises:
            ValueError: If ``cost_usd`` is negative.
        """
        if cost_usd < 0:
            raise ValueError(f"cost_usd must be non-negative; got {cost_usd}")
        key = self._key(tenant_id, period_key)
        async with self._client.pipeline(transaction=False) as pipe:
            pipe.incrbyfloat(key, cost_usd)
            if self._ttl is not None:
                pipe.expire(key, self._ttl, nx=True)
            await pipe.execute()
        logger.debug("redis ledger record tenant=%s key=%s +%.6f", tenant_id, period_key, cost_usd)

    async def close(self) -> None:
        """Close the client if this instance created it (``url=`` path).

        A caller-supplied ``client=`` is left open — the caller owns its
        lifecycle. Idempotent and safe in a ``finally``.
        """
        if self._owns_client:
            self._owns_client = False
            await self._client.aclose()
            logger.debug("RedisCostLedger: client closed.")


__all__ = ["RedisCostLedger"]
