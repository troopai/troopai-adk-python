"""``RedisSwarmCheckpointer`` — fast, TTL-aware swarm-run persistence via Redis.

Each checkpoint is a Redis hash under ``swarm:ckpt:<thread_id>`` holding a
JSON ``payload``, a ``lock_token`` fencing token, and ``updated_at``.
Optimistic locking compares and rotates the token atomically with a Lua
script; a stale token (concurrent writer) raises
:class:`~troopai.adk.exceptions.CheckpointConflictError`.

TTL is opt-in (``ttl_seconds``); the default keeps checkpoints until they
are explicitly deleted. An expired or evicted key reads back as ``None``
— a clean cold-miss — so the caller falls through to a fresh run. Keep the
TTL longer than a run's duration: if a key expires mid-run while a token
is cached, the next ``save`` raises ``CheckpointConflictError`` rather than
silently re-creating the row.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import TYPE_CHECKING, Any, TypedDict, cast

try:
    from redis.asyncio import Redis
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "RedisSwarmCheckpointer requires redis>=5.0: pip install 'troopai-adk-python[checkpointer-redis]'"
    ) from exc

from troopai.adk.exceptions import CheckpointConflictError
from troopai.adk.swarms.checkpointer import SwarmCheckpoint, SwarmHookRegistry

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from troopai.adk.swarms.swarm import Swarm


logger = logging.getLogger(__name__)

_KEY_PREFIX = "swarm:ckpt:"

# Atomic compare-and-set. Writes only if the stored ``lock_token`` equals
# the caller's expected token (empty string = "expect the key absent", for
# a first write). Returns the new token on success, or "CONFLICT".
_CAS_SCRIPT = """
local cur = redis.call('HGET', KEYS[1], 'lock_token')
if cur == false then
  if ARGV[1] ~= '' then return 'CONFLICT' end
elseif cur ~= ARGV[1] then
  return 'CONFLICT'
end
redis.call('HSET', KEYS[1], 'payload', ARGV[2], 'lock_token', ARGV[3], 'updated_at', ARGV[4])
if tonumber(ARGV[5]) > 0 then
  redis.call('PEXPIRE', KEYS[1], ARGV[5])
end
return ARGV[3]
"""


def _to_str(value: bytes | str) -> str:
    """Decode a Redis reply field to ``str`` (clients may return bytes)."""
    return value.decode() if isinstance(value, bytes) else value


class _CheckpointEnvelope(TypedDict):
    """JSON shape stored in the Redis hash ``payload`` field."""

    turn: int
    state: dict[str, Any]


class RedisSwarmCheckpointer:
    """Fast, TTL-aware swarm checkpointer backed by Redis hashes.

    Each ``thread_id`` maps to one hash with a JSON ``payload``, a
    ``lock_token`` fencing token, and ``updated_at``. ``save`` performs an
    atomic Lua compare-and-set against the token cached from the prior
    ``load`` or ``save``; a concurrent writer that rotates the token
    causes the losing ``save`` to raise
    :class:`~troopai.adk.exceptions.CheckpointConflictError`.

    Supply either a configured ``client`` or a ``url``. The caller owns
    the client lifecycle when ``client=`` is used.

    """

    def __init__(
        self,
        *,
        client: Redis | None = None,
        url: str | None = None,
        ttl_seconds: float | None = None,
        thread_id: str = "default",
    ) -> None:
        """Initialise the checkpointer with a client or a URL.

        Args:
            client: A pre-configured ``redis.asyncio.Redis`` (e.g. for tests
                or a shared pool). Mutually exclusive with ``url``.
            url: A Redis URL (``redis://host:port/db``) used to build a client
                when ``client`` is not supplied.
            ttl_seconds: Optional per-key expiry in seconds. ``None`` (the
                default) keeps checkpoints until explicitly deleted; the
                developer opts in to eviction.
            thread_id: Identifier used by :meth:`register`'s auto-save hook.
                Defaults to ``"default"`` when the caller does not supply an
                explicit id.

        Raises:
            ValueError: When both ``client`` and ``url`` are supplied, when
                neither is supplied, or when ``ttl_seconds`` is not positive.
        """
        if client is not None and url is not None:
            raise ValueError("RedisSwarmCheckpointer accepts client= or url=, not both.")
        if ttl_seconds is not None and ttl_seconds <= 0:
            raise ValueError(f"ttl_seconds must be positive or None; got {ttl_seconds!r}.")
        if client is not None:
            self._client: Redis = client
            self._owns_client = False
        elif url is not None:
            self._client = Redis.from_url(url)
            self._owns_client = True
        else:
            raise ValueError("RedisSwarmCheckpointer requires either client= or url=.")
        self._ttl_ms: int = int(ttl_seconds * 1000) if ttl_seconds is not None else 0
        self._thread_id = thread_id
        self._tokens: dict[str, str] = {}
        self._cas = self._client.register_script(_CAS_SCRIPT)
        logger.debug("RedisSwarmCheckpointer initialised (ttl_ms=%d).", self._ttl_ms)

    async def close(self) -> None:
        """Close the client if this checkpointer created it (``url=`` path).

        A caller-supplied ``client=`` is left open — the caller owns its
        lifecycle. Idempotent and safe in a ``finally``.
        """
        if self._owns_client:
            self._owns_client = False
            await self._client.aclose()
            logger.debug("RedisSwarmCheckpointer: client closed.")

    def register(self, registry: SwarmHookRegistry) -> None:
        """Subscribe a :class:`SwarmCheckpointerHooks` to ``registry``."""
        from troopai.adk.swarms.checkpointers.hooks import SwarmCheckpointerHooks

        registry.add(SwarmCheckpointerHooks(self, self._thread_id))
        logger.debug("RedisSwarmCheckpointer registered on SwarmHookRegistry.")

    async def save(self, checkpoint: SwarmCheckpoint) -> None:
        """Upsert ``checkpoint`` via an atomic compare-and-set.

        The first save for a ``thread_id`` (no cached token) requires the
        key to be absent; subsequent saves require the stored token to
        match the one cached from the prior ``load`` / ``save``. A losing
        race raises :class:`CheckpointConflictError`.

        Args:
            checkpoint: The snapshot to persist.

        Raises:
            CheckpointConflictError: When a concurrent writer has rotated
                the fencing token since this instance last observed it.
        """
        key = _KEY_PREFIX + checkpoint.thread_id
        payload = json.dumps(
            {
                "turn": checkpoint.turn,
                "state": checkpoint.state,
            },
            separators=(",", ":"),
        )
        new_token = str(uuid.uuid4())
        expected = self._tokens.get(checkpoint.thread_id, "")
        result = await self._cas(
            keys=[key],
            args=[expected, payload, new_token, str(time.time()), str(self._ttl_ms)],
        )
        if _to_str(result) != new_token:
            # The script returns the new token on success or "CONFLICT";
            # any other value also means the write did not land.
            raise CheckpointConflictError(checkpoint.thread_id)
        self._tokens[checkpoint.thread_id] = new_token
        logger.debug(
            "RedisSwarmCheckpointer.save: thread_id=%s turn=%s",
            checkpoint.thread_id,
            checkpoint.turn,
        )

    async def load(
        self,
        thread_id: str,
        swarm: Swarm[Any],
    ) -> SwarmCheckpoint | None:
        """Rehydrate the checkpoint for ``thread_id`` (``None`` if absent).

        A missing, expired, or evicted key returns ``None``. The observed
        ``lock_token`` is cached so a subsequent :meth:`save` can verify it.
        The ``swarm`` parameter is accepted for protocol parity; member-name
        resolution in :meth:`SwarmState.from_dict` is the de-facto integrity
        check at rehydration time.

        Args:
            thread_id: The logical run key.
            swarm: The :class:`Swarm` the checkpoint belongs to. Accepted
                for protocol parity; member validation happens at
                :meth:`SwarmState.from_dict` call time.

        Returns:
            A :class:`SwarmCheckpoint`, or ``None`` when no live
            checkpoint exists for ``thread_id``.
        """
        del swarm
        key = _KEY_PREFIX + thread_id
        # redis-py types hgetall as a shared sync/async union (ResponseT);
        # the async client always returns the awaitable at runtime.
        raw = await cast("Awaitable[dict[bytes, bytes]]", self._client.hgetall(key))
        if len(raw) == 0:
            # Key absent/expired/evicted — drop any cached token so the next
            # save() starts a clean insert rather than a false conflict.
            self._tokens.pop(thread_id, None)
            logger.debug("RedisSwarmCheckpointer.load: no checkpoint for thread_id=%s", thread_id)
            return None
        data = {_to_str(field): _to_str(value) for field, value in raw.items()}
        envelope: _CheckpointEnvelope = json.loads(data["payload"])
        self._tokens[thread_id] = data["lock_token"]
        logger.debug("RedisSwarmCheckpointer.load: thread_id=%s turn=%s", thread_id, envelope["turn"])
        return SwarmCheckpoint(thread_id=thread_id, state=envelope["state"], turn=envelope["turn"])

    async def list_checkpoints(self) -> list[str]:
        """Return a sorted list of thread ids currently stored."""
        out: list[str] = []
        async for key in self._client.scan_iter(match=_KEY_PREFIX + "*"):
            out.append(_to_str(key)[len(_KEY_PREFIX) :])
        return sorted(out)

    async def delete(self, thread_id: str) -> None:
        """Delete the checkpoint for ``thread_id`` (no-op if absent)."""
        # Returns the count of removed keys; unused.
        await self._client.delete(_KEY_PREFIX + thread_id)
        self._tokens.pop(thread_id, None)
        logger.debug("RedisSwarmCheckpointer.delete: thread_id=%s", thread_id)


__all__ = ["RedisSwarmCheckpointer"]
