"""Buffered session wrapper — write-coalescing decorator for SessionStore.

``BufferedSession`` wraps any inner :class:`~troopai.adk.types.session.SessionStore`
and accumulates :class:`~troopai.adk.session.session_event.SessionEvent` objects in
memory.  Buffered events are written to the inner store in **one batch** on an
explicit :meth:`flush` (or implicitly on :meth:`close`).

Design decisions
----------------
* **Events are buffered; state is not.**  :meth:`save_state` is delegated
  immediately to the inner store.  Buffering state would require either
  replaying mutations in order (complex, error-prone) or taking a full
  snapshot on each :meth:`save_state` call.  The simpler, explicit contract is:
  state writes are always durable; event writes are coalesced.

* **Consistent ``get()``.**  :meth:`get` merges persisted events (from the inner
  store) with the in-memory buffered tail so that the view is identical to what
  an unwrapped store would return after all events had been flushed.

* **Bounded buffer with auto-flush.**  When ``len(buffer) >= max_buffered_events``
  after an :meth:`add` call, :meth:`flush` is triggered automatically.  The
  default (``100``) is conservative enough for typical agent turns while
  preventing unbounded memory growth.

* **Crash semantics.**  Buffered events that have not been flushed are **lost**
  if the process terminates unexpectedly.  Call :meth:`flush` (or :meth:`close`)
  to persist all pending events before releasing the session handle.

Usage::

    inner = SQLiteMultiSessions()
    session = await inner.create("user-123")
    buffered = BufferedSession(session, max_buffered_events=50)
    try:
        # ... agent turns add events ...
        await buffered.flush()  # single-batch write to SQLite
    finally:
        await buffered.close()  # flushes any remaining events, then closes inner
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from troopai.adk.context.context_editing import ContextEditor

if TYPE_CHECKING:
    from troopai.adk.session.session_event import SessionEvent
    from troopai.adk.session.session_settings import SessionSettings
    from troopai.adk.types.session import SessionStore

logger = logging.getLogger(__name__)


def _drop_orphaned_tool_result_events(events: list[SessionEvent]) -> list[SessionEvent]:
    """Drop events whose content is a tool result with no in-window tool call.

    Even when the inner store returns a clean window, re-slicing the merged
    ``persisted + buffer`` list to the final limit can begin the window in
    the middle of a tool-call / tool-result exchange, orphaning the result
    (which Anthropic / Gemini reject with a 400).

    :meth:`ContextEditor.remove_orphaned_tool_results` operates on Layer-1
    content items and returns the very objects it keeps, so surviving events
    are recovered by content identity.

    Args:
        events: The windowed events, oldest first.

    Returns:
        The events whose content survives orphan removal, order preserved.
    """
    kept = ContextEditor.remove_orphaned_tool_results([event.content for event in events])
    kept_ids = {id(content) for content in kept}
    return [event for event in events if id(event.content) in kept_ids]


# Conservative default: large enough not to trigger auto-flush on typical
# agent runs (a few dozen events per session), small enough to guard against
# runaway loops that never flush.
_DEFAULT_MAX_BUFFERED_EVENTS: int = 100


class BufferedSession:
    """Write-coalescing wrapper around any :class:`~troopai.adk.types.session.SessionStore`.

    Accumulates :class:`~troopai.adk.session.session_event.SessionEvent` objects
    in memory via :meth:`add` and writes them to the inner store in a single
    batch when :meth:`flush` (or :meth:`close`) is called.

    :meth:`get` returns a **consistent** view that merges persisted events with
    the buffered tail, so reads behave identically whether or not the buffer has
    been flushed yet.

    .. warning::

        **Crash semantics**: buffered events that have not yet been flushed are
        **lost** if the process terminates abnormally.  Always call :meth:`flush`
        or :meth:`close` before releasing the session handle.

    Attributes:
        _inner: The wrapped :class:`~troopai.adk.types.session.SessionStore`.
        _buffer: In-memory list of unflushed events (oldest first).
        _max_buffered_events: Auto-flush threshold.  When the buffer length
            reaches this value after an :meth:`add`, :meth:`flush` is called
            automatically.
    """

    def __init__(
        self,
        inner: SessionStore,
        *,
        max_buffered_events: int = _DEFAULT_MAX_BUFFERED_EVENTS,
    ) -> None:
        """Wrap ``inner`` in a buffering layer.

        Args:
            inner: Any object satisfying the
                :class:`~troopai.adk.types.session.SessionStore` protocol.
            max_buffered_events: Auto-flush threshold.  When the number of
                pending (unflushed) events reaches this value after a call to
                :meth:`add`, :meth:`flush` is invoked automatically to keep
                memory growth bounded.  Must be a positive integer.
                Default is ``100``.

        Raises:
            ValueError: If ``max_buffered_events`` is less than 1.
        """
        if max_buffered_events < 1:
            raise ValueError(f"max_buffered_events must be >= 1, got {max_buffered_events}")
        self._inner = inner
        self._buffer: list[SessionEvent] = []
        self._max_buffered_events = max_buffered_events

    # ------------------------------------------------------------------
    # SessionStore protocol — identity delegation
    # ------------------------------------------------------------------

    @property
    def id(self) -> str:
        """Unique identifier for this session handle (delegated to inner)."""
        return self._inner.id

    @property
    def settings(self) -> SessionSettings | None:
        """Per-session configuration (delegated to inner)."""
        return self._inner.settings

    # ------------------------------------------------------------------
    # SessionStore protocol — core methods
    # ------------------------------------------------------------------

    async def get(self, limit: int | None = None) -> list[SessionEvent]:
        """Return a consistent view: persisted events merged with the buffered tail.

        The result is identical to what calling :meth:`flush` and then
        ``inner.get(limit)`` would return, without actually flushing.

        When a ``limit`` applies (from the argument or from
        ``settings.limit``), the N most-recent events across both the inner
        store and the buffer are returned, ordered oldest-first within that
        slice.

        Args:
            limit: Maximum number of events to return.  ``None`` falls back
                to ``settings.limit``, then returns all events.

        Returns:
            Combined list of :class:`~troopai.adk.session.session_event.SessionEvent`
            in chronological order (oldest first).
        """
        effective_limit = limit
        if effective_limit is None and self._inner.settings is not None:
            effective_limit = self._inner.settings.limit

        # Fetch persisted events bounded by the same effective limit we apply
        # after merging.  Passing ``effective_limit`` explicitly (instead of
        # ``None``) prevents the inner store's own ``settings.limit`` fallback
        # from truncating below what an explicit, larger ``limit`` asked for.
        # The inner returns the N most-recent persisted events oldest-first, so
        # the merge still yields the N most-recent across the boundary.
        persisted = await self._inner.get(limit=effective_limit)
        combined = persisted + self._buffer

        if effective_limit is not None:
            return _drop_orphaned_tool_result_events(combined[-effective_limit:])
        return combined

    async def add(self, events: list[SessionEvent]) -> None:
        """Accumulate events in the in-memory buffer.

        If the buffer length reaches ``max_buffered_events`` after this call,
        :meth:`flush` is triggered automatically to keep memory bounded.

        Args:
            events: Events to buffer.  An empty list is a no-op.
        """
        if not events:
            return

        self._buffer.extend(events)

        if len(self._buffer) >= self._max_buffered_events:
            logger.debug(
                "BufferedSession(id=%s): auto-flush triggered at %d buffered events",
                self._inner.id,
                len(self._buffer),
            )
            await self.flush()

    async def save_state(self) -> None:
        """Flush pending state changes to the inner store immediately.

        State is **not** buffered — it is delegated synchronously to the inner
        store so that state mutations are durable regardless of whether
        :meth:`flush` has been called.
        """
        await self._inner.save_state()

    async def close(self) -> None:
        """Flush any remaining buffered events, then close the inner store.

        After :meth:`close` is called, any buffered events that were not
        previously flushed are written to the inner store.  The inner store's
        :meth:`close` is then called to release backend resources.
        """
        await self.flush()
        await self._inner.close()

    # ------------------------------------------------------------------
    # Buffered-session-specific API
    # ------------------------------------------------------------------

    async def flush(self) -> None:
        """Write all buffered events to the inner store in a single batch.

        After a successful flush the in-memory buffer is cleared.  If the inner
        store's :meth:`add` raises, the buffer is **not** cleared so the events
        can be retried or inspected.
        """
        if not self._buffer:
            return

        batch = list(self._buffer)
        await self._inner.add(batch)
        # Drop only the snapshotted prefix — events appended by another
        # coroutine while the inner add() was suspended stay buffered.
        del self._buffer[: len(batch)]
        logger.debug(
            "BufferedSession(id=%s): flushed %d events to inner store",
            self._inner.id,
            len(batch),
        )

    @property
    def pending_count(self) -> int:
        """Number of events currently held in the buffer (not yet flushed).

        Returns:
            Count of unflushed :class:`~troopai.adk.session.session_event.SessionEvent`
            objects.
        """
        return len(self._buffer)
