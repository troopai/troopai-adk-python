"""Tests for BufferedSession — write-coalescing wrapper for SessionStore.

Covers:
- Buffered events are held in memory until flush().
- flush() writes all pending events to the inner store in a single batch.
- get() returns a CONSISTENT view: inner events + buffered tail (no flush needed).
- Auto-flush triggers when buffer reaches max_buffered_events.
- close() flushes remaining events then closes the inner store.
- save_state() delegates immediately to the inner store.
- Crash semantics: buffer is NOT cleared when flush fails (events survive for retry).
- max_buffered_events=0 raises ValueError at construction.
- pending_count reflects unflushed event count.
"""

from __future__ import annotations

import pytest

from troopai.adk.session.buffered_session import BufferedSession
from troopai.adk.session.session_event import SessionEvent, create_session_event
from troopai.adk.session.session_settings import SessionSettings

# ---------------------------------------------------------------------------
# In-memory inner store test double
# ---------------------------------------------------------------------------


class _InnerStore:
    """Minimal in-memory SessionStore double that records calls."""

    def __init__(self, session_id: str = "inner-session") -> None:
        self._session_id = session_id
        self._settings: SessionSettings | None = None
        self._events: list[SessionEvent] = []
        self.add_calls: list[list[SessionEvent]] = []
        self.save_state_calls: int = 0
        self.close_calls: int = 0
        self._add_should_raise: Exception | None = None

    @property
    def id(self) -> str:
        return self._session_id

    @property
    def settings(self) -> SessionSettings | None:
        return self._settings

    async def get(self, limit: int | None = None) -> list[SessionEvent]:
        if limit is not None:
            return self._events[-limit:]
        return list(self._events)

    async def add(self, events: list[SessionEvent]) -> None:
        if self._add_should_raise is not None:
            raise self._add_should_raise
        self.add_calls.append(list(events))
        self._events.extend(events)

    async def save_state(self) -> None:
        self.save_state_calls += 1

    async def close(self) -> None:
        self.close_calls += 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _event(text: str, author: str = "user") -> SessionEvent:
    return create_session_event(author=author, content={"role": author, "content": text})


def _tool_call_event(call_id: str) -> SessionEvent:
    return create_session_event(
        author="assistant",
        content={"type": "function_call", "call_id": call_id, "name": "t", "arguments": "{}"},
    )


def _tool_result_event(call_id: str) -> SessionEvent:
    return create_session_event(
        author="tool",
        content={"type": "function_call_output", "call_id": call_id, "output": "r"},
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestWindowedOrphanToolResult:
    """A limited get() whose final persisted+buffer slice cuts a tool pair must
    not return the orphaned tool result (Anthropic/Gemini reject it with 400).
    """

    @pytest.mark.asyncio
    async def test_boundary_orphan_result_is_dropped(self) -> None:
        inner = _InnerStore()
        # Persisted: a user turn then a tool call whose result is still buffered.
        inner._events = [_event("q"), _tool_call_event("c1")]
        buf = BufferedSession(inner)
        await buf.add([_tool_result_event("c1"), _event("next")])

        # get(limit=2): combined tail is [result(c1), next]; the tool call fell
        # outside the 2-window, orphaning result(c1) at the boundary.
        events = await buf.get(limit=2)
        assert all(e.content.get("type") != "function_call_output" for e in events)
        assert [e.content.get("content") for e in events] == ["next"]

    @pytest.mark.asyncio
    async def test_unbounded_get_keeps_everything(self) -> None:
        inner = _InnerStore()
        inner._events = [_tool_result_event("dangling")]
        buf = BufferedSession(inner)
        await buf.add([_event("q")])
        events = await buf.get()  # no limit → no windowing → no stripping
        assert len(events) == 2


class TestConstruction:
    def test_defaults_accepted(self) -> None:
        inner = _InnerStore()
        buf = BufferedSession(inner)
        assert buf.id == "inner-session"
        assert buf.settings is None
        assert buf.pending_count == 0

    def test_zero_max_buffered_raises(self) -> None:
        with pytest.raises(ValueError, match="max_buffered_events must be >= 1"):
            BufferedSession(_InnerStore(), max_buffered_events=0)

    def test_negative_max_buffered_raises(self) -> None:
        with pytest.raises(ValueError, match="max_buffered_events must be >= 1"):
            BufferedSession(_InnerStore(), max_buffered_events=-5)

    def test_settings_delegated(self) -> None:
        inner = _InnerStore()
        inner._settings = SessionSettings(limit=10)
        buf = BufferedSession(inner)
        assert buf.settings is inner._settings


# ---------------------------------------------------------------------------
# add() and pending_count
# ---------------------------------------------------------------------------


class TestAddAndPendingCount:
    @pytest.mark.asyncio
    async def test_add_accumulates_in_buffer(self) -> None:
        inner = _InnerStore()
        buf = BufferedSession(inner)
        await buf.add([_event("hello")])
        assert buf.pending_count == 1
        assert len(inner._events) == 0  # not yet persisted

    @pytest.mark.asyncio
    async def test_add_empty_list_is_noop(self) -> None:
        inner = _InnerStore()
        buf = BufferedSession(inner)
        await buf.add([])
        assert buf.pending_count == 0
        assert len(inner.add_calls) == 0

    @pytest.mark.asyncio
    async def test_multiple_adds_accumulate(self) -> None:
        inner = _InnerStore()
        buf = BufferedSession(inner)
        await buf.add([_event("a"), _event("b")])
        await buf.add([_event("c")])
        assert buf.pending_count == 3
        assert len(inner._events) == 0  # nothing flushed yet


# ---------------------------------------------------------------------------
# get() — consistent view
# ---------------------------------------------------------------------------


class TestGetConsistency:
    @pytest.mark.asyncio
    async def test_get_returns_inner_plus_buffer(self) -> None:
        """get() must return persisted events followed by buffered tail."""
        inner = _InnerStore()
        inner._events = [_event("persisted-1"), _event("persisted-2")]

        buf = BufferedSession(inner)
        await buf.add([_event("buffered-1")])

        events = await buf.get()
        assert len(events) == 3
        assert events[0].content["content"] == "persisted-1"
        assert events[1].content["content"] == "persisted-2"
        assert events[2].content["content"] == "buffered-1"

    @pytest.mark.asyncio
    async def test_get_with_limit_slices_combined_tail(self) -> None:
        """get(limit=N) returns the N most-recent events across both sources."""
        inner = _InnerStore()
        inner._events = [_event(f"p{i}") for i in range(5)]

        buf = BufferedSession(inner)
        await buf.add([_event("b0"), _event("b1")])

        events = await buf.get(limit=3)
        assert len(events) == 3
        # Most-recent 3 from [p0,p1,p2,p3,p4,b0,b1] are [p4,b0,b1]
        assert events[0].content["content"] == "p4"
        assert events[1].content["content"] == "b0"
        assert events[2].content["content"] == "b1"

    @pytest.mark.asyncio
    async def test_get_honours_settings_limit(self) -> None:
        """settings.limit applies when no explicit limit is passed."""
        inner = _InnerStore()
        inner._settings = SessionSettings(limit=2)
        inner._events = [_event(f"p{i}") for i in range(4)]

        buf = BufferedSession(inner)
        await buf.add([_event("b0")])

        events = await buf.get()
        # Combined length=5, limit=2 → last two
        assert len(events) == 2
        assert events[0].content["content"] == "p3"
        assert events[1].content["content"] == "b0"

    @pytest.mark.asyncio
    async def test_get_no_buffer_matches_inner_get(self) -> None:
        """With an empty buffer, get() is identical to inner.get()."""
        inner = _InnerStore()
        inner._events = [_event("x"), _event("y")]
        buf = BufferedSession(inner)
        assert await buf.get() == await inner.get()

    @pytest.mark.asyncio
    async def test_get_empty_both_returns_empty(self) -> None:
        inner = _InnerStore()
        buf = BufferedSession(inner)
        assert await buf.get() == []


# ---------------------------------------------------------------------------
# flush() — single-batch write
# ---------------------------------------------------------------------------


class TestFlush:
    @pytest.mark.asyncio
    async def test_flush_writes_all_buffered_events_in_one_call(self) -> None:
        """flush() must call inner.add() exactly ONCE with all buffered events."""
        inner = _InnerStore()
        buf = BufferedSession(inner)
        await buf.add([_event("a"), _event("b")])
        await buf.add([_event("c")])

        await buf.flush()

        assert len(inner.add_calls) == 1
        assert len(inner.add_calls[0]) == 3
        assert inner.add_calls[0][0].content["content"] == "a"

    @pytest.mark.asyncio
    async def test_flush_clears_buffer(self) -> None:
        inner = _InnerStore()
        buf = BufferedSession(inner)
        await buf.add([_event("x")])
        await buf.flush()
        assert buf.pending_count == 0

    @pytest.mark.asyncio
    async def test_flush_empty_buffer_is_noop(self) -> None:
        inner = _InnerStore()
        buf = BufferedSession(inner)
        await buf.flush()
        assert len(inner.add_calls) == 0

    @pytest.mark.asyncio
    async def test_flush_failure_preserves_buffer(self) -> None:
        """When inner.add() raises, the buffer must NOT be cleared.

        This allows the caller to retry the flush or inspect the lost events —
        consistent with the crash-semantics contract documented in the class.
        """
        inner = _InnerStore()
        inner._add_should_raise = RuntimeError("DB unavailable")
        buf = BufferedSession(inner)
        await buf.add([_event("important")])

        with pytest.raises(RuntimeError, match="DB unavailable"):
            await buf.flush()

        assert buf.pending_count == 1  # buffer is preserved for retry

    @pytest.mark.asyncio
    async def test_flush_then_get_shows_no_buffer(self) -> None:
        """After flush, get() returns only inner events (buffer is empty)."""
        inner = _InnerStore()
        buf = BufferedSession(inner)
        await buf.add([_event("msg")])
        await buf.flush()

        # Now the inner store has the event; buffer is empty.
        events = await buf.get()
        assert len(events) == 1
        assert events[0].content["content"] == "msg"


# ---------------------------------------------------------------------------
# Auto-flush at max_buffered_events
# ---------------------------------------------------------------------------


class TestAutoFlush:
    @pytest.mark.asyncio
    async def test_auto_flush_triggers_at_threshold(self) -> None:
        """When pending events reach max_buffered_events, flush fires automatically."""
        inner = _InnerStore()
        buf = BufferedSession(inner, max_buffered_events=3)

        await buf.add([_event("a"), _event("b")])
        assert len(inner.add_calls) == 0  # not yet

        await buf.add([_event("c")])  # reaches threshold → auto-flush
        assert len(inner.add_calls) == 1
        assert len(inner.add_calls[0]) == 3
        assert buf.pending_count == 0

    @pytest.mark.asyncio
    async def test_auto_flush_batch_contains_all_pending(self) -> None:
        """Auto-flush must write ALL pending events in one call, not just the new ones."""
        inner = _InnerStore()
        buf = BufferedSession(inner, max_buffered_events=2)

        await buf.add([_event("first")])
        await buf.add([_event("second")])  # threshold hit

        assert len(inner.add_calls) == 1
        contents = [e.content["content"] for e in inner.add_calls[0]]
        assert contents == ["first", "second"]

    @pytest.mark.asyncio
    async def test_no_auto_flush_below_threshold(self) -> None:
        inner = _InnerStore()
        buf = BufferedSession(inner, max_buffered_events=10)
        for i in range(9):
            await buf.add([_event(f"msg{i}")])
        assert len(inner.add_calls) == 0
        assert buf.pending_count == 9


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


class TestClose:
    @pytest.mark.asyncio
    async def test_close_flushes_remaining_events(self) -> None:
        inner = _InnerStore()
        buf = BufferedSession(inner)
        await buf.add([_event("pending")])

        await buf.close()

        assert buf.pending_count == 0
        assert len(inner._events) == 1

    @pytest.mark.asyncio
    async def test_close_calls_inner_close(self) -> None:
        inner = _InnerStore()
        buf = BufferedSession(inner)
        await buf.close()
        assert inner.close_calls == 1

    @pytest.mark.asyncio
    async def test_close_flushes_then_closes(self) -> None:
        """Flush must happen before inner.close() is called."""
        events_at_close_time: list[int] = []
        original_close = _InnerStore.close

        inner = _InnerStore()

        async def _recording_close(self: _InnerStore) -> None:
            events_at_close_time.append(len(self._events))
            await original_close(self)

        import types

        inner.close = types.MethodType(_recording_close, inner)  # type: ignore[method-assign]

        buf = BufferedSession(inner)
        await buf.add([_event("a"), _event("b")])
        await buf.close()

        # inner store must have 2 events WHEN close() is called
        assert events_at_close_time == [2]


# ---------------------------------------------------------------------------
# save_state() — immediate delegation
# ---------------------------------------------------------------------------


class TestSaveState:
    @pytest.mark.asyncio
    async def test_save_state_delegates_to_inner_immediately(self) -> None:
        inner = _InnerStore()
        buf = BufferedSession(inner)
        # Buffer some events without flushing
        await buf.add([_event("not-yet-flushed")])

        await buf.save_state()

        assert inner.save_state_calls == 1
        # Buffer is untouched
        assert buf.pending_count == 1

    @pytest.mark.asyncio
    async def test_save_state_multiple_calls_each_delegate(self) -> None:
        inner = _InnerStore()
        buf = BufferedSession(inner)
        await buf.save_state()
        await buf.save_state()
        assert inner.save_state_calls == 2


# ---------------------------------------------------------------------------
# SessionStore protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_session_store_protocol(self) -> None:
        from troopai.adk.types.session import SessionStore

        inner = _InnerStore()
        buf = BufferedSession(inner)
        assert isinstance(buf, SessionStore)


# ---------------------------------------------------------------------------
# get() must not be truncated by the inner store's own settings.limit fallback
# ---------------------------------------------------------------------------


class _SettingsAwareInnerStore(_InnerStore):
    """Inner double that mirrors SQLiteSession: a ``limit=None`` ``get()`` falls
    back to ``settings.limit`` and truncates to the N most-recent events.

    The plain :class:`_InnerStore` returns everything on ``limit=None``, so it
    cannot reproduce the truncation seen against a real settings-bound store.
    """

    async def get(self, limit: int | None = None) -> list[SessionEvent]:
        effective_limit = limit
        if effective_limit is None and self._settings is not None:
            effective_limit = self._settings.limit
        if effective_limit is not None:
            return self._events[-effective_limit:]
        return list(self._events)


class TestGetBypassesInnerSettingsTruncation:
    @pytest.mark.asyncio
    async def test_explicit_limit_larger_than_settings_surfaces_all_requested(self) -> None:
        """get(limit=N) with N > settings.limit must surface up to N events.

        Regression: BufferedSession previously called inner.get() with no
        limit, so a settings-bound inner silently truncated persisted events
        to settings.limit before the merge could apply the larger N.
        """
        inner = _SettingsAwareInnerStore()
        inner._settings = SessionSettings(limit=10)
        inner._events = [_event(f"p{i}") for i in range(100)]

        buf = BufferedSession(inner)  # buffer empty
        events = await buf.get(limit=50)

        # The caller asked for up to 50 and there are 100 persisted; the inner
        # settings.limit=10 must NOT cap the result below the explicit request.
        assert len(events) == 50
        assert events[0].content["content"] == "p50"
        assert events[-1].content["content"] == "p99"

    @pytest.mark.asyncio
    async def test_explicit_limit_with_buffer_surfaces_most_recent_across_boundary(self) -> None:
        """The N most-recent across persisted + buffer survive a settings-bound inner."""
        inner = _SettingsAwareInnerStore()
        inner._settings = SessionSettings(limit=10)
        inner._events = [_event(f"p{i}") for i in range(100)]

        buf = BufferedSession(inner)
        await buf.add([_event("b0"), _event("b1")])

        events = await buf.get(limit=5)
        contents = [e.content["content"] for e in events]
        # Most-recent 5 of [p0..p99, b0, b1] are [p97, p98, p99, b0, b1].
        assert contents == ["p97", "p98", "p99", "b0", "b1"]

    @pytest.mark.asyncio
    async def test_no_explicit_limit_still_honours_settings_limit(self) -> None:
        """The common path (limit arg None) is unchanged: settings.limit applies."""
        inner = _SettingsAwareInnerStore()
        inner._settings = SessionSettings(limit=2)
        inner._events = [_event(f"p{i}") for i in range(4)]

        buf = BufferedSession(inner)
        await buf.add([_event("b0")])

        events = await buf.get()
        contents = [e.content["content"] for e in events]
        # Combined effective limit = 2 → last two of [p0..p3, b0].
        assert contents == ["p3", "b0"]


async def test_flush_preserves_events_added_during_inner_add() -> None:
    """Events appended while the inner add() is suspended must survive.

    A clear() after the await would silently destroy them — neither
    persisted nor buffered.
    """
    inner = _InnerStore()
    buffered = BufferedSession(inner=inner, max_buffered_events=100)
    await buffered.add([_event("first")])

    original_add = inner.add

    async def add_with_reentry(events: list[SessionEvent]) -> None:
        await original_add(events)
        # Simulate a concurrent coroutine appending mid-flush.
        await buffered.add([_event("late")])

    inner.add = add_with_reentry  # type: ignore[method-assign]  # test seam
    await buffered.flush()
    inner.add = original_add  # type: ignore[method-assign]  # restore

    remaining = await buffered.get()
    texts = [e.content.get("content") for e in remaining if isinstance(e.content, dict)]
    assert "late" in texts, "the late event must still be visible (buffered), not destroyed"
    await buffered.flush()
    assert sum(len(batch) for batch in inner.add_calls) == 2, "late event must reach the inner store"
