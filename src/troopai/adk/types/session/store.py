"""SessionStore — structural protocol for conversation-session backends.

Any object that exposes the five methods below satisfies this protocol
without inheriting from it.  Concrete implementations (SQLite, Redis,
Postgres, in-memory) are structural members; no explicit subclassing is
required.

Runner call-site evidence
-------------------------
The Runner accesses sessions exclusively through:

- ``session.id``          — str property; included in tracing payloads.
- ``session.settings``    — ``SessionSettings | None``; drives the
                            ``get()`` limit when the caller omits it.
- ``await session.get(limit=limit)`` — load history before the LLM call.
- ``await session.add(events)``      — persist the turn after the LLM call.
- ``await session.save_state()``     — flush pending state changes after
                                       ``add()`` succeeds.
- ``await session.close()``          — release backend resources when the
                                       session handle is no longer needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from troopai.adk.session.session_event import SessionEvent
    from troopai.adk.session.session_settings import SessionSettings


@runtime_checkable
class SessionStore(Protocol):
    """Structural protocol for a single-session conversation backend.

    ``isinstance`` checks against this protocol confirm only that
    the attribute NAMES exist — Python's runtime protocol checks do
    not verify coroutine-ness, property descriptors, or signatures.
    Static type checking is the real conformance gate; the runtime
    check is a convenience for tests.

    Any concrete class that provides the five attributes/methods below
    satisfies this protocol without inheritance.

    Attributes:
        id: Unique identifier for this session handle.
        settings: Per-session configuration (history limit, etc.), or
            ``None`` to use defaults.
    """

    @property
    def id(self) -> str:
        """Unique identifier for this session handle."""
        ...

    @property
    def settings(self) -> SessionSettings | None:
        """Per-session configuration, or ``None`` to use defaults."""
        ...

    async def get(self, limit: int | None = None) -> list[SessionEvent]:
        """Retrieve stored events in chronological order (oldest first).

        Args:
            limit: Maximum number of events to return.  ``None`` means
                return all stored events (or fall back to
                ``settings.limit`` when set).

        Returns:
            List of :class:`~troopai.adk.session.session_event.SessionEvent`
            in chronological order.
        """
        ...

    async def add(self, events: list[SessionEvent]) -> None:
        """Append events to the session.

        Args:
            events: Events to persist.  An empty list is a no-op.
        """
        ...

    async def save_state(self) -> None:
        """Flush any pending in-memory state changes to the backend.

        Implementations that do not track dirty state may treat this as a
        no-op.
        """
        ...

    async def close(self) -> None:
        """Release backend resources held by this session handle.

        Implementations that share a connection pool or delegate resource
        management to a factory may treat this as a no-op.
        """
        ...
