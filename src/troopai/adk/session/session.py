"""Session ABC — behavioral contract for conversation persistence.

A ``Session`` represents one conversation thread.  The Runner calls
``get()``, ``add()``, and reads ``settings``/``id``/``state``.
Concrete implementations are produced by a manager (e.g.
:class:`SQLiteMultiSessions`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.session.session_event import SessionEvent
    from troopai.adk.session.session_settings import SessionSettings
    from troopai.adk.session.state import State


class Session(ABC):
    """Protocol for session implementations.

    A session represents the chronological sequence of messages and actions
    (events) for a single, ongoing interaction between a user and your agent
    system.
    """

    @property
    @abstractmethod
    def id(self) -> str:
        """The session id."""
        ...

    @property
    def app_name(self) -> str:
        """Application name this session belongs to."""
        return ""

    @property
    def user_id(self) -> str:
        """User identifier this session belongs to."""
        return ""

    @property
    def settings(self) -> SessionSettings | None:
        """Session settings, if any.  Subclasses may override."""
        return None

    @property
    @abstractmethod
    def state(self) -> State:
        """Mutable session state.  Subclasses must override to return the bound State."""
        ...

    @abstractmethod
    async def get(self, limit: int | None = None) -> list[SessionEvent]:
        """Retrieve the conversation history for this session.

        Args:
            limit: Maximum number of events to retrieve.
                If ``None``, retrieve all.

        Returns:
            List of :class:`SessionEvent` in chronological order
            (oldest first).
        """

    @abstractmethod
    async def add(self, events: list[SessionEvent]) -> None:
        """Append events to this session.

        Args:
            events: Events to add to the conversation history.
        """

    @abstractmethod
    async def pop_last(self) -> SessionEvent | None:
        """Remove and return the most recent event, or ``None`` if empty."""

    @abstractmethod
    async def clear(self) -> None:
        """Clear all conversation history from the session."""

    async def save_state(self) -> None:
        """Persist pending state changes.  No-op by default."""

    async def close(self) -> None:
        """Release resources.  No-op by default."""
