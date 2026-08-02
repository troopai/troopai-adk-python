"""MultiSessions ABC — abstract interface for session managers.

Defines the contract for session management backends.  Concrete
implementations (e.g. :class:`SQLiteMultiSessions`) provide storage
via specific databases.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from troopai.adk.session.session import Session
    from troopai.adk.session.session_settings import SessionSettings
    from troopai.adk.session.sqlite_multi_sessions import SessionInfo


class MultiSessions(ABC):
    """Abstract base class for session managers.

    A session manager provides CRUD operations on a collection of
    sessions.  Each returned session implements the :class:`Session`
    ABC and can be passed to the Runner.

    Subclasses must implement: ``create``, ``get``, ``get_or_create``,
    ``list``, ``delete``, ``count``, ``close``.
    """

    @abstractmethod
    async def create(
        self,
        session_id: str,
        user_id: str = "",
        state: dict[str, Any] | None = None,
        settings: SessionSettings | None = None,
    ) -> Session:
        """Create a new session.

        Args:
            session_id: Unique identifier for the session.
            user_id: User identifier for multi-tenant scoping.
            state: Optional initial state dict.
            settings: Per-session settings.

        Returns:
            A :class:`Session` instance bound to the new session.

        Raises:
            ValueError: If a session with this ID already exists.
        """

    @abstractmethod
    async def get(
        self,
        session_id: str,
        user_id: str = "",
    ) -> Session | None:
        """Retrieve an existing session.

        Args:
            session_id: The session to retrieve.
            user_id: User identifier.

        Returns:
            A :class:`Session` instance, or ``None`` if not found.
        """

    @abstractmethod
    async def get_or_create(
        self,
        session_id: str,
        user_id: str = "",
        state: dict[str, Any] | None = None,
        settings: SessionSettings | None = None,
    ) -> Session:
        """Retrieve a session, creating it if it doesn't exist.

        Args:
            session_id: The session to retrieve or create.
            user_id: User identifier.
            state: Initial state for new sessions.
            settings: Per-session settings for new sessions.

        Returns:
            A :class:`Session` instance.
        """

    @abstractmethod
    async def list(
        self,
        user_id: str | None = None,
    ) -> list[SessionInfo]:
        """List sessions.

        Args:
            user_id: If provided, filter by user.  If ``None``, list all.

        Returns:
            List of :class:`~troopai.adk.session.sqlite_multi_sessions.SessionInfo` objects.
        """

    @abstractmethod
    async def delete(
        self,
        session_id: str,
        user_id: str = "",
    ) -> bool:
        """Delete a session and all its data.

        Args:
            session_id: The session to delete.
            user_id: User identifier.

        Returns:
            ``True`` if deleted, ``False`` if not found.
        """

    @abstractmethod
    async def count(
        self,
        user_id: str | None = None,
    ) -> int:
        """Return the number of sessions.

        Args:
            user_id: If provided, count only this user's sessions.
                If ``None``, count all sessions for this app.

        Returns:
            Total number of matching sessions.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release resources."""
