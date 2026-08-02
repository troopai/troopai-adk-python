"""Session-level configuration settings."""

from __future__ import annotations

from pydantic import BaseModel


class SessionSettings(BaseModel):
    """Configuration for session behavior.

    Attributes:
        limit: Default maximum number of items to retrieve from
            the session via ``get()``. If ``None``, all items are returned.
    """

    limit: int | None = None
    """Default maximum number of items to retrieve from the session via ``get()``.
    If ``None``, all items are returned. Default is None (no limit)."""
