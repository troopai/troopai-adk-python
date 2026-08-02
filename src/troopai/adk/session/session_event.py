"""Structured session event — typed conversation entry.

A ``SessionEvent`` wraps a single conversation item with metadata:
author, timestamp, and optional state changes.  Events are immutable
once created (frozen dataclass).

Named ``SessionEvent`` (not ``Event``) to avoid collision with the
framework's stream events (``StreamEvent``, ``RunItemStreamEvent``, etc.).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from troopai.adk.types.input import LLMInputContentItem


@dataclass(frozen=True)
class SessionEvent:
    """A single conversation event in a session.

    Events are append-only and immutable once stored.  Each event wraps
    a single :data:`LLMInputContentItem` with metadata.

    Attributes:
        id: Unique event identifier (UUID).
        author: Who produced this event (``"user"``, ``"assistant"``,
            ``"tool"``, ``"system"``, or an agent name).
        content: The actual message or action (Layer 1 type).
        timestamp: Unix timestamp when the event was created.
        state_delta: State changes caused by this event.  Empty dict
            means no state changes.
    """

    id: str
    author: str
    content: LLMInputContentItem
    timestamp: float
    state_delta: dict[str, Any] = field(default_factory=dict)

    def to_content(self) -> LLMInputContentItem:
        """Extract the content item for LLM input construction.

        Returns:
            The :data:`LLMInputContentItem` carried by this event.
        """
        return self.content


def create_session_event(
    author: str,
    content: LLMInputContentItem,
    state_delta: dict[str, Any] | None = None,
) -> SessionEvent:
    """Create a SessionEvent with auto-generated id and timestamp.

    Args:
        author: Who produced this event.
        content: The message or action content.
        state_delta: Optional state changes from this event.

    Returns:
        A new :class:`SessionEvent` instance.
    """
    return SessionEvent(
        id=str(uuid.uuid4()),
        author=author,
        content=content,
        timestamp=time.time(),
        state_delta=state_delta or {},
    )
