"""Session state with delta tracking and temp prefix support.

The :class:`State` class is a dict-like object that tracks pending
changes (delta) separately from committed data.  Keys prefixed with
``temp:`` are stored in-memory only and stripped before persistence.

Inspired by Google ADK's ``State`` class.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, override

logger = logging.getLogger(__name__)

_DELETED = object()
"""Sentinel marking a key as deleted in the delta."""

TEMP_PREFIX = "temp:"
"""Keys with this prefix live in-memory only — never persisted."""

APP_PREFIX = "app:"
"""Keys with this prefix are app-scoped (shared across sessions)."""


class State:
    """Dict-like session state with delta tracking.

    Maintains three layers:
    - ``_data`` — committed/persisted state (loaded from DB).
    - ``_delta`` — pending changes (not yet persisted).
    - ``_temp`` — ephemeral ``temp:`` keys (in-memory only).

    Reads check ``_temp`` → ``_delta`` → ``_data`` (most-recent wins).
    Writes go to ``_delta`` (or ``_temp`` for ``temp:`` keys).

    Example::

        state = State.from_dict({"name": "Alice"})
        state["score"] = 42  # tracked in delta
        state["temp:scratch"] = "wip"  # in-memory only, not persisted
        state.to_persist()  # {"name": "Alice", "score": 42}
        state.commit()  # delta applied to _data, cleared
    """

    __slots__ = ("_data", "_delta", "_temp")

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        delta: dict[str, Any] | None = None,
    ) -> None:
        """Initialise a State with optional pre-loaded data and delta.

        Args:
            data: Previously persisted state to load as the committed
                base layer.  Defaults to an empty dict.
            delta: Pending changes to pre-populate the delta layer.
                Defaults to an empty dict.
        """
        self._data: dict[str, Any] = data if data is not None else {}
        self._delta: dict[str, Any] = delta if delta is not None else {}
        self._temp: dict[str, Any] = {}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> State:
        """Create a State from a persisted dict.

        Args:
            data: Mapping of key/value pairs to load as the committed
                base layer.

        Returns:
            A new :class:`State` with *data* as its committed layer
            and an empty delta.
        """
        return cls(data=dict(data))

    # ------------------------------------------------------------------
    # MutableMapping-like interface
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> Any:
        if key.startswith(TEMP_PREFIX) and key in self._temp:
            return self._temp[key]
        if key in self._delta:
            value = self._delta[key]
            if value is _DELETED:
                raise KeyError(key)
            return value
        if key in self._data:
            return self._data[key]
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key.startswith(TEMP_PREFIX):
            self._temp[key] = value
        else:
            self._delta[key] = value

    def __delitem__(self, key: str) -> None:
        if key.startswith(TEMP_PREFIX):
            if key in self._temp:
                del self._temp[key]
            else:
                raise KeyError(key)
        else:
            # Mark as deleted in delta (even if only in _data)
            if key in self._delta:
                if self._delta[key] is _DELETED:
                    raise KeyError(key)
                self._delta[key] = _DELETED
            elif key in self._data:
                self._delta[key] = _DELETED
            else:
                raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key.startswith(TEMP_PREFIX):
            return key in self._temp
        if key in self._delta:
            return self._delta[key] is not _DELETED
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        seen: set[str] = set()
        # Temp keys
        for key in self._temp:
            seen.add(key)
            yield key
        # Delta keys (skip deleted)
        for key, value in self._delta.items():
            if key not in seen and value is not _DELETED:
                seen.add(key)
                yield key
        # Data keys (skip overridden/deleted)
        for key in self._data:
            if key not in seen and key not in self._delta:
                seen.add(key)
                yield key

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value for *key*, or *default* if not present."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        """Return all visible keys."""
        return list(self)

    def values(self) -> list[Any]:
        """Return all visible values."""
        return [self[k] for k in self]

    def items(self) -> list[tuple[str, Any]]:
        """Return all visible (key, value) pairs."""
        return [(k, self[k]) for k in self]

    # ------------------------------------------------------------------
    # Delta & persistence
    # ------------------------------------------------------------------

    def delta(self) -> dict[str, Any]:
        """Return the pending changes (excludes temp keys).

        Deleted keys are represented as ``None`` values.
        """
        result: dict[str, Any] = {}
        for key, value in self._delta.items():
            result[key] = None if value is _DELETED else value
        return result

    def pending_delta(self) -> dict[str, Any]:
        """Return the raw pending changes, preserving the deletion sentinel.

        Unlike :meth:`delta`, deleted keys keep the module-level ``_DELETED``
        sentinel (not ``None``) so a persistence layer can distinguish
        "delete this key" from "store ``None`` under this key". Returns a copy;
        mutating it does not affect the live delta.
        """
        return dict(self._delta)

    def has_changes(self) -> bool:
        """Whether there are uncommitted changes."""
        return len(self._delta) > 0

    def to_persist(self) -> dict[str, Any]:
        """Return the full state to persist (data + delta, no temp or app keys).

        Deleted keys and ``app:``-prefixed keys are excluded from the result.
        App-scoped state is owned by the app-state table, not the sessions
        table; writing it into the session column would cause stale app-state
        values to shadow the canonical app-state table on reload.
        """
        merged = dict(self._data)
        for key, value in self._delta.items():
            if value is _DELETED:
                merged.pop(key, None)
            else:
                merged[key] = value
        return {k: v for k, v in merged.items() if not k.startswith(APP_PREFIX)}

    def commit(self) -> None:
        """Apply pending delta to data and clear delta."""
        for key, value in self._delta.items():
            if value is _DELETED:
                self._data.pop(key, None)
            else:
                self._data[key] = value
        self._delta.clear()
        logger.debug("State committed, %d keys", len(self._data))

    @override
    def __repr__(self) -> str:
        return f"State(data={self._data}, delta={len(self._delta)} pending, temp={len(self._temp)})"
