"""Recursive dict merge with last-write-wins on leaf values.

Used by manifest+capability composition to layer overrides: the
default manifest provides a base shape, capabilities mutate it,
the agent's local manifest takes precedence.
"""

from __future__ import annotations

from typing import TypeGuard

__all__ = ["deep_merge"]


def _is_string_object_dict(value: object) -> TypeGuard[dict[str, object]]:
    """Return True iff ``value`` is a dict whose keys are all strings."""
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def deep_merge(base: dict[str, object], overrides: dict[str, object]) -> dict[str, object]:
    """Recursively merge ``overrides`` into ``base`` and return a NEW dict.

    Both inputs MUST be string-keyed. When both sides have a dict at
    the same key, recurse; otherwise ``overrides`` wins. The result
    is a shallow copy of ``base`` plus the new keys — neither input
    is mutated.

    >>> deep_merge({"a": 1, "b": {"c": 2}}, {"b": {"d": 3}})
    {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    result = base.copy()
    for key, value in overrides.items():
        existing = result.get(key)
        if _is_string_object_dict(existing) and _is_string_object_dict(value):
            result[key] = deep_merge(existing, value)
        else:
            result[key] = value
    return result
