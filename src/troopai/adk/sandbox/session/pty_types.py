"""PTY-related constants + small helper types for sandbox sessions.

Backends that expose ``pty_start`` / ``pty_write_stdin`` /
``pty_terminate_all`` share the same yield-time clamping logic
and the same LRU eviction policy on stale PTY process IDs;
factoring it here means a new backend doesn't have to re-derive
the numbers.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass

from troopai.adk.sandbox.utils.token_truncation import formatted_truncate_text_with_token_count

logger = logging.getLogger(__name__)

__all__ = [
    "PTY_EMPTY_YIELD_TIME_MS_MIN",
    "PTY_PROCESSES_MAX",
    "PTY_PROCESSES_PROTECTED_RECENT",
    "PTY_PROCESSES_WARNING",
    "PTY_PROCESS_ID_MAX_EXCLUSIVE",
    "PTY_PROCESS_ID_MIN",
    "PTY_YIELD_TIME_MS_MAX",
    "PTY_YIELD_TIME_MS_MIN",
    "PtyExecUpdate",
    "allocate_pty_process_id",
    "clamp_pty_yield_time_ms",
    "process_id_to_prune_from_meta",
    "resolve_pty_write_yield_time_ms",
    "truncate_text_by_tokens",
]

PTY_YIELD_TIME_MS_MIN: int = 250
"""Floor on the per-poll PTY yield window."""

PTY_EMPTY_YIELD_TIME_MS_MIN: int = 5_000
"""Floor on the yield window when the input chunk is empty (drain mode)."""

PTY_YIELD_TIME_MS_MAX: int = 30_000
"""Ceiling on the per-poll PTY yield window."""

PTY_PROCESSES_MAX: int = 64
"""Maximum concurrent PTY processes per session."""

PTY_PROCESSES_WARNING: int = 60
"""Warn the operator when the active PTY count crosses this threshold."""

PTY_PROCESSES_PROTECTED_RECENT: int = 8
"""Number of most-recently-used PTYs protected from LRU eviction."""

PTY_PROCESS_ID_MIN: int = 1_000
"""Lower bound for synthesized PTY process IDs."""

PTY_PROCESS_ID_MAX_EXCLUSIVE: int = 100_000
"""Exclusive upper bound for synthesized PTY process IDs."""

_PTY_PROCESS_ID_RANGE: int = PTY_PROCESS_ID_MAX_EXCLUSIVE - PTY_PROCESS_ID_MIN


@dataclass(frozen=True)
class PtyExecUpdate:
    """One incremental update from an active PTY session.

    Attributes:
        process_id: Numeric ID scoped to the session (``None`` for
            metadata-only updates that don't correspond to a process).
        output: Raw output bytes captured since the previous update.
        exit_code: Process exit code when the process has finished;
            ``None`` while still running.
        original_token_count: Pre-truncation token estimate; ``None``
            when no truncation was applied to this update.
    """

    process_id: int | None
    """Numeric ID of the PTY process (``None`` for metadata-only updates)."""

    output: bytes
    """Raw output bytes since the previous update."""

    exit_code: int | None
    """Process exit code on completion; ``None`` while running."""

    original_token_count: int | None
    """Pre-truncation token estimate when output was capped."""


def clamp_pty_yield_time_ms(yield_time_ms: int) -> int:
    """Clamp ``yield_time_ms`` into the supported PTY-yield range."""
    return max(PTY_YIELD_TIME_MS_MIN, min(PTY_YIELD_TIME_MS_MAX, yield_time_ms))


def resolve_pty_write_yield_time_ms(*, yield_time_ms: int, input_empty: bool) -> int:
    """Pick a yield window — uses the drain-mode floor when ``input_empty`` is True."""
    normalized = clamp_pty_yield_time_ms(yield_time_ms)
    if input_empty:
        return max(normalized, PTY_EMPTY_YIELD_TIME_MS_MIN)
    return normalized


# Safety bound on the ``allocate_pty_process_id`` loop. The numeric
# range is 99k IDs; ``PTY_PROCESSES_MAX`` caps the in-use set at 64,
# so allocation succeeds with probability ≈99.94% on the first
# attempt. We cap at 10× the in-use ceiling to guard against a
# pathological RNG.
_MAX_PTY_PROCESS_ID_ATTEMPTS: int = 10 * PTY_PROCESSES_MAX


def allocate_pty_process_id(used_process_ids: set[int]) -> int:
    """Return a fresh PTY process ID not in ``used_process_ids``.

    Raises:
        RuntimeError: Failed to find a free ID after
            ``_MAX_PTY_PROCESS_ID_ATTEMPTS`` tries — either the
            caller forgot to release IDs or the numeric range is
            exhausted.
    """
    if len(used_process_ids) >= _PTY_PROCESS_ID_RANGE:
        raise RuntimeError(f"allocate_pty_process_id: numeric range exhausted ({_PTY_PROCESS_ID_RANGE} IDs)")
    for _ in range(_MAX_PTY_PROCESS_ID_ATTEMPTS):
        process_id = random.randrange(PTY_PROCESS_ID_MIN, PTY_PROCESS_ID_MAX_EXCLUSIVE)
        if process_id not in used_process_ids:
            return process_id
    raise RuntimeError(
        f"allocate_pty_process_id: failed to find a free ID in "
        f"{_MAX_PTY_PROCESS_ID_ATTEMPTS} attempts (used={len(used_process_ids)})"
    )


def process_id_to_prune_from_meta(
    meta: Sequence[tuple[int, float, bool]],
) -> int | None:
    """Pick a PTY process to prune via LRU + recency protection.

    Args:
        meta: Sequence of ``(process_id, last_used_epoch_ms, exited)``
            triples for currently-tracked PTY processes.

    Returns:
        The PTY process ID to evict, or ``None`` when nothing
        evictable remains (every entry is recent-protected and
        still running).
    """
    if len(meta) == 0:
        return None

    by_recency = sorted(meta, key=lambda item: item[1], reverse=True)
    protected = {process_id for process_id, _last_used, _exited in by_recency[:PTY_PROCESSES_PROTECTED_RECENT]}

    lru = sorted(meta, key=lambda item: item[1])

    for process_id, _last_used, exited in lru:
        if process_id in protected:
            continue
        if exited:
            return process_id

    for process_id, _last_used, _exited in lru:
        if process_id not in protected:
            return process_id

    # Every tracked PTY is protected-recent — caller can't reclaim a slot
    # without ejecting a still-active PTY by hand. Worth a WARNING so the
    # configuration anomaly surfaces (would normally require
    # PTY_PROCESSES_PROTECTED_RECENT >= len(meta), which is itself a
    # misconfiguration to chase).
    logger.warning(
        "process_id_to_prune_from_meta: nothing evictable (n=%d, protected=%d) — "
        "every tracked PTY is in the recent-protected set",
        len(meta),
        len(protected),
    )
    return None


def truncate_text_by_tokens(text: str, max_output_tokens: int | None) -> tuple[str, int | None]:
    """Re-export of the token-budget truncator used in PTY output paths."""
    return formatted_truncate_text_with_token_count(text, max_output_tokens)
