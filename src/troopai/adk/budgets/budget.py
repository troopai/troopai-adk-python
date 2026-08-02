"""Per-tenant budget policy + calendar period bucketing."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import assert_never


class BudgetPeriod(StrEnum):
    """Calendar bucket granularity for per-period budgets (UTC)."""

    HOUR = "hour"
    DAY = "day"
    MONTH = "month"


def period_key(period: BudgetPeriod, now: datetime) -> str:
    """Return the calendar bucket key for ``now`` (caller passes UTC).

    DAY -> ``"YYYY-MM-DD"``, HOUR -> ``"YYYY-MM-DDTHH"``,
    MONTH -> ``"YYYY-MM"``. One counter row per (tenant, key).

    Args:
        period: The calendar granularity to bucket by.
        now: A timezone-aware UTC datetime whose components are used to form
            the key. A naive datetime raises ``ValueError`` because naive
            local time produces wrong calendar buckets when the host timezone
            differs from UTC.

    Returns:
        A string key uniquely identifying the calendar window.

    Raises:
        ValueError: If ``now`` is a naive datetime (``tzinfo`` is ``None``).
    """
    if now.tzinfo is None:
        raise ValueError("period_key requires a timezone-aware datetime (UTC); got a naive datetime")
    if period is BudgetPeriod.HOUR:
        return now.strftime("%Y-%m-%dT%H")
    if period is BudgetPeriod.MONTH:
        return now.strftime("%Y-%m")
    if period is BudgetPeriod.DAY:
        return now.strftime("%Y-%m-%d")
    assert_never(period)


@dataclass(frozen=True)
class TenantBudget:
    """Dollar limits applied to a run's tenant (identity from RunContext).

    Attributes:
        dollars_per_run: Hard cap on a single run's accumulated cost.
        dollars_per_period: Cap on cross-run spend within ``period``;
            requires a ``CostLedger`` on the run config.
        period: Calendar bucket for ``dollars_per_period``.
        kill_on_exceed: ``True`` raises ``TenantBudgetExceeded``; ``False``
            logs a warning and continues. The breach event is emitted on
            BOTH paths (before the raise/warn).
    """

    dollars_per_run: float | None = None
    """Hard cap on a single run's accumulated cost. ``None`` = no limit."""

    dollars_per_period: float | None = None
    """Cross-run spend cap within ``period``; requires a ``CostLedger``. ``None`` = no limit."""

    period: BudgetPeriod = BudgetPeriod.DAY
    """Calendar bucket for ``dollars_per_period``."""

    kill_on_exceed: bool = True
    """``True`` raises ``TenantBudgetExceeded``; ``False`` warns and continues.
    The breach event is emitted on both paths."""

    def __post_init__(self) -> None:
        # Reject non-finite caps too: `<= 0` is False for both inf and nan,
        # so they would slip through and silently disable enforcement (inf is
        # never exceeded; every `cost > nan` is False).
        if self.dollars_per_run is not None and (not math.isfinite(self.dollars_per_run) or self.dollars_per_run <= 0):
            raise ValueError(f"dollars_per_run must be a positive finite number, got {self.dollars_per_run}")
        if self.dollars_per_period is not None and (
            not math.isfinite(self.dollars_per_period) or self.dollars_per_period <= 0
        ):
            raise ValueError(f"dollars_per_period must be a positive finite number, got {self.dollars_per_period}")
