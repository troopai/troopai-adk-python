"""``TokenBudget`` — typed token cap with explicit drop policy.

Used by :class:`HandoffConfig` to express the token-cap threshold
AND the drop-policy when over-budget as one structured value
instead of two implicitly-coupled fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

type TokenDropPolicy = Literal["oldest_first", "preserve_system"]
"""Policy for which messages to drop when the transferred history
exceeds ``TokenBudget.max_tokens``.

- ``"oldest_first"`` (strict FIFO): drop the oldest message first
  regardless of role. System messages get evicted along with
  everything else.
- ``"preserve_system"`` (default): drop oldest non-system messages
  first; keep the system message even when over budget.
"""


@dataclass(frozen=True)
class TokenBudget:
    """A token-count cap on transferred handoff history.

    Attributes:
        max_tokens: Maximum input-token budget. MUST be > 0.
        drop_policy: Which messages to drop when over budget.
            Default ``"preserve_system"`` keeps the leading system
            message and drops oldest non-system messages first.
    """

    max_tokens: int
    """Maximum input-token budget."""

    drop_policy: TokenDropPolicy = "preserve_system"
    """Which messages to drop when over budget."""

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError(f"TokenBudget.max_tokens must be positive, got {self.max_tokens}")
