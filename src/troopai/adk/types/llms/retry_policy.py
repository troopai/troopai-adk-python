"""Provider-agnostic retry policy for LLM calls.

Describes *how* transient LLM failures should be retried — not *what*
counts as a failure. Error-category detection is delegated to each
``LLM`` implementation because exception types differ per provider
SDK. The framework-level contract is:

- :data:`LLMRetryErrorKind` — the set of categories the runtime may
  produce.
- :class:`LLMRetryPolicy` — the policy parameters a developer sets on
  ``LLMConfig.retry_policy``.

An ``LLM`` implementation receives the ``LLMRetryPolicy`` through
config, catches its own provider-specific exceptions, maps them to a
:data:`LLMRetryErrorKind`, and decides whether to retry based on
:meth:`LLMRetryPolicy.should_retry`.

This is distinct from :attr:`LLMConfig.num_retries`, which is a
provider-side retry hint passed straight to the SDK. The retry
policy runs *outside* the SDK and owns jitter, backoff, and
category selection — knobs that are typically absent from SDK-level
retry parameters.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

LLMRetryErrorKind = Literal["rate_limit", "server_error", "timeout"]
"""Framework-level categories of retryable LLM errors.

Mapped from provider-specific exceptions by each ``LLM``
implementation. Authentication errors and permanent failures are
deliberately excluded — they are never retried.
"""

_ALL_KINDS: frozenset[LLMRetryErrorKind] = frozenset(["rate_limit", "server_error", "timeout"])


@dataclass(frozen=True)
class LLMRetryPolicy:
    """Exponential-backoff retry policy for transient LLM failures.

    Attributes:
        max_retries: Maximum number of retry attempts before giving up.
            ``max_retries=1`` (the default) means at most 2 total calls
            (1 initial + 1 retry). Set higher for more aggressive retry
            behaviour.
        initial_delay: Delay in seconds before the first retry.
        max_delay: Upper bound on the delay between retries. Backoff
            stops growing beyond this.
        multiplier: Exponential growth factor applied to the delay
            each retry.
        jitter: When ``True``, the delay is randomized in
            ``[0, computed_delay]`` to avoid thundering-herd problems.
        retry_on: Optional subset of error kinds to retry. Defaults to
            ``frozenset(["rate_limit"])`` — the narrowest useful set.
            Pass ``None`` to retry all :data:`LLMRetryErrorKind` kinds,
            or supply an explicit set to control which categories trigger
            a retry.

    Example::

        from troopai.adk.types.llms import LLMRetryPolicy
        from troopai.adk.llms import LLMConfig

        config = LLMConfig(
            retry_policy=LLMRetryPolicy(
                max_retries=5,
                initial_delay=1.0,
                retry_on=frozenset(["rate_limit", "timeout"]),
            )
        )
    """

    max_retries: int = 1
    """Maximum retry attempts before giving up (default 1 = one retry)."""

    initial_delay: float = 0.5
    """Delay in seconds before the first retry."""

    max_delay: float = 30.0
    """Upper bound on the delay between retries."""

    multiplier: float = 2.0
    """Exponential growth factor applied each retry."""

    jitter: bool = True
    """When ``True``, randomize the delay in ``[0, computed_delay]``."""

    retry_on: frozenset[LLMRetryErrorKind] | None = frozenset(["rate_limit"])
    """Subset of retryable error kinds. Defaults to rate-limit-only."""

    def should_retry(self, kind: LLMRetryErrorKind, attempt: int) -> bool:
        """Return ``True`` if an error of *kind* should be retried.

        Args:
            kind: The category the implementation mapped the exception to.
            attempt: Zero-indexed attempt number. ``attempt=0`` means the
                first retry would be the next call. ``attempt >= max_retries``
                means budget exhausted.

        Returns:
            ``True`` when another attempt should be made, ``False`` when the
            budget is exhausted or the error kind is not in the allowed set.
        """
        if attempt >= self.max_retries:
            return False
        allowed = self.retry_on if self.retry_on is not None else _ALL_KINDS
        return kind in allowed

    def compute_delay(self, attempt: int) -> float:
        """Return the delay in seconds before the next retry.

        Args:
            attempt: Zero-indexed attempt number. ``attempt=0`` yields
                :attr:`initial_delay` (±jitter).

        Returns:
            Delay in seconds, capped at :attr:`max_delay` and optionally
            randomized when :attr:`jitter` is ``True``.
        """
        raw = self.initial_delay * (self.multiplier ** max(0, attempt))
        capped = min(raw, self.max_delay)
        if self.jitter:
            return random.uniform(0.0, capped)
        return capped
