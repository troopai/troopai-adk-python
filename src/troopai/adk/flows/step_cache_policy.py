"""FlowStepCachePolicy — declarative result-caching for Flow steps.

When attached to a step via the decorator's ``cache=`` keyword,
the executor consults a per-step cache before running the step
body: a cache hit short-circuits the body, restores the cached
state snapshot, and dispatches successors as if the step had just
run. A miss runs the body normally and writes its post-body state
snapshot back to the cache under the computed key.

The cache scope is **per-executor** (one cache per
:meth:`Runner.arun_flow` call). Sharing a cache across runs would
require a backend store and is out of scope for this primitive —
the developer can compose checkpointing + an external KV cache on
top.

Snapshot semantics:

- Pydantic ``BaseModel`` states snapshot via
  :meth:`BaseModel.model_copy(deep=True)`.
- ``@dataclass`` states snapshot via :func:`copy.deepcopy`.
- Anything else raises :class:`FlowDefinitionError` at cache-write
  time so the developer sees the unsupported shape loudly rather
  than caching a shallow alias.

Router steps additionally cache the ``str`` route label they
returned so the cached hit re-dispatches the same downstream
listeners.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from troopai.adk.flows.step_context import FlowStepContext


FlowCacheKeyFn = Callable[
    ["FlowStepContext[Any]"],
    "str | Awaitable[str]",
]
"""Callable that maps a :class:`FlowStepContext` to a cache key string.

Sync or async; the executor awaits the result if it is awaitable.
The returned string is used verbatim as the cache key — developers
typically derive it from
:attr:`FlowStepContext.flow_state` fields they consider load-bearing
for the step's output, deliberately excluding fields that should
NOT participate in the hit/miss decision (timestamps, request ids,
...).
"""


@dataclass(frozen=True, kw_only=True)
class FlowStepCachePolicy:
    """Per-step result-cache configuration.

    Attached to a :class:`FlowStep` via the decorator's ``cache=``
    keyword. Independent of :class:`FlowConfig` because the cache
    is a step-local concern; the same flow class can have caching
    on some steps and not others.

    Attributes:
        cache_key_fn: Callable that produces the cache key from
            the current step context. Returns ``str`` or awaitable
            ``str``. Required — no inferred default because the
            developer must declare which state fields participate.
        ttl_seconds: Optional time-to-live for cache entries. When
            set, entries older than ``ttl_seconds`` (measured by
            ``time.monotonic()``) are evicted on lookup. ``None``
            (default) means entries never expire — bounded only by
            :attr:`max_entries`.
        max_entries: Maximum cache entries retained per step.
            LRU-evicted when exceeded. Cost-conservative default
            ``128`` — large enough for typical fan-out, small
            enough that misconfigured caches don't grow unbounded
            in memory.
    """

    cache_key_fn: FlowCacheKeyFn
    """Cache-key callable; sync or async."""

    ttl_seconds: float | None = None
    """Optional TTL in seconds; ``None`` ⇒ never expires."""

    max_entries: int = 128
    """LRU cap on entries retained per step."""

    def __post_init__(self) -> None:
        """Validate :class:`FlowStepCachePolicy` field values.

        Raises:
            ValueError: When ``cache_key_fn`` is not callable, when
                ``ttl_seconds`` is non-positive, or when ``max_entries``
                is less than 1.
        """
        if not callable(self.cache_key_fn):
            raise ValueError(
                f"FlowStepCachePolicy.cache_key_fn must be callable, got {type(self.cache_key_fn).__name__}",
            )
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            raise ValueError(
                f"FlowStepCachePolicy.ttl_seconds must be positive when set, got {self.ttl_seconds}",
            )
        if self.max_entries < 1:
            raise ValueError(
                f"FlowStepCachePolicy.max_entries must be >= 1, got {self.max_entries}",
            )
