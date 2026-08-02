"""Per-node reliability — effective-policy resolution and the
retry/per-attempt-timeout wrapper applied inside ``_invoke_node``.

The graph BSP driver is unchanged: a node's task internally retries
(with per-attempt timeout) and then either yields a ``NodeResult`` or
raises into the existing error path. A node that opted into neither a
retry policy nor a timeout re-raises its original exception unchanged.

Timeout retryability
--------------------
By default a :class:`~troopai.adk.exceptions.GraphNodeTimeoutError` is
terminal even when a retry policy is configured.  The new opt-in
semantic is:

- When ``policy.max_attempts > 1`` AND the timeout fires on an
  *intermediate* attempt, the timeout is treated as a retryable failure
  (subject to ``retry_on`` filtering: empty = retry-all, or
  ``TimeoutError`` must match).
- On the **final** attempt a timeout ALWAYS raises
  :class:`~troopai.adk.exceptions.GraphNodeTimeoutError` — the typed
  wrapper is always present when a timeout was the terminal cause.
- Without a retry policy (``max_attempts == 1``) a timeout raises
  :class:`~troopai.adk.exceptions.GraphNodeTimeoutError` immediately,
  preserving existing behaviour (zero behavioural change without an
  explicit policy).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, NoReturn

from troopai.adk.exceptions import GraphNodeTimeoutError, NodeRetriesExhaustedError
from troopai.adk.graphs.interrupt import InterruptException

if TYPE_CHECKING:
    from troopai.adk.graphs.config import NodeRetryPolicy
    from troopai.adk.graphs.graph import Graph
    from troopai.adk.graphs.node import GraphNode
    from troopai.adk.orchestration.executable import NodeResult


logger = logging.getLogger(__name__)


def resolve_node_reliability(
    graph: Graph[Any],
    node: GraphNode,
) -> tuple[NodeRetryPolicy, float | None]:
    """Return the effective ``(retry_policy, timeout)`` for ``node``.

    A per-node field overrides the graph-level default only when it is
    not ``None``; otherwise the graph default applies.

    Args:
        graph: The compiled :class:`~troopai.adk.graphs.graph.Graph`
            whose :attr:`~troopai.adk.graphs.graph.Graph.config` holds
            the graph-level defaults.
        node: The :class:`~troopai.adk.graphs.node.GraphNode` being
            evaluated. Its :attr:`~troopai.adk.graphs.node.GraphNode.retry`
            and :attr:`~troopai.adk.graphs.node.GraphNode.timeout` fields
            take precedence when set.

    Returns:
        A two-tuple ``(policy, timeout)`` where ``policy`` is the
        effective :class:`~troopai.adk.graphs.config.NodeRetryPolicy` and
        ``timeout`` is the effective per-attempt timeout in seconds, or
        ``None`` when no timeout applies.
    """
    policy = node.retry if node.retry is not None else graph.config.default_retry
    timeout = node.timeout if node.timeout is not None else graph.config.per_node_timeout
    logger.debug(
        "resolve_node_reliability: node=%s policy=%s timeout=%s",
        node.id,
        policy,
        timeout,
    )
    return policy, timeout


async def _invoke_once(
    *,
    invoke: Callable[[], Awaitable[NodeResult]],
    timeout: float | None,
    attempt: int,
) -> NodeResult:
    """Run one attempt under the optional timeout and stamp its count.

    The attempt count is stashed under an underscored metadata key so
    the graph loop can stamp the node span; observers read attempts off
    the span attribute, not the NodeResult metadata directly.
    """
    if timeout is not None:
        async with asyncio.timeout(timeout):
            result = await invoke()
    else:
        result = await invoke()
    result.metadata["__attempts__"] = attempt
    return result


async def run_node_with_reliability(
    *,
    node_id: str,
    policy: NodeRetryPolicy,
    timeout: float | None,
    invoke: Callable[[], Awaitable[NodeResult]],
) -> NodeResult:
    """Run ``invoke`` with per-attempt timeout and bounded retry.

    Bounded by ``policy.max_attempts``; ``asyncio.CancelledError`` is
    never caught (a fail-fast sibling cancel propagates cleanly).
    Timeouts on intermediate attempts retry like any other retryable
    exception (subject to ``retry_on``); with ``max_attempts == 1``
    behaviour is identical to having no retry policy.
    """
    retry_on: tuple[type[Exception], ...] = policy.retry_on
    backoff = policy.initial_backoff
    last_exc: Exception | None = None
    timed_out = False
    retryable = False

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await _invoke_once(invoke=invoke, timeout=timeout, attempt=attempt)
        except InterruptException:
            raise  # HITL cooperative pause signal; never retried, never wrapped
        except Exception as exc:  # CancelledError is BaseException -> not caught
            last_exc = exc
            # Only treat a TimeoutError as a node-reliability timeout when a
            # per-attempt timeout was actually configured — that is the sole
            # case where ``_invoke_once`` wraps the call in
            # ``asyncio.timeout``. A TimeoutError raised by the node body
            # itself with no configured timeout is an ordinary failure;
            # wrapping it as ``GraphNodeTimeoutError(timeout=0.0)`` would mask
            # the real error.
            is_timeout = isinstance(exc, TimeoutError) and timeout is not None
            retryable = len(retry_on) == 0 or isinstance(exc, retry_on)

            if retryable and attempt < policy.max_attempts:
                # Intermediate retryable failure — timeouts included.
                logger.warning(
                    "node %s attempt %d/%d failed (%s); retrying in %.1fs",
                    node_id,
                    attempt,
                    policy.max_attempts,
                    "timeout" if is_timeout else exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, policy.max_backoff)
                timed_out = False
                continue

            timed_out = is_timeout  # final attempt or non-retryable: exit loop
            break

    if last_exc is None:
        raise RuntimeError(f"run_node_with_reliability: unreachable — no attempt recorded for node {node_id!r}")
    _raise_final_failure(
        node_id=node_id,
        timeout=timeout,
        attempt=attempt,
        max_attempts=policy.max_attempts,
        timed_out=timed_out,
        retryable=retryable,
        last_exc=last_exc,
    )


def _raise_final_failure(
    *,
    node_id: str,
    timeout: float | None,
    attempt: int,
    max_attempts: int,
    timed_out: bool,
    retryable: bool,
    last_exc: Exception,
) -> NoReturn:
    """Raise the typed final-failure exception for an exhausted node.

    A timeout raises :class:`~troopai.adk.exceptions.GraphNodeTimeoutError`;
    an exhausted multi-attempt policy raises
    :class:`~troopai.adk.exceptions.NodeRetriesExhaustedError`; anything
    else re-raises the original exception unchanged.
    """
    if timed_out:
        raise GraphNodeTimeoutError(
            node_id=node_id,
            timeout=timeout if timeout is not None else 0.0,
            attempts=attempt,
        ) from last_exc
    if retryable and max_attempts > 1:
        raise NodeRetriesExhaustedError(
            node_id=node_id,
            attempts=max_attempts,
            last_error=last_exc,
        ) from last_exc
    raise last_exc


__all__ = ["resolve_node_reliability", "run_node_with_reliability"]
