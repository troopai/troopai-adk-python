"""Replay-safe tracing helpers for Temporal workflows.

Temporal replays workflow history deterministically when recovering from
failures.  During replay, side effects such as emitting OpenTelemetry spans
must be suppressed to avoid duplicate traces.  Timestamps and UUIDs must be
sourced from Temporal's deterministic clock and PRNG rather than the system
clock and :mod:`uuid`.

Each function in this module checks whether it is being called inside a
Temporal workflow and, if so, whether the workflow is currently replaying.
When ``temporalio`` is not installed the functions fall back transparently
to their non-workflow equivalents.

References:
    Temporal Python SDK — unsafe context utilities:
    https://python.temporal.io/temporalio.workflow.html#unsafe
    Temporal Python SDK — workflow.uuid4:
    https://python.temporal.io/temporalio.workflow.html#uuid4
    Temporal Python SDK — workflow.now:
    https://python.temporal.io/temporalio.workflow.html#now
"""

from __future__ import annotations

import logging
import time
import uuid

logger = logging.getLogger(__name__)


def should_emit_span() -> bool:
    """Return ``True`` when it is safe to emit an OpenTelemetry span.

    Suppresses span emission during Temporal workflow replay to prevent
    duplicate spans.  Returns ``True`` unconditionally when called outside
    a Temporal workflow or when ``temporalio`` is not installed.

    Returns:
        ``False`` during replay; ``True`` otherwise.
    """
    try:
        from temporalio import workflow

        if not workflow.in_workflow():
            logger.debug("should_emit_span: outside workflow — emitting")
            return True

        replaying = workflow.unsafe.is_replaying()
        if replaying:
            logger.debug("should_emit_span: workflow is replaying — suppressing span")
        else:
            logger.debug("should_emit_span: workflow is not replaying — emitting span")
        return not replaying

    except ImportError:
        logger.debug("should_emit_span: temporalio not installed — emitting")
        return True


def deterministic_timestamp() -> float:
    """Return a deterministic timestamp safe for use inside Temporal workflows.

    Inside a workflow the Temporal clock (:func:`temporalio.workflow.now`) is
    used so that replay produces the same value.  Outside a workflow or when
    ``temporalio`` is not installed, :func:`time.time` is used.

    Returns:
        A Unix timestamp as a float.
    """
    try:
        from temporalio import workflow

        if workflow.in_workflow():
            ts = workflow.now().timestamp()
            logger.debug("deterministic_timestamp: workflow clock → %s", ts)
            return ts

    except ImportError:
        pass

    ts = time.time()
    logger.debug("deterministic_timestamp: system clock → %s", ts)
    return ts


def deterministic_uuid() -> str:
    """Return a deterministic UUID string safe for use inside Temporal workflows.

    Inside a workflow :func:`temporalio.workflow.uuid4` is used so that replay
    produces the same identifier.  Outside a workflow or when ``temporalio`` is
    not installed, :func:`uuid.uuid4` is used.

    Returns:
        A UUID as a hyphenated lowercase string.
    """
    try:
        from temporalio import workflow

        if workflow.in_workflow():
            uid = str(workflow.uuid4())
            logger.debug("deterministic_uuid: workflow uuid4 → %s", uid)
            return uid

    except ImportError:
        pass

    uid = str(uuid.uuid4())
    logger.debug("deterministic_uuid: system uuid4 → %s", uid)
    return uid
