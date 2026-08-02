"""Audit-event delivery orchestration.

``Instrumentation`` is the bridge between sandbox backend
operations and the configured sink hierarchy. A backend emits a
``SandboxSessionEvent``; the instrumentation layer applies the
effective payload policy (default → per-op → per-sink, in that
precedence) and fans the redacted event out to every sink
honoring each sink's ``mode`` (sync / async / best_effort) and
``on_error`` (raise / log / ignore) contract.

Payload-policy layering relies on Pydantic v2 ``model_fields_set``
so an override only touches the fields the caller explicitly set —
a per-op ``EventPayloadPolicy(include_exec_output=True)`` does NOT
reset ``max_stdout_chars`` back to its default.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from troopai.adk.sandbox.session.events import (
    EventPayloadPolicy,
    SandboxSessionEvent,
    SandboxSessionFinishEvent,
)
from troopai.adk.sandbox.session.op_codes import OpName
from troopai.adk.sandbox.session.sinks import NEVER_DOWNGRADE_EXC, ChainedSink, EventSink
from troopai.adk.sandbox.session.utils import safe_decode_with_max_chars

logger = logging.getLogger(__name__)

__all__ = ["Instrumentation"]


class Instrumentation:
    """Deliver sandbox audit events to configured sinks with per-sink payload policies.

    Construct once per sandbox session. The same instance is safe
    to share across the session's operations; background deliveries
    are tracked so ``flush()`` can await them before teardown.
    """

    def __init__(
        self,
        *,
        sinks: Sequence[EventSink] | None = None,
        payload_policy: EventPayloadPolicy | None = None,
        payload_policy_by_op: dict[OpName, EventPayloadPolicy] | None = None,
    ) -> None:
        self._sinks: list[EventSink] = list(sinks) if sinks is not None else []
        self.payload_policy = payload_policy or EventPayloadPolicy()
        self.payload_policy_by_op = payload_policy_by_op or {}
        self._tasks: set[asyncio.Task[None]] = set()

    @property
    def sinks(self) -> list[EventSink]:
        """Snapshot copy of the configured sinks (mutation-safe view)."""
        return list(self._sinks)

    def add_sink(self, sink: EventSink) -> None:
        """Append a sink — takes effect for subsequent ``emit`` calls."""
        self._sinks.append(sink)

    async def emit(self, event: SandboxSessionEvent) -> None:
        """Deliver ``event`` to every configured sink, policy-redacted per sink.

        ``ChainedSink`` members are delivered strictly in order
        (the chain's contract), so each inner sink fully completes
        before the next observes the event regardless of its
        ``mode``.
        """
        for sink in self._sinks:
            if isinstance(sink, ChainedSink):
                for inner in sink.sinks:
                    policy = self._policy_for(event.op, inner)
                    per_sink_event = self._apply_policy(event, policy)
                    await self._deliver_chained(inner, per_sink_event)
            else:
                policy = self._policy_for(event.op, sink)
                per_sink_event = self._apply_policy(event, policy)
                await self._deliver(sink, per_sink_event)

    async def flush(self) -> None:
        """Await every in-flight background delivery (async / best_effort sinks).

        Background tasks consume their own failures via the sink's
        ``on_error`` policy. Anything that still escapes to here is,
        by definition, an *unhandled* background failure (a re-raised
        programmer-bug exception, an out-of-domain ``on_error``, or a
        ``BaseException`` the task body did not catch). Those MUST NOT
        be silently dropped — surface them at ERROR with the
        traceback so a teardown-time loss is visible (R7).
        """
        pending = tuple(self._tasks)
        if len(pending) == 0:
            return
        results = await asyncio.gather(*pending, return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                logger.error(
                    "sandbox background event delivery failed and was NOT handled by the sink's on_error policy: %r",
                    result,
                    exc_info=result,
                )

    def _policy_for(self, op: OpName, sink: EventSink) -> EventPayloadPolicy:
        """Resolve the effective policy: default → per-op → per-sink."""
        effective = self.payload_policy.model_copy(deep=True)

        op_policy = self.payload_policy_by_op.get(op)
        if op_policy is not None:
            effective = effective.model_copy(update=self._overrides(op_policy))

        sink_policy = sink.payload_policy
        if sink_policy is not None:
            effective = effective.model_copy(update=self._overrides(sink_policy))

        return effective

    def _overrides(self, policy: EventPayloadPolicy) -> dict[str, object]:
        """Return only the fields the caller explicitly set on ``policy``.

        Unset fields keep their lower-precedence value rather than
        being clobbered by the model default.
        """
        return {name: getattr(policy, name) for name in policy.model_fields_set}

    def _apply_policy(self, event: SandboxSessionEvent, policy: EventPayloadPolicy) -> SandboxSessionEvent:
        """Clone + redact ``event`` per ``policy`` (one clone per sink)."""
        out = event.model_copy(deep=True)

        if not policy.include_write_len:
            # pop(..., None) already no-ops when absent — no redundant `in` guard.
            out.data.pop("bytes", None)

        if isinstance(out, SandboxSessionFinishEvent):
            if not policy.include_exec_output:
                out.stdout = None
                out.stderr = None
                out.stdout_bytes = None
                out.stderr_bytes = None
            else:
                if out.stdout_bytes is not None and policy.max_stdout_chars == 0:
                    logger.debug(
                        "instrumentation: contradictory policy for event_id=%s "
                        "op=%s — include_exec_output=True but max_stdout_chars=0; "
                        "stdout fully redacted (empty string is NOT genuine empty output)",
                        out.event_id,
                        out.op,
                    )
                if out.stdout_bytes is not None:
                    out.stdout = safe_decode_with_max_chars(out.stdout_bytes, max_chars=policy.max_stdout_chars)
                if out.stderr_bytes is not None and policy.max_stderr_chars == 0:
                    logger.debug(
                        "instrumentation: contradictory policy for event_id=%s "
                        "op=%s — include_exec_output=True but max_stderr_chars=0; "
                        "stderr fully redacted (empty string is NOT genuine empty output)",
                        out.event_id,
                        out.op,
                    )
                if out.stderr_bytes is not None:
                    out.stderr = safe_decode_with_max_chars(out.stderr_bytes, max_chars=policy.max_stderr_chars)

        return out

    async def _deliver(self, sink: EventSink, event: SandboxSessionEvent) -> None:
        """Dispatch one event to one (non-chained) sink per its delivery mode."""
        if sink.mode == "sync":
            # `except Exception` (not BaseException): CancelledError /
            # KeyboardInterrupt / SystemExit intentionally escape so a
            # cancelled emit propagates to the caller rather than being
            # mis-handled as a sink failure.
            try:
                await sink.handle(event)
            except Exception as exc:
                self._handle_sink_error(sink, event, exc)
            return

        if sink.mode == "async":
            if sink.on_error == "raise":
                # Awaited inline so the failure surfaces synchronously,
                # wrapped + chained for a consistent error surface with
                # the sync path (see _handle_sink_error).
                try:
                    await sink.handle(event)
                except Exception as exc:
                    self._handle_sink_error(sink, event, exc)
                return
            self._spawn_background(sink, event, force_no_raise=False)
            return

        if sink.mode == "best_effort":
            self._spawn_background(sink, event, force_no_raise=True)
            return

        raise ValueError(f"Instrumentation: unknown sink.mode {sink.mode!r}")

    def _spawn_background(
        self,
        sink: EventSink,
        event: SandboxSessionEvent,
        *,
        force_no_raise: bool,
    ) -> None:
        async def _task() -> None:
            # `except Exception` only — a CancelledError on the detached
            # task escapes to flush()'s gather, which logs it at ERROR.
            try:
                await sink.handle(event)
            except Exception as exc:
                self._handle_sink_error(sink, event, exc, force_no_raise=force_no_raise)

        task = asyncio.create_task(_task())
        # Track so the task isn't GC'd mid-flight and flush() can await it.
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _deliver_chained(self, sink: EventSink, event: SandboxSessionEvent) -> None:
        """Deliver to a ChainedSink member — always awaited to preserve order.

        ``Instrumentation`` decomposes a ``ChainedSink`` (to apply each
        inner sink's own payload policy) and therefore bypasses
        ``ChainedSink.handle``. That means this method MUST itself
        replicate every guarantee the chain makes — the
        ``NEVER_DOWNGRADE_EXC`` re-raise and the
        ``on_error="ignore"`` breadcrumb — which it does via the
        shared ``_handle_sink_error``.
        """
        try:
            await sink.handle(event)
        except Exception as exc:
            force_no_raise = sink.mode == "best_effort"
            self._handle_sink_error(sink, event, exc, force_no_raise=force_no_raise)

    def _handle_sink_error(
        self,
        sink: EventSink,
        event: SandboxSessionEvent,
        exc: BaseException,
        *,
        force_no_raise: bool = False,
    ) -> None:
        """Enforce the sink's ``on_error`` contract on a delivery failure.

        Programmer-bug / interpreter-fault exceptions
        (``NEVER_DOWNGRADE_EXC``) are re-raised unconditionally —
        ``on_error="log"`` / ``"ignore"`` / ``best_effort`` MUST NOT
        bury a real defect. Every other failure follows the sink's
        ``on_error``: ``raise`` wraps + chains the original cause;
        ``log`` logs at ERROR with full event identity; ``ignore``
        (and any ``force_no_raise`` non-log) still leaves a DEBUG
        breadcrumb so the drop is traceable in post-mortem.
        """
        if isinstance(exc, NEVER_DOWNGRADE_EXC):
            raise exc

        sink_label = type(sink).__name__
        if force_no_raise or sink.on_error in ("log", "ignore"):
            if sink.on_error == "log":
                logger.error(
                    "sandbox event sink failed (ignored per on_error=log): sink=%s event_id=%s session=%s seq=%s op=%s",
                    sink_label,
                    event.event_id,
                    event.session_id,
                    event.seq,
                    event.op,
                    exc_info=exc,
                )
            else:
                logger.debug(
                    "sandbox event sink failed — dropped (on_error=%s mode=%s): "
                    "sink=%s event_id=%s session=%s seq=%s op=%s",
                    sink.on_error,
                    sink.mode,
                    sink_label,
                    event.event_id,
                    event.session_id,
                    event.seq,
                    event.op,
                )
            return
        raise RuntimeError(f"sandbox event sink failed: {sink_label} while handling event {event.event_id}") from exc
