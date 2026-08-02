"""FlowExecutor — drives Flow execution per :class:`FlowConfig`.

The executor is created per :meth:`Runner.arun_flow` call. It:

1. Builds a :class:`FlowTransitionTable` from the Flow class's
   :class:`FlowStepRegistry` (built once by :class:`FlowMeta`).
2. Seeds a pending queue with every ``@flow_start`` method.
3. In each iteration: pops the entire pending batch, invokes each step
   in parallel via :func:`asyncio.gather`, and resolves successors
   (direct listeners + routers + AND/OR gates) for each completed
   step.
4. Tracks total step invocations against :attr:`FlowConfig.max_steps`,
   halting cleanly with ``status="halted_max_steps"`` on overflow.
5. Handles step errors per :attr:`FlowConfig.error_policy` — either
   halts on first failure (``"halt"``) or routes to a ``@flow_listen("__error__")``
   handler (``"route_to_error_handler"``).
6. Evaluates per-step ``enabled`` / ``requires_approval`` gates and
   wraps step bodies in the configured ``timeout`` + ``max_retries``
   loop before successor dispatch. ``requires_approval`` short-circuits
   the run into ``status="deferred"`` with a populated checkpoint.
7. Builds the final :class:`FlowRunResult` from the accumulated state.

The executor is single-use: one instance per ``Runner.arun_flow`` call.
NOT thread-safe; designed for single-loop asyncio execution.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import inspect
import json
import logging
import time
from collections import deque
from collections.abc import Awaitable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from troopai.adk.flows.checkpoint import FlowCheckpoint
from troopai.adk.flows.deferred import FlowDeferredStep
from troopai.adk.flows.events import (
    FlowEndEvent,
    FlowEvent,
    FlowRouteEvaluatedEvent,
    FlowStartEvent,
    FlowStepDeferredEvent,
    FlowStepEndEvent,
    FlowStepErrorEvent,
    FlowStepRejectedEvent,
    FlowStepSkippedEvent,
    FlowStepStartEvent,
)
from troopai.adk.flows.exceptions import (
    FlowAgentDeferred,
    FlowDefinitionError,
    FlowStepDeferred,
    FlowStepGovernanceError,
    FlowStepGuardrailTripped,
    FlowStepRateLimitExceeded,
    FlowStepRejected,
    FlowStepSkipped,
)
from troopai.adk.flows.flow_wrappers import FlowRole, FlowStep
from troopai.adk.flows.registry import build_transition_table
from troopai.adk.flows.result import FlowRunResult, FlowRunStatus
from troopai.adk.flows.step_context import FlowStepContext
from troopai.adk.flows.step_guardrails import FlowStepGuardrailVerdict
from troopai.adk.flows.triggers import FLOW_ERROR_TRIGGER, FlowTriggerEvent, FlowTriggerKind
from troopai.adk.run.governance import emit_guardrail_audit
from troopai.adk.types.tokens.llm_usage import LLMUsage

if TYPE_CHECKING:
    from collections.abc import Callable

    from troopai.adk.flows.config import FlowConfig
    from troopai.adk.flows.definition import FlowDefinition
    from troopai.adk.flows.flow import Flow
    from troopai.adk.flows.registry import FlowTransitionTable


logger = logging.getLogger(__name__)


_RATE_LIMIT_RETRY_HEADROOM = 8
"""Extra iterations beyond ``rpm`` allowed inside :meth:`FlowExecutor._acquire_rate_limit`.

The bound is :attr:`FlowStepRateLimit.rpm` plus this constant — small
enough that a misconfigured rate limit fails loud rather than spinning,
large enough that legitimate burst-then-wait acquires complete cleanly.
"""


_FLOW_INTERNAL_SIGNALS: tuple[type[BaseException], ...] = (
    FlowStepDeferred,
    FlowAgentDeferred,
    FlowStepSkipped,
    FlowStepGuardrailTripped,
    FlowStepRejected,
)
"""Executor control-flow signals — NOT errors.

These are raised to drive the executor's own state machine (HITL
deferral, enablement skip, guardrail/rejection routing) and are caught at
the batch-processing boundary. They MUST propagate untouched through
:meth:`FlowExecutor._run_body`'s retry loop: retrying a control-flow
signal re-runs the step body and its inner agent (double billing) and
captures duplicate deferrals (corrupted checkpoint) — the same reason
cancellation is never retried. Grouped here so every error/retry boundary
shares one exclusion set.
"""


class FlowExecutor[StateT]:
    """Drives a :class:`Flow` to completion under a :class:`FlowConfig`.

    One executor per ``Runner.arun_flow`` call. The executor reads the
    flow's :class:`FlowStepRegistry`, compiles a
    :class:`FlowTransitionTable`, and walks the step graph.

    The optional ``on_event`` callback is invoked once per
    :class:`FlowEvent`. The streaming runner hooks into this to push
    events to a queue; the non-streaming runner leaves it as a no-op.

    HITL contract — mirrors :class:`FunctionTool.requires_approval` /
    :class:`RunState.approve` exactly:

    1. A step whose ``requires_approval`` gate trips emits
       :class:`FlowStepDeferredEvent`, captures a
       :class:`FlowDeferredStep`, and halts the run with
       ``status="deferred"`` — both on the streaming and non-streaming
       paths. There is NO live-inject channel; the streaming path
       behaves exactly like the non-streaming path (the stream simply
       ends after the deferred event).
    2. The developer records decisions on the returned
       :class:`FlowCheckpoint` via
       :meth:`FlowCheckpoint.approve` /
       :meth:`FlowCheckpoint.reject` — same shape as
       :meth:`RunState.approve` / :meth:`RunState.reject`.
    3. Resume via :meth:`Runner.arun_flow_from_checkpoint(flow, checkpoint)`
       — the checkpoint carries the decisions; there is NO separate
       ``approvals=`` kwarg.

    Args:
        flow: The :class:`Flow` instance to run.
        config: :class:`FlowConfig` bounds for the run.
        on_event: Optional event sink. ``None`` = no-op.
    """

    flow: Flow[StateT]
    """The :class:`Flow` instance being executed."""

    config: FlowConfig
    """Bounds for this run (max_steps, error policy, fan-out cap)."""

    definition: FlowDefinition
    """Pure-data topology snapshot — roles, step names, gate topology."""

    table: FlowTransitionTable
    """Precomputed dispatch table built from the flow's registry."""

    completed_steps: list[str]
    """Step method names that ran (in completion order). Skip slots NOT included."""

    step_count: int
    """Total step invocations attempted; checked against ``config.max_steps``."""

    and_arrivals: dict[str, set[str]]
    """``gate_id → set of arrived trigger names`` for AND-gate tracking."""

    consumed_gates: set[str]
    """``gate_id`` set for OR and AND gates that have already fired once."""

    errored: tuple[str, BaseException] | None
    """``(step_name, exception)`` when a step raised under ``"halt"`` policy; else ``None``."""

    on_event: Callable[[FlowEvent], None]
    """Event sink — streaming runner pushes to a queue; non-streaming uses no-op."""

    pending_triggers: dict[str, list[FlowTriggerEvent]]
    """``step_name → list of triggers`` that scheduled this step's next firing."""

    deferred_steps: list[FlowDeferredStep]
    """Steps captured by a ``requires_approval`` gate during this run."""

    pending_queue_snapshot: tuple[str, ...]
    """Snapshot of the pending queue at the moment a deferral halted the run."""

    last_invocation_triggers: dict[str, tuple[FlowTriggerEvent, ...]]
    """``step_name → triggers tuple`` captured by the most recent :meth:`_build_step_context`.

    Read by :meth:`_capture_agent_deferral` so the post-invocation
    rescue path can recover the trigger list the step body saw,
    even after :meth:`_build_step_context` has popped the
    :attr:`pending_triggers` entry.
    """

    rate_limit_buckets: dict[str, deque[float]]
    """Per-step ``deque`` of monotonic timestamps for sliding-window rate limiting."""

    step_caches: dict[str, _StepCache]
    """Per-step :class:`_StepCache` instance for :class:`FlowStepCachePolicy`."""

    per_step_usage: dict[str, LLMUsage]
    """``step_name → LLMUsage`` delta accumulated as each step finalises.

    Recorded on the success path (and as an empty :class:`LLMUsage` for a
    cache hit) so :meth:`_build_result` can surface per-step cost attribution
    on :attr:`FlowRunResult.per_step_usage`. Best-effort for concurrently
    executed steps, whose shared-context spend interleaves.
    """

    def __init__(
        self,
        flow: Flow[StateT],
        *,
        config: FlowConfig,
        on_event: Callable[[FlowEvent], None] | None = None,
    ) -> None:
        from troopai.adk.flows.definition import build_flow_definition
        from troopai.adk.flows.flow import collect_step_descriptions

        registry = flow.get_registry()
        descs = collect_step_descriptions(type(flow).__dict__)
        self.definition = build_flow_definition(registry, descriptions=descs)
        self.flow = flow
        self.config = config
        self.table = build_transition_table(
            registry,
            max_listeners_per_step=config.max_listeners_per_step,
        )
        # Immutable record of the class-level @flow_start method names from the
        # definition. _seed_executor_from_checkpoint replaces table.starts with
        # checkpoint.pending_steps, but FlowStartEvent.start_steps must always
        # reflect the declared @flow_start methods — not the resume pending
        # queue — so we capture the value here and never overwrite it.
        self._class_starts: tuple[str, ...] = tuple(sorted(self.definition.starts))
        self.completed_steps = []
        self.step_count = 0
        self.and_arrivals = {}
        self.consumed_gates = set()
        self.errored = None
        self.on_event = on_event or (lambda _event: None)
        self.pending_triggers = {}
        self.deferred_steps = []
        self.pending_queue_snapshot = ()
        self.last_invocation_triggers = {}
        self.rate_limit_buckets = {}
        self.step_caches = {}
        self.per_step_usage = {}

    async def run(self) -> FlowRunResult[StateT]:
        """Run the flow to completion and return the final result.

        Drives one BSP-like loop: drain pending, run in parallel,
        resolve successors, repeat. Bounded by
        :attr:`FlowConfig.max_steps`.

        Returns:
            A :class:`FlowRunResult` whose ``status`` reflects how the
            run terminated. Failures and deferrals surface via ``status``,
            not as raised exceptions; ``CancelledError`` /
            ``KeyboardInterrupt`` / ``SystemExit`` still propagate so
            cancellation and shutdown are handled correctly.
        """
        self.on_event(
            FlowStartEvent(
                flow_id=self.flow.flow_id,
                # Always use the class-level @flow_start names regardless of
                # whether table.starts was replaced by checkpoint.pending_steps
                # during a resume (see _seed_executor_from_checkpoint).
                start_steps=self._class_starts,
            )
        )

        pending: deque[str] = deque(self.table.starts)
        for start in self.table.starts:
            # setdefault, NOT assignment: on a checkpoint resume the runner has
            # already restored pending_triggers for the pending steps (which now
            # occupy table.starts). A plain assignment would reset that restored
            # provenance to empty, so a gate callable that inspects ctx.triggers
            # would behave differently on resume than on the cold-start path.
            self.pending_triggers.setdefault(start, [])

        while len(pending) > 0:
            batch = list(pending)
            pending.clear()
            if self.config.max_total_tokens is not None:
                run_ctx = self.flow.run_context
                if run_ctx is not None:
                    used = run_ctx.usage.input_tokens + run_ctx.usage.output_tokens
                    if used >= self.config.max_total_tokens:
                        return self._build_result(status="halted_max_tokens")
            if self.step_count + len(batch) > self.config.max_steps:
                return self._build_result(status="halted_max_steps")
            self.step_count += len(batch)
            results = await asyncio.gather(
                *(self._invoke_step(name) for name in batch),
                return_exceptions=True,
            )
            terminal = self._process_batch_results(batch, results, pending)
            if terminal is not None:
                return terminal
        self._warn_unconsumed_agent_resolutions()
        return self._build_result(status="completed")

    def _process_batch_results(
        self,
        batch: list[str],
        results: list[BaseException | str | None],
        pending: deque[str],
    ) -> FlowRunResult[StateT] | None:
        """Process one parallel batch's results; return a terminal result on halt.

        Updates :attr:`completed_steps`, dispatches successors into
        ``pending`` for non-error results, and enforces
        :attr:`FlowConfig.error_policy` for exceptions.

        Args:
            batch: The step names that just ran (parallel batch).
            results: ``asyncio.gather(return_exceptions=True)`` output.
            pending: Mutable queue to extend with successors.

        Returns:
            A terminal :class:`FlowRunResult` when the run should halt
            (error under ``"halt"`` policy with no recovery handler, or
            any deferred step in this batch); ``None`` to continue.
        """
        saw_defer = False
        saw_halt = False
        halted_error_steps: list[str] = []
        for step_name, outcome in zip(batch, results, strict=True):
            if isinstance(outcome, FlowStepSkipped):
                continue
            if isinstance(outcome, FlowStepDeferred):
                saw_defer = True
                continue
            if isinstance(outcome, FlowAgentDeferred):
                self._capture_agent_deferral(outcome)
                saw_defer = True
                continue
            if isinstance(outcome, FlowStepRejected):
                # Record the halt but DON'T early-return: a sibling in this same
                # batch may have deferred, and a deferral must win (see below).
                # _handle_rejection's side effects (events, error routing) still
                # run; only the terminal-result construction is deferred so that
                # FlowEndEvent fires at most once for the batch.
                if self._handle_rejection(outcome, pending):
                    saw_halt = True
                continue
            if isinstance(outcome, BaseException):
                if self._handle_error(step_name, outcome, pending):
                    saw_halt = True
                    halted_error_steps.append(step_name)
            else:
                self.completed_steps.append(step_name)
                pending.extend(self._resolve_next(step_name, outcome))
        # Prefer a deferral over an error/rejection halt from the same batch. A
        # deferred step's captured checkpoint (inner-agent run_state) is
        # unrecoverable if we return the error terminal instead, whereas an
        # errored/rejected sibling is NOT in completed_steps and re-runs on
        # resume — so preferring the deferral keeps the run recoverable without
        # permanently masking the error. `_build_result` emits FlowEndEvent, so
        # it is called at most once per batch (after the loop) regardless of how
        # many steps deferred or halted — duplicate calls would double-emit and
        # confuse streaming consumers tracking run end.
        if saw_defer:
            # A deferral preempts an errored sibling's halt (above). That errored
            # step is NOT in completed_steps and would otherwise vanish: it is
            # neither a deferred step nor in `pending`, so the resume checkpoint
            # would never re-fire it and its branch would be silently dropped.
            # Re-queue it so resume re-runs it (the deferral kept the run
            # recoverable precisely so nothing is lost). Rejected siblings are
            # deliberately NOT re-queued — their decision is already consumed.
            for name in halted_error_steps:
                pending.append(name)
            # Snapshot the post-batch pending queue NOW: a successful sibling in
            # this batch may have scheduled successors into `pending` (via
            # _resolve_next above) that must survive into the checkpoint built by
            # the deferred-result path below. Capturing in run() only happens
            # after this method returns — too late for _build_checkpoint.
            self.pending_queue_snapshot = tuple(pending)
            return self._build_result(status="deferred")
        if saw_halt:
            return self._build_result(status="failed")
        return None

    def _capture_agent_deferral(self, exc: FlowAgentDeferred) -> None:
        """Record a :class:`FlowAgentDeferred` as a :class:`FlowDeferredStep`.

        Pops triggers from :attr:`pending_triggers` (same contract as
        :meth:`_build_step_context`) so cyclic re-fires don't
        accumulate stale entries; emits a WARNING when the step name
        has no recorded triggers so a misconfigured ``defer_key``
        surfaces in logs rather than silently producing an
        empty-trigger deferral.

        Guards an empty :attr:`exc.defer_key` because resumption
        keys the agent-bridge map on this string — an empty key
        would silently send the resume down the cold-start path.
        """
        if len(exc.defer_key) == 0:
            raise FlowDefinitionError(
                f"FlowAgentDeferred for step {exc.step_name!r} has an empty defer_key; "
                f"pass an explicit defer_key=... to arun_flow_agent.",
            )
        # `_build_step_context` already popped the entry from
        # `pending_triggers` before the body ran — recover via the
        # back-channel `last_invocation_triggers` so the deferral
        # carries the same provenance the step body saw.
        if exc.step_name not in self.last_invocation_triggers:
            logger.warning(
                "Flow %s: agent deferral on step %r has no recorded triggers — "
                "defer_key=%r may not match any registered step.",
                self.flow.flow_id,
                exc.step_name,
                exc.defer_key,
            )
        ctx_triggers = self.last_invocation_triggers.pop(exc.step_name, ())
        self.deferred_steps.append(
            FlowDeferredStep(
                step_name=exc.step_name,
                kind="approval",
                triggers=ctx_triggers,
                policy=self._get_step_descriptor(exc.step_name).approval_policy,
                defer_key=exc.defer_key,
                agent_run_state=exc.run_state_data,
            )
        )
        self.on_event(
            FlowStepDeferredEvent(
                flow_id=self.flow.flow_id,
                step_name=exc.step_name,
                triggers=ctx_triggers,
                completed_steps=tuple(self.completed_steps),
            )
        )

    def _handle_error(
        self,
        step_name: str,
        exc: BaseException,
        pending: deque[str],
    ) -> bool:
        """Apply :attr:`FlowConfig.error_policy` to a step exception.

        Args:
            step_name: Step whose body OR pre-body governance hook
                (rate limit, guardrails, cache) raised.
                :class:`FlowStepGovernanceError` carries the hook
                breadcrumb so operators can tell the source apart.
            exc: The raised exception.
            pending: Queue to extend with the error-handler listener
                under ``"route_to_error_handler"`` policy.

        Returns:
            ``True`` when the run must halt (no handler routed the
            error); ``False`` when the run continues via error handler.
            The terminal :class:`FlowRunResult` is built once by the
            caller after the batch loop so :class:`FlowEndEvent` fires
            at most once even when several steps fail in one batch.
        """
        # Critical exceptions MUST propagate — never route through the
        # error policy. asyncio.gather(return_exceptions=True) captures
        # BaseException including CancelledError / KeyboardInterrupt /
        # SystemExit; if we routed those through @flow_listen("__error__"),
        # an operator Ctrl+C would silently fire a recovery branch
        # instead of stopping the run.
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise exc
        self.on_event(
            FlowStepErrorEvent(
                flow_id=self.flow.flow_id,
                step_name=step_name,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        )
        if self.config.error_policy == "halt":
            self.errored = (step_name, exc)
            return True
        error_listeners = self.table.direct_listeners.get(FLOW_ERROR_TRIGGER, ())
        if len(error_listeners) == 0:
            self.errored = (step_name, exc)
            logger.warning(
                "Flow %s step %r raised under error_policy=route_to_error_handler but no "
                "@flow_listen('__error__') is registered; halting.",
                self.flow.flow_id,
                step_name,
            )
            return True
        # Record a trigger event for each error listener so its FlowStepContext
        # carries non-empty ctx.triggers naming the failed step as source_step.
        # Error listeners are queued directly here (never via _resolve_arrival),
        # so without this the handler would fire with an empty trigger tuple and
        # no provenance for which step failed.
        error_event = FlowTriggerEvent(name=FLOW_ERROR_TRIGGER, source_step=step_name, kind="step_completion")
        for listener in error_listeners:
            self.pending_triggers.setdefault(listener, []).append(error_event)
        pending.extend(error_listeners)
        return False

    def _handle_rejection(
        self,
        exc: FlowStepRejected,
        pending: deque[str],
    ) -> bool:
        """Route a streaming-HITL rejection through ``error_policy``.

        Emits :class:`FlowStepRejectedEvent` and then defers to the
        same policy machinery as a step exception, so consumers can
        register ``@flow_listen("__error__")`` handlers that observe
        rejections alongside ordinary failures.

        Returns ``True`` when the rejection halts the run; ``False``
        when it routes to an error handler.
        """
        self.on_event(
            FlowStepRejectedEvent(
                flow_id=self.flow.flow_id,
                step_name=exc.step_name,
                message=exc.decision_message,
            )
        )
        return self._handle_error(exc.step_name, exc, pending)

    async def _invoke_step(self, name: str) -> str | None:
        """Invoke one step body and return its router label or ``None``.

        Evaluation order (see ``docs/flows/flows.md`` for the full
        discussion):

        1. Build the :class:`FlowStepContext`.
        2. Gate chain — ``enabled`` → resume-decision → ``requires_approval``.
        3. Resolve the cache key ONCE (used for both lookup and write).
        4. Cache lookup — hit short-circuits the body, restores
           the cached state snapshot, emits a balanced
           ``FlowStepStartEvent`` + ``FlowStepEndEvent`` pair.
        5. ``rate_limit`` acquire (only on a miss).
        6. ``guardrails.pre`` chain.
        7. Body invocation, wrapped in ``timeout`` + ``max_retries``.
        8. ``guardrails.post`` chain.
        9. Cache write (soft — logged but does not fail the step).

        Args:
            name: The step method name to invoke.

        Returns:
            The router's returned label (``str``) for ``@flow_router``
            methods; ``None`` for ``@flow_start`` / ``@flow_listen``.

        Raises:
            FlowDefinitionError: When a router returns a non-string or
                empty-string value.
            FlowStepSkipped: Internal — caught by the executor when
                ``enabled`` returns ``False``.
            FlowStepDeferred: Internal — caught by the executor when
                ``requires_approval`` returns ``True``.
            FlowStepRejected: Internal — caught by the executor when a
                resume-path decision is ``approved=False``.
        """
        # Validate the step name against the registry BEFORE invoking,
        # so a tampered checkpoint cannot trigger arbitrary methods on
        # the Flow class.
        role = self._role_of(name)
        step = self._get_step_descriptor(name)
        ctx = self._build_step_context(name)
        await self._gate_step(name, step, ctx)
        # Resolve the cache key ONCE — the lookup and write must
        # share the same key so a body that mutates state never
        # causes the cache to write under a different key than the
        # lookup queried.
        cache_key = await self._resolve_step_cache_key(step, ctx)
        if cache_key is not None:
            hit = self._lookup_cache(step, ctx, cache_key)
            if hit is not None:
                self.per_step_usage[name] = LLMUsage()
                # Emit Start before End so streaming consumers always see
                # balanced pairs — the miss path emits its Start inside
                # _run_step_with_governance, so the hit path must emit its own.
                self.on_event(
                    FlowStepStartEvent(
                        flow_id=self.flow.flow_id,
                        step_name=name,
                        step_count=self.step_count,
                    )
                )
                self.on_event(
                    FlowStepEndEvent(
                        flow_id=self.flow.flow_id,
                        step_name=name,
                        next_steps=(),
                        usage=LLMUsage(),
                    )
                )
                return hit.route_label
        return await self._run_step_with_governance(role, name, step, ctx, cache_key)

    def _usage_totals(self) -> tuple[int, int, int, int]:
        """Scalar snapshot of the shared run context's cumulative usage."""
        run_ctx = self.flow.run_context
        if run_ctx is None:
            return (0, 0, 0, 0)
        usage = run_ctx.usage
        return (usage.requests, usage.input_tokens, usage.output_tokens, usage.total_tokens)

    def _usage_since(self, before: tuple[int, int, int, int]) -> LLMUsage:
        """Scalar usage delta accumulated on the shared context since ``before``.

        Attribution is exact for sequentially executed steps. Steps that
        run concurrently against one shared context interleave their
        spend, in which case a step's delta can include a concurrent
        sibling's usage.
        """
        after = self._usage_totals()
        return LLMUsage(
            requests=max(0, after[0] - before[0]),
            input_tokens=max(0, after[1] - before[1]),
            output_tokens=max(0, after[2] - before[2]),
            total_tokens=max(0, after[3] - before[3]),
        )

    async def _run_step_with_governance(
        self,
        role: FlowRole,
        name: str,
        step: FlowStep,
        ctx: FlowStepContext[StateT],
        cache_key: str | None,
    ) -> str | None:
        """Run the rate-limit / pre-guardrail / body / post-guardrail / cache-write chain.

        Extracted so :meth:`_invoke_step` stays under the 60-line
        function cap and the post-gate sequence reads top-to-bottom
        in one place. ``cache_key`` was resolved once in
        :meth:`_invoke_step` and is reused on the write path.

        Every :class:`FlowStepStartEvent` is guaranteed to be balanced
        by a :class:`FlowStepEndEvent` via the ``finally`` block, even
        when the body raises or a guardrail trips. This ensures streaming
        consumers never see dangling start events.
        """
        await self._acquire_rate_limit(name, step)
        usage_before = self._usage_totals()
        await self._run_guardrails(step, ctx, phase="pre")
        end_emitted = False
        try:
            self.on_event(
                FlowStepStartEvent(
                    flow_id=self.flow.flow_id,
                    step_name=name,
                    step_count=self.step_count,
                )
            )
            method = getattr(self.flow, name)
            result = await self._run_body(method, step)
            await self._run_guardrails(step, ctx, phase="post")
            if cache_key is not None:
                self._store_cache(role, step, ctx, cache_key, result)
            final = self._finalize_step(role, name, result, usage=self._usage_since(usage_before))
            end_emitted = True
            return final
        finally:
            if not end_emitted:
                self.on_event(
                    FlowStepEndEvent(
                        flow_id=self.flow.flow_id,
                        step_name=name,
                        next_steps=(),
                        usage=self._usage_since(usage_before),
                    )
                )

    async def _gate_step(self, name: str, step: FlowStep, ctx: FlowStepContext[StateT]) -> None:
        """Evaluate ``enabled`` + resume-decision + ``requires_approval`` gates.

        Evaluation order — IMPORTANT:

        1. ``enabled`` is checked FIRST, on every invocation (cold
           start AND resume). A pre-approved-but-now-disabled step
           (state changed between defer and resume, or the
           ``enabled`` callable's underlying flag flipped) is
           silently skipped just like any cold-start disabled step.
        2. Pre-queued :class:`FlowApprovalDecision` from the resume
           checkpoint short-circuits the ``requires_approval`` gate
           — and the decision is *consumed* (popped from
           ``flow._pending_approvals``) so a cyclic re-fire of the
           same step does NOT silently auto-approve.
        3. ``requires_approval`` runs on the cold-start path only;
           on resume it never fires because the consumed decision
           already settled the question.

        Mirrors the tool gate chain in
        :mod:`troopai.adk.run.tools_executor`: enablement and approval
        are orthogonal gates with different semantics — one is a
        feature flag / dynamic skip, the other is a policy gate.
        """
        if not await step.check_enabled(ctx):
            self.on_event(
                FlowStepSkippedEvent(
                    flow_id=self.flow.flow_id,
                    step_name=name,
                    triggers=ctx.triggers,
                )
            )
            raise FlowStepSkipped(name)
        decision = self.flow.consume_pending_approval(name)
        if decision is not None:
            if not decision.approved:
                raise FlowStepRejected(name, message=decision.message)
            return
        if await step.check_requires_approval(ctx):
            self.deferred_steps.append(
                FlowDeferredStep(step_name=name, triggers=ctx.triggers, policy=step.approval_policy),
            )
            self.on_event(
                FlowStepDeferredEvent(
                    flow_id=self.flow.flow_id,
                    step_name=name,
                    triggers=ctx.triggers,
                    completed_steps=tuple(self.completed_steps),
                )
            )
            raise FlowStepDeferred(name)

    async def _run_body(self, method: Callable[..., Any], step: FlowStep) -> Any:
        """Invoke ``method`` honouring ``step.timeout`` + ``step.max_retries``.

        Retry semantics:
        - ``max_retries`` is the count of EXTRA attempts after the
          initial call. Total tries = ``max_retries + 1``.
        - Retries are skipped for cancellation-class exceptions
          (``CancelledError`` / ``KeyboardInterrupt`` / ``SystemExit``).
        - On exhaustion, every intermediate failure is logged at
          WARNING (operators see flaky steps without enabling DEBUG)
          and the final exception is raised. The exception chain is
          preserved via ``raise … from prev`` so the traceback
          surfaces the causal sequence — a TimeoutError on attempt N
          does not erase the RuntimeErrors on attempts 1..N-1.
        """
        attempts = (step.max_retries or 0) + 1
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            try:
                if step.timeout is not None:
                    return await asyncio.wait_for(method(), timeout=step.timeout)
                return await method()
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except _FLOW_INTERNAL_SIGNALS:
                # Control-flow signals (deferral / skip / guardrail / rejection)
                # are not failures — propagate immediately so the executor's
                # state machine handles them. Retrying would re-run the body +
                # inner agent (double billing) and duplicate the deferral.
                raise
            except BaseException as exc:
                if last_exc is not None:
                    exc.__cause__ = last_exc
                last_exc = exc
                if attempt + 1 >= attempts:
                    break
                logger.warning(
                    "Flow %s step %r attempt %d/%d failed (%s: %s); retrying.",
                    self.flow.flow_id,
                    step.__name__,
                    attempt + 1,
                    attempts,
                    type(exc).__name__,
                    exc,
                )
        if last_exc is None:
            # Unreachable — the loop either returned on success or set
            # last_exc on every failure. Explicit guard rather than
            # ``assert`` so the boundary check survives ``python -O``.
            raise FlowDefinitionError(
                f"Internal: _run_body for {step.__name__!r} exited without success or exception.",
            )
        raise last_exc

    async def _acquire_rate_limit(self, name: str, step: FlowStep) -> None:
        """Acquire a slot in the step's sliding-window rate limit, if configured.

        Maintains :attr:`rate_limit_buckets[name]` — a deque of
        monotonic timestamps. Drops timestamps older than 60s on
        each acquire. On saturation: ``"wait"`` (default) sleeps
        until the oldest timestamp expires, ``"error"`` raises
        :class:`FlowStepRateLimitExceeded`. The ``max_wait_seconds``
        cap converts ``"wait"`` semantics to ``"error"`` when the
        cumulative wait would exceed it.
        """
        cfg = step.rate_limit
        if cfg is None:
            return
        bucket = self.rate_limit_buckets.setdefault(name, deque())
        window = 60.0
        waited = 0.0
        # Bounded retry loop — every iteration either acquires (returns)
        # or sleeps for a positive ``retry_after``, so the bound
        # ``cfg.rpm + _RATE_LIMIT_RETRY_HEADROOM`` is provably
        # sufficient. Decoupled from ``FlowConfig.max_steps`` because
        # the two limits govern unrelated dimensions (flow depth vs.
        # acquire retries).
        for _ in range(cfg.rpm + _RATE_LIMIT_RETRY_HEADROOM):
            now = time.monotonic()
            while len(bucket) > 0 and now - bucket[0] >= window:
                bucket.popleft()
            if len(bucket) < cfg.rpm:
                bucket.append(now)
                return
            if cfg.behavior == "error":
                raise FlowStepRateLimitExceeded(name, cfg.rpm)
            retry_after = window - (now - bucket[0])
            if cfg.max_wait_seconds is not None and waited + retry_after > cfg.max_wait_seconds:
                raise FlowStepRateLimitExceeded(name, cfg.rpm)
            await asyncio.sleep(retry_after)
            waited += retry_after
        raise FlowStepRateLimitExceeded(name, cfg.rpm)

    async def _run_guardrails(
        self,
        step: FlowStep,
        ctx: FlowStepContext[StateT],
        *,
        phase: Literal["pre", "post"],
    ) -> None:
        """Evaluate the configured guardrail chain for ``phase`` ∈ {"pre", "post"}.

        First non-allow verdict short-circuits. Reject verdicts
        raise :class:`FlowStepGuardrailTripped`; raise-exception
        verdicts surface the carried exception directly. Empty
        guardrail tuples no-op.
        """
        bundle = step.guardrails
        if bundle is None:
            return
        chain = bundle.pre if phase == "pre" else bundle.post
        for guardrail in chain:
            try:
                raw_verdict = guardrail(ctx)
                if inspect.isawaitable(raw_verdict):
                    verdict: FlowStepGuardrailVerdict = await raw_verdict
                else:
                    verdict = raw_verdict
            except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
                raise
            except FlowStepGuardrailTripped:
                raise
            except BaseException as exc:
                raise FlowStepGovernanceError(
                    step_name=ctx.step_name,
                    hook="guardrail",
                    phase=phase,
                ) from exc
            if not isinstance(verdict, FlowStepGuardrailVerdict):
                raise FlowDefinitionError(
                    f"FlowStepGuardrails {phase}-callable returned {type(verdict).__name__}, "
                    f"expected FlowStepGuardrailVerdict.",
                )
            run_ctx = self.flow.run_context
            if run_ctx is not None:
                emit_guardrail_audit(
                    run_ctx,
                    level="flow_pre" if phase == "pre" else "flow_post",
                    agent_name=None,
                    guardrail_name=ctx.step_name,
                    action=verdict.resolved_action(),
                    checked=ctx.step_name,
                )
            if verdict.allowed:
                continue
            if verdict.exception is not None:
                raise verdict.exception
            raise FlowStepGuardrailTripped(
                step_name=ctx.step_name,
                phase=phase,
                message=verdict.message,
            )

    async def _resolve_step_cache_key(
        self,
        step: FlowStep,
        ctx: FlowStepContext[StateT],
    ) -> str | None:
        """Resolve the cache key for ``step`` against the current ``ctx``.

        Returns ``None`` when the step declares no cache policy —
        callers branch on ``None`` to skip both lookup and write.
        Otherwise returns the resolved ``str`` key.

        Resolved ONCE per step invocation (before body execution).
        Both the lookup and the post-body write reuse this value so a
        body that mutates state never causes the cache to write under
        a different key than the one looked up. Wraps any
        developer-supplied callable raise as
        :class:`FlowStepGovernanceError` so the resulting
        :class:`FlowStepErrorEvent` clearly identifies the failure
        source (cache hook, not step body).
        """
        cfg = step.cache
        if cfg is None:
            return None
        try:
            return await _resolve_cache_key(cfg.cache_key_fn(ctx))
        except (asyncio.CancelledError, KeyboardInterrupt, SystemExit):
            raise
        except FlowDefinitionError:
            raise
        except BaseException as exc:
            raise FlowStepGovernanceError(
                step_name=ctx.step_name,
                hook="cache_key_fn",
            ) from exc

    def _lookup_cache(
        self,
        step: FlowStep,
        ctx: FlowStepContext[StateT],
        key: str,
    ) -> _CacheHit | None:
        """Look up ``key`` in the step's cache; restore state on hit.

        Cache-hit semantics: cache hits skip the body AND the
        rate-limit acquire (no real resource is consumed). Document
        this when a step's :class:`FlowStepRateLimit` is meant to
        protect an external service from accidental amplification:
        cache hits do not count toward the ``rpm`` cap.
        """
        cache = self._get_step_cache(step, ctx)
        if cache is None:
            return None
        entry = cache.get(key)
        if entry is None:
            return None
        snapshot, route_label = entry
        _apply_cached_state(self.flow, snapshot)
        logger.debug(
            "Flow %s step %r served from cache (key=%r).",
            self.flow.flow_id,
            ctx.step_name,
            key,
        )
        return _CacheHit(route_label=route_label, key=key)

    def _store_cache(
        self,
        role: FlowRole,
        step: FlowStep,
        ctx: FlowStepContext[StateT],
        key: str,
        result: Any,
    ) -> None:
        """Write a post-body state snapshot into the cache under ``key``.

        ``key`` is the value resolved by :meth:`_resolve_step_cache_key`
        pre-body — passing it explicitly guarantees lookup/write
        consistency even when ``cache_key_fn`` reads mutable state.

        Only a ``@flow_router`` step's string return is cached as a
        route label. Non-router return values are ignored on the live
        path (:meth:`_finalize_step` returns ``None`` for them), so they
        MUST NOT be cached as a route label either — otherwise a cache
        hit would replay a spurious route the cache-miss path never
        produced.

        Cache-write failures (typically ``_snapshot_state`` raising
        :class:`FlowDefinitionError` on an un-serialisable state) are
        soft: they log at ERROR but DO NOT mark the step failed.
        Caching is an optimisation; a successful body must not be
        invalidated by a cache-infrastructure problem.
        """
        cache = self._get_step_cache(step, ctx)
        if cache is None:
            return
        try:
            snapshot = _snapshot_state(self.flow.state)
        except Exception:
            # Caching is a soft optimisation: a successful body must NEVER be
            # invalidated by a snapshot failure. That covers both the declared
            # FlowDefinitionError (unsupported state type) and a copy.deepcopy /
            # model_copy raising on an un-copyable field (a lock, an open handle,
            # a generator). CancelledError / KeyboardInterrupt / SystemExit are
            # BaseException, not Exception, so they still propagate.
            logger.exception(
                "Flow %s step %r cache write failed (state not snapshotable); body succeeded, dropping cache write.",
                self.flow.flow_id,
                ctx.step_name,
            )
            return
        route_label = result if (role == "router" and isinstance(result, str)) else None
        cache.put(key, snapshot, route_label)

    def _get_step_cache(
        self,
        step: FlowStep,
        ctx: FlowStepContext[StateT],
    ) -> _StepCache | None:
        """Return the per-step :class:`_StepCache`, creating it on first access.

        Extracted so the lookup and write paths share the same
        creation site — the invariant "lookup precedes write"
        is encoded in the call sequence inside :meth:`_invoke_step`,
        not in duplicated ``setdefault`` calls.
        """
        cfg = step.cache
        if cfg is None:
            return None
        return self.step_caches.setdefault(
            ctx.step_name,
            _StepCache(max_entries=cfg.max_entries, ttl_seconds=cfg.ttl_seconds),
        )

    def _finalize_step(self, role: FlowRole, name: str, result: Any, *, usage: LLMUsage) -> str | None:
        """Emit the post-body events for one step and validate router output.

        Returns the router label (str) for ``@flow_router`` steps;
        ``None`` for ``@flow_start`` / ``@flow_listen``.

        Args:
            role: The step's flow role (start / listen / router).
            name: The step method name.
            result: The body's return value.
            usage: Scalar usage delta attributed to this step, emitted on
                the :class:`FlowStepEndEvent`.
        """
        self.per_step_usage[name] = usage
        if role == "router":
            if not isinstance(result, str) or len(result) == 0:
                # Truncate result repr to bound log noise — flow_router
                # outputs may originate from LLMs and could be long.
                truncated = repr(result)[:200]
                raise FlowDefinitionError(
                    f"Router {name!r} must return a non-empty string label; got {truncated}.",
                )
            self.on_event(
                FlowStepEndEvent(
                    flow_id=self.flow.flow_id,
                    step_name=name,
                    next_steps=(),
                    usage=usage,
                )
            )
            return result
        self.on_event(
            FlowStepEndEvent(
                flow_id=self.flow.flow_id,
                step_name=name,
                next_steps=(),
                usage=usage,
            )
        )
        return None

    def _warn_unconsumed_agent_resolutions(self) -> None:
        """Emit a warning when agent_resolutions outlive the run.

        :meth:`Runner.arun_flow_from_checkpoint(agent_resolutions=...)`
        may legitimately supply more decisions than the resumed flow
        consumes (e.g. when a sibling deferral was already handled
        out of band). Surface the leftovers so developers can
        reconcile their out-of-band approval inventory; never raise.
        """
        leftover = self.flow.pending_agent_resolution_keys()
        if len(leftover) > 0:
            logger.warning(
                "Flow %s completed with %d unconsumed agent_resolutions: %r",
                self.flow.flow_id,
                len(leftover),
                leftover,
            )

    def _build_step_context(self, name: str) -> FlowStepContext[StateT]:
        """Construct the :class:`FlowStepContext` snapshot for one step invocation.

        Pops the accumulated triggers for ``name`` so a step that
        fires more than once in a run (rare but possible via routed
        cycles) gets a fresh trigger list each iteration. Records
        the popped tuple onto :attr:`last_invocation_triggers` so
        the agent-bridge rescue path can recover them after the
        step body raises :class:`FlowAgentDeferred`.
        """
        triggers = tuple(self.pending_triggers.pop(name, ()))
        self.last_invocation_triggers[name] = triggers
        return FlowStepContext[StateT](
            step_name=name,
            flow_id=self.flow.flow_id,
            flow_state=self.flow.state,
            context=self.flow.run_context,
            completed_steps=tuple(self.completed_steps),
            triggers=triggers,
        )

    def _get_step_descriptor(self, name: str) -> FlowStep:
        """Return the unbound :class:`FlowStep` descriptor for ``name``.

        Reads ``type(self.flow).__dict__`` so we get the class-level
        descriptor (with the gate config) rather than a bound copy
        produced by ``flow.<name>`` attribute access.
        """
        descriptor = type(self.flow).__dict__.get(name)
        if not isinstance(descriptor, FlowStep):
            raise FlowDefinitionError(
                f"Internal: step {name!r} resolves to non-FlowStep descriptor.",
            )
        return descriptor

    def _role_of(self, name: str) -> FlowRole:
        """Look up the role of a step by name from the definition.

        Args:
            name: Step method name.

        Returns:
            One of ``"start"`` / ``"listen"`` / ``"router"`` — typed as
            :data:`FlowRole` so callers can ``match`` exhaustively.

        Raises:
            FlowDefinitionError: When ``name`` is not in the definition —
                indicates an internal inconsistency, not user error.
        """
        role = self.definition.roles.get(name)
        if role == "start":
            return "start"
        if role == "router":
            return "router"
        if role == "listen":
            return "listen"
        raise FlowDefinitionError(
            f"Internal: unknown step name {name!r} not in definition.",
        )

    def _resolve_next(
        self,
        completed_step: str,
        route_label: str | None,
    ) -> list[str]:
        """Return the next step names to fire after ``completed_step``.

        Also records the :class:`FlowTriggerEvent` that scheduled each
        successor into :attr:`pending_triggers` so the next iteration's
        :class:`FlowStepContext` reflects accurate provenance.

        The returned fire list is deduplicated preserving first-occurrence
        order: when ``route_label`` equals an existing step's method name,
        the step-completion arrival and the route-label arrival resolve
        the same listener, and firing it twice in one batch would run its
        body twice. Both provenance trigger events are STILL recorded in
        :attr:`pending_triggers` — only the fire list is deduped.
        """
        nxt = self._resolve_arrival(completed_step, kind="step_completion")
        if route_label is not None:
            triggered = tuple(
                self._resolve_arrival(route_label, kind="route_label", source=completed_step),
            )
            nxt.extend(triggered)
            self.on_event(
                FlowRouteEvaluatedEvent(
                    flow_id=self.flow.flow_id,
                    router_step=completed_step,
                    route_label=route_label,
                    triggered_steps=triggered,
                )
            )
        return list(dict.fromkeys(nxt))

    def _resolve_arrival(
        self,
        trigger: str,
        *,
        kind: FlowTriggerKind,
        source: str | None = None,
    ) -> list[str]:
        """Return listener / router method names that fire on ``trigger`` arrival.

        ``kind`` is one of ``"step_completion"`` / ``"route_label"`` and
        is stamped on every :class:`FlowTriggerEvent` recorded into
        :attr:`pending_triggers`. ``source`` overrides the trigger's
        ``source_step`` for the route-label case (where the source is
        the router method, not the route literal).
        """
        fire: list[str] = []
        source_step = source if source is not None else trigger
        event = FlowTriggerEvent(name=trigger, source_step=source_step, kind=kind)
        for target in self.table.direct_listeners.get(trigger, ()):
            fire.append(target)
            self.pending_triggers.setdefault(target, []).append(event)
        for target in self.table.routers_for.get(trigger, ()):
            fire.append(target)
            self.pending_triggers.setdefault(target, []).append(event)
        for gate_id in self.table.and_gates_for.get(trigger, ()):
            if gate_id in self.consumed_gates:
                continue
            spec = self.table.and_gates[gate_id]
            arrivals = self.and_arrivals.setdefault(gate_id, set())
            arrivals.add(trigger)
            self.pending_triggers.setdefault(spec.listener_name, []).append(event)
            if arrivals >= spec.triggers:
                fire.append(spec.listener_name)
                self.consumed_gates.add(gate_id)
        for gate_id in self.table.or_gates_for.get(trigger, ()):
            if gate_id in self.consumed_gates:
                continue
            spec = self.table.or_gates[gate_id]
            fire.append(spec.listener_name)
            self.pending_triggers.setdefault(spec.listener_name, []).append(event)
            self.consumed_gates.add(gate_id)
        return fire

    def _build_checkpoint(self) -> FlowCheckpoint:
        """Capture the live executor state into a serialisable checkpoint.

        Called once when a deferral terminates the run, so the
        developer can persist the checkpoint and later resume via
        :meth:`Runner.arun_flow_from_checkpoint`.

        The pending-steps tuple includes both the carry-over queue
        (steps scheduled but not yet run) and the deferred step
        names — so the resume executor re-fires the deferred steps
        first, picks up their now-recorded :class:`FlowApprovalDecision`
        from the checkpoint, and proceeds.
        """
        state_data = encode_state(self.flow.state)
        state_type_name = type(self.flow.state).__module__ + "." + type(self.flow.state).__qualname__
        deferred_names_set = {d.step_name for d in self.deferred_steps}
        deferred_names = tuple(d.step_name for d in self.deferred_steps)
        pending = deferred_names + tuple(name for name in self.pending_queue_snapshot if name not in deferred_names_set)
        # Serialize pending_triggers for non-deferred pending steps so that
        # ctx.triggers is correctly reconstructed on resume. Without this,
        # gate callables that branch on ctx.triggers behave differently on
        # resume than on cold start.
        non_deferred_triggers = {
            name: list(triggers) for name, triggers in self.pending_triggers.items() if name not in deferred_names_set
        }
        return FlowCheckpoint.capture(
            flow_id=self.flow.flow_id,
            completed_steps=tuple(self.completed_steps),
            pending_steps=pending,
            and_gate_arrivals=self.and_arrivals,
            consumed_gates=self.consumed_gates,
            state_data=state_data,
            state_type_name=state_type_name,
            deferred_steps=tuple(self.deferred_steps),
            pending_step_triggers=non_deferred_triggers,
        )

    def _build_result(self, *, status: FlowRunStatus) -> FlowRunResult[StateT]:
        """Assemble the final :class:`FlowRunResult`.

        Reads ``flow.run_context.usage`` for cumulative LLM usage if the
        developer wired their inner :meth:`Runner.arun` calls to share
        the context; defaults to an empty :class:`LLMUsage` otherwise.
        """
        error_msg: str | None = None
        if self.errored is not None:
            step_name, exc = self.errored
            error_msg = f"Step {step_name!r} raised: {type(exc).__name__}: {exc}"
        run_ctx = self.flow.run_context
        cumulative = run_ctx.usage if run_ctx is not None else LLMUsage()
        checkpoint = self._build_checkpoint() if status == "deferred" else None
        self.on_event(
            FlowEndEvent(
                flow_id=self.flow.flow_id,
                status=status,
                completed_steps=tuple(self.completed_steps),
                cumulative_usage=cumulative,
            )
        )
        return FlowRunResult(
            final_state=self.flow.state,
            flow_id=self.flow.flow_id,
            status=status,
            completed_steps=tuple(self.completed_steps),
            cumulative_usage=cumulative,
            per_step_usage=dict(self.per_step_usage),
            error=error_msg,
            deferred_steps=tuple(self.deferred_steps),
            checkpoint=checkpoint,
            guardrail_audit=run_ctx.collect_guardrail_audit() if run_ctx is not None else (),
        )


async def _resolve_cache_key(raw_key: str | Awaitable[str]) -> str:
    """Resolve a possibly-awaitable cache key to a concrete ``str``.

    Extracted so the executor's lookup and write paths share the
    exact same narrowing, and so the type-checker sees a uniform
    ``str`` value at both call sites without a ``cast``.
    """
    if inspect.isawaitable(raw_key):
        resolved = await raw_key
    else:
        resolved = raw_key
    if not isinstance(resolved, str):
        raise FlowDefinitionError(
            f"FlowStepCachePolicy.cache_key_fn must return str (or Awaitable[str]); got {type(resolved).__name__}.",
        )
    return resolved


def _snapshot_state(state: Any) -> Any:
    """Capture a deep copy of ``state`` for caching.

    Pydantic ``BaseModel`` → :meth:`BaseModel.model_copy(deep=True)`.
    ``@dataclass`` → :func:`copy.deepcopy`. Other types raise
    :class:`FlowDefinitionError` rather than caching a shallow
    alias (silent state aliasing would let post-hit step bodies
    corrupt the cache).
    """
    if isinstance(state, BaseModel):
        return state.model_copy(deep=True)
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        return copy.deepcopy(state)
    raise FlowDefinitionError(
        f"FlowStepCachePolicy: state type {type(state).__module__}.{type(state).__qualname__} "
        f"is not cacheable. Use a Pydantic BaseModel or @dataclass for StateT.",
    )


def _restore_state(snapshot: Any) -> Any:
    """Return a deep copy of a cached snapshot for restoration onto ``flow.state``.

    Mirrors :func:`_snapshot_state` — re-copying is mandatory so a
    cache hit does not silently alias the cached entry into the
    live flow state (the next step's mutations would then corrupt
    the cache).
    """
    if isinstance(snapshot, BaseModel):
        return snapshot.model_copy(deep=True)
    return copy.deepcopy(snapshot)


def _apply_cached_state(flow: Flow[Any], snapshot: Any) -> None:
    """Restore a cached state snapshot onto ``flow.state`` without a wholesale rebind.

    Copies the (freshly deep-copied) snapshot's fields onto the EXISTING
    ``flow.state`` object in place so a sibling step running concurrently in
    the same parallel batch — which captured the same ``flow.state`` reference
    via its :class:`FlowStepContext` — does not lose its writes to an object
    swap. Falls back to rebinding the reference when the state rejects in-place
    assignment (a frozen dataclass / frozen ``BaseModel``).

    Residual risk: the snapshot carries the FULL state, so a concurrent sibling
    writing a field the snapshot also captures can still race with this copy.
    The framework tracks no per-step field ownership; parallel steps must write
    disjoint fields or guard shared state with their own lock (the Flow
    parallel-listener contract).
    """
    restored = _restore_state(snapshot)
    live = flow.state
    if type(live) is type(restored) and isinstance(restored, BaseModel):
        names: tuple[str, ...] = tuple(type(restored).model_fields.keys())
    elif type(live) is type(restored) and dataclasses.is_dataclass(restored) and not isinstance(restored, type):
        names = tuple(f.name for f in dataclasses.fields(restored))
    else:
        flow.state = restored
        return
    try:
        for name in names:
            setattr(live, name, getattr(restored, name))
    except (AttributeError, TypeError, ValueError):
        # Frozen dataclass / frozen BaseModel reject in-place setattr; a
        # wholesale rebind is the only option there (concurrent-sibling writes
        # may be lost — the documented residual risk above).
        flow.state = restored


@dataclass(frozen=True, kw_only=True)
class _CacheHit:
    """Result of a successful step-cache lookup.

    Carries the router label when the cached step was a
    ``@flow_router`` (``None`` otherwise); the executor uses this
    to skip body invocation while still dispatching the cached
    successors.

    Attributes:
        route_label: Cached router label, or ``None`` for non-router
            cached steps. Used to replay the same successor dispatch
            without re-invoking the body.
        key: Resolved cache key at the time of lookup (informational —
            useful for logging and metrics).
    """

    route_label: str | None
    """Cached router label, or ``None`` for non-router cached steps."""

    key: str
    """Resolved cache key (informational — useful for logging / metrics)."""


@dataclass
class _StepCache:
    """Per-step LRU + TTL result cache.

    Stored on :attr:`FlowExecutor.step_caches`; one instance per
    step that declares a :class:`FlowStepCachePolicy`. Entries are
    deep-copied state snapshots (Pydantic ``model_copy(deep=True)``
    for BaseModel, :func:`copy.deepcopy` for dataclasses) so that
    cache writes capture an immutable view at body-exit time.

    Eviction is LRU when ``len > max_entries``; expiry is by
    ``time.monotonic`` when ``ttl_seconds`` is set.

    Attributes:
        max_entries: LRU cap on entries retained.
        ttl_seconds: Optional TTL in seconds; ``None`` ⇒ entries
            never expire, bounded only by :attr:`max_entries`.
        entries: ``key → (created_at_monotonic, state_snapshot,
            route_label)`` ordered dict providing LRU semantics.
    """

    max_entries: int
    """LRU cap on entries retained."""

    ttl_seconds: float | None
    """Optional TTL in seconds; ``None`` ⇒ never expires."""

    entries: dict[str, tuple[float, Any, str | None]] = field(default_factory=dict)
    """``key → (created_at_monotonic, state_snapshot, route_label)``."""

    def get(self, key: str) -> tuple[Any, str | None] | None:
        """Return the cached ``(state_snapshot, route_label)`` for ``key``, if any.

        Honours ``ttl_seconds`` (evicts on expiry) and refreshes the
        entry's LRU position on hit.
        """
        entry = self.entries.get(key)
        if entry is None:
            return None
        created_at, snapshot, route_label = entry
        if self.ttl_seconds is not None and time.monotonic() - created_at > self.ttl_seconds:
            del self.entries[key]
            return None
        # Refresh LRU position.
        del self.entries[key]
        self.entries[key] = entry
        return snapshot, route_label

    def put(self, key: str, snapshot: Any, route_label: str | None) -> None:
        """Insert ``(snapshot, route_label)`` under ``key`` and refresh LRU position.

        Removes any prior entry for the key BEFORE re-inserting so a
        re-write moves the entry to the most-recent position
        (Python dict overwrite preserves the original insertion
        order, which would silently violate the LRU contract).
        """
        if key in self.entries:
            del self.entries[key]
        self.entries[key] = (time.monotonic(), snapshot, route_label)
        while len(self.entries) > self.max_entries:
            oldest = next(iter(self.entries))
            del self.entries[oldest]


def encode_state(state: Any) -> str:
    """Serialise a Flow state instance to a JSON string.

    Pydantic ``BaseModel`` → ``model_dump_json()`` (handles datetimes,
    sets, and enums natively). ``@dataclass`` → ``json.dumps(asdict(state),
    default=str)``: ``default=str`` coerces the exotic values ``asdict``
    leaves in place (``datetime``, ``set``, ``Enum``, a nested
    ``BaseModel``) to strings rather than raising ``TypeError``. This runs
    inside :meth:`FlowExecutor._build_checkpoint`, PAST the
    ``error_policy`` boundary — a bare ``json.dumps`` would let that
    ``TypeError`` bubble unchecked out of :meth:`FlowExecutor.run`, so the
    coercion is mandatory even though it is lossy for round-tripping.
    Anything that is neither a BaseModel nor a dataclass raises
    :class:`FlowDefinitionError` with flow-specific context.
    """
    if isinstance(state, BaseModel):
        return state.model_dump_json()
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        return json.dumps(dataclasses.asdict(state), default=str)
    raise FlowDefinitionError(
        f"Flow state type {type(state).__module__}.{type(state).__qualname__} is not "
        f"checkpointable. Use a Pydantic BaseModel or @dataclass for StateT — the framework "
        f"refuses to fall back to a bare json.dumps that would lose semantics for non-trivial "
        f"shapes (datetimes, sets, enums, …).",
    )
