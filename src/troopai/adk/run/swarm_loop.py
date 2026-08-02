"""Swarm driver loop — orchestrates multi-agent swarm execution.

This module is the **composition layer** on top of ``run_agent_loop``:
it owns the cross-agent control flow, while the single-agent runner
still owns per-turn execution. The seam between them is narrow:

- The driver computes per-turn ``extra_tools`` / ``swarm_tool_names``
  from the active :class:`SwarmPolicy` and threads them through
  ``run_agent_loop``.
- When the inner loop detects a policy-injected tool call it returns a
  :class:`RunResult` with ``swarm_yield`` set (see
  :class:`~troopai.adk.run.next_step.NextStepSwarmYield`).
- The driver inspects ``result.swarm_yield``, updates state, runs
  :class:`~troopai.adk.swarms.termination.TerminationCondition` checks,
  and calls :meth:`SwarmPolicy.select_next` for the next turn.

Deliberate non-goals:

- No parallel intra-swarm execution. Agents take serial turns.
- No auto-injection of system prompts. If a member needs swarm
  awareness the developer calls
  :func:`troopai.adk.swarms.swarm_prompt.prompt_with_swarm_instructions`
  explicitly.
- No provider-specific wire types. Inputs are Layer 1
  :class:`~troopai.adk.types.input.LLMInputContentItem`; outputs are
  Layer 3 :class:`~troopai.adk.types.items.items.RunItem`.

Hard guards (``SwarmConfig.max_handoffs`` / ``max_total_tokens``) and
soft conditions (``TerminationCondition``) both live above this loop;
the driver merely evaluates them.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import time
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from troopai.adk.exceptions import AgentToolDeferral
from troopai.adk.graphs.interrupt import InterruptException, NestedAgentInterrupt
from troopai.adk.run.context import RunContext
from troopai.adk.run.llm_calls import resolve_compaction_llm, resolve_model_name
from troopai.adk.run.loop import (
    build_initial_messages,
    inject_system_prompt,
    run_agent_loop,
)
from troopai.adk.run.swarm_resume import run_resumed_hitl_turn, run_resumed_nested_turn
from troopai.adk.swarms.hooks import HookRegistry
from troopai.adk.swarms.shared_context import prepare_turn_input
from troopai.adk.swarms.state import SwarmState
from troopai.adk.swarms.stop_reason import StopReason
from troopai.adk.swarms.yield_signal import SwarmDone, SwarmHandoff
from troopai.adk.tracing.spans import Span, swarm_turn_span
from troopai.adk.types.items.items import ItemHelpers
from troopai.adk.types.tokens.llm_usage import LLMUsage
from troopai.adk.types.tracing.span_data import CustomSpanData, SwarmTurnSpanData

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.types import UserPrompt
    from troopai.adk.swarms.checkpointer import SwarmCheckpointer
    from troopai.adk.swarms.interrupt import SwarmResume
    from troopai.adk.swarms.result import SwarmRunResult
    from troopai.adk.swarms.swarm import Swarm
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.items.items import RunItem
    from troopai.adk.types.run.run_result import RunResult


logger = logging.getLogger(__name__)


def _usage_delta(before: LLMUsage, after: LLMUsage) -> LLMUsage:
    """Compute a field-wise delta ``after - before``.

    :class:`LLMUsage` defines ``__add__`` but not ``__sub__`` — the
    driver needs the delta for per-member attribution. Only the four
    numeric top-level counters are diffed; nested ``*_details`` are
    not accumulated in the per-member dict because cache-hit /
    reasoning-token breakdowns are meaningful per-request, not
    aggregated across runs.
    """
    return LLMUsage(
        requests=max(0, after.requests - before.requests),
        total_tokens=max(0, after.total_tokens - before.total_tokens),
        input_tokens=max(0, after.input_tokens - before.input_tokens),
        output_tokens=max(0, after.output_tokens - before.output_tokens),
    )


def _snapshot_usage(usage: LLMUsage) -> LLMUsage:
    """Shallow copy of the counters we need for delta computation.

    The ``usage`` attribute on :class:`RunContext` is rebound each
    time ``context.usage = context.usage + response.usage`` runs, so
    a simple reference grab would be stable. We still take a snapshot
    to make intent explicit and to survive any future in-place mutation.
    """
    return LLMUsage(
        requests=usage.requests,
        total_tokens=usage.total_tokens,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )


def user_prompt_text(user_prompt: UserPrompt) -> str:
    """Extract a string representation of the initial user prompt.

    Used when the swarm driver synthesizes a ``SwarmHandoff`` for
    policy-driven transitions (see the ``else`` branch in
    :func:`run_swarm_loop`).

    ``UserPrompt`` is ``str | list[LLMInputContentItem]``. When it is a
    list, the helper walks the items in **reverse** and returns the
    text of the **most-recent user-role message** (multi-turn replay:
    the last user prompt in the sequence is the one the target agent
    should answer). Content may be a plain string or a list of content
    blocks; for the block-list case, the first ``LLMInputText`` block
    (discriminator ``type == "input_text"``) wins. Non-text blocks
    (images, audio) and non-message items in the union
    (``LLMResponseFunctionToolCallParam``, ``FunctionToolCallResultParam``,
    reasoning, provider items) are skipped — their ``.get("role")``
    either returns ``None`` or a non-``"user"`` value.

    Empty string is returned only when no user-role text is reachable
    at all. An empty synthesized message would leave the target agent
    with ``messages=[]`` again (``_prepare_scoped`` skips zero-length
    handoff payloads), so callers of ``run_swarm_loop`` should prefer
    non-empty prompts. LAST_N / FULL_BROADCAST strategies read from
    ``state.shared_history`` directly and do not depend on this helper.
    """
    if isinstance(user_prompt, str):
        return user_prompt
    if isinstance(user_prompt, list):
        for msg in reversed(user_prompt):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "input_text":
                        text = block.get("text", "")
                        if isinstance(text, str) and len(text) > 0:
                            return text
    return ""


def _accumulate(
    per_member_usage: dict[str, LLMUsage],
    agent_name: str,
    delta: LLMUsage,
) -> None:
    """Add ``delta`` into the per-member running total, in place.

    First write for an agent creates the entry; subsequent writes add
    via the stock ``LLMUsage.__add__`` operator so the details slots
    are preserved as zeros.
    """
    existing = per_member_usage.get(agent_name)
    if existing is None:
        per_member_usage[agent_name] = delta
    else:
        per_member_usage[agent_name] = existing + delta


def _fold_turn_usage(
    state: SwarmState,
    per_member_usage: dict[str, LLMUsage],
    agent_name: str,
    usage_before: LLMUsage,
    ctx_wrapper: RunContext,
) -> None:
    """Fold a (possibly partial) turn's usage into cumulative + per-member totals.

    Used by the interrupt / deferral handlers so a member's pre-suspend tokens
    are still attributed even though the turn returned before the normal
    step-8 accumulation ran. Without it, the parked turn's tokens vanish from
    ``cumulative_usage`` (undercounting the ``max_total_tokens`` guard on
    resume) and from the returned per-member breakdown.
    """
    delta = _usage_delta(usage_before, _snapshot_usage(ctx_wrapper.usage))
    state.cumulative_usage = state.cumulative_usage + delta
    _accumulate(per_member_usage, agent_name, delta)


def _stamp_turn_span(
    turn_span: Span[SwarmTurnSpanData],
    *,
    status: str,
    monotonic_start: float,
    tracing_enabled: bool,
    metrics_enabled: bool = False,
) -> None:
    """Stamp terminal attributes on ``turn_span`` and finish it.

    When both tracing and metrics are disabled the span is a
    :class:`NoOpSpan` whose ``data`` field is the typed
    :class:`SwarmTurnSpanData` payload — NOT a :class:`CustomSpanData`
    envelope. The ``cast()`` itself is a no-op at runtime; it is the
    subsequent dict subscript ``.data["..."]`` that would raise
    ``TypeError`` because ``SwarmTurnSpanData`` is a dataclass, not a
    dict. Guard on either flag so the disabled path only calls
    ``finish()`` (which is a no-op).

    Args:
        turn_span: The per-turn span returned by
            :func:`swarm_turn_span`.
        status: Terminal status to record on the span payload.
        monotonic_start: ``time.monotonic()`` snapshot taken at
            span-open; used to compute ``duration_ms``.
        tracing_enabled: Mirror of ``RunConfig.tracing_enabled`` —
            passed explicitly (not derived from ``config``) so the
            helper is unit-testable in isolation.
        metrics_enabled: Mirror of ``RunConfig.metrics_enabled``.
            When ``True``, the span carries real data for metric recording.
    """
    if tracing_enabled or metrics_enabled:
        payload = cast(CustomSpanData, turn_span.data).data
        payload["status"] = status
        payload["duration_ms"] = int((time.monotonic() - monotonic_start) * 1000)
    turn_span.finish()


async def _run_member_turn_guarded(
    *,
    agent: Agent,
    guardrail_input: UserPrompt,
    ctx_wrapper: RunContext,
    hooks: RunHooks,
    config: RunConfig,
    max_turns: int,
    turn_messages: list[LLMInputContentItem],
    extra_tools: list[FunctionTool] | None,
    swarm_tool_names: set[str] | None,
) -> RunResult:
    """Run one normal swarm member turn wrapped in input + output guardrails.

    The streamed driver runs each member turn through ``Runner._run_streamed``,
    which applies the agent's input and output guardrails; the synchronous
    driver called ``run_agent_loop`` directly and skipped them entirely. This
    bracket restores parity: blocking input guardrails before the turn,
    parallel input guardrails alongside it, and output guardrails on the
    member's result. A tripwire propagates out and aborts the swarm, matching
    the streamed driver (whose inner stream re-raises it on drain).
    """
    from troopai.adk.run.guardrails_executor import (
        run_blocking_input_guardrails,
        run_output_guardrails,
        run_parallel_input_guardrails,
    )
    from troopai.adk.run.runner import apply_output_transform

    await run_blocking_input_guardrails(
        agent,
        guardrail_input,
        ctx_wrapper,
        hooks,
        config.guardrails.input,
        tracing_enabled=config.tracing_enabled,
        metrics_enabled=config.metrics_enabled,
    )
    parallel_task = asyncio.create_task(
        run_parallel_input_guardrails(
            agent,
            guardrail_input,
            ctx_wrapper,
            hooks,
            config.guardrails.input,
            tracing_enabled=config.tracing_enabled,
            metrics_enabled=config.metrics_enabled,
        )
    )
    try:
        result = await run_agent_loop(
            agent=agent,
            user_prompt=guardrail_input,
            context=ctx_wrapper,
            ctx_wrapper=ctx_wrapper,
            hooks=hooks,
            max_turns=max_turns,
            config=config,
            initial_messages=turn_messages,
            initial_new_items=None,
            extra_tools=extra_tools,
            swarm_tool_names=swarm_tool_names,
        )
        await parallel_task
    except BaseException:
        if not parallel_task.done():
            parallel_task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await parallel_task
        raise
    if not result.requires_action:
        output_agent = result.last_agent or agent
        await run_output_guardrails(
            output_agent,
            result.final_output,
            ctx_wrapper,
            hooks,
            config.guardrails.output,
            on_transform=lambda replacement: apply_output_transform(result, replacement),
            tracing_enabled=config.tracing_enabled,
            metrics_enabled=config.metrics_enabled,
        )
    return result


async def run_swarm_loop(
    swarm: Swarm,
    user_prompt: UserPrompt,
    ctx_wrapper: RunContext,
    hooks: RunHooks,
    config: RunConfig,
    initial_state: SwarmState | None = None,
    swarm_resume: SwarmResume | None = None,
    swarm_id: str | None = None,
    checkpointer: SwarmCheckpointer | None = None,
) -> SwarmRunResult:
    """Execute a swarm run end-to-end.

    Sequence per turn:

    1. Check termination condition against current state. Fire
       :class:`StopReason` and exit if it says stop.
    2. Check hard guards (``max_handoffs``, ``max_total_tokens``).
    3. Increment ``state.total_turns``; fire ``on_swarm_turn_start``.
    4. Build the next turn's messages. Turn 1 uses
       :func:`build_initial_messages` (system + user prompt). Turn N
       uses the :class:`SharedContextStrategy` to build the body,
       with a fresh system prompt injected for the current agent.
    5. Ask the policy for ``extra_tools``; derive
       ``swarm_tool_names`` for the runner's yield-detection path.
    6. Snapshot ``ctx_wrapper.usage`` for per-member delta.
    7. Call :func:`run_agent_loop` and inspect ``result.swarm_yield``.
    8. Extend ``state.shared_history`` and per-agent scratch; update
       ``cumulative_usage``.
    9. Dispatch the yield:
       - :class:`SwarmHandoff` — record, advance agent, bump counters.
       - :class:`SwarmDone` — record on state; termination check
         catches it at step 1 of the next iteration via
         :class:`ExplicitDoneTermination`.
       - ``None`` (free-form turn) — pass control to ``policy.select_next``.
    10. Fire ``on_swarm_turn_end`` + emit ``SwarmTurnEndEvent``.

    This driver never raises ``MaxTurnsExceeded`` itself — that
    surfaces from the inner loop when ``RunConfig.max_total_turns``
    is exceeded. The swarm's own max-handoffs / max-total-tokens
    guards instead stop the loop cleanly with a
    :class:`~troopai.adk.swarms.stop_reason.StopReason`
    (``kind="max_handoffs"`` / ``kind="max_total_tokens"``) so the
    caller still receives a well-formed ``SwarmRunResult``.

    Args:
        swarm: The :class:`Swarm` configuration (roster, policy,
            termination, shared-context strategy, budgets).
        user_prompt: The initial user input (string or Layer 1 item list).
        ctx_wrapper: The run context. Usage is accumulated across
            the whole swarm run — same instance is passed into each
            inner ``run_agent_loop`` call.
        hooks: Run-level hooks. ``SwarmHooks`` on the swarm fire in
            addition to ``RunHooks``.
        config: Run configuration (tracing, usage limits, ctx_mgmt).
            ``config.max_total_turns`` still applies as the absolute
            safety net on LLM-call count.
        initial_state: Optional :class:`SwarmState` carried over from a
            prior checkpoint. When supplied, ``total_turns`` /
            ``shared_history`` / ``per_agent_scratch`` continue from
            the parked turn boundary. Without ``swarm_resume``, any
            parked interrupts and nested-agent snapshots are dropped
            and the parked turn re-runs from scratch.
        swarm_resume: Optional :class:`SwarmResume` carrying per-member
            replies. When provided alongside ``initial_state``, the
            splice at step 7 substitutes a deep-resume call for the
            parked member: a nested-agent-defer reply is applied via
            :meth:`AgentExecutable.resume_from_snapshot`, and a pure
            HITL reply is seeded onto the run context for the parked
            member's tool to consume on its re-fire.

    Returns:
        A :class:`SwarmRunResult` with the terminal output, the final
        :class:`SwarmState`, per-member usage, and the triggering
        :class:`StopReason`.

    Raises:
        MaxTurnsExceeded: Propagated from the inner runner when
            ``config.max_total_turns`` is exceeded.
        ValueError: When the active policy returns an agent not in
            ``swarm.members`` (validated here to avoid silent
            misrouting).
    """
    from troopai.adk.swarms.result import SwarmRunResult

    # ── Hook registry ──────────────────────────────────────────────
    # A single registry fans all lifecycle events to every subscriber.
    # swarm.hooks is a single observer; checkpointer.register adds its
    # own persistence hook. An empty registry is a no-op, so the
    # None-guard on swarm.hooks is intentionally dropped throughout.
    hook_registry = HookRegistry()
    if swarm.hooks is not None:
        hook_registry.add(swarm.hooks)
    if checkpointer is not None:
        checkpointer.register(hook_registry)

    # ── Initial state ─────────────────────────────────────────────
    # When ``initial_state`` is supplied (resume path via
    # ``Runner.arun_swarm_from_checkpoint``) carry over total_turns,
    # shared_history, etc. Parked ``pending_interrupts`` /
    # ``nested_agent_snapshots`` are dropped here only when the caller
    # did NOT pass a ``swarm_resume`` payload — that's the
    # clear-and-restart path. With a payload, the parked entries stay
    # intact so the step-7 splice can apply the typed reply.
    state: SwarmState
    if initial_state is not None:
        state = initial_state
        if swarm_resume is None:
            state.pending_interrupts.clear()
            state.nested_agent_snapshots.clear()
        state.status = "running"
        if state.current_agent_name not in state.per_agent_scratch:
            state.per_agent_scratch[state.current_agent_name] = []
    else:
        state = SwarmState(
            swarm=swarm,
            current_agent=swarm.entry,
            current_agent_name=swarm.entry.name,
        )
        state.per_agent_scratch[swarm.entry.name] = []

    # Effective swarm_id: prefer the loaded state's id (resume), then the
    # kwarg (runner-supplied), else generate a fresh one defensively so
    # the turn-span factory always has a non-empty id.
    if initial_state is not None and initial_state.swarm_id is not None:
        effective_swarm_id = initial_state.swarm_id
    elif swarm_id is not None:
        effective_swarm_id = swarm_id
    else:
        effective_swarm_id = str(uuid.uuid4())
    state.swarm_id = effective_swarm_id

    per_member_usage: dict[str, LLMUsage] = {}
    final_output: Any = None

    # Fire swarm-start hook (RunHooks + SwarmHooks)
    await hook_registry.on_swarm_start(ctx_wrapper, state)

    is_first_turn = initial_state is None
    # Resuming an interrupt that fired during the very first turn: that turn
    # never completed, so its scoped scratch is empty and SCOPED/broadcast
    # rebuilds would hand the member an empty body (no question). total_turns
    # == 1 on the loaded state marks a turn-1 interrupt (a completed turn 1
    # advances past 1 before parking later). Rebuild from the opening prompt.
    resume_first_turn = (
        initial_state is not None
        and swarm_resume is not None
        and initial_state.total_turns <= 1
        and len(initial_state.pending_interrupts) > 0
    )

    while True:
        # ── Step 1: termination (soft) ─────────────────────────────
        reason = swarm.termination.should_stop(state)
        if reason is not None:
            logger.info(
                "Swarm terminated by condition: kind=%s detail=%s",
                reason.kind,
                reason.detail,
            )
            break

        # ── Step 2: hard guards ────────────────────────────────────
        if state.handoff_count >= swarm.config.max_handoffs:
            reason = StopReason(
                kind="max_handoffs",
                detail=(f"Hit the {swarm.config.max_handoffs}-handoff hard guard (observed {state.handoff_count})."),
            )
            break
        if (
            swarm.config.max_total_tokens is not None
            and state.cumulative_usage.total_tokens >= swarm.config.max_total_tokens
        ):
            reason = StopReason(
                kind="max_total_tokens",
                detail=(f"Consumed {state.cumulative_usage.total_tokens}/{swarm.config.max_total_tokens} tokens."),
            )
            break

        # ── Step 3: turn start ─────────────────────────────────────
        state.total_turns += 1
        current_agent = state.current_agent

        # Open the per-turn span BEFORE the start hook so any spans
        # the hook emits via the framework's contextvar-tracked tracer
        # nest correctly under this turn span. Closure is unified in
        # the finally below so every exit path stamps + finishes the
        # span exactly once (covers steps 4-9 exceptions, the two
        # except clauses below, the out-of-roster continue, the
        # policy_error break, and the success path).
        turn_span = swarm_turn_span(
            swarm_id=effective_swarm_id,
            index=state.total_turns,
            member=current_agent.name,
            disabled=not (config.tracing_enabled or config.metrics_enabled),
        )
        turn_span.start()
        turn_start_monotonic = time.monotonic()

        turn_status: str | None = None
        try:
            await hook_registry.on_swarm_turn_start(ctx_wrapper, state, current_agent.name)

            # ── Step 4: build this turn's messages ─────────────────
            turn_messages: list[LLMInputContentItem]
            if is_first_turn or resume_first_turn:
                turn_messages = await build_initial_messages(
                    current_agent,
                    user_prompt,
                    ctx_wrapper,
                )
                # Record the opening prompt once so cross-agent broadcast
                # strategies keep the question visible on later turns without
                # duplicating it into shared_history / new_items.
                if len(state.initial_input_items) == 0:
                    state.initial_input_items = list(
                        ItemHelpers.messages_to_run_items(ItemHelpers.input_to_new_input_list(user_prompt))
                    )
            else:
                body = await prepare_turn_input(
                    state=state,
                    next_agent=current_agent,
                    last_yield=state.last_yield,
                    config=swarm.config.shared_context,
                    compaction_llm=resolve_compaction_llm(current_agent, config),
                    compaction_model=resolve_model_name(current_agent, config),
                    context=ctx_wrapper,
                )
                # inject_system_prompt replaces/inserts the system message in-place
                turn_messages = await inject_system_prompt(
                    current_agent,
                    list(body),
                    ctx_wrapper,
                )

            # ── Step 5: policy tool injection ──────────────────────
            extra_tools = swarm.policy.build_step_tools(state)
            swarm_tool_names: set[str] = {t.name for t in extra_tools}

            # ── Step 6: per-member usage snapshot ──────────────────
            usage_before = _snapshot_usage(ctx_wrapper.usage)

            # ── Step 7: delegate to the single-agent runner ────────
            # Run with max_turns pulled from the agent; the driver
            # relies on max_total_turns + SwarmConfig.max_handoffs to
            # bound the overall run. Each inner run is a single "turn"
            # from the swarm's perspective even though the inner loop
            # may itself call the LLM multiple times for tool chaining.
            max_turns = getattr(current_agent, "max_turns", None) or 10
            try:
                parked_interrupt = state.pending_interrupts.get(current_agent.name)
                parked_snap = state.nested_agent_snapshots.get(current_agent.name)
                if parked_interrupt is not None and swarm_resume is not None and parked_snap is not None:
                    result = await run_resumed_nested_turn(
                        member=current_agent,
                        swarm_resume=swarm_resume,
                        state=state,
                        ctx_wrapper=ctx_wrapper,
                        config=config,
                    )
                elif parked_interrupt is not None and swarm_resume is not None:
                    result = await run_resumed_hitl_turn(
                        member=current_agent,
                        swarm_resume=swarm_resume,
                        state=state,
                        ctx_wrapper=ctx_wrapper,
                        config=config,
                        hooks=hooks,
                        user_prompt=user_prompt,
                        is_first_turn=is_first_turn,
                        turn_messages=turn_messages,
                        extra_tools=list(extra_tools) if len(extra_tools) > 0 else None,
                        swarm_tool_names=swarm_tool_names if len(swarm_tool_names) > 0 else None,
                        max_turns=max_turns,
                    )
                else:
                    result = await _run_member_turn_guarded(
                        agent=current_agent,
                        guardrail_input=user_prompt if is_first_turn else "",
                        ctx_wrapper=ctx_wrapper,
                        hooks=hooks,
                        config=config,
                        max_turns=max_turns,
                        turn_messages=turn_messages,
                        extra_tools=list(extra_tools) if len(extra_tools) > 0 else None,
                        swarm_tool_names=swarm_tool_names if len(swarm_tool_names) > 0 else None,
                    )
            except InterruptException as exc:
                # HITL pause via request_human_input from inside the
                # member's tool. Park the interrupt under the member's
                # name and exit with stop_reason.kind == "interrupted".
                from troopai.adk.swarms.result import SwarmRunResult

                turn_status = "interrupted"
                _fold_turn_usage(state, per_member_usage, current_agent.name, usage_before, ctx_wrapper)
                state.pending_interrupts[current_agent.name] = exc.interrupt
                state.status = "interrupted"
                await hook_registry.on_swarm_turn_interrupt(ctx_wrapper, state, current_agent.name, exc.interrupt)
                reason = StopReason(
                    kind="interrupted",
                    detail=(
                        f"member {current_agent.name!r} suspended on "
                        f"{type(exc.interrupt).__name__}({exc.interrupt.kind!r})"
                    ),
                )
                return SwarmRunResult(
                    final_output=None,
                    stop_reason=reason,
                    user_prompt=user_prompt,
                    new_items=list(state.shared_history),
                    state=state,
                    last_agent=current_agent,
                    context=ctx_wrapper,
                    per_member_usage=per_member_usage,
                    total_turns=state.total_turns,
                    handoff_count=state.handoff_count,
                    interrupts=(exc.interrupt,),
                )
            except AgentToolDeferral as deferral:
                # Nested-agent defer: lift to NestedAgentInterrupt +
                # park the deferring agent's RunState in
                # nested_agent_snapshots.
                from troopai.adk.swarms.result import SwarmRunResult

                turn_status = "interrupted"
                _fold_turn_usage(state, per_member_usage, current_agent.name, usage_before, ctx_wrapper)
                interrupt = NestedAgentInterrupt.from_deferral(
                    node_id=current_agent.name,
                    deferral=deferral,
                )
                state.pending_interrupts[current_agent.name] = interrupt
                state.nested_agent_snapshots[current_agent.name] = deferral.state
                state.status = "interrupted"
                await hook_registry.on_swarm_turn_interrupt(ctx_wrapper, state, current_agent.name, interrupt)
                reason = StopReason(
                    kind="interrupted",
                    detail=(
                        f"member {current_agent.name!r} suspended on nested-agent "
                        f"defer ({len(deferral.deferred_requests.approvals)} tool call(s))"
                    ),
                )
                return SwarmRunResult(
                    final_output=None,
                    stop_reason=reason,
                    user_prompt=user_prompt,
                    new_items=list(state.shared_history),
                    state=state,
                    last_agent=current_agent,
                    context=ctx_wrapper,
                    per_member_usage=per_member_usage,
                    total_turns=state.total_turns,
                    handoff_count=state.handoff_count,
                    interrupts=(interrupt,),
                )

            is_first_turn = False
            resume_first_turn = False

            if config.tracing_enabled or config.metrics_enabled:
                resume_count = state.resume_counts.get(current_agent.name)
                if resume_count is not None and resume_count > 0:
                    cast(CustomSpanData, turn_span.data).data["resume_attempt"] = resume_count

            # ── Step 8: accumulate state ───────────────────────────
            turn_items: list[RunItem] = list(result.new_items)
            state.shared_history.extend(turn_items)
            if current_agent.name not in state.per_agent_scratch:
                state.per_agent_scratch[current_agent.name] = []
            state.per_agent_scratch[current_agent.name].extend(turn_items)

            usage_after = _snapshot_usage(ctx_wrapper.usage)
            delta = _usage_delta(usage_before, usage_after)
            _accumulate(per_member_usage, current_agent.name, delta)
            state.cumulative_usage = state.cumulative_usage + delta

            # Populate structured routing input if the agent had a
            # schema and produced structured output. The runner's
            # final_output already carries the parsed Pydantic model
            # in that case.
            if current_agent.output_schema is not None and result.final_output is not None:
                state.last_structured_output = result.final_output
            else:
                state.last_structured_output = None

            # ── Step 9: dispatch the yield ─────────────────────────
            yield_signal = result.swarm_yield

            if isinstance(yield_signal, SwarmHandoff):
                # Resolve target by name within the swarm's roster
                target_agent = None
                for m in swarm.members:
                    if m.name == yield_signal.target:
                        target_agent = m
                        break
                if target_agent is None:
                    # Target not in roster — surface as a stop
                    # condition ("handoff_to" termination may want to
                    # fire if the user configured it for out-of-roster
                    # targets).
                    logger.warning(
                        "swarm handoff target %r not found in swarm roster; "
                        "incrementing handoff_count so max_handoffs guard remains effective.",
                        yield_signal.target,
                    )
                    state.handoff_count += 1
                    state.last_yield = yield_signal
                    swarm.policy.record_yield(state, yield_signal)
                    await hook_registry.on_swarm_handoff(
                        ctx_wrapper,
                        state,
                        current_agent.name,
                        yield_signal.target,
                        yield_signal.message,
                    )
                    await hook_registry.on_swarm_turn_end(
                        ctx_wrapper,
                        state,
                        turn_items,
                    )
                    turn_status = "success"
                    continue

                state.handoff_count += 1
                state.last_yield = yield_signal
                # Seed the target's scratch with the handoff message
                # so SCOPED strategy surfaces it verbatim on the next
                # turn.
                if target_agent.name not in state.per_agent_scratch:
                    state.per_agent_scratch[target_agent.name] = []
                # Advance current agent. The next turn's
                # prepare_turn_input will surface SwarmHandoff.message
                # via SCOPED.
                state.advance_to(target_agent)

                # Notify policy + hooks
                swarm.policy.record_yield(state, yield_signal)
                await hook_registry.on_swarm_handoff(
                    ctx_wrapper,
                    state,
                    current_agent.name,
                    yield_signal.target,
                    yield_signal.message,
                )

            elif isinstance(yield_signal, SwarmDone):
                # Fill in final_output from the terminal agent's last
                # message — the runner emits swarm_done before
                # producing a final string, so use result.final_output
                # when present.
                resolved_output = result.final_output
                if resolved_output is None:
                    resolved_output = ItemHelpers.extract_last_text(turn_items)
                done_signal = replace(yield_signal, final_output=resolved_output)
                state.last_yield = done_signal
                final_output = resolved_output
                swarm.policy.record_yield(state, done_signal)

            else:
                # No explicit yield this turn — the agent finished
                # without routing. Fall back to the policy to decide
                # the next speaker (RoundRobin continues rotation;
                # LLMHandoffPolicy keeps the same agent, letting the
                # termination condition decide; StructuredRoutingPolicy
                # reads last_structured_output).
                if result.final_output is not None:
                    # Preserve the latest final_output so a
                    # termination condition that stops without
                    # SwarmDone (e.g. MaxTurnsTermination) can surface
                    # something useful.
                    final_output = result.final_output
                try:
                    next_agent = await swarm.policy.select_next(state, ctx_wrapper)
                except Exception as exc:
                    logger.warning(
                        "Policy select_next raised %s — terminating swarm with policy_error",
                        exc,
                    )
                    reason = StopReason(
                        kind="policy_error",
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                    await hook_registry.on_swarm_turn_end(
                        ctx_wrapper,
                        state,
                        turn_items,
                    )
                    turn_status = "policy_error"
                    break
                if next_agent not in swarm.members:
                    raise ValueError(
                        f"Policy {type(swarm.policy).__name__}.select_next returned "
                        f"agent '{next_agent.name}' which is not in Swarm.members."
                    )
                if next_agent is not current_agent:
                    state.handoff_count += 1
                    # SCOPED strategy feeds the next agent from
                    # per-agent scratch + the SwarmHandoff.message
                    # addressed to it. Policy-driven transitions
                    # (StructuredRoutingPolicy, RoundRobinPolicy,
                    # CustomPolicy) don't emit a yield, so a target
                    # with empty scratch would see [] messages and
                    # fail at the provider. Seed a synthetic handoff
                    # carrying the user's original prompt so SCOPED
                    # has something to hand over.
                    synthesized_yield = SwarmHandoff(
                        target=next_agent.name,
                        message=user_prompt_text(user_prompt),
                    )
                    state.last_yield = synthesized_yield
                    state.advance_to(next_agent)
                    # Mirror the explicit-yield path: notify the
                    # policy so subclasses that track routing history
                    # (CustomPolicy overrides, cycle-count-aware
                    # implementations) stay consistent across both
                    # transition shapes, and fire on_swarm_handoff so
                    # SwarmHooks subscribers observe every agent
                    # transition uniformly — not just the ones the
                    # LLM voluntarily emitted via transfer_to_<name>.
                    swarm.policy.record_yield(state, synthesized_yield)
                    await hook_registry.on_swarm_handoff(
                        ctx_wrapper,
                        state,
                        current_agent.name,
                        synthesized_yield.target,
                        synthesized_yield.message,
                    )

            # ── Step 10: turn end ──────────────────────────────────
            turn_status = "success"

            await hook_registry.on_swarm_turn_end(
                ctx_wrapper,
                state,
                turn_items,
            )
        finally:
            # ``turn_status is None`` here means an exception is propagating out
            # of the turn body (uncaught LLMError, guardrail tripwire, hook
            # exception, the out-of-roster ValueError, …) — the interrupt paths
            # return with an explicit status and the success/policy paths set
            # one. Record the swarm as failed so a resumed/checkpointed state
            # reflects the crash rather than a perpetual "running".
            if turn_status is None:
                pending_exc = sys.exc_info()[1]
                state.status = "failed"
                state.error = (
                    f"{type(pending_exc).__name__}: {pending_exc}" if pending_exc is not None else "unknown error"
                )
            # Single span-close site: defaults to "error" when no explicit
            # status was set so the span never leaks open and OTel parent-child
            # chaining for subsequent turns stays intact.
            _stamp_turn_span(
                turn_span,
                status=turn_status if turn_status is not None else "error",
                monotonic_start=turn_start_monotonic,
                tracing_enabled=config.tracing_enabled,
                metrics_enabled=config.metrics_enabled,
            )

    # Normal loop exit (a termination condition or hard guard broke the loop).
    # The interrupt paths return early with status "interrupted"; reaching here
    # with status still "running" means the run finished cleanly.
    if state.status == "running":
        state.status = "completed"

    # Terminal hook
    await hook_registry.on_swarm_done(
        ctx_wrapper,
        state,
        reason,
        final_output,
    )

    return SwarmRunResult(
        final_output=final_output,
        stop_reason=reason,
        user_prompt=user_prompt,
        new_items=list(state.shared_history),
        state=state,
        last_agent=state.current_agent,
        context=ctx_wrapper,
        per_member_usage=per_member_usage,
        total_turns=state.total_turns,
        handoff_count=state.handoff_count,
    )


__all__ = ["run_swarm_loop"]
