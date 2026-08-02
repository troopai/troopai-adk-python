"""Streamed analog of run_swarm_loop.

Mirrors the synchronous swarm-driver loop structurally: same
termination check, same hard guards, same per-turn lifecycle,
the deep-resume splice from swarm_resume.py, and the turn-span
tracing/metrics seam gates. The only difference
is that every documented seam emits an event into the supplied
SwarmRunResultStreaming's queue, and the per-turn member execution
is delegated to a streamed helper that drains the inner agent
runner's events through the queue.

Returns None; populates fields on the passed-in result and
calls result.complete() in its own finally so the consumer's
stream_events() exits cleanly even on crash.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from troopai.adk.exceptions import AgentToolDeferral
from troopai.adk.graphs.interrupt import InterruptException, NestedAgentInterrupt
from troopai.adk.llms.llm_usage import LLMUsage
from troopai.adk.run.llm_calls import resolve_compaction_llm, resolve_model_name
from troopai.adk.run.loop import build_initial_messages, inject_system_prompt
from troopai.adk.run.stream import CancelMode
from troopai.adk.run.swarm_loop import user_prompt_text
from troopai.adk.run.swarm_resume import (
    run_resumed_hitl_turn,
    run_resumed_nested_turn,
)
from troopai.adk.swarms.events import (
    SwarmDoneEvent,
    SwarmHandoffEvent,
    SwarmStartEvent,
    SwarmTurnEndEvent,
    SwarmTurnInterruptEvent,
    SwarmTurnStartEvent,
)
from troopai.adk.swarms.hooks import HookRegistry
from troopai.adk.swarms.shared_context import prepare_turn_input
from troopai.adk.swarms.state import SwarmState
from troopai.adk.swarms.stop_reason import StopReason
from troopai.adk.swarms.yield_signal import SwarmDone, SwarmHandoff
from troopai.adk.tracing.spans import swarm_turn_span
from troopai.adk.types.items.items import ItemHelpers
from troopai.adk.types.tracing.span_data import CustomSpanData

if TYPE_CHECKING:
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.types import UserPrompt
    from troopai.adk.swarms.checkpointer import SwarmCheckpointer
    from troopai.adk.swarms.interrupt import SwarmResume
    from troopai.adk.swarms.result import SwarmRunResultStreaming
    from troopai.adk.swarms.swarm import Swarm
    from troopai.adk.types.run.run_result import RunResult


logger = logging.getLogger(__name__)


async def run_swarm_loop_streamed(
    *,
    swarm: Swarm[Any],
    user_prompt: UserPrompt,
    ctx_wrapper: RunContext[Any],
    hooks: RunHooks[Any],
    config: RunConfig,
    result: SwarmRunResultStreaming[Any],
    initial_state: SwarmState[Any] | None = None,
    swarm_resume: SwarmResume | None = None,
    swarm_id: str | None = None,
    checkpointer: SwarmCheckpointer | None = None,
) -> None:
    """Streamed swarm driver. Returns None; mutates ``result``.

    Non-Interrupt exceptions are stored via ``result.set_exception()``
    so the consumer's ``stream_events()`` re-raises after drain.
    Always calls ``result.complete()`` in finally so the consumer
    exits cleanly on every path.
    """
    try:
        await _run_streamed_body(
            swarm=swarm,
            user_prompt=user_prompt,
            ctx_wrapper=ctx_wrapper,
            hooks=hooks,
            config=config,
            result=result,
            initial_state=initial_state,
            swarm_resume=swarm_resume,
            swarm_id=swarm_id,
            checkpointer=checkpointer,
        )
    except Exception as exc:
        result.set_exception(exc)
    except BaseException as exc:
        # asyncio.CancelledError and friends. A developer-issued immediate
        # cancel() is a clean, requested stop, so record the exception for the
        # consumer only when the cancellation came from outside (cancel_mode
        # not IMMEDIATE) — otherwise stream_events() would re-raise a spurious
        # CancelledError out of a naive ``cancel(); break``. Re-raise either
        # way to preserve cancellation propagation semantics.
        if result.cancel_mode != CancelMode.IMMEDIATE:
            result.set_exception(exc)
        raise
    finally:
        await result.complete()


async def _run_streamed_body(
    *,
    swarm: Swarm[Any],
    user_prompt: UserPrompt,
    ctx_wrapper: RunContext[Any],
    hooks: RunHooks[Any],
    config: RunConfig,
    result: SwarmRunResultStreaming[Any],
    initial_state: SwarmState[Any] | None,
    swarm_resume: SwarmResume | None,
    swarm_id: str | None,
    checkpointer: SwarmCheckpointer | None,
) -> None:
    # ── Hook registry ────────────────────────────────────────────────
    hook_registry = HookRegistry()
    if swarm.hooks is not None:
        hook_registry.add(swarm.hooks)
    if checkpointer is not None:
        checkpointer.register(hook_registry)

    # ── Initial state setup (mirrors run_swarm_loop) ─────────────────
    state: SwarmState[Any]
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

    # Resolve effective swarm_id, preferring an existing state id for cross-resume correlation.
    if initial_state is not None and initial_state.swarm_id is not None:
        effective_swarm_id = initial_state.swarm_id
    elif swarm_id is not None:
        effective_swarm_id = swarm_id
    else:
        effective_swarm_id = str(uuid.uuid4())
    state.swarm_id = effective_swarm_id

    # Emit SwarmStartEvent.
    await result.put_event(
        SwarmStartEvent(
            entry_agent=swarm.entry.name,
            member_names=tuple(m.name for m in swarm.members),
        )
    )

    # Fire swarm-start hook.
    await hook_registry.on_swarm_start(ctx_wrapper, state)

    is_first_turn = initial_state is None
    # Resuming an interrupt that fired during the first turn: rebuild the opening
    # prompt (build_initial_messages) instead of a SCOPED body that would be
    # empty (turn 1 never completed). Mirrors the sync driver. total_turns == 1
    # on the loaded state marks a turn-1 interrupt.
    resume_first_turn = (
        initial_state is not None
        and swarm_resume is not None
        and initial_state.total_turns <= 1
        and len(initial_state.pending_interrupts) > 0
    )
    reason: StopReason | None = None
    final_output: Any = None
    per_member_usage: dict[str, LLMUsage] = {}

    while True:
        # ── Step 1: termination check ────────────────────────────────
        reason = swarm.termination.should_stop(state)
        if reason is not None:
            logger.info(
                "Swarm terminated by condition: kind=%s detail=%s",
                reason.kind,
                reason.detail,
            )
            break

        # ── Step 2: hard guards ──────────────────────────────────────
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

        # ── Step 3: turn start ───────────────────────────────────────
        state.total_turns += 1
        current_agent = state.current_agent
        turn_span = swarm_turn_span(
            swarm_id=effective_swarm_id,
            index=state.total_turns,
            member=current_agent.name,
            disabled=not (config.tracing_enabled or config.metrics_enabled),
        )
        turn_span.start()
        turn_start_monotonic = time.monotonic()
        await hook_registry.on_swarm_turn_start(ctx_wrapper, state, current_agent.name)
        await result.put_event(
            SwarmTurnStartEvent(
                agent=current_agent.name,
                turn=state.total_turns,
            )
        )

        # ── Step 6: per-member usage snapshot (mirrors sync swarm loop) ──
        # Snapshot the shared context's usage BEFORE the turn so the
        # post-turn delta captures every token the turn consumed,
        # regardless of which dispatch path ran. Resume helpers fold
        # their tokens directly into ``ctx_wrapper.usage``; the normal
        # streamed path folds the fresh inner context's usage into
        # ``ctx_wrapper.usage`` below before the delta is taken.
        usage_before = _snapshot_usage(ctx_wrapper.usage)

        turn_status: str | None = None
        _step7_completed = False
        try:
            # ── Step 7: splice dispatch ──────────────────────────────
            parked_interrupt = state.pending_interrupts.get(current_agent.name)
            parked_snap = state.nested_agent_snapshots.get(current_agent.name)
            if parked_interrupt is not None and swarm_resume is not None and parked_snap is not None:
                inner_result = await run_resumed_nested_turn(
                    member=current_agent,
                    swarm_resume=swarm_resume,
                    state=state,
                    ctx_wrapper=ctx_wrapper,
                    config=config,
                )
            elif parked_interrupt is not None and swarm_resume is not None:
                # Build the turn input + step tools the sync loop builds (its
                # Steps 4-5) before resuming. run_resumed_hitl_turn does NOT
                # build them, so passing []/None left the resumed member with
                # no messages and no swarm tools.
                hitl_extra_tools = swarm.policy.build_step_tools(state)
                hitl_tool_names = {t.name for t in hitl_extra_tools}
                if is_first_turn or resume_first_turn:
                    hitl_turn_messages = await build_initial_messages(current_agent, user_prompt, ctx_wrapper)
                    _record_initial_input(state, user_prompt)
                else:
                    hitl_body = await prepare_turn_input(
                        state=state,
                        next_agent=current_agent,
                        last_yield=state.last_yield,
                        config=swarm.config.shared_context,
                        compaction_llm=resolve_compaction_llm(current_agent, config),
                        compaction_model=resolve_model_name(current_agent, config),
                        context=ctx_wrapper,
                    )
                    hitl_turn_messages = await inject_system_prompt(current_agent, list(hitl_body), ctx_wrapper)
                inner_result = await run_resumed_hitl_turn(
                    member=current_agent,
                    swarm_resume=swarm_resume,
                    state=state,
                    ctx_wrapper=ctx_wrapper,
                    config=config,
                    hooks=hooks,
                    user_prompt=user_prompt,
                    is_first_turn=is_first_turn,
                    turn_messages=hitl_turn_messages,
                    extra_tools=hitl_extra_tools if len(hitl_extra_tools) > 0 else None,
                    swarm_tool_names=hitl_tool_names if len(hitl_tool_names) > 0 else None,
                    max_turns=getattr(current_agent, "max_turns", None) or 10,
                )
            else:
                # ── Steps 4-5: build messages + inject policy tools ──────
                # Mirrors the sync swarm loop (swarm_loop.py:403-430).
                # Must be done here, before delegating to the inner
                # runner, so SharedContextStrategy and swarm_done /
                # transfer_to_<name> tools are visible to the LLM.
                if is_first_turn:
                    normal_turn_messages = await build_initial_messages(current_agent, user_prompt, ctx_wrapper)
                    _record_initial_input(state, user_prompt)
                else:
                    normal_body = await prepare_turn_input(
                        state=state,
                        next_agent=current_agent,
                        last_yield=state.last_yield,
                        config=swarm.config.shared_context,
                        compaction_llm=resolve_compaction_llm(current_agent, config),
                        compaction_model=resolve_model_name(current_agent, config),
                        context=ctx_wrapper,
                    )
                    normal_turn_messages = await inject_system_prompt(current_agent, list(normal_body), ctx_wrapper)
                normal_extra_tools = swarm.policy.build_step_tools(state)
                normal_swarm_tool_names: set[str] = {t.name for t in normal_extra_tools}

                inner_result = await _stream_member_turn(
                    member=current_agent,
                    user_prompt=user_prompt,
                    ctx_wrapper=ctx_wrapper,
                    hooks=hooks,
                    config=config,
                    is_first_turn=is_first_turn,
                    result=result,
                    initial_messages=normal_turn_messages,
                    extra_tools=normal_extra_tools if normal_extra_tools else None,
                    swarm_tool_names=normal_swarm_tool_names if normal_swarm_tool_names else None,
                )

            # Stamp resume_attempt on the turn span if the splice fired.
            if config.tracing_enabled or config.metrics_enabled:
                resume_count = state.resume_counts.get(current_agent.name)
                if resume_count is not None and resume_count > 0:
                    cast(CustomSpanData, turn_span.data).data["resume_attempt"] = resume_count
            _step7_completed = True

        except InterruptException as exc:
            turn_status = "interrupted"
            _fold_turn_usage(state, per_member_usage, current_agent.name, usage_before, ctx_wrapper)
            _stamp_turn_span_end(turn_span, turn_status, turn_start_monotonic, config)
            state.pending_interrupts[current_agent.name] = exc.interrupt
            state.status = "interrupted"
            await hook_registry.on_swarm_turn_interrupt(
                ctx_wrapper,
                state,
                current_agent.name,
                exc.interrupt,
            )
            await result.put_event(
                SwarmTurnInterruptEvent(
                    agent=current_agent.name,
                    turn=state.total_turns,
                    interrupt=exc.interrupt,
                )
            )
            reason = StopReason(
                kind="interrupted",
                detail=(
                    f"member {current_agent.name!r} suspended on {type(exc.interrupt).__name__}({exc.interrupt.kind!r})"
                ),
            )
            result.state = state
            result.stop_reason = reason
            result.interrupts = (exc.interrupt,)
            result.last_agent = current_agent
            result.context = ctx_wrapper
            result.total_turns = state.total_turns
            result.handoff_count = state.handoff_count
            result.per_member_usage = per_member_usage
            await result.put_event(SwarmDoneEvent(reason=reason, final_output=None))
            return
        except AgentToolDeferral as deferral:
            turn_status = "interrupted"
            _fold_turn_usage(state, per_member_usage, current_agent.name, usage_before, ctx_wrapper)
            _stamp_turn_span_end(turn_span, turn_status, turn_start_monotonic, config)
            lifted = NestedAgentInterrupt.from_deferral(
                node_id=current_agent.name,
                deferral=deferral,
            )
            state.pending_interrupts[current_agent.name] = lifted
            state.nested_agent_snapshots[current_agent.name] = deferral.state
            state.status = "interrupted"
            await hook_registry.on_swarm_turn_interrupt(
                ctx_wrapper,
                state,
                current_agent.name,
                lifted,
            )
            await result.put_event(
                SwarmTurnInterruptEvent(
                    agent=current_agent.name,
                    turn=state.total_turns,
                    interrupt=lifted,
                )
            )
            reason = StopReason(
                kind="interrupted",
                detail=(
                    f"member {current_agent.name!r} suspended on nested-agent "
                    f"defer ({len(deferral.deferred_requests.approvals)} tool call(s))"
                ),
            )
            result.state = state
            result.stop_reason = reason
            result.interrupts = (lifted,)
            result.last_agent = current_agent
            result.context = ctx_wrapper
            result.total_turns = state.total_turns
            result.handoff_count = state.handoff_count
            result.per_member_usage = per_member_usage
            await result.put_event(SwarmDoneEvent(reason=reason, final_output=None))
            return
        finally:
            # Safety net: an exception propagating out of the step-7 body (not
            # the Interrupt / Deferral paths, which return above) leaves
            # turn_status unset. Mark the run failed and stamp the span so it
            # doesn't leak. Skip when step 7 completed normally — the steps-8-10
            # block below owns the span close for that path.
            if turn_status is None and not _step7_completed:
                _mark_turn_failed(state, result)
                _stamp_turn_span_end(turn_span, "error", turn_start_monotonic, config)

        is_first_turn = False
        resume_first_turn = False

        # Steps 8-10 run outside step 7's try/except; a failure here (the
        # out-of-roster ValueError, a hook / event error, …) would otherwise
        # leak the open turn span. Wrap them so the span is always closed.
        try:
            # ── Step 8: accumulate state ─────────────────────────────────
            turn_items = list(inner_result.new_items)
            state.shared_history.extend(turn_items)
            if current_agent.name not in state.per_agent_scratch:
                state.per_agent_scratch[current_agent.name] = []
            state.per_agent_scratch[current_agent.name].extend(turn_items)

            # Accumulate this turn's token usage via a snapshot/delta on the
            # SHARED ctx_wrapper.usage — the same pattern the sync swarm loop
            # uses. Member turns run on the driver's RunContext (passed as
            # shared_run_context), so a turn's usage already accrued live on
            # ctx_wrapper and inner_result.context IS ctx_wrapper — the identity
            # guard below then correctly SKIPS the fold (the delta already
            # captures those live writes). The guard folds only the defensive
            # case where a turn result carries a DIFFERENT context whose usage
            # has not yet reached ctx_wrapper. Either way the delta captures
            # every path exactly once and reaches both the cumulative total the
            # max_total_tokens guard checks and the per-member breakdown.
            turn_ctx = inner_result.context
            if turn_ctx is not None and turn_ctx is not ctx_wrapper:
                ctx_wrapper.usage = ctx_wrapper.usage + turn_ctx.usage
            turn_delta = _usage_delta(usage_before, ctx_wrapper.usage)
            state.cumulative_usage = state.cumulative_usage + turn_delta
            existing = per_member_usage.get(current_agent.name)
            per_member_usage[current_agent.name] = turn_delta if existing is None else existing + turn_delta

            # ── Step 9: dispatch yield ───────────────────────────────────
            yield_signal = inner_result.swarm_yield
            if isinstance(yield_signal, SwarmHandoff):
                target_agent = None
                for m in swarm.members:
                    if m.name == yield_signal.target:
                        target_agent = m
                        break
                if target_agent is None:
                    # Target not in roster — surface as a routing attempt
                    # visible to consumers and hooks, then let the termination
                    # check on the next iteration decide whether to stop. Mirror
                    # the sync loop: bump handoff_count so the max_handoffs hard
                    # guard still trips on a member that loops on the same
                    # out-of-roster target, and notify the policy so routing
                    # state stays consistent across sync/streamed execution.
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
                    await result.put_event(
                        SwarmHandoffEvent(
                            from_agent=current_agent.name,
                            to_agent=yield_signal.target,
                            message=yield_signal.message,
                        )
                    )
                else:
                    state.handoff_count += 1
                    state.last_yield = yield_signal
                    if target_agent.name not in state.per_agent_scratch:
                        state.per_agent_scratch[target_agent.name] = []
                    state.advance_to(target_agent)
                    swarm.policy.record_yield(state, yield_signal)
                    await hook_registry.on_swarm_handoff(
                        ctx_wrapper,
                        state,
                        current_agent.name,
                        yield_signal.target,
                        yield_signal.message,
                    )
                    await result.put_event(
                        SwarmHandoffEvent(
                            from_agent=current_agent.name,
                            to_agent=yield_signal.target,
                            message=yield_signal.message,
                        )
                    )
            elif isinstance(yield_signal, SwarmDone):
                resolved_output = inner_result.final_output
                if resolved_output is None:
                    resolved_output = ItemHelpers.extract_last_text(turn_items)
                # Match the sync loop: store the SwarmDone with its final_output
                # resolved, not the raw signal (whose final_output is often None
                # because swarm_done is emitted before the terminal string).
                done_signal = replace(yield_signal, final_output=resolved_output)
                state.last_yield = done_signal
                final_output = resolved_output
                swarm.policy.record_yield(state, done_signal)
            else:
                if inner_result.final_output is not None:
                    final_output = inner_result.final_output
                try:
                    next_agent = await swarm.policy.select_next(state, ctx_wrapper)
                except Exception as exc:
                    # Mirror the sync loop: a policy that raises terminates the
                    # swarm gracefully with a policy_error stop reason instead of
                    # escaping uncaught (which the streamed driver would otherwise
                    # surface only as a bare set_exception).
                    logger.warning(
                        "Policy select_next raised %s — terminating swarm with policy_error",
                        exc,
                    )
                    reason = StopReason(kind="policy_error", detail=f"{type(exc).__name__}: {exc}")
                    _stamp_turn_span_end(turn_span, "policy_error", turn_start_monotonic, config)
                    await hook_registry.on_swarm_turn_end(ctx_wrapper, state, turn_items)
                    await result.put_event(SwarmTurnEndEvent(agent=current_agent.name, items=tuple(turn_items)))
                    break
                if next_agent not in swarm.members:
                    raise ValueError(
                        f"Policy {type(swarm.policy).__name__}.select_next returned "
                        f"agent {next_agent.name!r} which is not in Swarm.members."
                    )
                if next_agent is not current_agent:
                    state.handoff_count += 1
                    # Seed the synthetic handoff with the user's prompt text (the
                    # sync loop's contract) so a SCOPED policy hands the target a
                    # non-empty first message — message="" broke the target's
                    # first turn.
                    synth = SwarmHandoff(
                        target=next_agent.name,
                        message=user_prompt_text(user_prompt),
                    )
                    state.last_yield = synth
                    state.advance_to(next_agent)
                    swarm.policy.record_yield(state, synth)
                    await hook_registry.on_swarm_handoff(
                        ctx_wrapper,
                        state,
                        current_agent.name,
                        next_agent.name,
                        synth.message,
                    )
                    await result.put_event(
                        SwarmHandoffEvent(
                            from_agent=current_agent.name,
                            to_agent=next_agent.name,
                            message=synth.message,
                        )
                    )

            # ── Step 10: turn end ────────────────────────────────────────
            turn_status = "success"
            _stamp_turn_span_end(turn_span, turn_status, turn_start_monotonic, config)
            await hook_registry.on_swarm_turn_end(ctx_wrapper, state, turn_items)
            await result.put_event(SwarmTurnEndEvent(agent=current_agent.name, items=tuple(turn_items)))
        except BaseException:
            # Steps 8-10 raised after the member turn completed (the
            # out-of-roster ValueError, a hook / event error, …). Step 10's
            # success stamp never ran, so close the span here rather than
            # leaking it, and mark the run failed before propagating.
            _mark_turn_failed(state, result)
            if turn_status != "success":
                _stamp_turn_span_end(turn_span, "error", turn_start_monotonic, config)
            raise

    # ── Terminal: emit SwarmDoneEvent ────────────────────────────────
    # Guard before calling hooks so they never receive reason=None and
    # raise a misleading AttributeError on reason.kind / reason.detail.
    if reason is None:
        raise RuntimeError(
            "swarm streamed loop exited the while loop without setting a "
            "stop reason — every break path must assign reason"
        )
    # Normal loop exit (a termination condition or hard guard broke the loop).
    # The interrupt paths return early with status "interrupted"; reaching here
    # with status still "running" means the run finished cleanly.
    if state.status == "running":
        state.status = "completed"

    await hook_registry.on_swarm_done(ctx_wrapper, state, reason, final_output)
    result.state = state
    result.stop_reason = reason
    result.last_agent = state.current_agent
    result.context = ctx_wrapper
    result.final_output = final_output
    result.total_turns = state.total_turns
    result.handoff_count = state.handoff_count
    result.per_member_usage = per_member_usage
    await result.put_event(SwarmDoneEvent(reason=reason, final_output=final_output))


async def _stream_member_turn(
    *,
    member: Any,
    user_prompt: Any,
    ctx_wrapper: Any,
    hooks: Any,
    config: Any,
    is_first_turn: bool,
    result: SwarmRunResultStreaming[Any],
    initial_messages: list[Any] | None = None,
    extra_tools: list[Any] | None = None,
    swarm_tool_names: set[str] | None = None,
) -> RunResult[Any]:
    """Run one member turn via Runner._run_streamed and pipe events.

    Drains the inner stream's events into ``result``'s queue
    between this turn's ``SwarmTurnStartEvent`` and
    ``SwarmTurnEndEvent`` boundaries, then materialises the
    terminal :class:`RunResult` from the inner streaming result's
    populated fields so the outer loop's post-step-8 block can
    consume it uniformly with a fresh ``run_agent_loop`` call.

    Args:
        initial_messages: Pre-built turn messages (Step 4 output from
            ``prepare_turn_input`` / ``build_initial_messages``).
        extra_tools: Policy-injected tools (Step 5: ``transfer_to_*``
            and ``swarm_done``).
        swarm_tool_names: Names of the swarm-injected tools so that
            ``resolve_swarm_yield_step`` can recognise routing calls
            inside the streamed loop.

    Raises:
        InterruptException: Propagated from the inner stream when
            a member's tool raises ``request_human_input_in_swarm``;
            caught by ``run_swarm_loop_streamed``'s except clause.
        AgentToolDeferral: Propagated from the inner stream when a
            sub-agent defers; caught upstream.
    """
    from troopai.adk.run.runner import Runner
    from troopai.adk.types.run.run_result import RunResult as _RunResult

    inner_streaming = Runner._run_streamed(
        agent=member,
        user_prompt=user_prompt if is_first_turn else "",
        context=ctx_wrapper.context,
        # Share the driver's RunContext so this member turn accumulates cost /
        # usage onto it — the per-run dollar budget and usage limits then accrue
        # cumulatively across the swarm (matching the sync swarm) instead of
        # resetting on a fresh context each turn.
        shared_run_context=ctx_wrapper,
        hooks=hooks,
        run_config=config,
        initial_messages=initial_messages,
        extra_tools=extra_tools,
        swarm_tool_names=swarm_tool_names,
        # The swarm revisits this member across turns; disposing its
        # toolsets per turn would leave an MCP member with no tools on
        # revisit. run_swarm_loop_streamed's driver disposes every member
        # once in its own finally instead.
        dispose_toolsets=False,
    )

    # Drain inner events through the swarm queue. If the inner run raises (a
    # HITL interrupt or nested-agent defer), the member turn's pre-suspend usage
    # already accrued live on the shared ctx_wrapper (the member ran on it
    # directly), so the driver's interrupt handler attributes those tokens with
    # no fold. The identity guard below folds only the defensive case where the
    # inner context is somehow NOT ctx_wrapper.
    try:
        async for inner_event in inner_streaming.stream_events():
            await result.put_event(inner_event)
    except BaseException:
        inner_ctx = inner_streaming.context
        if inner_ctx is not None and inner_ctx is not ctx_wrapper:
            ctx_wrapper.usage = ctx_wrapper.usage + inner_ctx.usage
        raise

    # Inner stream completed — materialise the RunResult-shape value the outer
    # swarm loop's step-8 block consumes. Carry the inner streaming context
    # (normally the shared ctx_wrapper itself, since the member ran on it) so
    # step-8's delta reads this turn's usage; fall back to ctx_wrapper only if
    # the inner context is somehow unset. Because the context is shared, the
    # loop's identity guard skips the extra fold — the usage is already live on
    # ctx_wrapper.
    inner_context = inner_streaming.context if inner_streaming.context is not None else ctx_wrapper
    return _RunResult(
        final_output=inner_streaming.final_output,
        user_prompt=user_prompt if is_first_turn else "",
        new_items=list(inner_streaming.new_items),
        context=inner_context,
        last_agent=member,
        swarm_yield=inner_streaming.swarm_yield,
    )


def _snapshot_usage(usage: LLMUsage) -> LLMUsage:
    """Shallow copy of the counters needed for a per-turn usage delta.

    Mirrors the sync swarm loop's snapshot: ``ctx_wrapper.usage`` is
    rebound (not mutated in place) on every accumulation, but a copy
    makes the snapshot intent explicit and survives any future
    in-place mutation.
    """
    return LLMUsage(
        requests=usage.requests,
        total_tokens=usage.total_tokens,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
    )


def _usage_delta(before: LLMUsage, after: LLMUsage) -> LLMUsage:
    """Compute a field-wise delta ``after - before``.

    :class:`LLMUsage` defines ``__add__`` but not ``__sub__``; the
    swarm driver needs the delta for per-member attribution. Only the
    four numeric top-level counters are diffed (nested ``*_details``
    are per-request, not meaningfully aggregated across turns), and
    each is floored at zero so a non-monotonic snapshot can never
    subtract from the cumulative total.
    """
    return LLMUsage(
        requests=max(0, after.requests - before.requests),
        total_tokens=max(0, after.total_tokens - before.total_tokens),
        input_tokens=max(0, after.input_tokens - before.input_tokens),
        output_tokens=max(0, after.output_tokens - before.output_tokens),
    )


def _record_initial_input(state: SwarmState[Any], user_prompt: UserPrompt) -> None:
    """Record the run's opening prompt once for cross-agent broadcast strategies.

    ``shared_history`` holds only items each member *produced*, so the strategies
    that read it (``FULL_BROADCAST`` / ``LAST_N`` / ``SUMMARIZED``) never see the
    user's question on turn 2 onward. Recording it here on the first turn
    (idempotent — a no-op once populated, including on resume) keeps the
    question visible without duplicating it into ``shared_history`` /
    ``new_items``.
    """
    if len(state.initial_input_items) == 0:
        state.initial_input_items = list(
            ItemHelpers.messages_to_run_items(ItemHelpers.input_to_new_input_list(user_prompt))
        )


def _mark_turn_failed(state: SwarmState[Any], result: SwarmRunResultStreaming[Any]) -> None:
    """Tag the swarm state failed from the live exception and surface it.

    Called from the turn-body exception paths so a crashed streamed run reports
    ``status == "failed"`` (with the error message) on ``result.state`` instead
    of a perpetual ``"running"`` — matching the sync driver. A ``BaseException``
    that is not an ``Exception`` (a cancel, ``KeyboardInterrupt``) is left
    untagged: it is a control-flow signal, not a run failure.
    """
    exc = sys.exc_info()[1]
    if isinstance(exc, Exception):
        state.status = "failed"
        state.error = f"{type(exc).__name__}: {exc}"
        result.state = state


def _fold_turn_usage(
    state: SwarmState[Any],
    per_member_usage: dict[str, LLMUsage],
    agent_name: str,
    usage_before: LLMUsage,
    ctx_wrapper: RunContext[Any],
) -> None:
    """Fold a partial (interrupted / deferred) turn's usage into the totals.

    The interrupt and deferral handlers return before the normal step-8
    accumulation runs, so without this the parked turn's pre-suspend tokens
    vanish from ``cumulative_usage`` (undercounting the ``max_total_tokens``
    guard on resume) and from the per-member breakdown.
    """
    delta = _usage_delta(usage_before, _snapshot_usage(ctx_wrapper.usage))
    state.cumulative_usage = state.cumulative_usage + delta
    existing = per_member_usage.get(agent_name)
    per_member_usage[agent_name] = delta if existing is None else existing + delta


def _stamp_turn_span_end(
    turn_span: Any,
    status: str,
    monotonic_start: float,
    config: Any,
) -> None:
    """Close a turn span with status + duration."""
    if config.tracing_enabled or config.metrics_enabled:
        payload = cast(CustomSpanData, turn_span.data).data
        payload["status"] = status
        payload["duration_ms"] = int((time.monotonic() - monotonic_start) * 1000)
    turn_span.finish()


__all__ = ["run_swarm_loop_streamed"]
