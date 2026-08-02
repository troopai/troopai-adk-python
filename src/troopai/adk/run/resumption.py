"""HITL state resumption — sync, streamed, and nested agent-tool.

Handles resuming interrupted runs after human approval/rejection
decisions have been made. Supports nested agent-tool deferral
resumption via lazy import of ``Runner.arun()``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from troopai.adk.exceptions import AgentToolDeferral, MaxTurnsExceeded
from troopai.adk.run.config import DEFAULT_MAX_TURNS, DEFAULT_RUN_CONFIG, get_messages
from troopai.adk.run.llm_calls import find_agent_by_name
from troopai.adk.types.items.items import ItemHelpers
from troopai.adk.verbose.hooks import (
    emit_hitl_approval_granted,
    emit_hitl_approval_rejected,
)

# FunctionToolCallResultParam imported lazily inside functions to avoid circular import

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext, TContext
    from troopai.adk.run.state import RunState
    from troopai.adk.run.stream import RunResultStreaming
    from troopai.adk.tools.deferred_tool import DeferredToolCall
    from troopai.adk.types.input import FunctionToolCallResultParam, LLMInputContentItem
    from troopai.adk.types.items.items import RunItem
    from troopai.adk.types.run import RunResult

logger = logging.getLogger(__name__)


def _build_redeferral_call(original_call_id: str, deferral: AgentToolDeferral) -> DeferredToolCall:
    """Rebuild a still-pending nested-agent deferral, preserving the call_id.

    Reusing ``original_call_id`` (the parent tool_use id) rather than a freshly
    synthesised id keeps the eventual ``function_call_output`` paired with the
    parent tool_use block once the nested approval finally resolves. A synthetic
    id would leave the parent tool_use unpaired and the new result orphaned.
    """
    from troopai.adk.tools.deferred_tool import (
        DeferredToolCall,
        DeferredToolCallMetadata,
        NestedDeferredToolRequests,
    )

    return DeferredToolCall(
        tool_call_id=original_call_id,
        tool_name=deferral.agent_name,
        tool_arguments={},
        raw_arguments="{}",
        metadata=DeferredToolCallMetadata(
            nested_agent=True,
            nested_agent_name=deferral.agent_name,
            nested_state=deferral.state.to_dict(),
            nested_deferred_requests=NestedDeferredToolRequests(
                approvals=[
                    DeferredToolCall(
                        tool_call_id=req.tool_call_id,
                        tool_name=req.tool_name,
                        tool_arguments=req.tool_arguments,
                        raw_arguments=req.raw_arguments,
                    )
                    for req in deferral.deferred_requests.approvals
                ],
            ),
        ),
    )


async def _resolve_approved_output(
    approved_tool: DeferredToolCall,
    agent: Agent,
    ctx_wrapper: RunContext[Any],
    hooks: RunHooks[TContext],
    config: RunConfig,
    context: Any,
    raw_hooks: RunHooks[TContext] | None,
    max_turns: int,
    messages: list[LLMInputContentItem],
) -> FunctionToolCallResultParam:
    """Produce the tool-result param for one approved deferred tool.

    A nested agent-tool resumes its sub-agent — which may raise
    ``AgentToolDeferral`` if it defers again; the caller keeps the original
    call_id so the eventual result stays paired with the parent tool_use.
    A regular tool is resolved and executed; an unresolvable tool yields a
    "not found" result so every tool_use keeps a matching tool_result
    (Anthropic requires the pairing).
    """
    from troopai.adk.run.llm_calls import resolve_function_tool
    from troopai.adk.run.tools_executor import execute_approved_tool
    from troopai.adk.types.input import FunctionToolCallResultParam

    if approved_tool.metadata is not None and approved_tool.metadata.nested_agent:
        result = await resume_nested_agent_tool(
            parent_agent=agent,
            deferred_tool=approved_tool,
            context=context,
            hooks=raw_hooks,
            max_turns=max_turns,
            config=config,
        )
        return FunctionToolCallResultParam(
            type="function_call_output",
            call_id=approved_tool.tool_call_id,
            output=str(result),
        )

    tool = await resolve_function_tool(agent, approved_tool.tool_name, ctx_wrapper)
    if tool is None or tool.on_invoke is None:
        return FunctionToolCallResultParam(
            type="function_call_output",
            call_id=approved_tool.tool_call_id,
            output=get_messages(config).tool_not_found_on_resumption(approved_tool.tool_name),
        )

    result_content, _ = await execute_approved_tool(
        agent=agent,
        approved_tool=approved_tool,
        ctx_wrapper=ctx_wrapper,
        hooks=hooks,
        config=config,
        context=context,
        messages=messages,
    )
    return FunctionToolCallResultParam(
        type="function_call_output",
        call_id=approved_tool.tool_call_id,
        output=result_content,
    )


def _build_redeferred_run_result(
    state: RunState,
    current_agent: Agent,
    messages: list[LLMInputContentItem],
    new_items: list[RunItem],
    redeferrals: list[DeferredToolCall],
) -> RunResult:
    """Build the interrupted RunResult for nested agent-tools that re-deferred.

    The remaining approved/rejected/external decisions were already applied to
    ``messages`` before this fires, so only the still-pending nested approvals
    surface as new deferred requests.
    """
    from troopai.adk.run.context import RunContext
    from troopai.adk.tools.deferred_tool import DeferredToolRequests
    from troopai.adk.types.run import RunResult

    new_deferred = DeferredToolRequests(approvals=list(redeferrals))
    new_state_obj = state.__class__(
        conversation_history=list(ItemHelpers.messages_to_run_items(messages)),
        context=state.context,
        deferred_tool_requests=new_deferred,
        original_user_prompt=state.original_user_prompt,
        current_agent_name=current_agent.name,
        turn_count=state.turn_count,
    )
    return RunResult(
        final_output=None,
        user_prompt=state.original_user_prompt,
        new_items=new_items,
        context=RunContext(context=state.context),
        last_agent=current_agent,
        deferred_requests=new_deferred,
        state=new_state_obj,
    )


def _apply_streamed_redeferral(
    streaming_result: RunResultStreaming,
    state: RunState,
    current_agent: Agent,
    messages: list[LLMInputContentItem],
    new_items: list[RunItem],
    redeferrals: list[DeferredToolCall],
) -> None:
    """Mark a streamed result interrupted by nested-agent re-deferrals.

    Mirrors :func:`_build_redeferred_run_result` for the streamed path: the
    remaining decisions were already applied to ``messages`` above, so only the
    still-pending nested approvals surface as new deferred requests.
    """
    from troopai.adk.run.state import RunState as _RunState
    from troopai.adk.tools.deferred_tool import DeferredToolRequests

    new_deferred = DeferredToolRequests(approvals=list(redeferrals))
    streaming_result.new_items.extend(new_items)
    streaming_result.deferred_requests = new_deferred
    streaming_result.state = _RunState(
        conversation_history=list(ItemHelpers.messages_to_run_items(messages)),
        context=state.context,
        deferred_tool_requests=new_deferred,
        original_user_prompt=state.original_user_prompt,
        current_agent_name=current_agent.name,
        turn_count=state.turn_count,
    )
    streaming_result.final_output = None


async def resume_from_state(
    agent: Agent,
    state: RunState,
    hooks: RunHooks[TContext] | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    config: RunConfig | None = None,
    context: Any = None,
) -> RunResult:
    """Resume execution from a saved RunState.

    This handles the resumption of an interrupted run after human
    approval decisions have been made.

    Args:
        agent: The agent to run.
        state: The RunState containing conversation history and approvals.
        hooks: Optional lifecycle hooks.
        max_turns: Maximum turns in the agent loop.
        config: Optional execution configuration.
        context: Optional caller-supplied context value. When provided,
            overrides ``state.context`` in the constructed
            :class:`RunContext`. This allows the flow agent bridge to
            thread the flow's shared ``run_context.context`` through
            resumed inner agent runs so cumulative usage is aggregated
            correctly.

    Returns:
        RunResult with final output or new deferred requests.
    """
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.guardrails_executor import run_output_guardrails
    from troopai.adk.run.loop import run_agent_loop
    from troopai.adk.run.runner import apply_output_transform, wrap_hooks_with_verbose
    from troopai.adk.types.input import FunctionToolCallResultParam

    config = config or DEFAULT_RUN_CONFIG
    # Preserve the unwrapped user hooks for nested-agent delegation:
    # resume_nested_agent_tool -> Runner.arun -> resume_from_state wraps
    # exactly once, so passing the already-wrapped chain there would
    # double-compose VerboseHooks.
    raw_hooks = hooks
    hooks = wrap_hooks_with_verbose(hooks, config)

    # Create context wrapper: caller-supplied context takes precedence so
    # the flow agent bridge can propagate the flow's shared run_context.
    # tenant_id flows from config so the per-tenant tool allowlist/budget
    # gates stay in force across HITL resume.
    effective_context = context if context is not None else state.context
    ctx_wrapper = RunContext.make(effective_context, tenant_id=config.tenant_id)

    # Get the current agent (may have changed due to handoffs)
    current_agent = agent
    if state.current_agent_name and state.current_agent_name != agent.name:
        found = find_agent_by_name(agent, state.current_agent_name)
        if found is not None:
            current_agent = found

    await hooks.on_agent_start(ctx_wrapper, current_agent)
    if current_agent.hooks is not None:
        await current_agent.hooks.on_start(ctx_wrapper, current_agent)

    # Convert RunItems (Layer 3) back to Layer 1 params
    messages: list[LLMInputContentItem] = ItemHelpers.run_items_to_params(state.conversation_history)
    new_items: list[RunItem] = []
    # Nested agent-tools that deferred AGAIN during this resume, keyed by their
    # ORIGINAL call_id. Collected instead of aborting the loop so the remaining
    # approved/rejected/external decisions are still applied.
    redeferrals: list[DeferredToolCall] = []

    try:
        # Process approved tools
        for approved_tool in state.approved_tools:
            # Close the pending HITL gate with an approved verdict
            # *before* we execute the tool. approval_metadata is optional
            # so we also gracefully handle auto-approved tools (no audit
            # record attached).
            approval_meta = state.approval_metadata.get(approved_tool.tool_call_id)
            emit_hitl_approval_granted(
                hooks,
                current_agent,
                approved_tool.tool_name,
                approved_tool.tool_call_id,
                approver_id=(approval_meta.approver_id if approval_meta is not None else None),
                reason=(approval_meta.reason if approval_meta is not None else None),
            )
            try:
                msg = await _resolve_approved_output(
                    approved_tool,
                    current_agent,
                    ctx_wrapper,
                    hooks,
                    config,
                    effective_context,
                    raw_hooks,
                    max_turns,
                    messages,
                )
            except AgentToolDeferral as deferral:
                # Sub-agent deferred again. Keep the original call_id so the
                # eventual result pairs with the parent tool_use, and keep
                # processing the other decisions rather than dropping them.
                redeferrals.append(_build_redeferral_call(approved_tool.tool_call_id, deferral))
                continue
            messages.append(msg)
            new_items.extend(ItemHelpers.message_to_run_items(msg, current_agent.name))

        # Process rejected tools
        _msgs = get_messages(config)
        for rejected_tool, message in state.rejected_tools:
            # Close the pending HITL gate with a rejected verdict.
            approval_meta = state.approval_metadata.get(rejected_tool.tool_call_id)
            emit_hitl_approval_rejected(
                hooks,
                current_agent,
                rejected_tool.tool_name,
                rejected_tool.tool_call_id,
                approver_id=(approval_meta.approver_id if approval_meta is not None else None),
                message=message,
            )
            # Check for nested agent-tool rejection
            if rejected_tool.metadata is not None and rejected_tool.metadata.nested_agent:
                agent_name = rejected_tool.metadata.nested_agent_name or "sub-agent"
                rejection_msg = message or _msgs.nested_agent_rejected(agent_name)
            else:
                rejection_msg = message or _msgs.tool_rejected
            msg = FunctionToolCallResultParam(
                type="function_call_output",
                call_id=rejected_tool.tool_call_id,
                output=rejection_msg,
            )
            messages.append(msg)
            new_items.extend(ItemHelpers.message_to_run_items(msg, current_agent.name))

        # Process external tool results
        for ext_result in state.external_results:
            msg = FunctionToolCallResultParam(
                type="function_call_output",
                call_id=ext_result.call_id,
                output=str(ext_result.output),
            )
            messages.append(msg)
            new_items.extend(ItemHelpers.message_to_run_items(msg, current_agent.name))

        # One or more nested agent-tools re-deferred: surface an interrupted
        # RunResult now (remaining decisions already applied above) instead of
        # continuing the loop with unresolved tool_use blocks.
        if len(redeferrals) > 0:
            return _build_redeferred_run_result(state, current_agent, messages, new_items, redeferrals)

        # Determine if we need to reset tool_choice for the resumed loop.
        # After HITL approval/rejection, if the agent uses "required" + reset_tool_choice,
        # the next LLM call should use "auto" to prevent infinite loops.
        initial_override: str | None = None
        agent_llm_config = current_agent.llm_config
        if (
            agent_llm_config is not None
            and agent_llm_config.tool_choice == "required"
            and agent_llm_config.reset_tool_choice is not False
        ):
            initial_override = "auto"

        # Reject resume when the state has already consumed all allowed turns.
        remaining_turns = max_turns - state.turn_count
        if remaining_turns <= 0:
            raise MaxTurnsExceeded(f"Resume rejected: state already consumed {state.turn_count} of {max_turns} turns")

        # Continue the agent loop from where we left off. Pass the SAME
        # ctx_wrapper as ``context`` (the main runner path does this via
        # from_run_context): the loop accumulates usage/cost on ``context``,
        # and hooks/tools/guardrails read ``ctx_wrapper`` — a second, fresh
        # RunContext here would leave that ctx_wrapper's usage at zero.
        run_result = await run_agent_loop(
            agent=current_agent,
            user_prompt=state.original_user_prompt,
            context=ctx_wrapper,
            ctx_wrapper=ctx_wrapper,
            hooks=hooks,
            max_turns=remaining_turns,
            config=config,
            initial_messages=messages,
            initial_new_items=new_items,
            initial_tool_choice_override=initial_override,
        )

        # Run output guardrails on the agent that produced the output.
        # last_agent tracks the final agent after handoffs; falls back to
        # the agent resolved from state if no handoff occurred during
        # the resumed loop.
        if not run_result.requires_action:
            output_agent = run_result.last_agent or current_agent
            output_results = await run_output_guardrails(
                output_agent,
                run_result.final_output,
                ctx_wrapper,
                hooks,
                config.guardrails.output,
                on_transform=lambda replacement: apply_output_transform(run_result, replacement),
                tracing_enabled=config.tracing_enabled,
                metrics_enabled=config.metrics_enabled,
            )
            run_result.guardrail_results.output = tuple(output_results)

        run_result.guardrail_audit = ctx_wrapper.collect_guardrail_audit()

        await hooks.on_agent_end(ctx_wrapper, current_agent, run_result)
        if current_agent.hooks is not None:
            await current_agent.hooks.on_end(ctx_wrapper, current_agent, run_result.final_output)

        return run_result

    except AgentToolDeferral as deferral:
        # A deferral surfacing from the resumed loop itself (not tied to a
        # specific approved tool) has no original call_id to reuse, so mint an
        # opaque one. Approved-tool re-deferrals never reach here — they are
        # handled per-iteration above with their real parent call_id.
        synthetic = _build_redeferral_call(f"nested_{id(deferral):x}", deferral)
        return _build_redeferred_run_result(state, current_agent, messages, new_items, [synthetic])

    except Exception as e:
        logger.error("Error during resumed execution: %s", e)
        raise


def resume_from_state_streamed(
    agent: Agent,
    state: RunState,
    hooks: RunHooks[TContext] | None = None,
    max_turns: int = DEFAULT_MAX_TURNS,
    config: RunConfig | None = None,
    context: Any = None,
) -> RunResultStreaming:
    """Resume a streamed execution from a saved RunState.

    Mirrors ``resume_from_state()`` but returns a ``RunResultStreaming``
    for progressive event delivery during HITL resumption.

    Args:
        agent: The agent to run.
        state: The RunState containing conversation history and approvals.
        hooks: Optional lifecycle hooks.
        max_turns: Maximum turns in the agent loop.
        config: Optional execution configuration.
        context: Caller-supplied user context. When provided, overrides
            ``state.context`` so the flow agent bridge can propagate the
            flow's shared run context through resumed inner agent runs.

    Returns:
        RunResultStreaming with stream_events() async iterator.
    """
    from troopai.adk.run.context import RunContext
    from troopai.adk.run.runner import wrap_hooks_with_verbose
    from troopai.adk.run.stream import (
        RunItemStreamEvent,
        RunItemType,
        RunResultStreaming,
    )

    config = config or DEFAULT_RUN_CONFIG
    # Preserve unwrapped hooks for nested-agent delegation so the inner
    # Runner.arun -> resume_from_state wraps VerboseHooks exactly once.
    raw_hooks = hooks
    hooks = wrap_hooks_with_verbose(hooks, config)

    # Get the current agent (may have changed due to handoffs)
    current_agent = agent
    if state.current_agent_name and state.current_agent_name != agent.name:
        found = find_agent_by_name(agent, state.current_agent_name)
        if found is not None:
            current_agent = found

    # Caller-supplied context takes precedence so the flow agent bridge
    # can propagate the flow's shared run_context through resumed inner runs.
    effective_context = context if context is not None else state.context

    # Reject resume when the state has already consumed all allowed turns.
    remaining_turns = max_turns - state.turn_count
    if remaining_turns <= 0:
        from troopai.adk.run.stream import RunResultStreaming

        result = RunResultStreaming(
            current_agent=current_agent,
            current_turn=state.turn_count,
            max_turns=max_turns,
            user_prompt=state.original_user_prompt,
            context=RunContext.make(effective_context, tenant_id=config.tenant_id),
        )
        exc = MaxTurnsExceeded(f"Resume rejected: state already consumed {state.turn_count} of {max_turns} turns")
        result.set_exception(exc)

        async def _fail_impl() -> None:
            raise exc

        result.set_deferred_run_impl(_fail_impl)
        return result

    # Create run context from effective context. tenant_id flows from
    # config so the per-tenant tool allowlist/budget gates stay in force
    # across streamed HITL resume.
    run_context = RunContext.make(effective_context, tenant_id=config.tenant_id)

    # Create streaming result
    streaming_result = RunResultStreaming(
        current_agent=current_agent,
        current_turn=state.turn_count,
        max_turns=max_turns,
        user_prompt=state.original_user_prompt,
        context=run_context,
    )

    async def run_impl():
        from troopai.adk.run.guardrails_executor import run_output_guardrails
        from troopai.adk.run.loop import run_agent_loop_streamed
        from troopai.adk.run.runner import apply_output_transform
        from troopai.adk.types.input import FunctionToolCallResultParam

        # Convert RunItems (Layer 3) back to Layer 1 params
        messages: list[LLMInputContentItem] = ItemHelpers.run_items_to_params(state.conversation_history)
        new_items: list[RunItem] = []
        # Nested agent-tools that deferred AGAIN during this streamed resume,
        # keyed by their ORIGINAL call_id (mirrors the sync path).
        redeferrals: list[DeferredToolCall] = []

        try:
            ctx_wrapper = RunContext.from_run_context(run_context)
            await hooks.on_agent_start(ctx_wrapper, current_agent)
            if current_agent.hooks is not None:
                await current_agent.hooks.on_start(ctx_wrapper, current_agent)

            # Process approved tools
            for approved_tool in state.approved_tools:
                # Close the pending HITL gate (streamed path).
                approval_meta = state.approval_metadata.get(approved_tool.tool_call_id)
                emit_hitl_approval_granted(
                    hooks,
                    current_agent,
                    approved_tool.tool_name,
                    approved_tool.tool_call_id,
                    approver_id=(approval_meta.approver_id if approval_meta is not None else None),
                    reason=(approval_meta.reason if approval_meta is not None else None),
                )
                try:
                    msg = await _resolve_approved_output(
                        approved_tool,
                        current_agent,
                        ctx_wrapper,
                        hooks,
                        config,
                        effective_context,
                        raw_hooks,
                        max_turns,
                        messages,
                    )
                except AgentToolDeferral as deferral:
                    # Sub-agent deferred again. Keep the original call_id and
                    # keep processing the remaining decisions (no TOOL_OUTPUT
                    # for a still-pending call).
                    redeferrals.append(_build_redeferral_call(approved_tool.tool_call_id, deferral))
                    continue
                messages.append(msg)
                new_items.extend(ItemHelpers.message_to_run_items(msg, current_agent.name))
                await streaming_result.put_event(
                    RunItemStreamEvent(
                        name=RunItemType.TOOL_OUTPUT,
                        item=msg,
                    )
                )

            # Process rejected tools
            _msgs = get_messages(config)
            for rejected_tool, message in state.rejected_tools:
                # Close the pending HITL gate (streamed path).
                approval_meta = state.approval_metadata.get(rejected_tool.tool_call_id)
                emit_hitl_approval_rejected(
                    hooks,
                    current_agent,
                    rejected_tool.tool_name,
                    rejected_tool.tool_call_id,
                    approver_id=(approval_meta.approver_id if approval_meta is not None else None),
                    message=message,
                )
                if rejected_tool.metadata is not None and rejected_tool.metadata.nested_agent:
                    agent_name = rejected_tool.metadata.nested_agent_name or "sub-agent"
                    rejection_msg = message or _msgs.nested_agent_rejected(agent_name)
                else:
                    rejection_msg = message or _msgs.tool_rejected
                msg = FunctionToolCallResultParam(
                    type="function_call_output",
                    call_id=rejected_tool.tool_call_id,
                    output=rejection_msg,
                )
                messages.append(msg)
                new_items.extend(ItemHelpers.message_to_run_items(msg, current_agent.name))
                await streaming_result.put_event(
                    RunItemStreamEvent(
                        name=RunItemType.TOOL_OUTPUT,
                        item=msg,
                    )
                )

            # Process external tool results (mirrors the sync path at lines 206-214)
            for ext_result in state.external_results:
                msg = FunctionToolCallResultParam(
                    type="function_call_output",
                    call_id=ext_result.call_id,
                    output=str(ext_result.output),
                )
                messages.append(msg)
                new_items.extend(ItemHelpers.message_to_run_items(msg, current_agent.name))
                await streaming_result.put_event(
                    RunItemStreamEvent(
                        name=RunItemType.TOOL_OUTPUT,
                        item=msg,
                    )
                )

            # One or more nested agent-tools re-deferred: surface the interrupt
            # now (remaining decisions already applied + streamed above) instead
            # of continuing the loop with unresolved tool_use blocks.
            if len(redeferrals) > 0:
                _apply_streamed_redeferral(streaming_result, state, current_agent, messages, new_items, redeferrals)
                return

            # Determine if we need to reset tool_choice for the resumed loop.
            initial_override_s: str | None = None
            agent_llm_config_s = current_agent.llm_config
            if (
                agent_llm_config_s is not None
                and agent_llm_config_s.tool_choice == "required"
                and agent_llm_config_s.reset_tool_choice is not False
            ):
                initial_override_s = "auto"

            # Continue the agent loop with streaming
            streaming_result.new_items.extend(new_items)
            await run_agent_loop_streamed(
                agent=current_agent,
                user_prompt=state.original_user_prompt,
                result=streaming_result,
                ctx_wrapper=ctx_wrapper,
                hooks=hooks,
                config=config,
                initial_messages=messages,
                initial_tool_choice_override=initial_override_s,
            )

            # Output guardrails on the agent that produced the output.
            # current_agent is always set (initialized to the starting agent,
            # updated on each handoff during the streamed loop).
            if not streaming_result.requires_action:
                output_agent = streaming_result.current_agent
                output_results = await run_output_guardrails(
                    output_agent,
                    streaming_result.final_output,
                    ctx_wrapper,
                    hooks,
                    config.guardrails.output,
                    on_transform=lambda replacement: apply_output_transform(streaming_result, replacement),
                    tracing_enabled=config.tracing_enabled,
                    metrics_enabled=config.metrics_enabled,
                )
                streaming_result.guardrail_results.output = tuple(output_results)

            streaming_result.guardrail_audit = ctx_wrapper.collect_guardrail_audit()

            await hooks.on_agent_end(ctx_wrapper, current_agent, streaming_result)
            if current_agent.hooks is not None:
                await current_agent.hooks.on_end(ctx_wrapper, current_agent, streaming_result.final_output)

        except AgentToolDeferral as deferral:
            # A deferral surfacing from the streamed loop itself has no original
            # call_id to reuse (approved-tool re-deferrals are handled
            # per-iteration above with their real parent call_id), so mint an
            # opaque one.
            synthetic = _build_redeferral_call(f"nested_{id(deferral):x}", deferral)
            _apply_streamed_redeferral(streaming_result, state, current_agent, messages, new_items, [synthetic])
        except Exception as e:
            streaming_result.set_exception(e)
        finally:
            await streaming_result.complete()

    # Schedule the task
    try:
        loop = asyncio.get_running_loop()
        streaming_result.set_run_task(loop.create_task(run_impl()))
    except RuntimeError:
        # No running loop — store for lazy creation in stream_events()
        streaming_result.set_deferred_run_impl(run_impl)

    return streaming_result


async def resume_nested_agent_tool(
    parent_agent: Agent,
    deferred_tool: DeferredToolCall,
    context: Any,
    hooks: RunHooks[TContext] | None,
    max_turns: int,
    config: RunConfig,
) -> str:
    """Resume a nested agent-tool deferral.

    When an agent wrapped via as_tool() encounters a tool requiring
    approval, the deferral is stored in the parent's DeferredToolCall
    metadata. This method reconstructs the sub-agent's RunState,
    applies the approval, and runs the sub-agent to completion.

    Args:
        parent_agent: The parent agent that owns the tool.
        deferred_tool: The DeferredToolCall with nested state in metadata.
        context: The user context.
        hooks: Lifecycle hooks.
        max_turns: Maximum turns for the sub-agent.
        config: Run configuration.

    Returns:
        The sub-agent's final output as a string.
    """
    from troopai.adk.run.state import RunState

    metadata = deferred_tool.metadata
    if metadata is None:
        raise ValueError("resume_nested_agent_tool requires metadata on the deferred tool call")
    if metadata.nested_state is None:
        raise ValueError("resume_nested_agent_tool requires nested_state in metadata")

    # Reconstruct sub-agent's RunState
    nested_state = RunState.from_dict(metadata.nested_state)

    # Apply the approval to the nested state's deferred requests.
    for req in list(nested_state.deferred_tool_requests.approvals):
        nested_state.approve(req)

    # Find the sub-agent instance via the shared resolver.
    from troopai.adk.run.llm_calls import resolve_function_tool

    tool = await resolve_function_tool(parent_agent, deferred_tool.tool_name)

    sub_agent = None
    if tool is not None:
        sub_agent = tool.get_delegate_agent()

    if sub_agent is None:
        return f"Error: Could not find sub-agent for tool '{deferred_tool.tool_name}' to resume."

    # Lazy import to avoid circular dependency: resumption → Runner → loop → ...
    from troopai.adk.run.runner import Runner

    # Resume the sub-agent
    sub_result = await Runner.arun(
        sub_agent,
        nested_state,
        context=context,
        hooks=hooks,
        max_turns=max_turns,
        run_config=config,
    )

    # If the sub-agent defers again, propagate it
    if sub_result.requires_action:
        if sub_result.deferred_requests is None:
            raise RuntimeError("Sub-agent result requires action but has no deferred requests")
        if sub_result.state is None:
            raise RuntimeError("Sub-agent result requires action but has no resumable state")
        raise AgentToolDeferral(
            agent_name=sub_agent.name,
            deferred_requests=sub_result.deferred_requests,
            state=sub_result.state,
        )

    return str(sub_result.final_output)
