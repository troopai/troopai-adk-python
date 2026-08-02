"""Agent loop — turn-by-turn execution cycle.

The core execution loop that drives agent interactions: LLM calls,
tool execution, handoff processing, and final output detection.
Extracted from ``Runner`` to isolate the loop logic from the
public API facade.
"""

from __future__ import annotations

import dataclasses
import logging
from inspect import isawaitable
from typing import TYPE_CHECKING, Any

from troopai.adk.exceptions import MaxTurnsExceeded, ModelRefusalError, UsageLimitExceeded
from troopai.adk.llms.llm_usage import LLMUsage
from troopai.adk.run.cost import check_usage_limits
from troopai.adk.run.llm_calls import (
    call_llm,
    call_llm_streamed,
    call_llm_streamed_with_routing,
    call_llm_with_routing,
    enforce_tenant_budget,
    record_tenant_spend,
    resolve_compaction_llm,
    resolve_llm,
    resolve_model_name,
)
from troopai.adk.run.tools_executor import (
    execute_tool_calls,
    execute_tool_calls_streamed,
)
from troopai.adk.tracing import generation_span
from troopai.adk.types.items.items import ItemHelpers
from troopai.adk.verbose.hooks import (
    emit_budget_exceeded,
    emit_context_compacted,
    emit_stream_end,
    emit_stream_start,
    emit_turn_end,
    emit_turn_start,
    emit_usage_recorded,
)
from troopai.adk.verbose.run_bridge import active_verbose_agent, active_verbose_hooks

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.context.context_manager import ContextManager
    from troopai.adk.context.directives import DirectiveStore
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.run.agent_middleware import AgentBlockOutcome
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext, TContext
    from troopai.adk.run.stream import (
        RunResultStreaming,
    )
    from troopai.adk.run.tools_executor import FunctionToolFailureCounts
    from troopai.adk.run.types import UserPrompt
    from troopai.adk.tools.function_tool import FunctionTool
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.input.llm_input_easy_message import LLMInputEasyMessage
    from troopai.adk.types.items.items import RunItem
    from troopai.adk.types.run import RunResult

logger = logging.getLogger(__name__)


async def _activate_lazy_skills(
    tool_results: list,
    skill_tool_map: dict[str, str],
    activated_skills: set[str],
    agent: Agent,
    messages: list[LLMInputContentItem],
    hooks: RunHooks[TContext],
    ctx_wrapper: RunContext[TContext],
) -> None:
    """Check tool results for LAZY skill activation.

    When a tool belonging to a not-yet-activated skill is called
    for the first time, inject that skill's instructions into the
    system prompt message.

    Args:
        tool_results: Results from tool execution.
        skill_tool_map: Mapping of tool name → skill name.
        activated_skills: Set of already-activated skill names (mutated).
        agent: The current agent (to look up skill objects).
        messages: The message list (system message is modified in-place).
        hooks: Run hooks for ``on_skill_activated`` callback.
        ctx_wrapper: Run context for hooks.
    """
    from troopai.adk.skills.skill_rendering import render_single_skill_instruction

    for tr in tool_results:
        tool_name = getattr(tr, "tool_name", None) or getattr(tr, "name", None)
        if tool_name is None:
            continue
        skill_name = skill_tool_map.get(tool_name)
        if skill_name is None or skill_name in activated_skills:
            continue

        activated_skills.add(skill_name)

        # Find the skill object
        skill_obj = next((s for s in agent.skills if s.name == skill_name), None)
        if skill_obj is None or not skill_obj.instructions:
            continue

        # Inject instructions into the system message
        instruction_text = render_single_skill_instruction(skill_obj)
        if instruction_text is not None and len(messages) > 0:
            system_msg = messages[0]
            if isinstance(system_msg, dict) and system_msg.get("role") == "system":
                current_content = str(system_msg.get("content", ""))

                # Add "## Available Skills" header if not present
                if "## Available Skills" not in current_content:
                    new_content = current_content + "\n\n## Available Skills" + instruction_text
                else:
                    new_content = current_content + instruction_text

                # Replace system message with updated content
                new_system: LLMInputEasyMessage = {"role": "system", "content": new_content}
                messages[0] = new_system

                logger.info(
                    "LAZY skill '%s' activated — instructions injected into system prompt",
                    skill_name,
                )

        # Fire hook
        await hooks.on_skill_activated(ctx_wrapper, agent, skill_name)


def _detect_jit_directives(agent: Agent) -> DirectiveStore | None:
    """Detect JIT directive store from agent's tools.

    Returns the DirectiveStore if the agent has a JITContextAwareTool,
    or None otherwise.  Called at loop start and after each handoff.

    Only ``ManageContextAwareTool`` ever writes directives, and each JIT
    tool owns a separate store, so the store of the ``ManageContextAwareTool``
    is selected directly when present. Otherwise the first JIT tool's store
    is returned, making detection independent of tool ordering.
    """
    if len(agent.tools) > 0:
        from troopai.adk.tools.builtin.jit_context_aware_tool import (
            JITContextAwareTool as _JITTool,
            ManageContextAwareTool as _ManageTool,
        )

        first_jit: _JITTool | None = None
        for _tool in agent.tools:
            if isinstance(_tool, _ManageTool):
                return _tool.directives
            if first_jit is None and isinstance(_tool, _JITTool):
                first_jit = _tool
        if first_jit is not None:
            return first_jit.directives
    return None


def _adjust_context_end(context_end: int, len_before: int, len_after: int) -> int:
    """Shift the temporal-slicing boundary by a rebind's net length change.

    The per-turn JIT / context-manager / history-processor / input-filter
    steps may rebind the message list to a list of a different length.
    ``context_end`` indexes the split between an agent's inherited context and
    its own output; it moves by the net length change so a later handoff split
    does not slice at a stale index. Clamped to ``[0, len_after]`` so the split
    is always in range.
    """
    shifted = context_end + (len_after - len_before)
    if shifted < 0:
        return 0
    if shifted > len_after:
        return len_after
    return shifted


async def resolve_system_prompt(
    agent: Agent,
    ctx_wrapper: RunContext[TContext],
) -> str:
    """Resolve the agent's system_prompt to a plain string.

    Handles ``str``, ``SystemPrompt``, and callables (sync or async)
    that receive ``(RunContext, Agent)`` and return either type.

    For EAGER skill activation, skill instructions are appended to
    the resolved prompt.
    """
    from troopai.adk.prompts.system_prompt import DynamicSystemPromptData, SystemPrompt
    from troopai.adk.skills.activation import SkillActivation

    raw_prompt: Any = agent.system_prompt
    if callable(raw_prompt):
        raw_prompt = raw_prompt(DynamicSystemPromptData(context=ctx_wrapper, agent=agent))
        if isawaitable(raw_prompt):
            raw_prompt = await raw_prompt
    prompt_str = raw_prompt.generate() if isinstance(raw_prompt, SystemPrompt) else str(raw_prompt)

    # Append EAGER skill instructions
    if agent.skills and agent.skill_activation == SkillActivation.EAGER:
        from troopai.adk.skills.skill_rendering import render_skill_instructions

        skill_section = await render_skill_instructions(agent.skills, ctx_wrapper)
        if skill_section is not None:
            prompt_str = prompt_str + "\n\n" + skill_section
            logger.debug("Injected EAGER skill instructions into system prompt")

    # TR.3 + TR.4: Sandbox capability-instruction composition.
    # When a sandbox session is bound to this run, swap the placeholder
    # for the ADK default sandbox prompt + capability fragments + the
    # rendered filesystem tree from the manifest.
    prompt_str = await _maybe_compose_sandbox_prompt(prompt_str, ctx_wrapper)

    return prompt_str


async def _maybe_compose_sandbox_prompt(
    prompt_str: str,
    ctx_wrapper: RunContext[TContext],
) -> str:
    """Delegate to ``runner_integration.instructions_composer`` when a
    sandbox session is attached to the run context. No-op otherwise.
    """
    handle = getattr(ctx_wrapper, "_sandbox_handle", None)
    if handle is None:
        return prompt_str
    from troopai.adk.sandbox.runner_integration.instructions_composer import (
        compose_sandbox_prompt,
    )

    return await compose_sandbox_prompt(
        prompt_str,
        capabilities=handle.capabilities,
        manifest=handle.manifest,
    )


async def build_initial_messages(
    agent: Agent,
    user_prompt: UserPrompt,
    ctx_wrapper: RunContext[TContext],
) -> list[LLMInputContentItem]:
    """Build initial messages for LLM call."""
    system_content = await resolve_system_prompt(agent, ctx_wrapper)

    system_msg: LLMInputEasyMessage = {"role": "system", "content": system_content}
    messages: list[LLMInputContentItem] = [system_msg]

    if isinstance(user_prompt, str):
        user_msg: LLMInputEasyMessage = {"role": "user", "content": user_prompt}
        messages.append(user_msg)
    else:
        # user_prompt is already a message list
        messages.extend(user_prompt)

    return messages


async def inject_system_prompt(
    agent: Agent,
    messages: list[LLMInputContentItem],
    ctx_wrapper: RunContext[TContext],
) -> list[LLMInputContentItem]:
    """Replace or prepend the specialist agent's system prompt after handoff.

    After a handoff, ``prepare_handoff_input()`` returns the filtered
    history from the *source* agent — which includes the source's system
    prompt, not the target's.  This function swaps in the target agent's
    system prompt so the specialist operates with its own instructions.
    """
    system_content = await resolve_system_prompt(agent, ctx_wrapper)
    new_system: LLMInputEasyMessage = {"role": "system", "content": system_content}
    if len(messages) > 0 and messages[0].get("role") == "system":
        messages[0] = new_system
    else:
        messages.insert(0, new_system)
    return messages


async def _apply_call_model_input_filter(
    *,
    agent: Agent,
    config: RunConfig,
    ctx_wrapper: RunContext[TContext],
    messages: list[LLMInputContentItem],
) -> list[LLMInputContentItem]:
    """Apply ``RunConfig.call_model_input_filter`` if set.

    Runs immediately before the LLM call, after context management and
    history processors. Passes the filter a shallow copy of ``messages``
    wrapped in ``ModelInputData`` together with the current agent and
    unwrapped run context. The filter may be sync or async and must
    return ``ModelInputData``; anything else raises ``TypeError``.

    Returns:
        The (possibly rewritten) messages list. A ``None`` filter is a
        pass-through and returns the input reference unchanged — no copy
        is allocated on the hot path when the hook is unused.
    """
    filter_fn = config.call_model_input_filter
    if filter_fn is None:
        return messages

    from troopai.adk.run.config import CallModelData, ModelInputData

    model_data = ModelInputData(input=list(messages))  # shallow copy
    payload: CallModelData = CallModelData(
        model_data=model_data,
        agent=agent,
        context=ctx_wrapper.context,
    )

    try:
        result = filter_fn(payload)
        if isawaitable(result):
            result = await result
    except Exception:
        logger.exception(
            "call_model_input_filter raised (agent=%s)",
            agent.name,
        )
        raise

    if not isinstance(result, ModelInputData):
        raise TypeError(f"call_model_input_filter must return ModelInputData, got {type(result).__name__}")
    return result.input


async def run_agent_block(
    agent: Agent,
    messages: list[LLMInputContentItem],
    context_end: int,
    user_prompt: UserPrompt,
    context: RunContext[TContext],
    ctx_wrapper: RunContext[TContext],
    hooks: RunHooks[TContext],
    max_turns: int,
    config: RunConfig,
    new_items: list[RunItem],
    tool_failure_counts: FunctionToolFailureCounts,
    initial_tool_choice_override: str | None,
    extra_tools: list[FunctionTool] | None,
    swarm_tool_names: set[str] | None,
    ctx_mgr: ContextManager | None,
    jit_directives: DirectiveStore | None,
    skill_tool_map: dict[str, str],
    activated_skills: set[str],
    turn_offset: int,
    starting_total_turns: int,
) -> AgentBlockOutcome:
    """Run one agent's tenure within a run loop.

    Drives the per-turn cycle (LLM call -> tool execution -> step
    resolution) for a single agent until that agent's contribution
    transitions to one of: final output, handoff to another agent,
    HITL interruption, or swarm yield. Returns an
    :class:`AgentBlockOutcome` describing the transition.

    The outer :func:`run_agent_loop` re-invokes this function (through
    the agent-middleware chain) each time a handoff outcome surfaces,
    so per-agent observability via ``Agent.middleware.agents`` re-
    fires on every transition.

    Args:
        agent: The current agent for this block.
        messages: The mutable message list. Mutations (assistant
            messages, tool results) accumulate across the block.
        context_end: Where the agent's prior context ends in
            ``messages`` and where the agent's output begins.
        user_prompt: The original user prompt for the run.
        context: The user context (carries cumulative usage).
        ctx_wrapper: The run-context wrapper passed to hooks and
            tool execution.
        hooks: Run-level lifecycle hooks.
        max_turns: Absolute maximum turns for the whole run.
        config: Run configuration.
        new_items: Mutable list of run items accumulated across the
            run. Mutations in this block are visible in the outer
            loop and surface on the final ``RunResult``.
        tool_failure_counts: Per-tool failure counter. Mutated by
            tool execution; reset by the outer loop on handoff.
        initial_tool_choice_override: Optional override for the
            first LLM call's ``tool_choice``. Cleared after the first
            call; updated by ``NextStepRunAgain``.
        extra_tools: Optional swarm-injected tools for this turn.
        swarm_tool_names: Optional set of swarm tool names that
            indicate a yield when called.
        ctx_mgr: Optional ``ContextManager`` shared across the run.
        jit_directives: Optional JIT directive store for this agent.
        skill_tool_map: Run-level mapping of tool name -> skill name.
        activated_skills: Run-level set of activated skill names.
        turn_offset: Turns consumed by prior blocks in the run.
        starting_total_turns: Cumulative turn counter at block entry.

    Returns:
        :class:`AgentBlockOutcome` describing the transition.

    Raises:
        :class:`MaxTurnsExceeded`: If absolute ``max_turns`` is
            reached and ``RunConfig.on_max_turns`` does not salvage
            a final output.
        :class:`UsageLimitExceeded`: If ``RunConfig.usage_limits``
            is exceeded after an LLM call.
    """
    from troopai.adk.run.agent_middleware import AgentBlockOutcome
    from troopai.adk.types.run import RunResult

    turn = 0
    total_turns = starting_total_turns
    tool_choice_override = initial_tool_choice_override

    while turn_offset + turn < max_turns:
        turn += 1
        total_turns += 1

        # Swarm safety: check cross-agent cumulative turn limit
        if config.max_total_turns is not None and total_turns > config.max_total_turns:
            raise MaxTurnsExceeded(
                f"Cross-agent total turns ({total_turns}) exceeded max_total_turns ({config.max_total_turns})"
            )

        # Emit turn boundary (verbose hooks only — zero cost otherwise).
        # Paired with the matching emit_turn_end at every exit path below.
        emit_turn_start(hooks, agent, turn_offset + turn)

        # Temporal-slicing boundary maintenance. The JIT / context-manager /
        # history-processor / input-filter steps below may rebind ``messages``
        # to a list of a different length (compaction shrinks it; a processor
        # may drop or add items). ``context_end`` indexes the split between this
        # agent's inherited context and its own output, so it must shift by the
        # net length change or the handoff split slices at a stale index.
        len_before_rebind = len(messages)

        # Apply JIT directives (before ContextManager, works independently)
        if jit_directives is not None and jit_directives.count > 0:
            from troopai.adk.context.directives import apply_directives

            llm_model_d = resolve_model_name(agent, config)
            compaction_llm_d = resolve_compaction_llm(agent, config)
            messages = await apply_directives(
                messages,
                jit_directives,
                compaction_llm_d,
                llm_model_d,
                config,
                context=context,
            )

        # Apply context management before LLM call
        if ctx_mgr is not None:
            llm_model = resolve_model_name(agent, config)
            compaction_llm = resolve_compaction_llm(agent, config)
            # Measure pre/post token delta so the verbose layer can
            # surface context compaction as a first-class event. The
            # count is advisory — exceptions are swallowed at DEBUG to
            # preserve loop correctness.
            pre_tokens = 0
            try:
                pre_tokens = ctx_mgr.get_token_usage(messages, llm_model)["used"]
            except Exception as exc:
                logger.debug("pre-compact token estimate failed: %s", exc)
            messages = await ctx_mgr.prepare_messages(
                messages,
                compaction_llm,
                llm_model,
                config,
                context=context,
            )
            try:
                post_tokens = ctx_mgr.get_token_usage(messages, llm_model)["used"]
                if post_tokens < pre_tokens:
                    emit_context_compacted(
                        hooks,
                        agent,
                        pre_tokens,
                        post_tokens,
                    )
            except Exception as exc:
                logger.debug("post-compact token estimate failed: %s", exc)

        # Check if context pressure requires forcing tool use (once per turn).
        # Uses "required" (any tool) rather than a named tool choice
        # for cross-provider compatibility (Gemini doesn't support named choice).
        # The pressure_feedback message guides the LLM toward manage_context.
        # Consumed immediately to prevent re-triggering on subsequent loop iterations.
        force_tool_override: str | None = None
        if ctx_mgr is not None and ctx_mgr.force_tool is not None:
            force_tool_override = "required"
            ctx_mgr.consume_force_tool()

        # Apply history processors (after context management, before LLM call)
        # Processors work with Layer 3 RunItems; convert at boundary.
        if config.history_processors is not None:
            run_items = list(ItemHelpers.messages_to_run_items(messages))
            for processor in config.history_processors:
                run_items = processor(run_items)
            messages = ItemHelpers.run_items_to_params(run_items)

        # Apply call_model_input_filter (after history processors, before LLM)
        messages = await _apply_call_model_input_filter(
            agent=agent,
            config=config,
            ctx_wrapper=ctx_wrapper,
            messages=messages,
        )

        # Shift the context boundary by the net length change the rebinds
        # applied so the handoff split below stays aligned with the current
        # (possibly compacted) message list.
        context_end = _adjust_context_end(context_end, len_before_rebind, len(messages))

        # Call LLM
        llm_model_name = resolve_model_name(agent, config)
        llm = resolve_llm(agent, config)
        if config.router is None:
            await enforce_tenant_budget(agent, config, context, messages, llm_model_name, llm, hooks)
        await hooks.on_llm_start(ctx_wrapper, agent, messages)
        if agent.hooks is not None:
            await agent.hooks.on_llm_start(ctx_wrapper, agent, messages)
        response = None  # set before on_llm_end so the finally block always has it
        try:
            with generation_span(
                model=llm_model_name,
                tenant_id=context.tenant_id,
                disabled=not (config.tracing_enabled or config.metrics_enabled),
            ) as gen_span:
                if config.router is not None:
                    # When a router and tenant_budget are both set, the per-candidate budget gate
                    # runs inside call_llm_with_routing (after on_llm_start). A budget kill on
                    # a routed call fires on_llm_start without a paired on_llm_end. The non-routed
                    # path gates before on_llm_start. This narrow edge is accepted.
                    outcome = await call_llm_with_routing(
                        config.router,
                        agent,
                        messages,
                        config,
                        hooks,
                        context=ctx_wrapper,
                        tool_failure_counts=tool_failure_counts,
                        tool_choice_override=force_tool_override or tool_choice_override,
                        extra_tools=extra_tools,
                    )
                    response = outcome.response
                    llm_model_name = outcome.model
                    llm = outcome.llm
                    gen_span.data = dataclasses.replace(gen_span.data, model=llm_model_name)
                else:
                    response = await call_llm(
                        agent,
                        messages,
                        config,
                        context=ctx_wrapper,
                        tool_failure_counts=tool_failure_counts,
                        tool_choice_override=force_tool_override or tool_choice_override,
                        extra_tools=extra_tools,
                    )
                    # Update to actual serving model — litellm may have used a fallback
                    # model that differs from the configured primary. The router path
                    # already does this via outcome.model; mirror that here.
                    if response.model:
                        llm_model_name = response.model
                        gen_span.data = dataclasses.replace(gen_span.data, model=llm_model_name)
                if response.usage is not None:
                    call_cost = llm.cost(llm_model_name, response.usage)
                    gen_span.data = dataclasses.replace(
                        gen_span.data,
                        usage=dataclasses.asdict(response.usage),
                        cost_usd=call_cost,
                    )
                    if call_cost is not None:
                        context.cost_usd += call_cost
                        await record_tenant_spend(config, context, call_cost)
        finally:
            # Always fire on_llm_end to keep hook pairs balanced, even
            # when call_llm raises (network error, budget kill, etc.).
            # response is None only when the exception fires before any
            # LLM call completes — pass it through; hooks that track
            # latency timers must tolerate a None response.
            await hooks.on_llm_end(ctx_wrapper, agent, response)
            if agent.hooks is not None:
                await agent.hooks.on_llm_end(ctx_wrapper, agent, response)

        # Clear the override after it's been applied
        tool_choice_override = None

        # Track usage. Some providers or test doubles may omit token usage,
        # but the framework still made one model request.
        usage_delta = response.usage if response.usage is not None else LLMUsage(requests=1)
        context.usage = context.usage + usage_delta
        # Surface cumulative usage to the verbose layer after
        # every LLM call. Cheap — single attribute read + string
        # format on the verbose side.
        emit_usage_recorded(hooks, agent, context.usage)

        # Check usage limits
        if config.usage_limits is not None:
            try:
                check_usage_limits(config.usage_limits, context.usage)
            except UsageLimitExceeded as exc:
                # Announce the breach through the verbose layer
                # before the exception propagates. ``limit_type`` is
                # extracted from the message verbatim — the raw field
                # is not exposed on the exception.
                emit_budget_exceeded(hooks, agent, str(exc))
                raise

        # ── Resolve next step via shared turn resolution ──────────
        from troopai.adk.run.next_step import (
            NextStepFinalOutput,
            NextStepHandoff,
            NextStepInterruption,
            NextStepRunAgain,
            NextStepSwarmYield,
        )
        from troopai.adk.run.turn_resolution import (
            resolve_handoff_step,
            resolve_structured_output_step,
            resolve_swarm_yield_step,
            resolve_tool_results_step,
        )

        # 1. Structured output path
        step = await resolve_structured_output_step(
            agent,
            response,
            messages,
            new_items,
            context_end,
            context,
            ctx_wrapper,
            hooks,
            config,
        )

        # 2. If no structured output decision, process tool calls
        if step is None:
            # Add assistant message to history
            response_items = ItemHelpers.response_to_run_items(response, agent.name)
            for ri in response_items:
                messages.append(ri.to_param())
            new_items.extend(response_items)

            tool_calls = response.tool_calls

            if tool_calls is None or len(tool_calls) == 0:
                # If the model returned a content-policy refusal with no
                # text content, raise a typed error so callers can
                # distinguish a refusal from a clean empty output or an
                # HITL interruption.  Check refusal before content so a
                # response that carries BOTH a refusal part and text
                # (unusual but possible) still surfaces the refusal.
                refusal_text = response.refusal
                if refusal_text is not None and response.content is None:
                    raise ModelRefusalError(refusal_text)
                step = NextStepFinalOutput(output=response.content)
            else:
                # Check for swarm yield first — a policy-injected
                # transfer_to_<name> or swarm_done must win over any
                # coincidentally-named regular handoff tool because
                # we dispatched this turn with the swarm tools merged
                # in. Guarded on swarm_tool_names so non-swarm turns
                # pay zero cost.
                swarm_step: NextStepSwarmYield | None = None
                if swarm_tool_names is not None and len(swarm_tool_names) > 0:
                    swarm_step = resolve_swarm_yield_step(
                        agent,
                        tool_calls,
                        swarm_tool_names,
                        messages,
                        new_items,
                        context_end,
                        config,
                    )
                if swarm_step is not None:
                    step = swarm_step
                else:
                    # Check for handoff
                    handoff_step = await resolve_handoff_step(
                        agent,
                        tool_calls,
                        messages,
                        new_items,
                        context_end,
                        context,
                        ctx_wrapper,
                        hooks,
                        config,
                    )
                    if handoff_step is not None:
                        step = handoff_step
                    else:
                        # Execute tools
                        llm_model_tc = resolve_model_name(agent, config)
                        agent_llm_config = agent.llm_config
                        is_parallel = (
                            agent_llm_config is not None and agent_llm_config.tool_execution_mode == "parallel"
                        )
                        tool_results, deferred = await execute_tool_calls(
                            agent=agent,
                            tool_calls=tool_calls,
                            ctx_wrapper=ctx_wrapper,
                            hooks=hooks,
                            config=config,
                            tool_failure_counts=tool_failure_counts,
                            model=llm_model_tc,
                            messages=messages,
                            turn=turn,
                            parallel=is_parallel,
                        )

                        # LAZY skill activation: inject instructions for
                        # newly-activated skills after tool execution
                        if len(skill_tool_map) > 0 and len(tool_results) > 0:
                            await _activate_lazy_skills(
                                tool_results,
                                skill_tool_map,
                                activated_skills,
                                agent,
                                messages,
                                hooks,
                                ctx_wrapper,
                            )

                        step = await resolve_tool_results_step(
                            agent,
                            tool_results,
                            deferred,
                            tool_calls,
                            messages,
                            new_items,
                            user_prompt,
                            context,
                            ctx_wrapper,
                            turn_offset + turn,
                        )

        # ── Apply the resolved step ──────────────────────────────
        # Emit turn-end on every control-flow exit. Paired with
        # emit_turn_start at the top of the loop so verbose panels
        # always close cleanly — including final-output, handoff,
        # interruption, and re-run branches. A bare ``continue`` /
        # fall-through also closes the current turn's block before
        # the next iteration opens its own.
        match step:
            case NextStepFinalOutput(output=output):
                emit_turn_end(hooks, agent, turn_offset + turn)
                return AgentBlockOutcome(
                    kind="final",
                    result=RunResult(
                        final_output=output,
                        user_prompt=user_prompt,
                        new_items=new_items,
                        context=context,
                        last_agent=agent,
                    ),
                    turn=turn,
                    total_turns_consumed=total_turns,
                )
            case NextStepHandoff(new_agent=new_agent, new_messages=new_msgs, context_end=new_end):
                emit_turn_end(hooks, agent, turn_offset + turn)
                return AgentBlockOutcome(
                    kind="handoff",
                    handoff_target=new_agent,
                    next_messages=new_msgs,
                    next_context_end=new_end,
                    turn=turn,
                    total_turns_consumed=total_turns,
                )
            case NextStepInterruption(deferred=deferred, state=run_state):
                emit_turn_end(hooks, agent, turn_offset + turn)
                return AgentBlockOutcome(
                    kind="final",
                    result=RunResult(
                        final_output=None,
                        user_prompt=user_prompt,
                        new_items=new_items,
                        context=context,
                        last_agent=agent,
                        deferred_requests=deferred,
                        state=run_state,
                    ),
                    turn=turn,
                    total_turns_consumed=total_turns,
                )
            case NextStepRunAgain(tool_choice_override=override):
                emit_turn_end(hooks, agent, turn_offset + turn)
                tool_choice_override = override
                # Continue to next turn within this block
            case NextStepSwarmYield(signal=yield_signal):
                # Swarm driver seam: hand control back to
                # ``run_swarm_loop`` with the explicit yield signal.
                # The driver inspects ``result.swarm_yield`` to decide
                # whether to advance to the next member (SwarmHandoff)
                # or terminate the run (SwarmDone).
                emit_turn_end(hooks, agent, turn_offset + turn)
                return AgentBlockOutcome(
                    kind="final",
                    result=RunResult(
                        final_output=None,
                        user_prompt=user_prompt,
                        new_items=new_items,
                        context=context,
                        last_agent=agent,
                        swarm_yield=yield_signal,
                    ),
                    turn=turn,
                    total_turns_consumed=total_turns,
                )

    # Max turns reached — give the on_max_turns handler a chance to
    # salvage a final output before raising.
    if config.on_max_turns is not None:
        logger.info(
            "Invoking on_max_turns handler (agent=%s, turn=%d, total=%d)",
            agent.name,
            turn_offset + turn,
            total_turns,
        )
        salvaged = await config.on_max_turns(agent, turn_offset + turn)
        if salvaged is not None:
            return AgentBlockOutcome(
                kind="final",
                result=RunResult(
                    final_output=salvaged,
                    user_prompt=user_prompt,
                    new_items=new_items,
                    context=context,
                    last_agent=agent,
                ),
                turn=turn,
                total_turns_consumed=total_turns,
            )
    raise MaxTurnsExceeded(f"Agent loop exceeded {max_turns} turns")


async def run_agent_block_streamed(
    agent: Agent,
    messages: list[LLMInputContentItem],
    result: RunResultStreaming,
    user_prompt: UserPrompt,
    ctx_wrapper: RunContext[TContext],
    hooks: RunHooks[TContext],
    config: RunConfig,
    tool_failure_counts: FunctionToolFailureCounts,
    initial_tool_choice_override: str | None,
    extra_tools: list[FunctionTool] | None,
    swarm_tool_names: set[str] | None,
    ctx_mgr: ContextManager | None,
    jit_directives: DirectiveStore | None,
    skill_tool_map: dict[str, str],
    activated_skills: set[str],
    context_end: int,
    starting_total_turns: int,
) -> AgentBlockOutcome:
    """Run one agent's tenure within the streaming loop.

    Streaming sibling of :func:`run_agent_block`. Drives the per-turn
    cycle (LLM call → tool execution → step resolution) for a single
    agent until that agent's contribution transitions to one of: final
    output, handoff, HITL interruption, swarm yield, or cancellation.

    Unlike :func:`run_agent_block`, the streaming block mutates
    ``result`` directly (`final_output`, `deferred_requests`, `state`,
    `swarm_yield`, `current_turn`, `current_agent`) rather than
    constructing a ``RunResult``. The returned
    :class:`AgentBlockOutcome` carries the discriminator the driver
    needs to decide whether to continue (handoff) or stop (final);
    interruption / swarm-yield / cancellation all surface as
    ``kind="final"`` because they all terminate the streaming run.
    The driver does NOT read ``outcome.result`` on the streaming
    path — that field is left ``None``.

    Args:
        agent: The current agent for this block.
        messages: The mutable message list for this block.
        result: The streaming result object — mutated throughout
            (event queue, current_turn, current_agent, final_output,
            deferred_requests, state, swarm_yield).
        user_prompt: The original user prompt for the run.
        ctx_wrapper: Run context wrapper.
        hooks: Run-level lifecycle hooks.
        config: Run configuration.
        tool_failure_counts: Per-tool failure counter for retry
            budget. Mutated by tool execution; reset by the driver on
            handoff.
        initial_tool_choice_override: Optional override for the first
            LLM call's ``tool_choice``.
        extra_tools: Optional swarm-injected tools.
        swarm_tool_names: Optional set of swarm tool names that
            indicate a yield when called.
        ctx_mgr: Optional ``ContextManager`` shared across the run.
        jit_directives: Optional JIT directive store for this agent.
        skill_tool_map: Tool-name → skill-name map for LAZY activation.
        activated_skills: Set of activated skill names (run-level).
        context_end: Index in ``messages`` where the agent's prior
            context ends and its output begins.
        starting_total_turns: Cumulative turn counter at block entry.
    """
    from troopai.adk.run.agent_middleware import AgentBlockOutcome
    from troopai.adk.run.stream import (
        AgentUpdatedStreamEvent,
        CancelMode,
        RunItemStreamEvent,
        RunItemType,
    )

    turn = 0
    total_turns = starting_total_turns
    tool_choice_override = initial_tool_choice_override

    while result.current_turn < result.max_turns:
        result.current_turn += 1
        turn += 1
        total_turns += 1
        result.current_agent = agent

        # Swarm safety: cross-agent cumulative turn limit
        if config.max_total_turns is not None and total_turns > config.max_total_turns:
            raise MaxTurnsExceeded(
                f"Cross-agent total turns ({total_turns}) exceeded max_total_turns ({config.max_total_turns})"
            )

        # Cancellation gate. Both modes break out — the streaming
        # contract is "stop after the in-flight tool batch finishes",
        # which has already happened by the time control returns to
        # the top of the loop.
        if result.cancel_mode == CancelMode.IMMEDIATE:
            return AgentBlockOutcome(kind="final", turn=turn, total_turns_consumed=total_turns)
        if result.cancel_mode == CancelMode.AFTER_TURN:
            return AgentBlockOutcome(kind="final", turn=turn, total_turns_consumed=total_turns)

        emit_turn_start(hooks, agent, result.current_turn)

        # Temporal-slicing boundary maintenance — mirrors the non-streaming
        # block: the rebind steps below may change ``messages`` length, so
        # record it now and shift ``context_end`` by the net delta afterward.
        len_before_rebind = len(messages)

        if jit_directives is not None and jit_directives.count > 0:
            from troopai.adk.context.directives import apply_directives

            llm_model_d = resolve_model_name(agent, config)
            compaction_llm_d = resolve_compaction_llm(agent, config)
            messages = await apply_directives(
                messages,
                jit_directives,
                compaction_llm_d,
                llm_model_d,
                config,
                context=ctx_wrapper,
            )

        if ctx_mgr is not None:
            llm_model = resolve_model_name(agent, config)
            compaction_llm = resolve_compaction_llm(agent, config)
            pre_tokens = 0
            try:
                pre_tokens = ctx_mgr.get_token_usage(messages, llm_model)["used"]
            except Exception as exc:
                logger.debug("pre-compact (streamed) estimate failed: %s", exc)
            messages = await ctx_mgr.prepare_messages(
                messages,
                compaction_llm,
                llm_model,
                config,
                context=ctx_wrapper,
            )
            try:
                post_tokens = ctx_mgr.get_token_usage(messages, llm_model)["used"]
                if post_tokens < pre_tokens:
                    emit_context_compacted(hooks, agent, pre_tokens, post_tokens)
            except Exception as exc:
                logger.debug("post-compact (streamed) estimate failed: %s", exc)

        force_tool_override: str | None = None
        if ctx_mgr is not None and ctx_mgr.force_tool is not None:
            force_tool_override = "required"
            ctx_mgr.consume_force_tool()

        if config.history_processors is not None:
            run_items = list(ItemHelpers.messages_to_run_items(messages))
            for processor in config.history_processors:
                run_items = processor(run_items)
            messages = ItemHelpers.run_items_to_params(run_items)

        messages = await _apply_call_model_input_filter(
            agent=agent,
            config=config,
            ctx_wrapper=ctx_wrapper,
            messages=messages,
        )

        # Shift the context boundary by the net length change the rebinds
        # applied so the handoff split stays aligned with the current list.
        context_end = _adjust_context_end(context_end, len_before_rebind, len(messages))

        llm_model_name = resolve_model_name(agent, config)
        llm = resolve_llm(agent, config)
        if config.router is None and result.context is not None:
            await enforce_tenant_budget(agent, config, result.context, messages, llm_model_name, llm, hooks)
        await hooks.on_llm_start(ctx_wrapper, agent, messages)
        if agent.hooks is not None:
            await agent.hooks.on_llm_start(ctx_wrapper, agent, messages)

        # Bracket the streaming Live widget. ``emit_stream_start`` opens
        # the Live panel; the ContextVar bridge then lets the
        # per-chunk emitter inside ``call_llm_streamed`` find the active
        # hooks chain without requiring it as a positional argument.
        emit_stream_start(hooks, agent)
        hooks_token = active_verbose_hooks.set(hooks)
        agent_token = active_verbose_agent.set(agent)
        final_response = None  # set before on_llm_end so the finally block always has it
        try:
            with generation_span(
                model=llm_model_name,
                tenant_id=result.context.tenant_id if result.context is not None else None,
                disabled=not (config.tracing_enabled or config.metrics_enabled),
            ) as gen_span:
                if config.router is not None:
                    # When a router and tenant_budget are both set, the per-candidate budget gate
                    # runs inside call_llm_streamed_with_routing (after on_llm_start). A budget
                    # kill on a routed call fires on_llm_start without a paired on_llm_end. The
                    # non-routed path gates before on_llm_start. This narrow edge is accepted.
                    stream_outcome = await call_llm_streamed_with_routing(
                        config.router,
                        agent,
                        messages,
                        config,
                        hooks,
                        result=result,
                        context=ctx_wrapper,
                        tool_failure_counts=tool_failure_counts,
                        tool_choice_override=force_tool_override or tool_choice_override,
                        extra_tools=extra_tools,
                    )
                    final_response = stream_outcome.response
                    llm_model_name = stream_outcome.model
                    llm = stream_outcome.llm
                    gen_span.data = dataclasses.replace(gen_span.data, model=llm_model_name)
                else:
                    final_response = await call_llm_streamed(
                        agent=agent,
                        messages=messages,
                        config=config,
                        result=result,
                        context=ctx_wrapper,
                        tool_failure_counts=tool_failure_counts,
                        tool_choice_override=force_tool_override or tool_choice_override,
                        extra_tools=extra_tools,
                    )
                    # Update to actual serving model — litellm may have used a fallback
                    # model that differs from the configured primary. The router path
                    # already does this via stream_outcome.model; mirror that here.
                    if final_response.model:
                        llm_model_name = final_response.model
                        gen_span.data = dataclasses.replace(gen_span.data, model=llm_model_name)
                if final_response.usage is not None:
                    call_cost = llm.cost(llm_model_name, final_response.usage)
                    gen_span.data = dataclasses.replace(
                        gen_span.data,
                        usage=dataclasses.asdict(final_response.usage),
                        cost_usd=call_cost,
                    )
                    # result.context is RunContext | None on the streaming path; the
                    # non-streaming context is always set, so that path needs no guard.
                    if result.context is not None and call_cost is not None:
                        result.context.cost_usd += call_cost
                        await record_tenant_spend(config, result.context, call_cost)
        finally:
            active_verbose_agent.reset(agent_token)
            active_verbose_hooks.reset(hooks_token)
            emit_stream_end(hooks, agent)
            # Always fire on_llm_end to keep hook pairs balanced, even
            # when call_llm_streamed raises (network error, budget kill).
            await hooks.on_llm_end(ctx_wrapper, agent, final_response)
            if agent.hooks is not None:
                await agent.hooks.on_llm_end(ctx_wrapper, agent, final_response)

        tool_choice_override = None

        if result.context is not None:
            usage_delta = final_response.usage if final_response.usage is not None else LLMUsage(requests=1)
            result.context.usage = result.context.usage + usage_delta
            emit_usage_recorded(hooks, agent, result.context.usage)

        if config.usage_limits is not None and result.context is not None:
            try:
                check_usage_limits(config.usage_limits, result.context.usage)
            except UsageLimitExceeded as exc:
                emit_budget_exceeded(hooks, agent, str(exc))
                raise

        # Resolve next step via shared turn resolution
        from troopai.adk.run.next_step import (
            NextStepFinalOutput,
            NextStepHandoff,
            NextStepInterruption,
            NextStepRunAgain,
            NextStepSwarmYield,
        )
        from troopai.adk.run.turn_resolution import (
            resolve_handoff_step,
            resolve_structured_output_step,
            resolve_swarm_yield_step,
            resolve_tool_results_step,
        )

        step = await resolve_structured_output_step(
            agent,
            final_response,
            messages,
            result.new_items,
            context_end,
            result.context,
            ctx_wrapper,
            hooks,
            config,
        )

        if step is None:
            response_items_s = ItemHelpers.response_to_run_items(final_response, agent.name)
            for ri in response_items_s:
                messages.append(ri.to_param())
            result.new_items.extend(response_items_s)

            # Only emit MESSAGE_OUTPUT_CREATED when the response
            # actually carries text or a refusal — tool-call-only turns
            # have no assistant message content and the event would be
            # null-content noise for streaming consumers.  This mirrors
            # the non-streaming path where response_to_run_items only
            # produces a MessageOutputItem when text/refusal parts exist.
            if final_response.content is not None or final_response.refusal is not None:
                await result.put_event(
                    RunItemStreamEvent(
                        name=RunItemType.MESSAGE_OUTPUT_CREATED,
                        item={"content": final_response.content, "role": "assistant"},
                    )
                )

            tool_calls = final_response.tool_calls

            if tool_calls is None or len(tool_calls) == 0:
                refusal_text_s = final_response.refusal
                if refusal_text_s is not None and final_response.content is None:
                    raise ModelRefusalError(refusal_text_s)
                step = NextStepFinalOutput(output=final_response.content)
            else:
                swarm_step_s: NextStepSwarmYield | None = None
                if swarm_tool_names is not None and len(swarm_tool_names) > 0:
                    swarm_step_s = resolve_swarm_yield_step(
                        agent,
                        tool_calls,
                        swarm_tool_names,
                        messages,
                        result.new_items,
                        context_end,
                        config,
                    )
                if swarm_step_s is not None:
                    step = swarm_step_s
                else:
                    handoff_step = await resolve_handoff_step(
                        agent,
                        tool_calls,
                        messages,
                        result.new_items,
                        context_end,
                        result.context,
                        ctx_wrapper,
                        hooks,
                        config,
                    )
                    if handoff_step is not None:
                        step = handoff_step
                    else:
                        llm_model_tc_s = resolve_model_name(agent, config)
                        tool_results, deferred = await execute_tool_calls_streamed(
                            agent=agent,
                            tool_calls=tool_calls,
                            ctx_wrapper=ctx_wrapper,
                            hooks=hooks,
                            config=config,
                            result=result,
                            tool_failure_counts=tool_failure_counts,
                            model=llm_model_tc_s,
                            messages=messages,
                            turn=result.current_turn,
                        )

                        if len(skill_tool_map) > 0 and len(tool_results) > 0:
                            await _activate_lazy_skills(
                                tool_results,
                                skill_tool_map,
                                activated_skills,
                                agent,
                                messages,
                                hooks,
                                ctx_wrapper,
                            )

                        step = await resolve_tool_results_step(
                            agent,
                            tool_results,
                            deferred,
                            tool_calls,
                            messages,
                            result.new_items,
                            user_prompt,
                            result.context,
                            ctx_wrapper,
                            result.current_turn,
                        )

        match step:
            case NextStepFinalOutput(output=output):
                emit_turn_end(hooks, agent, result.current_turn)
                result.final_output = output
                return AgentBlockOutcome(kind="final", turn=turn, total_turns_consumed=total_turns)
            case NextStepHandoff(new_agent=new_agent):
                emit_turn_end(hooks, agent, result.current_turn)
                await result.put_event(
                    RunItemStreamEvent(
                        name=RunItemType.HANDOFF_OCCURRED,
                        item={
                            "from_agent": agent.name,
                            "to_agent": new_agent.name,
                        },
                    )
                )
                await result.put_event(AgentUpdatedStreamEvent(new_agent=new_agent))
                return AgentBlockOutcome(
                    kind="handoff",
                    handoff_target=new_agent,
                    next_messages=step.new_messages,
                    next_context_end=step.context_end,
                    turn=turn,
                    total_turns_consumed=total_turns,
                )
            case NextStepInterruption(deferred=deferred, state=run_state):
                emit_turn_end(hooks, agent, result.current_turn)
                # ``execute_tool_calls_streamed`` already emitted a TOOL_OUTPUT
                # event for every completed result in this batch (the same
                # results carried on ``step.completed_tool_results``). Re-emitting
                # here would duplicate each one for streaming consumers, so only
                # the deferral state is surfaced.
                result.deferred_requests = deferred
                result.state = run_state
                result.final_output = None
                return AgentBlockOutcome(kind="final", turn=turn, total_turns_consumed=total_turns)
            case NextStepRunAgain(tool_choice_override=override):
                emit_turn_end(hooks, agent, result.current_turn)
                tool_choice_override = override
            case NextStepSwarmYield(signal=yield_signal):
                emit_turn_end(hooks, agent, result.current_turn)
                result.swarm_yield = yield_signal
                result.final_output = None
                return AgentBlockOutcome(kind="final", turn=turn, total_turns_consumed=total_turns)

    # Genuine turn exhaustion for this block. Every terminal pathway (final
    # output, handoff, interruption, swarm yield, cancel) returns from inside
    # the loop, so reaching here means the turn budget ran out mid-tool-loop.
    # Mirror the non-streaming ``run_agent_block``: give ``on_max_turns`` a
    # chance to salvage a final output, otherwise raise here. Raising from the
    # block (rather than signalling the driver to re-derive exhaustion from
    # ``result`` state) is what lets a legitimate empty final output
    # (``final_output is None``) on the last turn return cleanly.
    if config.on_max_turns is not None:
        logger.info(
            "Invoking on_max_turns handler (streamed, agent=%s, turns=%d)",
            agent.name,
            result.current_turn,
        )
        salvaged = await config.on_max_turns(agent, result.current_turn)
        if salvaged is not None:
            result.final_output = salvaged
            return AgentBlockOutcome(kind="final", turn=turn, total_turns_consumed=total_turns)
    raise MaxTurnsExceeded(f"Agent loop exceeded {result.max_turns} turns")


async def run_agent_loop(
    agent: Agent,
    user_prompt: UserPrompt,
    context: RunContext[TContext],
    ctx_wrapper: RunContext[TContext],
    hooks: RunHooks[TContext],
    max_turns: int,
    config: RunConfig,
    initial_messages: list[LLMInputContentItem] | None = None,
    initial_new_items: list[RunItem] | None = None,
    initial_tool_choice_override: str | None = None,
    extra_tools: list[FunctionTool] | None = None,
    swarm_tool_names: set[str] | None = None,
) -> RunResult:
    """Execute the main agent loop with per-agent middleware.

    The outer iteration applies ``Agent.middleware.agents`` around
    each per-agent block (one block = one agent's tenure until
    handoff or final). The chain re-fires on every handoff / swarm
    transition, giving users true per-agent observability that
    ``RunHooks.on_agent_start`` / ``on_agent_end`` cannot provide
    (those frame the whole run, not each agent's contribution).

    The loop continues until:
    1. Agent produces final output (no tool calls or handoffs)
    2. Max turns is reached
    3. An error occurs
    4. Tools are deferred for approval (HITL)
    5. A swarm-injected tool is called (only when ``swarm_tool_names``
       is non-empty) — returns a ``RunResult`` with ``swarm_yield`` set
       so the swarm driver can advance members.
    """
    from troopai.adk.run.agent_middleware import compose_agent_middleware

    current_agent = agent
    messages: list[LLMInputContentItem] = initial_messages or await build_initial_messages(
        current_agent, user_prompt, ctx_wrapper
    )
    new_items: list[RunItem] = list(initial_new_items) if initial_new_items is not None else []

    # Temporal slicing: tracks where context ends and output begins.
    # Everything before context_end existed before the agent's turn;
    # everything from context_end onward was produced during this turn.
    # Reset after each handoff when messages is rebuilt.
    context_end: int = len(messages)

    # Track per-tool failure counts for retry budget. Reset on handoff.
    tool_failure_counts: FunctionToolFailureCounts = {}

    # Track whether tool_choice should be overridden for the next LLM call.
    # Set to "auto" after tools execute when reset_tool_choice is True.
    # initial_tool_choice_override is used by resumption after HITL rejection.
    tool_choice_override: str | None = initial_tool_choice_override

    # Build context manager once for the whole run (run-level state)
    ctx_mgr = None
    if config.context_management is not None:
        from troopai.adk.context import ContextManager

        ctx_mgr = ContextManager(config.context_management)

    # Detect JIT directive store (re-detected per agent on handoff)
    jit_directives = _detect_jit_directives(current_agent)

    # LAZY skill activation: track which skills have been activated
    # and map tool names to skill names for activation on first call.
    # Run-level state — preserved across handoffs to match prior
    # behaviour (existing code never rebuilt the map on handoff).
    activated_skills: set[str] = set()
    skill_tool_map: dict[str, str] = {}
    if current_agent.skills:
        from troopai.adk.skills.activation import SkillActivation

        if current_agent.skill_activation == SkillActivation.LAZY:
            for _sk in current_agent.skills:
                for _t in _sk.tools:
                    _tname = getattr(_t, "name", None)
                    if _tname is not None:
                        skill_tool_map[_tname] = _sk.name

    # Cumulative turn counter across blocks (for the absolute
    # ``max_turns`` check inside ``run_agent_block`` and
    # ``RunConfig.max_total_turns`` for swarm safety).
    total_turns = 0
    turn_offset = 0

    while True:
        # The terminal of the agent-middleware chain calls
        # ``run_agent_block`` for the current agent. Bound here so the
        # chain's ``next`` callable matches the ``(agent, messages) ->
        # outcome`` shape even though the block needs the full set of
        # outer-loop state to run a turn. Default-arg binding captures
        # this iteration's *per-agent* values (which reset on
        # handoff) so a stored / forked chain never resolves to
        # mutated outer-scope state on a later call.
        #
        # NOT captured in default args (intentional): ``new_items``,
        # ``activated_skills``, ``skill_tool_map``, ``ctx_mgr``,
        # ``context``, ``ctx_wrapper``, ``hooks``, ``config``,
        # ``user_prompt``, ``extra_tools``, ``swarm_tool_names``.
        # These are run-level (cross-agent) shared state — the same
        # object identity must be visible to every block so item
        # accumulation, skill activation tracking, and context
        # management persist across handoffs. A future cache /
        # replay middleware that stores ``block_terminal`` and
        # invokes it later WILL see the live shared lists, by design.
        async def block_terminal(
            block_agent: Agent,
            block_messages: list[LLMInputContentItem],
            *,
            _context_end: int = context_end,
            _tool_failure_counts: FunctionToolFailureCounts = tool_failure_counts,
            _tool_choice_override: str | None = tool_choice_override,
            _jit_directives: Any = jit_directives,
            _turn_offset: int = turn_offset,
            _starting_total_turns: int = total_turns,
        ) -> AgentBlockOutcome:
            return await run_agent_block(
                agent=block_agent,
                messages=block_messages,
                context_end=_context_end,
                user_prompt=user_prompt,
                context=context,
                ctx_wrapper=ctx_wrapper,
                hooks=hooks,
                max_turns=max_turns,
                config=config,
                new_items=new_items,
                tool_failure_counts=_tool_failure_counts,
                initial_tool_choice_override=_tool_choice_override,
                extra_tools=extra_tools,
                swarm_tool_names=swarm_tool_names,
                ctx_mgr=ctx_mgr,
                jit_directives=_jit_directives,
                skill_tool_map=skill_tool_map,
                activated_skills=activated_skills,
                turn_offset=_turn_offset,
                starting_total_turns=_starting_total_turns,
            )

        chain = compose_agent_middleware(
            current_agent.middleware.agents,
            block_terminal,
            context=ctx_wrapper,
        )

        # ``AgentMiddlewareTermination`` is caught and unwrapped inside
        # ``compose_agent_middleware`` so the chain returns a normal
        # ``AgentBlockOutcome``; this loop receives the cached/synthetic
        # outcome like any other return value.
        outcome = await chain(current_agent, messages)

        # Carry cumulative turn counters forward so the next block's
        # absolute ``max_turns`` check sees the right offset.
        total_turns = outcome.total_turns_consumed
        turn_offset = total_turns

        if outcome.kind == "final":
            if outcome.result is None:
                raise RuntimeError("AgentBlockOutcome(kind='final') must carry a result")
            return outcome.result

        # handoff — reset per-agent state and re-enter the chain
        if outcome.handoff_target is None:
            raise RuntimeError("AgentBlockOutcome(kind='handoff') must carry a target")
        if outcome.next_messages is None:
            raise RuntimeError("AgentBlockOutcome(kind='handoff') must carry next_messages")
        if outcome.next_context_end is None:
            raise RuntimeError("AgentBlockOutcome(kind='handoff') must carry next_context_end")
        current_agent = outcome.handoff_target
        messages = outcome.next_messages
        context_end = outcome.next_context_end
        tool_failure_counts = {}
        jit_directives = _detect_jit_directives(current_agent)
        # ``tool_choice_override`` was already cleared inside the block
        # before exit; the new agent starts with no override.
        tool_choice_override = None


async def run_agent_loop_streamed(
    agent: Agent,
    user_prompt: UserPrompt,
    result: RunResultStreaming,
    ctx_wrapper: RunContext[TContext],
    hooks: RunHooks[TContext],
    config: RunConfig,
    initial_messages: list[LLMInputContentItem] | None = None,
    initial_tool_choice_override: str | None = None,
    extra_tools: list[FunctionTool] | None = None,
    swarm_tool_names: set[str] | None = None,
) -> None:
    """Execute the streaming agent loop with per-agent middleware.

    Mirrors the non-streaming :func:`run_agent_loop` driver pattern:
    each iteration builds the agent-middleware chain around
    :func:`run_agent_block_streamed` and dispatches on the returned
    outcome. Per-agent observability via ``Agent.middleware.agents``
    re-fires on every handoff / swarm transition, just like the
    non-streaming path.

    The streaming-only pathways (interruption, swarm yield, cancel,
    final output) all surface as ``AgentBlockOutcome(kind="final")``
    because the streaming driver only needs the binary
    "continue (handoff) or stop" decision; per-pathway state has
    already been mutated onto ``result`` inside the block.

    Args:
        agent: The agent to run.
        user_prompt: The original user prompt.
        result: The streaming result to emit events to.
        ctx_wrapper: The run context wrapper.
        hooks: Lifecycle hooks.
        config: Run configuration.
        initial_messages: Optional pre-built message list for resumption
            from a saved RunState. When provided, skips
            ``build_initial_messages()`` and uses these directly.
        initial_tool_choice_override: If provided, overrides the
            agent's tool choice for the first LLM call. Used by
            resumption after HITL rejection with ``reset_tool_choice``.
        extra_tools: Optional per-turn ephemeral tools injected by the
            swarm driver. Appended to ``build_tools`` output verbatim.
        swarm_tool_names: Names of the tools the swarm policy injected
            for this turn. Non-empty triggers ``resolve_swarm_yield_step``
            before handoff resolution so the driver wins the dispatch.
    """
    from troopai.adk.run.agent_middleware import compose_agent_middleware

    current_agent = agent
    messages: list[LLMInputContentItem] = initial_messages or await build_initial_messages(
        current_agent, user_prompt, ctx_wrapper
    )

    # Temporal slicing: tracks where context ends and output begins.
    context_end: int = len(messages)

    # Track per-tool failure counts for retry budget. Reset on handoff.
    tool_failure_counts: FunctionToolFailureCounts = {}

    # Track whether tool_choice should be overridden for the next LLM call.
    tool_choice_override: str | None = initial_tool_choice_override

    # Build context manager if configured (run-level)
    ctx_mgr = None
    if config.context_management is not None:
        from troopai.adk.context import ContextManager

        ctx_mgr = ContextManager(config.context_management)

    # Detect JIT directive store (re-detected per agent on handoff)
    jit_directives = _detect_jit_directives(current_agent)

    # LAZY skill activation: run-level state (preserved across handoffs)
    activated_skills: set[str] = set()
    skill_tool_map: dict[str, str] = {}
    if current_agent.skills:
        from troopai.adk.skills.activation import SkillActivation

        if current_agent.skill_activation == SkillActivation.LAZY:
            for _sk in current_agent.skills:
                for _t in _sk.tools:
                    _tname = getattr(_t, "name", None)
                    if _tname is not None:
                        skill_tool_map[_tname] = _sk.name

    # Cumulative turn counter across blocks (drives the absolute
    # max_turns check inside run_agent_block_streamed and
    # RunConfig.max_total_turns for swarm safety).
    total_turns = 0

    while True:
        # Build the per-block terminal closure. Per-agent values
        # (jit_directives, tool_failure_counts, tool_choice_override,
        # context_end) reset on handoff so each iteration captures
        # them via default-arg binding to keep the chain's `next`
        # callable shape (agent, messages) -> outcome.
        async def block_terminal(
            block_agent: Agent,
            block_messages: list[LLMInputContentItem],
            *,
            _context_end: int = context_end,
            _tool_failure_counts: FunctionToolFailureCounts = tool_failure_counts,
            _tool_choice_override: str | None = tool_choice_override,
            _jit_directives: Any = jit_directives,
            _starting_total_turns: int = total_turns,
        ) -> AgentBlockOutcome:
            return await run_agent_block_streamed(
                agent=block_agent,
                messages=block_messages,
                result=result,
                user_prompt=user_prompt,
                ctx_wrapper=ctx_wrapper,
                hooks=hooks,
                config=config,
                tool_failure_counts=_tool_failure_counts,
                initial_tool_choice_override=_tool_choice_override,
                extra_tools=extra_tools,
                swarm_tool_names=swarm_tool_names,
                ctx_mgr=ctx_mgr,
                jit_directives=_jit_directives,
                skill_tool_map=skill_tool_map,
                activated_skills=activated_skills,
                context_end=_context_end,
                starting_total_turns=_starting_total_turns,
            )

        chain = compose_agent_middleware(
            current_agent.middleware.agents,
            block_terminal,
            context=ctx_wrapper,
        )

        # AgentMiddlewareTermination is caught and unwrapped inside
        # compose_agent_middleware; this driver receives the
        # cached/synthetic outcome like any other return value.
        outcome = await chain(current_agent, messages)
        total_turns = outcome.total_turns_consumed

        if outcome.kind == "final":
            # Every terminal pathway (final output, HITL interruption, swarm
            # yield, cancellation, or an ``on_max_turns``-salvaged exhaustion)
            # has already mutated ``result`` inside the block. Genuine turn
            # exhaustion raises ``MaxTurnsExceeded`` from within the block
            # (mirroring the non-streaming ``run_agent_block``), so the driver
            # never re-derives it from ``result`` state — a heuristic that
            # could not tell a legitimate empty final output
            # (``final_output is None``) apart from real exhaustion.
            return

        # Handoff — reset per-agent state and re-enter the chain.
        if outcome.handoff_target is None or outcome.next_messages is None or outcome.next_context_end is None:
            raise RuntimeError("AgentBlockOutcome(kind='handoff') must carry target / next_messages / next_context_end")
        current_agent = outcome.handoff_target
        result.current_agent = current_agent
        messages = outcome.next_messages
        context_end = outcome.next_context_end
        tool_failure_counts = {}
        jit_directives = _detect_jit_directives(current_agent)
        tool_choice_override = None
