"""Turn resolution — determine the next step after an LLM response.

Extracts the decision logic from the agent loop into pure functions,
following the NextStep discriminated union pattern.  Both streaming
and non-streaming loops call these functions instead of duplicating
the decision logic inline.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from troopai.adk.exceptions import HandoffRejection
from troopai.adk.run.handoffs_executor import (
    apply_handoff_budget,
    execute_deterministic_handoff,
    execute_llm_handoff,
    prepare_handoff_input,
    prepare_handoff_input_from_data,
)
from troopai.adk.run.llm_calls import resolve_compaction_llm, resolve_model_name, resolve_output_schema
from troopai.adk.run.next_step import (
    NextStep,
    NextStepFinalOutput,
    NextStepHandoff,
    NextStepInterruption,
    NextStepRunAgain,
    NextStepSwarmYield,
)
from troopai.adk.run.tools_executor import check_tool_use_behavior
from troopai.adk.types.items.items import ItemHelpers

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext, TContext
    from troopai.adk.run.types import UserPrompt
    from troopai.adk.tools.deferred_tool import DeferredToolRequests
    from troopai.adk.types.input import LLMInputContentItem, LLMInputEasyMessage
    from troopai.adk.types.items.items import RunItem
    from troopai.adk.types.output import FunctionToolCallResult
    from troopai.adk.types.responses import LLMResponse

logger = logging.getLogger(__name__)


async def resolve_structured_output_step(
    current_agent: Agent,
    response: LLMResponse,
    messages: list[LLMInputContentItem],
    new_items: list[RunItem],
    context_end: int,
    context: Any,
    ctx_wrapper: RunContext[TContext],
    hooks: RunHooks[TContext],
    config: RunConfig,
) -> NextStep | None:
    """Resolve next step when agent uses structured output.

    Returns NextStepFinalOutput or NextStepHandoff if a deterministic
    handoff matches, or None if no structured output was produced.
    """
    if current_agent.output_schema is None:
        return None

    # Structured output validation: response.content carries the raw
    # JSON string the LLM emitted under response_format. Validate +
    # parse it into the agent's declared Pydantic type here so every
    # downstream consumer (handoff routing, swarm StructuredRoutingPolicy,
    # RunResult.final_output) sees a typed instance instead of a string.
    #
    # On validation failure we log + fall back to the raw string rather
    # than crash: downstream consumers (HandoffRoute.resolve, StructuredRoutingPolicy)
    # do isinstance() checks and route to their fallback branch when the
    # object isn't the declared type. Hard-failing here would turn every
    # malformed-JSON LLM response into a run-ending exception. Future
    # work: emit NextStepRunAgain with the validation error appended to
    # history so the LLM can self-correct within `max_retries`.
    schema = resolve_output_schema(current_agent)
    raw_content = response.content

    # If the LLM returned only tool calls (no text content) while the
    # agent has an output_schema set, allow the normal tool-execution
    # path to handle them instead of silently dropping them.  This
    # happens on non-strict response_format providers where the model
    # decides to call a tool on a continuation turn.
    if raw_content is None and len(response.tool_calls) > 0:
        logger.warning(
            "output_schema agent %r returned tool calls with no text content — "
            "deferring to tool-execution path so tool calls are not silently dropped.",
            current_agent.name,
        )
        return None

    if schema is None or raw_content is None:
        structured_output: Any = raw_content
    elif schema.is_plain_text():
        structured_output = raw_content
    else:
        try:
            structured_output = schema.validate_json(raw_content)
        # Catch Exception (not just ValueError) because AgentOutputSchemaBase
        # is abstract and user-supplied subclasses may raise other exception
        # types. The concrete AgentOutputSchema in this repo raises only
        # ValueError, but the contract can't enforce that across subclasses.
        except Exception as exc:
            logger.warning(
                "Agent %r output_schema validation failed on %d-char content "
                "(%s: %s); falling back to raw string so the run does not crash.",
                current_agent.name,
                len(raw_content),
                type(exc).__name__,
                exc,
            )
            structured_output = raw_content

    # Check for deterministic handoff based on Intent routing
    if current_agent.handoffs is not None and not isinstance(current_agent.handoffs, list):
        target = await current_agent.handoffs.resolve(structured_output, context)
        if target is not None:
            # Build the triage message as a proper ``LLMInputEasyMessage``
            # so the append + downstream helper see the declared Layer 1
            # shape.  ``thinking_blocks`` is a wire-format passthrough
            # Anthropic carries on assistant messages: it's read back out
            # in ``ItemHelpers.messages_to_run_items`` but not modeled on
            # the TypedDict.  Stash it via an ``Any``-typed alias so the
            # write lands on the runtime dict without broadening the
            # declared type of ``triage_msg``.  Add a typed slot on
            # ``LLMInputEasyMessage`` to remove this alias.
            triage_msg: LLMInputEasyMessage = {
                "role": "assistant",
                "content": response.content if response.content is not None else "",
            }
            # ``thinking_parts`` are ``LLMResponseReasoning`` dataclasses;
            # the ``thinking_blocks`` slot is the wire shape both the
            # converter passthrough and ``messages_to_run_items`` consume —
            # a ``list[dict]`` keyed ``type``/``thinking``/``signature``.
            # Emit that shape so extended-thinking replay survives the
            # deterministic-handoff path. ``encrypted_content`` (redacted
            # thinking) stands in for the signature when present.
            thinking_blocks: list[dict[str, str]] = []
            for part in response.thinking_parts:
                block: dict[str, str] = {"type": "thinking", "thinking": part.thinking}
                signature = part.encrypted_content if part.encrypted_content is not None else part.signature
                if signature is not None:
                    block["signature"] = signature
                thinking_blocks.append(block)
            if len(thinking_blocks) > 0:
                triage_any: Any = triage_msg
                triage_any["thinking_blocks"] = thinking_blocks
            messages.append(triage_msg)
            new_items.extend(_msg_to_items_impl(triage_msg, current_agent.name))

            # Execute handoff with temporal slicing.  If the target's
            # input_filter or on_handoff callback raises and the config
            # is on_error="reject_with_message", HandoffTarget.invoke
            # wraps the error in HandoffRejection.  On the deterministic
            # path there is no tool-result slot to write the rejection
            # to, so surface it as the agent's final output so the run
            # exits cleanly rather than crashing.
            context_items = tuple(ItemHelpers.messages_to_run_items(messages[:context_end]))
            output_items = tuple(ItemHelpers.messages_to_run_items(messages[context_end:]))
            try:
                new_agent, handoff_data = await execute_deterministic_handoff(
                    from_agent=current_agent,
                    target=target,
                    intent=structured_output,
                    context_msgs=context_items,
                    output_msgs=output_items,
                    context=context,
                    ctx_wrapper=ctx_wrapper,
                    hooks=hooks,
                    tracing_enabled=config.tracing_enabled,
                    metrics_enabled=config.metrics_enabled,
                )
            except HandoffRejection as rejection:
                logger.info(
                    "Deterministic handoff '%s' rejected: %s",
                    getattr(getattr(target, "target", None), "name", repr(target)),
                    rejection.tool_message,
                )
                # Remove the triage message already appended to avoid
                # leaving history in a half-mutated state.
                if messages and messages[-1] is triage_msg:
                    messages.pop()
                    if new_items and new_items[-1] is not None:
                        # Remove the items mirrored from triage_msg
                        items_from_triage = _msg_to_items_impl(triage_msg, current_agent.name)
                        del new_items[-len(items_from_triage) :]
                return NextStepFinalOutput(output=rejection.tool_message)
            llm_model = resolve_model_name(new_agent, config)
            compaction_llm = resolve_compaction_llm(new_agent, config)
            new_messages = await prepare_handoff_input(
                target,
                handoff_data,
                llm=compaction_llm,
                model=llm_model,
                context=ctx_wrapper,
            )
            # Apply the target's ``HandoffConfig.budget`` (truncation) AFTER
            # strategy/filter selection but BEFORE system-prompt injection,
            # matching the LLM-orchestrated path below. ``HandoffTarget``
            # exposes the same ``.config.budget`` field as ``Handoff`` — the
            # truncation logic only reads ``handoff.config.budget`` and the
            # name accessor, both shared between the two types.
            new_messages = await apply_handoff_budget(new_messages, target, llm_model)
            new_messages = await _inject_system_prompt_impl(new_agent, new_messages, ctx_wrapper)

            return NextStepHandoff(
                new_agent=new_agent,
                new_messages=new_messages,
                context_end=len(new_messages),
                target=target,
                is_deterministic=True,
            )

    # No handoff — structured output is the final answer
    return NextStepFinalOutput(output=structured_output)


def resolve_swarm_yield_step(
    current_agent: Agent,
    tool_calls: list,
    swarm_tool_names: set[str],
    messages: list[LLMInputContentItem],
    new_items: list[RunItem],
    context_end: int,
    config: RunConfig,
) -> NextStepSwarmYield | None:
    """Detect a policy-injected swarm tool call and convert to a yield.

    This runs BEFORE ``resolve_handoff_step`` on turns that were
    dispatched with a non-empty ``swarm_tool_names`` set. Detection
    keys on the exact tool-name set the policy produced for this
    turn, not on name prefixes — so a non-swarm agent's genuine
    ``transfer_to_foo`` handoff tool cannot be mistaken for a swarm
    transfer.

    On match, the swarm-injected tool call is acknowledged with a
    synthetic ``function_call_output`` (mirroring the handoff path),
    any sibling tool calls are skipped with the standard
    ``handoff_skipped`` message, and the yield signal is returned.
    The single-agent loop then surfaces the
    :class:`NextStepSwarmYield` back up to the swarm driver in
    ``run/swarm_loop.py``.

    Args:
        current_agent: The agent that produced the yielding tool call.
        tool_calls: All tool calls in the current LLM response.
        swarm_tool_names: Set of tool names injected by the swarm
            policy for this turn. An empty set means this turn was
            not dispatched as a swarm turn and this helper should
            never have been called.
        messages: The provider-agnostic message list being built for
            the next turn. Mutated in place to append ack/skip
            outputs.
        new_items: Layer-3 item list being accumulated for the run.
            Mutated in place to mirror ``messages``.
        context_end: Current context boundary — forwarded verbatim on
            the yield for the swarm driver.
        config: Run configuration, for ``get_messages`` localization.

    Returns:
        ``NextStepSwarmYield`` on the first matching tool call, or
        ``None`` if no swarm tool was called this turn (the LLM
        refused or produced other tool calls only).
    """
    if len(swarm_tool_names) == 0:
        return None

    import json as _json

    from troopai.adk.handoffs.handoff import HANDOFF_TOOL_PREFIX
    from troopai.adk.run.config import get_messages
    from troopai.adk.swarms.yield_signal import (
        SWARM_DONE_TOOL_NAME,
        SwarmDone,
        SwarmHandoff,
        SwarmYieldSignal as _SwarmYieldSignal,
    )
    from troopai.adk.types.input import FunctionToolCallResultParam as _ToolResultParam

    msgs = get_messages(config)

    for tool_call in tool_calls:
        if tool_call.name not in swarm_tool_names:
            continue

        # Parse arguments — tool_call.arguments is a JSON string per LLMResponseFunctionToolCall
        try:
            parsed: dict[str, Any] = _json.loads(tool_call.arguments) if tool_call.arguments else {}
        except (ValueError, TypeError) as e:
            logger.warning(
                "Swarm tool %r produced unparseable arguments %r: %s — treating as empty",
                tool_call.name,
                tool_call.arguments,
                e,
            )
            parsed = {}

        signal: _SwarmYieldSignal
        if tool_call.name == SWARM_DONE_TOOL_NAME:
            reason_raw = parsed.get("reason")
            reason: str = reason_raw if isinstance(reason_raw, str) else ""
            signal = SwarmDone(reason=reason, final_output=None)
            ack_output = f"swarm_done acknowledged: {reason}" if len(reason) > 0 else "swarm_done acknowledged"
            logger.info("Swarm done signal from %s: %s", current_agent.name, reason)
        elif tool_call.name.startswith(HANDOFF_TOOL_PREFIX):
            target = tool_call.name[len(HANDOFF_TOOL_PREFIX) :]
            message_raw = parsed.get("message")
            message: str = message_raw if isinstance(message_raw, str) else ""
            signal = SwarmHandoff(target=target, message=message)
            ack_output = msgs.handoff_transferred(target)
            logger.info(
                "Swarm handoff from %s to %s (message length=%d)",
                current_agent.name,
                target,
                len(message),
            )
        else:
            logger.warning(
                "Tool %r listed in swarm_tool_names but unrecognized shape — skipping",
                tool_call.name,
            )
            continue

        # Acknowledge the triggering call; skip siblings using the same pattern
        # as the handoff path so the LLM sees a coherent tool-result batch.
        for other_tc in tool_calls:
            if other_tc.call_id == tool_call.call_id:
                continue
            skip_msg = _ToolResultParam(
                type="function_call_output",
                call_id=other_tc.call_id,
                output=msgs.handoff_skipped,
            )
            messages.append(skip_msg)
            new_items.extend(_msg_to_items_impl(skip_msg, current_agent.name))

        ack_msg = _ToolResultParam(
            type="function_call_output",
            call_id=tool_call.call_id,
            output=ack_output,
        )
        messages.append(ack_msg)
        new_items.extend(_msg_to_items_impl(ack_msg, current_agent.name))

        return NextStepSwarmYield(signal=signal, context_end=context_end)

    return None


async def resolve_handoff_step(
    current_agent: Agent,
    tool_calls: list,
    messages: list[LLMInputContentItem],
    new_items: list[RunItem],
    context_end: int,
    context: Any,
    ctx_wrapper: RunContext[TContext],
    hooks: RunHooks[TContext],
    config: RunConfig,
) -> NextStepHandoff | NextStepRunAgain | None:
    """Check if any tool call is a handoff and execute it.

    Returns:
        - ``NextStepHandoff`` if a handoff was found and succeeded.
        - ``NextStepRunAgain`` if a handoff matched but was rejected
          via ``HandoffConfig.on_error == "reject_with_message"`` (the
          LLM sees the rejection as a tool result and gets to react).
        - ``None`` if no handoff tool call matched.
    """
    if current_agent.handoffs is None or not isinstance(current_agent.handoffs, list):
        return None

    from troopai.adk.handoffs.handoff_helpers import (
        find_handoff_target as _find_handoff_target,
        normalize_handoffs as _normalize_handoffs,
    )
    from troopai.adk.run.config import get_messages
    from troopai.adk.types.input import FunctionToolCallResultParam as _ToolResultParam

    normalized = await _normalize_handoffs(current_agent.handoffs)
    msgs = get_messages(config)

    for tool_call in tool_calls:
        target = await _find_handoff_target(normalized, tool_call.name, context)
        if target is None:
            continue

        # Try the handoff BEFORE emitting any tool-result messages.
        # If invoke raises HandoffRejection, we'll emit the rejection
        # message instead of "transferred" so the LLM sees a coherent
        # tool-result list.
        context_items = tuple(ItemHelpers.messages_to_run_items(messages[:context_end]))
        output_items = tuple(ItemHelpers.messages_to_run_items(messages[context_end:]))
        try:
            new_agent, handoff_data = await execute_llm_handoff(
                from_agent=current_agent,
                target=target,
                tool_call=tool_call,
                context_msgs=context_items,
                output_msgs=output_items,
                context=context,
                ctx_wrapper=ctx_wrapper,
                hooks=hooks,
                tracing_enabled=config.tracing_enabled,
                metrics_enabled=config.metrics_enabled,
            )
        except HandoffRejection as rejection:
            # Emit the rejection as the matched tool-result; skip the
            # parallel tool calls just like the success path. Return
            # NextStepRunAgain so the same agent is re-invoked with
            # the rejection visible in the conversation.
            for other_tc in tool_calls:
                if other_tc.call_id != tool_call.call_id:
                    skip_msg = _ToolResultParam(
                        type="function_call_output",
                        call_id=other_tc.call_id,
                        output=msgs.handoff_skipped,
                    )
                    messages.append(skip_msg)
                    new_items.extend(_msg_to_items_impl(skip_msg, current_agent.name))
            rejection_msg = _ToolResultParam(
                type="function_call_output",
                call_id=tool_call.call_id,
                output=rejection.tool_message,
            )
            messages.append(rejection_msg)
            new_items.extend(_msg_to_items_impl(rejection_msg, current_agent.name))
            logger.info(
                "Handoff '%s' rejected: %s",
                target.get_name(),
                rejection.tool_message,
            )
            return NextStepRunAgain()

        # Success — emit skip messages for parallel calls, then the
        # "transferred" message for the matched call.
        for other_tc in tool_calls:
            if other_tc.call_id != tool_call.call_id:
                skip_msg = _ToolResultParam(
                    type="function_call_output",
                    call_id=other_tc.call_id,
                    output=msgs.handoff_skipped,
                )
                messages.append(skip_msg)
                new_items.extend(_msg_to_items_impl(skip_msg, current_agent.name))

        handoff_msg = _ToolResultParam(
            type="function_call_output",
            call_id=tool_call.call_id,
            output=msgs.handoff_transferred(target.target.name),
        )
        messages.append(handoff_msg)
        new_items.extend(_msg_to_items_impl(handoff_msg, current_agent.name))

        llm_model = resolve_model_name(new_agent, config)
        compaction_llm = resolve_compaction_llm(new_agent, config)
        new_messages = await prepare_handoff_input_from_data(
            target,
            handoff_data,
            llm=compaction_llm,
            model=llm_model,
            context=ctx_wrapper,
        )
        new_messages = await apply_handoff_budget(new_messages, target, llm_model)
        new_messages = await _inject_system_prompt_impl(new_agent, new_messages, ctx_wrapper)

        return NextStepHandoff(
            new_agent=new_agent,
            new_messages=new_messages,
            context_end=len(new_messages),
            target=target,
        )

    return None


async def resolve_tool_results_step(
    current_agent: Agent,
    tool_results: list[FunctionToolCallResult],
    deferred: DeferredToolRequests | None,
    tool_calls: list,
    messages: list[LLMInputContentItem],
    new_items: list[RunItem],
    user_prompt: UserPrompt,
    context: Any,  # noqa: ARG001
    ctx_wrapper: RunContext[TContext],
    turn_count: int,
) -> NextStep:
    """Resolve next step after tool execution.

    Handles HITL interruption, tool_use_behavior short-circuit,
    and normal continuation.

    ``turn_count`` is the run-cumulative turn number (turns consumed by
    prior agent blocks plus the current block's turn), NOT the block-local
    turn. It is stamped onto the ``RunState`` built for a HITL interruption
    so a resumed run computes the correct remaining-turn budget after a
    handoff. Both loop drivers pass the cumulative value.
    """
    from troopai.adk.run.state import RunState
    from troopai.adk.types.input import FunctionToolCallResultParam

    # HITL interruption
    if deferred is not None and not deferred.is_empty():
        # Append completed results before snapshotting
        for tr in tool_results:
            msg = FunctionToolCallResultParam(
                type="function_call_output",
                call_id=tr.call_id,
                output=tr.output,
            )
            messages.append(msg)
            new_items.extend(_msg_to_items_impl(msg, current_agent.name))

        run_state = RunState(
            conversation_history=list(ItemHelpers.messages_to_run_items(messages)),
            context=ctx_wrapper.context,
            deferred_tool_requests=deferred,
            original_user_prompt=user_prompt,
            current_agent_name=current_agent.name,
            turn_count=turn_count,
        )
        return NextStepInterruption(
            deferred=deferred,
            state=run_state,
            completed_tool_results=tool_results,
        )

    # tool_use_behavior short-circuit
    if len(tool_results) > 0:
        behavior_output = await check_tool_use_behavior(
            agent=current_agent,
            tool_results=tool_results,
            tool_calls=tool_calls,
            ctx_wrapper=ctx_wrapper,
        )
        if behavior_output is not None:
            # Append results for observability
            for tr in tool_results:
                msg = FunctionToolCallResultParam(
                    type="function_call_output",
                    call_id=tr.call_id,
                    output=tr.output,
                )
                messages.append(msg)
                new_items.extend(_msg_to_items_impl(msg, current_agent.name))
            return NextStepFinalOutput(output=behavior_output)

    # Normal continuation — append results and loop
    for tr in tool_results:
        msg = FunctionToolCallResultParam(
            type="function_call_output",
            call_id=tr.call_id,
            output=tr.output,
        )
        messages.append(msg)
        new_items.extend(_msg_to_items_impl(msg, current_agent.name))

    # Check reset_tool_choice
    agent_llm_config = current_agent.llm_config
    override = None
    if (
        agent_llm_config is not None
        and agent_llm_config.tool_choice == "required"
        and agent_llm_config.reset_tool_choice is not False
    ):
        override = "auto"
        logger.debug("reset_tool_choice: next LLM call will use 'auto' instead of 'required'")

    return NextStepRunAgain(tool_choice_override=override)


# ---------------------------------------------------------------------------
# Helpers — avoid circular imports by using function-level imports
# ---------------------------------------------------------------------------


def _msg_to_items_impl(msg: LLMInputContentItem, agent_name: str | None = None) -> list[RunItem]:
    """Convert a message dict to RunItems via ItemHelpers."""
    from troopai.adk.types.items.items import ItemHelpers

    return ItemHelpers.message_to_run_items(msg, agent_name)


async def _inject_system_prompt_impl(
    agent: Agent,
    messages: list[LLMInputContentItem],
    ctx_wrapper: RunContext[TContext],
) -> list[LLMInputContentItem]:
    """Delegate to ``loop.inject_system_prompt`` without importing at module level."""
    from troopai.adk.run.loop import inject_system_prompt

    return await inject_system_prompt(agent, messages, ctx_wrapper)
