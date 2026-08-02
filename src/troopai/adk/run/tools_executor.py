"""Tool execution with the three-layer system.

Tenant gate: per-tenant tool allowlist (before lookup in the main path so
  builtins are gated; after lookup on HITL resume — always fires either way)
Layer 0: Permission check (can_use_tool callback)
Layer 1: Input guardrails
Layer 2: HITL check (requires_approval) → May defer for approval
Layer 3: Execute tool + output guardrails

Extracted from ``Runner`` with full typing.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import json
import logging
from collections.abc import AsyncIterator
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, Literal, Protocol, assert_never, runtime_checkable

from troopai.adk.exceptions import (
    AgentToolDeferral,
    ToolGuardrailTripwireTriggered,
    ToolRetry,
    ToolTimeoutError,
    UsageLimitExceeded,
)
from troopai.adk.run.config import DEFAULT_MODEL
from troopai.adk.run.cost import apply_result_limits, check_tool_call_limits_before_dispatch, minify_json
from troopai.adk.run.governance import emit_audit, emit_guardrail_audit, enforce_tenant_allowlist
from troopai.adk.tracing import function_span
from troopai.adk.types.tools.tool_stream_event import ToolStreamEvent
from troopai.adk.verbose.hooks import (
    emit_cache_hit,
    emit_cache_miss,
    emit_hitl_approval_requested,
    emit_retry,
    emit_tool_error,
)

type FunctionToolFailureCounts = dict[str, int]
"""Per-tool failure counter for the retry budget system.

Maps tool name → number of failures.  When the count exceeds
``FunctionTool.max_retries``, the tool is excluded from the LLM's
tool list so it can't waste more turns.
"""


@runtime_checkable
class ToolStreamSink(Protocol):
    """Sink that receives partial events from a streaming function tool.

    Plumbing-only sink — no policy verdict. The streaming tool
    executor sets ``RunResultStreaming`` (which already exposes
    ``put_event``) as the sink for the duration of a streaming-tool
    call. The non-streaming executor leaves the sink unset, so
    drains during ``Runner.arun()`` (without ``stream=True``) silently
    discard partial events while still surfacing the final value to
    the LLM.
    """

    async def put_event(self, event: Any) -> None:
        """Enqueue one stream event (typically a ``RunItemStreamEvent``)."""


_TOOL_STREAM_SINK: contextvars.ContextVar[ToolStreamSink | None] = contextvars.ContextVar(
    "troopai_tool_stream_sink",
    default=None,
)
"""Carries the active stream sink across the middleware chain.

Set by ``execute_tool_calls_streamed`` for each streaming-tool call;
unset (default ``None``) when the call runs under
``execute_tool_calls`` or any other non-streaming path. Read inside
``drain_streaming_tool_value`` so the drain target can change
without threading a ``stream_sink`` parameter through every layer of
the chain.
"""


def _check_tool_batch_limits(config: RunConfig, ctx_wrapper: RunContext[Any], pending: int) -> None:
    """Raise before dispatching a tool batch that exceeds configured limits."""
    if pending > config.max_tool_calls_per_turn:
        raise UsageLimitExceeded(f"Tool calls per turn exceeded: {pending} > {config.max_tool_calls_per_turn}")
    if config.usage_limits is not None:
        check_tool_call_limits_before_dispatch(config.usage_limits, ctx_wrapper.usage, pending)


async def drain_streaming_tool_value(
    value: Any,
    tool_name: str,
) -> Any:
    """Drain a streaming tool's iterator into a final scalar value.

    Pass-through when ``value`` is not an async iterator — already
    drained by an inner middleware terminal, or simply a plain return
    from a non-streaming tool. This makes the helper safe to call
    unconditionally from each terminal closure: only the innermost
    one (the one wrapping the user's async-gen) actually drains.

    Reads the active :class:`ToolStreamSink` from the
    ``_TOOL_STREAM_SINK`` contextvar. ``None`` (the non-streaming
    path) silences partial-event forwarding and emits one warning so
    the operator notices their streaming tool's chunks are being
    discarded.

    On any exit path — normal completion, exception, or cancellation
    — the iterator is explicitly ``aclose()``-d via
    :func:`contextlib.aclosing` so the producer's ``finally`` blocks
    run promptly. Without this, an in-flight cancel or a raise from
    ``sink.put_event`` would abandon the user's generator until GC,
    leaking any sockets / file handles / DB cursors it holds open
    for streaming.
    """
    if not hasattr(value, "__aiter__"):
        return value

    from troopai.adk.run.stream import RunItemStreamEvent, RunItemType

    sink = _TOOL_STREAM_SINK.get()
    if sink is None:
        logger.warning(
            "Streaming tool '%s' invoked under non-streaming path; "
            "partial events discarded (final value still returned).",
            tool_name,
        )

    final_value: Any = None
    iterator: AsyncIterator[Any] = value
    try:
        async for event in iterator:
            if isinstance(event, ToolStreamEvent) and event.type == "done":
                final_value = event.response
                continue
            if sink is not None:
                await sink.put_event(
                    RunItemStreamEvent(
                        name=RunItemType.TOOL_PARTIAL_OUTPUT,
                        item={"name": tool_name, "event": event},
                    )
                )
    finally:
        # Async generators expose ``aclose()`` for prompt cleanup of
        # ``finally`` blocks (sockets, file handles, DB cursors). The
        # ``__aiter__`` check earlier accepted any async iterator, so
        # gate on the attribute — generator-shaped iterators get
        # explicit cleanup; bare ``__aiter__``-only iterators degrade
        # gracefully to GC. Without this, a sink raise or an outer
        # cancel would abandon the producer's ``finally`` until GC.
        aclose = getattr(iterator, "aclose", None)
        if aclose is not None:
            await aclose()
    return final_value


def maybe_wrap_with_agent_middleware(
    tool: Any,
    middleware: list[Any],
) -> Any:
    """Return an invoke callable wrapped by the agent-global middleware chain.

    Returns ``tool.on_invoke`` unchanged when ``middleware`` is empty
    (zero-overhead path). When middleware is configured, returns a
    new callable with the same ``(ctx, raw_args_str) -> Any`` shape
    that runs through the middleware chain before calling
    ``tool.on_invoke``.

    Returns ``None`` when ``tool.on_invoke`` itself is ``None`` so the
    caller can short-circuit with the existing "no implementation"
    branch.
    """
    if tool.on_invoke is None:
        return None
    is_streaming = bool(getattr(tool, "streaming", False))
    if len(middleware) == 0 and not is_streaming:
        return tool.on_invoke

    from troopai.adk.tools.tool_middleware import (
        ToolMiddlewareTermination,
        compose_tool_middleware,
    )
    from troopai.adk.types.output.function_tool_call_result import FunctionToolCallResult

    original = tool.on_invoke
    tool_name = tool.name

    if len(middleware) == 0:
        # Streaming tool with no middleware: drain wrapper only.
        # Skipping the middleware-chain machinery keeps overhead at
        # one extra await + the drain itself.
        async def streaming_invoke(ctx_inner: Any, raw_args: str) -> Any:
            value = await original(ctx_inner, raw_args)
            return await drain_streaming_tool_value(value, tool_name)

        return streaming_invoke

    async def terminal(
        ctx_inner: Any,
        tool_inner: Any,
        args_inner: dict[str, Any],
    ) -> Any:
        # The actual tool call. Re-serialise the (possibly mutated)
        # args dict back to JSON so ``original`` sees the
        # JSON-string interface it was authored for.
        del tool_inner
        # Serialise the (possibly mutated) args dict back to JSON. An empty
        # dict must round-trip to "{}" (valid empty JSON object), never ""
        # (invalid JSON) — the main executor passes "{}" for empty args, and
        # a tool authored against the JSON-string interface may choke on "".
        raw = json.dumps(args_inner)
        out = await original(ctx_inner, raw)
        # Drain streaming-tool iterators so middleware sees the
        # final accumulated value, not chunks. Pass-through for
        # already-drained values (the typical non-streaming case).
        out = await drain_streaming_tool_value(out, tool_name)
        if isinstance(out, FunctionToolCallResult):
            return out
        # ``content_and_artifact`` tools return a 2-tuple
        # (content_str, artifact); preserve the artifact in the result
        # so ``wrapped`` (and the executor) can forward it instead of
        # stringifying the whole tuple into ``output``.
        if isinstance(out, tuple) and len(out) == 2:
            content, artifact = out
            return FunctionToolCallResult(
                type="function_call_output",
                call_id=ctx_inner.tool_call_id or "",
                output=content if isinstance(content, str) else str(content),
                artifact=artifact,
            )
        return FunctionToolCallResult(
            type="function_call_output",
            call_id=ctx_inner.tool_call_id or "",
            output=out if isinstance(out, str) else str(out),
        )

    chain = compose_tool_middleware(middleware, terminal)

    async def wrapped(ctx_inner: Any, raw_args: str) -> Any:
        try:
            args_dict: dict[str, Any] = json.loads(raw_args) if len(raw_args) > 0 else {}
        except json.JSONDecodeError:
            # Defer to the original tool, which has its own JSON error path.
            # Drain streaming-tool iterators here too so a streaming tool's
            # async generator is reduced to its final value — matching the
            # terminal path — instead of leaking an undrained iterator that
            # the executor would then stringify to a generator repr.
            out = await original(ctx_inner, raw_args)
            return await drain_streaming_tool_value(out, tool_name)
        try:
            result_obj = await chain(ctx_inner, tool, args_dict)
        except ToolMiddlewareTermination as term:
            result_obj = term.result
        # Unwrap to the raw value the surrounding executor expects
        # (the "result" variable later goes through
        # FunctionToolCallResult construction with full post-
        # processing — minify, max_result_tokens, etc.). For
        # ``content_and_artifact`` tools the artifact must survive the
        # chain: return (output, artifact) so the executor rebuilds the
        # result with both fields intact instead of dropping the
        # artifact.
        if result_obj.artifact is not None:
            return (result_obj.output, result_obj.artifact)
        return result_obj.output

    return wrapped


if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext, TContext
    from troopai.adk.run.stream import (
        RunResultStreaming,
    )
    from troopai.adk.tools.deferred_tool import (
        DeferredToolCall,
        DeferredToolRequests,
    )
    from troopai.adk.tools.tool_context import ToolContext
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.output import FunctionToolCallResult
    from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HITL resumption: execute a previously-approved tool call
# ---------------------------------------------------------------------------


async def _invoke_tool_via_middleware(
    tool: Any,
    tool_ctx: Any,
    raw_args: str,
    tool_name: str,
    middleware: list[Any],
) -> Any:
    """Invoke a tool through the agent-global middleware chain and its timeout.

    Shared entry point so a human-approved (HITL) resumption executes a tool
    with the same semantics as the main loop: agent-global tool middleware
    wrapping, the per-tool ``timeout``, streaming-iterator drain, and
    ``content_and_artifact`` tuple unwrapping. Calling ``tool.on_invoke``
    directly — as the resume path once did — skipped all four.

    ``ToolRetry`` and ``TimeoutError`` propagate to the caller, which owns the
    retry-hint and timeout-behavior policy. For a ``content_and_artifact`` tool
    the content string is returned and the artifact dropped, because the
    resumption contract returns a plain string.
    """
    actual_invoke = maybe_wrap_with_agent_middleware(tool, middleware)
    if actual_invoke is None:
        return f"Tool '{tool_name}' has no implementation"
    tool_timeout = tool.timeout
    if tool_timeout is not None:
        result = await asyncio.wait_for(actual_invoke(tool_ctx, raw_args), timeout=tool_timeout)
    else:
        result = await actual_invoke(tool_ctx, raw_args)
    # Non-streaming + no middleware returns on_invoke's value as-is (no
    # drain), so drain unconditionally here; it is a pass-through for
    # already-drained values.
    result = await drain_streaming_tool_value(result, tool_name)
    if tool.response_format == "content_and_artifact" and isinstance(result, tuple) and len(result) == 2:
        return result[0]
    return result


async def execute_approved_tool(
    agent: Agent,
    approved_tool: DeferredToolCall,
    ctx_wrapper: RunContext[TContext],
    hooks: RunHooks[TContext],
    config: RunConfig,
    context: Any,
    messages: list[LLMInputContentItem] | None = None,
) -> tuple[str, bool]:
    """Execute a tool that was previously deferred and then approved by a human.

    Handles: tenant allowlist gate (cannot be bypassed via approval) →
    permission re-check → enabled check → ToolContext creation → execution →
    audit emit → output guardrails → cost optimization.

    Args:
        agent: The agent whose tool is being executed.
        approved_tool: The previously deferred tool call now approved.
        ctx_wrapper: Run-level context wrapper.
        hooks: Lifecycle hooks.
        config: Run configuration.
        context: Caller-supplied user context passed through to ToolContext.
        messages: Optional conversation history (Layer 1 params) used to
            construct :class:`ExecutionAwareToolContext` or
            :class:`HistoryAwareToolContext` when the tool opts in via
            ``tool.execution_aware`` or ``tool.history_aware``.

    Returns:
        A tuple of (result_content: str, success: bool).
        ``success`` is False when the tool was denied or disabled.
    """
    from troopai.adk.run.config import get_messages
    from troopai.adk.run.llm_calls import resolve_model_name
    from troopai.adk.tools.tool_context import ToolContext
    from troopai.adk.tools.tool_guardrails import ToolOutputGuardrailData

    msgs = get_messages(config)
    tool_name = approved_tool.tool_name

    # Find the tool via the shared resolver — handles wired ShellTool /
    # ApplyPatchTool without relying on agent.tools being mutated.
    from troopai.adk.run.llm_calls import resolve_function_tool

    tool = await resolve_function_tool(agent, tool_name, ctx_wrapper)

    if tool is None or tool.on_invoke is None:
        return msgs.tool_not_found_on_resumption(tool_name), False

    # --- Per-tenant tool allowlist gate (cannot be bypassed via approval) ---
    denial_msg = await enforce_tenant_allowlist(
        config,
        tenant_id=ctx_wrapper.tenant_id,
        agent_name=agent.name,
        tool_name=tool_name,
        call_id=approved_tool.tool_call_id,
        raw_args=approved_tool.raw_arguments,
    )
    if denial_msg is not None:
        return denial_msg, False

    # --- Permission check (re-run on resumption) ---
    if config.can_use_tool is not None:
        permission_ctx = ToolContext(
            tool_name=tool_name,
            tool_call_id=approved_tool.tool_call_id,
            tool_arguments=approved_tool.tool_arguments,
            raw_arguments=approved_tool.raw_arguments,
            context=context,
            _run_config=config,
        )
        try:
            allowed = config.can_use_tool(agent, tool_name, permission_ctx)
            if isawaitable(allowed):
                allowed = await allowed
        except Exception as e:
            logger.warning(
                "can_use_tool callback raised for tool '%s' on resumption (agent '%s'): %s — "
                "suppressing tool call; fix the callback to avoid spurious denials",
                tool_name,
                agent.name,
                e,
                exc_info=True,
            )
            return msgs.tool_permission_check_error(tool_name), False
        if not allowed:
            return msgs.tool_permission_denied(tool_name), False

    # --- Enabled check ---
    if not await tool.check_enabled(ctx_wrapper):
        return msgs.tool_disabled(tool_name), False

    # --- Execute ---
    # Mirror _execute_single_tool_call: build the richest context type the
    # tool has opted into. history_aware implies execution_aware.
    if tool.history_aware:
        from copy import copy

        from troopai.adk.context.token_counter import TokenCounter
        from troopai.adk.llms.llm_usage import LLMUsage
        from troopai.adk.tools.tool_context import HistoryAwareToolContext
        from troopai.adk.types.items.items import ItemHelpers

        history_msg_count = len(messages) if messages is not None else 0
        model = resolve_model_name(agent, config)
        history_token_est = TokenCounter.count_messages(messages, model) if messages is not None else 0
        usage_snapshot: LLMUsage = copy(ctx_wrapper.usage)
        run_items = ItemHelpers.messages_to_run_items(messages) if messages is not None else ()
        tool_ctx: ToolContext[Any] = HistoryAwareToolContext(
            tool_name=tool_name,
            tool_call_id=approved_tool.tool_call_id,
            tool_arguments=approved_tool.tool_arguments,
            raw_arguments=approved_tool.raw_arguments,
            context=context,
            _run_config=config,
            usage=usage_snapshot,
            turns=0,
            messages=history_msg_count,
            tokens=history_token_est,
            history=tuple(run_items),
        )
    elif tool.execution_aware:
        from copy import copy

        from troopai.adk.context.token_counter import TokenCounter
        from troopai.adk.llms.llm_usage import LLMUsage
        from troopai.adk.tools.tool_context import ExecutionAwareToolContext

        msg_count = len(messages) if messages is not None else 0
        model = resolve_model_name(agent, config)
        token_est = TokenCounter.count_messages(messages, model) if messages is not None else 0
        usage_snapshot = copy(ctx_wrapper.usage)
        tool_ctx = ExecutionAwareToolContext(
            tool_name=tool_name,
            tool_call_id=approved_tool.tool_call_id,
            tool_arguments=approved_tool.tool_arguments,
            raw_arguments=approved_tool.raw_arguments,
            context=context,
            _run_config=config,
            usage=usage_snapshot,
            turns=0,
            messages=msg_count,
            tokens=token_est,
        )
    else:
        tool_ctx = ToolContext(
            tool_name=tool_name,
            tool_call_id=approved_tool.tool_call_id,
            tool_arguments=approved_tool.tool_arguments,
            raw_arguments=approved_tool.raw_arguments,
            context=context,
            _run_config=config,
        )

    await hooks.on_tool_start(ctx_wrapper, agent, tool_name, approved_tool.tool_arguments)
    if agent.hooks is not None:
        await agent.hooks.on_tool_start(ctx_wrapper, agent, tool_name, approved_tool.tool_arguments)

    audit_outcome: Literal["ok", "error"] = "ok"
    try:
        # Route through the shared wrapped path so resumption honors
        # agent-global tool middleware, the per-tool timeout,
        # content_and_artifact unwrapping, and streaming drain — none of
        # which fired when on_invoke was called directly.
        result = await _invoke_tool_via_middleware(
            tool,
            tool_ctx,
            approved_tool.raw_arguments,
            tool_name,
            agent.middleware.tools,
        )
    except ToolRetry as retry:
        # Cooperative retry hint — surface it to the LLM verbatim rather
        # than recording a failure (mirrors the main executor now that the
        # tool wrapper re-raises ToolRetry instead of swallowing it).
        result = retry.hint
    except TimeoutError:
        # The wrapped invoke applies ``tool.timeout``; honor the same
        # timeout_behavior policy the main executor uses.
        tool_timeout = tool.timeout
        if tool_timeout is None:
            raise RuntimeError("Tool timeout fired without a configured timeout") from None
        timeout_err = ToolTimeoutError(tool_name=tool_name, timeout=tool_timeout)
        if tool.timeout_behavior == "raise_exception":
            await emit_audit(
                config,
                tenant_id=ctx_wrapper.tenant_id,
                agent_name=agent.name,
                tool_name=tool_name,
                call_id=approved_tool.tool_call_id,
                args=approved_tool.raw_arguments,
                outcome="error",
            )
            raise timeout_err from None
        if tool.timeout_error is not None:
            from troopai.adk.run.context import RunContext

            run_ctx = RunContext(context=ctx_wrapper.context)
            err_result = tool.timeout_error(run_ctx, timeout_err)
            result = await err_result if isawaitable(err_result) else err_result
        else:
            result = msgs.tool_timeout(tool_name, tool_timeout)
        audit_outcome = "error"
    except Exception as e:
        if config.fail_on_tool_error:
            await emit_audit(
                config,
                tenant_id=ctx_wrapper.tenant_id,
                agent_name=agent.name,
                tool_name=tool_name,
                call_id=approved_tool.tool_call_id,
                args=approved_tool.raw_arguments,
                outcome="error",
            )
            raise
        logger.warning("Approved tool '%s' failed on resumption: %s", tool_name, e)
        result = msgs.tool_execution_error(tool_name, str(e))
        audit_outcome = "error"

    # --- Output guardrails (before on_tool_end / audit) ---
    # Run output guardrails FIRST so hooks and the audit log observe the
    # final result that the LLM actually receives.
    if tool.guardrails is not None and tool.guardrails.output is not None and len(tool.guardrails.output) > 0:
        guardrail_data = ToolOutputGuardrailData(
            context=tool_ctx,
            agent=agent,
            output=result,
        )
        for guardrail in tool.guardrails.output:
            try:
                g_output = await guardrail.run(guardrail_data)
            except Exception as e:
                if config.fail_on_tool_error:
                    raise
                logger.warning(
                    "Output guardrail %s raised on resumption: %s",
                    guardrail.get_name(),
                    e,
                )
                continue
            behavior = g_output.behavior
            transformed_message: str | None = None
            if behavior["type"] == "reject_content":
                transformed_message = behavior["message"]
            emit_guardrail_audit(
                ctx_wrapper,
                level="tool_output",
                agent_name=agent.name,
                guardrail_name=guardrail.get_name(),
                action=g_output.resolved_action(),
                checked=result,
                transformed=transformed_message,
            )
            if behavior["type"] == "reject_content":
                result = behavior["message"]
                break
            elif behavior["type"] == "raise_exception":
                raise ToolGuardrailTripwireTriggered(
                    guardrail_name=guardrail.get_name(),
                    output_info=g_output.output_info,
                )

    await hooks.on_tool_end(ctx_wrapper, agent, tool_name, result)
    if agent.hooks is not None:
        await agent.hooks.on_tool_end(ctx_wrapper, agent, tool_name, result)

    await emit_audit(
        config,
        tenant_id=ctx_wrapper.tenant_id,
        agent_name=agent.name,
        tool_name=tool_name,
        call_id=approved_tool.tool_call_id,
        args=approved_tool.raw_arguments,
        outcome=audit_outcome,
        result=result,
    )

    # --- Post-processing ---
    # Apply minify + the ``max_result_tokens`` cap to non-str results too,
    # after stringifying — a cost cap the developer configured MUST hold
    # regardless of the tool's return type. The previous ``str(result)`` at
    # return time skipped the cap for any non-str value.
    model = resolve_model_name(agent, config)
    result_str = result if isinstance(result, str) else str(result)
    result_str = minify_json(result_str)
    result_str = apply_result_limits(result_str, tool, model)

    return result_str, True


# ---------------------------------------------------------------------------
# Shared single-tool-call executor (all layers)
# ---------------------------------------------------------------------------


async def _execute_single_tool_call(
    agent: Agent,
    tool_call: LLMResponseFunctionToolCall,
    ctx_wrapper: RunContext[TContext],
    hooks: RunHooks[TContext],
    config: RunConfig,
    tool_failure_counts: FunctionToolFailureCounts | None,
    model: str,
    messages: list[LLMInputContentItem] | None,
    turn: int,
) -> FunctionToolCallResult | DeferredToolCall:
    """Execute all layers for a single tool call.

    Handles, in execution order:
    1. Tool lookup → error result if not found
    2. Permission check (Layer 0) → denial result if ``can_use_tool`` blocks it
    3. Enabled check → error result if disabled
    4. Retry budget check → error result if exhausted
    4b. Rate-limit gate (per-tool sliding window) → sleeps or returns
        error result on saturation, depending on ``rate_limit.behavior``
    4c. Deferred-loading execution gate
    5. ToolContext / ExecutionAwareToolContext creation
    6. Input guardrails (Layer 1)
    7. HITL check (Layer 2) → returns DeferredToolCall
    8. Cache check
    9. on_tool_start hook
    10. Tool execution with timeout (Layer 3)
    11. AgentToolDeferral → returns DeferredToolCall with metadata
    12. on_tool_end hook
    13. Output guardrails
    14. Structured tool-output conversion (text/image/file parts)
    15. String-result post-processing (minify + truncate)

    Returns:
        ``FunctionToolCallResult`` for completed calls, or
        ``DeferredToolCall`` for deferred (HITL) calls.
    """
    from copy import copy

    from troopai.adk.run.context import RunContext
    from troopai.adk.tools.deferred_tool import (
        DeferredToolCall,
        DeferredToolCallMetadata,
        NestedDeferredToolRequests,
    )
    from troopai.adk.tools.tool_context import ExecutionAwareToolContext, ToolContext
    from troopai.adk.tools.tool_guardrails import (
        ToolInputGuardrailData,
        ToolOutputGuardrailData,
    )
    from troopai.adk.types.items.items import ItemHelpers
    from troopai.adk.types.output import FunctionToolCallResult

    call_id = tool_call.call_id
    tool_name = tool_call.name
    raw_args = tool_call.arguments or "{}"
    # LLMs occasionally emit malformed JSON in tool-call arguments
    # (truncation, trailing commas, unescaped characters). Degrade to an
    # empty dict for context/HITL/hook construction rather than crashing
    # the whole run; the tool's own ``on_invoke`` re-parses ``raw_args``
    # and surfaces a recoverable error the LLM can retry.
    try:
        tool_input = json.loads(raw_args)
    except json.JSONDecodeError:
        logger.warning(
            "Tool %r produced malformed JSON arguments %r — treating as empty",
            tool_name,
            raw_args,
        )
        tool_input = {}

    # --- Per-tenant tool allowlist gate (before lookup, so builtins are gated too) ---
    denial_msg = await enforce_tenant_allowlist(
        config,
        tenant_id=ctx_wrapper.tenant_id,
        agent_name=agent.name,
        tool_name=tool_name,
        call_id=call_id,
        raw_args=raw_args,
    )
    if denial_msg is not None:
        return FunctionToolCallResult(call_id=call_id, output=denial_msg)

    # --- Step 1: Tool lookup ---
    # Shared resolver handles agent.tools, skill tools, wired
    # ShellTool/ApplyPatchTool, and toolset materialisation, without
    # mutating agent state.
    from troopai.adk.run.llm_calls import resolve_function_tool

    tool = await resolve_function_tool(agent, tool_name, ctx_wrapper)

    if tool is None:
        from troopai.adk.run.config import get_messages

        return FunctionToolCallResult(
            call_id=call_id,
            output=get_messages(config).tool_not_found(tool_name),
        )

    # --- Step 2: Permission check (Layer 0) ---
    if config.can_use_tool is not None:
        permission_ctx = ToolContext(
            tool_name=tool_name,
            tool_call_id=call_id,
            tool_arguments=tool_input,
            raw_arguments=raw_args,
            context=ctx_wrapper.context,
            _run_config=config,
        )
        try:
            allowed = config.can_use_tool(agent, tool_name, permission_ctx)
            if isawaitable(allowed):
                allowed = await allowed
        except Exception as e:
            logger.warning(
                "can_use_tool callback raised for tool '%s' (agent '%s'): %s — "
                "suppressing tool call; fix the callback to avoid spurious denials",
                tool_name,
                agent.name,
                e,
                exc_info=True,
            )
            from troopai.adk.run.config import get_messages

            return FunctionToolCallResult(
                call_id=call_id,
                output=get_messages(config).tool_permission_check_error(tool_name),
            )
        if not allowed:
            from troopai.adk.run.config import get_messages

            return FunctionToolCallResult(
                call_id=call_id,
                output=get_messages(config).tool_permission_denied(tool_name),
            )

    # --- Step 3: Enabled check ---
    if not await tool.check_enabled(ctx_wrapper):
        from troopai.adk.run.config import get_messages

        return FunctionToolCallResult(
            call_id=call_id,
            output=get_messages(config).tool_disabled(tool_name),
        )

    # --- Step 4: Retry budget check ---
    if (
        tool_failure_counts is not None
        and tool.max_retries is not None
        and tool_failure_counts.get(tool_name, 0) > tool.max_retries
    ):
        from troopai.adk.run.config import get_messages

        logger.info("Tool '%s' disabled after %d failed attempt(s)", tool_name, tool.max_retries)
        return FunctionToolCallResult(
            call_id=call_id,
            output=get_messages(config).tool_retry_exhausted(tool_name, tool.max_retries),
        )

    # --- Step 4b: Rate limit check ---
    # Sliding-window enforcement is handled inside the tool. ``"wait"``
    # behavior sleeps until a slot opens (capped by max_wait_seconds);
    # ``"error"`` returns False so the executor surfaces a clear error
    # to the LLM instead of silently throttling. Skipped when
    # tool.rate_limit is None.
    if tool.rate_limit is not None and not await tool.acquire_rate_slot():
        return FunctionToolCallResult(
            call_id=call_id,
            output=(f"Tool '{tool_name}' is rate-limited (max {tool.rate_limit.rpm} calls/min). Try again shortly."),
        )

    # --- Step 4c: Deferred-loading execution gate ---
    # ``build_tools()`` already filters non-revealed deferred tools out
    # of the LLM's per-step tool list, but a misbehaving or
    # prompt-injected LLM can still emit a function-call to any tool
    # name it has seen in context. Refuse execution here so visibility
    # filtering doubles as a capability-gate; without this check
    # ``defer_loading`` would be a token-cost optimisation only.
    if tool.defer_loading:
        from troopai.adk.tools.tool_search import find_revealed_deferred_tools

        revealed = find_revealed_deferred_tools(agent.tools)
        if tool_name not in revealed:
            from troopai.adk.run.config import get_messages

            return FunctionToolCallResult(
                call_id=call_id,
                output=get_messages(config).tool_not_found(tool_name),
            )

    # --- Step 5: Create ToolContext ---
    if tool.history_aware:
        # HistoryAwareToolContext: execution state + read-only history snapshot.
        # Convert Layer 1 wire types → Layer 3 RunItems to preserve the
        # three-layer type boundary (tools never see wire-format message types).
        from troopai.adk.context.token_counter import TokenCounter
        from troopai.adk.llms.llm_usage import LLMUsage
        from troopai.adk.tools.tool_context import HistoryAwareToolContext

        history_msg_count: int = len(messages) if messages is not None else 0
        history_token_est: int = TokenCounter.count_messages(messages, model) if messages is not None else 0
        usage_snapshot: LLMUsage = copy(ctx_wrapper.usage)
        run_items = ItemHelpers.messages_to_run_items(messages) if messages is not None else ()

        tool_ctx: ToolContext[Any] = HistoryAwareToolContext(
            tool_name=tool_name,
            tool_call_id=call_id,
            tool_arguments=tool_input,
            raw_arguments=raw_args,
            context=ctx_wrapper.context,
            _run_config=config,
            usage=usage_snapshot,
            turns=turn,
            messages=history_msg_count,
            tokens=history_token_est,
            history=tuple(run_items),
        )
    elif tool.execution_aware:
        from troopai.adk.context.token_counter import TokenCounter
        from troopai.adk.llms.llm_usage import LLMUsage

        msg_count = len(messages) if messages is not None else 0
        token_est = TokenCounter.count_messages(messages, model) if messages is not None else 0
        # Copy usage to prevent mutation by tools (preserves all 7 fields)
        usage_snapshot = copy(ctx_wrapper.usage)

        tool_ctx = ExecutionAwareToolContext(
            tool_name=tool_name,
            tool_call_id=call_id,
            tool_arguments=tool_input,
            raw_arguments=raw_args,
            context=ctx_wrapper.context,
            _run_config=config,
            usage=usage_snapshot,
            turns=turn,
            messages=msg_count,
            tokens=token_est,
        )
    else:
        tool_ctx = ToolContext(
            tool_name=tool_name,
            tool_call_id=call_id,
            tool_arguments=tool_input,
            raw_arguments=raw_args,
            context=ctx_wrapper.context,
            _run_config=config,
        )

    # --- Step 5: Input guardrails (Layer 1) ---
    if tool.guardrails is not None and tool.guardrails.input is not None and len(tool.guardrails.input) > 0:
        guardrail_data = ToolInputGuardrailData(
            context=tool_ctx,
            agent=agent,
        )

        for guardrail in tool.guardrails.input:
            try:
                output = await guardrail.run(guardrail_data)
            except Exception as e:
                if config.fail_on_tool_error:
                    raise
                logger.warning(
                    "Input guardrail %s raised an exception and was skipped: %s",
                    guardrail.get_name(),
                    e,
                )
                continue

            behavior = output.behavior
            input_replacement: str | None = None
            if behavior["type"] == "reject_content":
                input_replacement = behavior["message"]
            emit_guardrail_audit(
                ctx_wrapper,
                level="tool_input",
                agent_name=agent.name,
                guardrail_name=guardrail.get_name(),
                action=output.resolved_action(),
                checked=raw_args,
                transformed=input_replacement,
            )
            if behavior["type"] == "reject_content":
                return FunctionToolCallResult(
                    call_id=call_id,
                    output=behavior["message"],
                )
            elif behavior["type"] == "raise_exception":
                raise ToolGuardrailTripwireTriggered(
                    guardrail_name=guardrail.get_name(),
                    output_info=output.output_info,
                )

    # --- Step 6: HITL check (Layer 2) ---
    try:
        needs_approval = await tool.check_requires_approval(tool_ctx)
    except Exception as e:
        logger.warning("Error checking approval condition for tool '%s': %s", tool_name, e)
        needs_approval = tool.requires_approval if isinstance(tool.requires_approval, bool) else True

    if needs_approval:
        # Surface the HITL approval gate through the verbose layer
        # before returning the deferral. ``tool_input`` is forwarded so
        # the panel shows operators *what* they are approving (redaction
        # happens inside the emit helper). A nested_path is left empty
        # at this call site — resumption.py supplies the breadcrumb when
        # an ``as_tool()`` boundary is involved.
        emit_hitl_approval_requested(
            hooks,
            agent,
            tool_name,
            call_id,
            tool_input=tool_input,
        )
        return DeferredToolCall(
            tool_call_id=call_id,
            tool_name=tool_name,
            tool_arguments=tool_input,
            raw_arguments=raw_args,
        )

    # --- Step 7: Cache check ---
    result: Any = ""  # Default; overwritten by cache hit or execution
    artifact = None
    cache_hit = False
    audit_outcome: Literal["ok", "error"] = "ok"
    cache_policy = tool.resolve_cache_policy()
    cached_value = tool.get_cached(raw_args, tool_ctx, ctx_wrapper) if cache_policy is not None else None
    if cached_value is not None:
        # Unpack (result, artifact) tuple when artifact was cached
        if isinstance(cached_value, tuple) and len(cached_value) == 2:
            result, artifact = cached_value
        else:
            result = cached_value
        cache_hit = True
        logger.debug("Cache hit for tool '%s'", tool_name)
        emit_cache_hit(hooks, agent, tool_name)
    elif cache_policy is not None:
        # Cache lookup ran but produced nothing — a miss. We only emit
        # misses when caching is actually enabled; a tool with caching
        # disabled is not "missing" on every call.
        emit_cache_miss(hooks, agent, tool_name)

    # --- Step 8: on_tool_start hook ---
    await hooks.on_tool_start(ctx_wrapper, agent, tool_name, tool_input)
    if agent.hooks is not None:
        await agent.hooks.on_tool_start(ctx_wrapper, agent, tool_name, tool_input)

    # --- Step 9–11: Execute tool (Layer 3) — skip if cached ---
    skip_cache = False
    tool_timeout = tool.timeout  # Capture for use in except block (narrowing)
    # Resolve the actual invoke callable. When agent-global tool
    # middleware is configured, wrap ``tool.on_invoke`` with the
    # middleware chain. Toolset-scoped middleware on the materialised
    # tool has already been applied at ``WrapperToolset.get_tools``
    # time, so the agent-global chain composes outside it.
    actual_invoke = maybe_wrap_with_agent_middleware(tool, agent.middleware.tools)
    if not cache_hit:
        try:
            if actual_invoke is not None:
                with function_span(
                    name=tool_name,
                    input=raw_args,
                    disabled=not (config.tracing_enabled or config.metrics_enabled),
                ) as tool_span:
                    if tool_timeout is not None:
                        result = await asyncio.wait_for(
                            actual_invoke(tool_ctx, raw_args),
                            timeout=tool_timeout,
                        )
                    else:
                        result = await actual_invoke(tool_ctx, raw_args)
                    tool_span.data = dataclasses.replace(
                        tool_span.data,
                        output=str(result),
                    )
            else:
                result = f"Tool '{tool_name}' has no implementation"
        except TimeoutError:
            # tool_timeout is guaranteed non-None here: TimeoutError only raised
            # from asyncio.wait_for which is only called when tool_timeout is not None
            if tool_timeout is None:
                raise RuntimeError("Tool timeout fired without a configured timeout")
            timeout_err = ToolTimeoutError(tool_name=tool_name, timeout=tool_timeout)
            if tool.timeout_behavior == "raise_exception":
                await emit_audit(
                    config,
                    tenant_id=ctx_wrapper.tenant_id,
                    agent_name=agent.name,
                    tool_name=tool_name,
                    call_id=call_id,
                    args=raw_args,
                    outcome="error",
                )
                raise timeout_err
            # error_as_result path
            if tool.timeout_error is not None:
                run_ctx = RunContext(context=ctx_wrapper.context)
                err_result = tool.timeout_error(run_ctx, timeout_err)
                if isawaitable(err_result):
                    result = await err_result
                else:
                    result = err_result
            else:
                from troopai.adk.run.config import get_messages

                result = get_messages(config).tool_timeout(tool_name, tool_timeout)
            audit_outcome = "error"
            # Count timeout as failure for retry budget
            if tool_failure_counts is not None and tool.max_retries is not None:
                tool_failure_counts[tool_name] = tool_failure_counts.get(tool_name, 0) + 1
                logger.debug(
                    "Tool '%s' failure %d/%s",
                    tool_name,
                    tool_failure_counts[tool_name],
                    tool.max_retries,
                )
        except AgentToolDeferral as deferral:
            # Sub-agent (via as_tool()) requires human approval.
            return DeferredToolCall(
                tool_call_id=call_id,
                tool_name=tool_name,
                tool_arguments=tool_input,
                raw_arguments=raw_args,
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
        except ToolRetry as retry:
            # Tool requested LLM retry with a hint — NOT counted as a failure
            result = retry.hint
            skip_cache = True
            logger.debug("Tool '%s' requested retry with hint: %s", tool_name, retry.hint)
            # Attempt number is best-effort from the retry budget
            # counter — it is the count of failures seen so far, so the
            # emitted panel reads "retry N: reason". A purely-cooperative
            # retry (no prior failure) reports attempt=1.
            attempt = 1
            if tool_failure_counts is not None:
                attempt = tool_failure_counts.get(tool_name, 0) + 1
            emit_retry(hooks, agent, tool_name, attempt, retry.hint or "tool requested retry")
        except Exception as e:
            # Announce the failure through the verbose layer before
            # either re-raising or converting to an error-as-result
            # message. The Panel renderer closes any open tool block
            # with a red border; the line renderer emits a one-liner.
            emit_tool_error(hooks, agent, tool_name, e)
            if config.fail_on_tool_error:
                await emit_audit(
                    config,
                    tenant_id=ctx_wrapper.tenant_id,
                    agent_name=agent.name,
                    tool_name=tool_name,
                    call_id=call_id,
                    args=raw_args,
                    outcome="error",
                )
                raise
            from troopai.adk.run.config import get_messages

            result = get_messages(config).tool_execution_error(tool_name, str(e))
            audit_outcome = "error"
            # Count exception as failure for retry budget
            if tool_failure_counts is not None and tool.max_retries is not None:
                tool_failure_counts[tool_name] = tool_failure_counts.get(tool_name, 0) + 1
                logger.debug(
                    "Tool '%s' failure %d/%s",
                    tool_name,
                    tool_failure_counts[tool_name],
                    tool.max_retries,
                )

        # Handle content_and_artifact response format
        artifact = None
        if tool.response_format == "content_and_artifact" and isinstance(result, tuple) and len(result) == 2:
            result, artifact = result

        # Store in cache on successful execution only (not on error/timeout/retry)
        if not skip_cache and cache_policy is not None:
            should_cache = True
            if tool.cache_function is not None:
                result_str = str(result) if not isinstance(result, str) else result
                try:
                    should_cache = tool.cache_function(raw_args, result_str)
                except Exception as e:
                    logger.warning("cache_function for tool '%s' raised: %s", tool_name, e)
                    should_cache = False
            if should_cache and isinstance(result, str):
                if artifact is not None:
                    tool.set_cached(raw_args, (result, artifact), tool_ctx, ctx_wrapper)
                else:
                    tool.set_cached(raw_args, result, tool_ctx, ctx_wrapper)

    # --- Step 11: Output guardrails (before on_tool_end / audit) ---
    # Run output guardrails FIRST so hooks and the audit log observe the
    # final result that the LLM actually receives, not the raw pre-guardrail
    # value.  A PII-masking guardrail must be visible to the audit trail.
    if tool.guardrails is not None and tool.guardrails.output is not None and len(tool.guardrails.output) > 0:
        guardrail_data = ToolOutputGuardrailData(
            context=tool_ctx,
            agent=agent,
            output=result,
        )

        for agent_output_guardrail in tool.guardrails.output:
            try:
                output_result = await agent_output_guardrail.run(guardrail_data)
            except Exception as e:
                if config.fail_on_tool_error:
                    raise
                logger.warning(
                    "Output guardrail %s raised an exception and was skipped: %s",
                    agent_output_guardrail.get_name(),
                    e,
                )
                continue

            output_behavior = output_result.behavior
            output_replacement: str | None = None
            if output_behavior["type"] == "reject_content":
                output_replacement = output_behavior["message"]
            emit_guardrail_audit(
                ctx_wrapper,
                level="tool_output",
                agent_name=agent.name,
                guardrail_name=agent_output_guardrail.get_name(),
                action=output_result.resolved_action(),
                checked=result,
                transformed=output_replacement,
            )
            if output_behavior["type"] == "reject_content":
                result = output_behavior["message"]
                break
            elif output_behavior["type"] == "raise_exception":
                raise ToolGuardrailTripwireTriggered(
                    guardrail_name=agent_output_guardrail.get_name(),
                    output_info=output_result.output_info,
                )

    # --- Step 12: on_tool_end hook (post-guardrail result) ---
    await hooks.on_tool_end(ctx_wrapper, agent, tool_name, result)
    if agent.hooks is not None:
        await agent.hooks.on_tool_end(ctx_wrapper, agent, tool_name, result)

    await emit_audit(
        config,
        tenant_id=ctx_wrapper.tenant_id,
        agent_name=agent.name,
        tool_name=tool_name,
        call_id=call_id,
        args=raw_args,
        outcome=audit_outcome,
        result=result,
    )

    # --- Step 14: Handle structured tool output types ---
    from troopai.adk.types.tools.tool_output_types import ToolOutputFileContent, ToolOutputImage, ToolOutputText

    if isinstance(result, (ToolOutputText, ToolOutputImage, ToolOutputFileContent)):
        result = [result]
    if (
        isinstance(result, list)
        and result
        and isinstance(result[0], (ToolOutputText, ToolOutputImage, ToolOutputFileContent))
    ):
        # Convert to multimodal output parts
        from troopai.adk.types.input.llm_input_image import LLMInputImage
        from troopai.adk.types.input.llm_input_text import LLMInputText

        output_parts: list[LLMInputText | LLMInputImage] = []
        for part in result:
            if isinstance(part, ToolOutputText):
                output_parts.append(LLMInputText(type="input_text", text=part.text))
            elif isinstance(part, ToolOutputImage):
                output_parts.append(LLMInputImage(type="input_image", image_url=part.image_url))
            elif isinstance(part, ToolOutputFileContent):
                output_parts.append(LLMInputText(type="input_text", text=f"[File: {part.filename or 'unknown'}]"))
            else:
                output_parts.append(LLMInputText(type="input_text", text=str(part)))
        return FunctionToolCallResult(call_id=call_id, output=output_parts, artifact=artifact)

    # --- Step 15: Post-process results (string or stringified) ---
    # Apply minify + the ``max_result_tokens`` cap to non-str results too,
    # after stringifying. A cost cap the developer configured MUST hold
    # regardless of the tool's return type; the previous bare ``str(result)``
    # let a large non-str result bypass the budget entirely.
    result = result if isinstance(result, str) else str(result)
    result = minify_json(result)
    result = apply_result_limits(result, tool, model)

    return FunctionToolCallResult(call_id=call_id, output=result, artifact=artifact)


# ---------------------------------------------------------------------------
# Non-streaming wrapper
# ---------------------------------------------------------------------------


async def execute_tool_calls(
    agent: Agent,
    tool_calls: list[LLMResponseFunctionToolCall],
    ctx_wrapper: RunContext[TContext],
    hooks: RunHooks[TContext],
    config: RunConfig,
    tool_failure_counts: FunctionToolFailureCounts | None = None,
    model: str = DEFAULT_MODEL,
    messages: list[LLMInputContentItem] | None = None,
    turn: int = 0,
    parallel: bool = False,
) -> tuple[list[FunctionToolCallResult], DeferredToolRequests | None]:
    """Execute tool calls with guardrails and HITL support.

    Iterates the tool calls and delegates each to
    ``_execute_single_tool_call``, which runs the full per-call
    pipeline (tenant gate, permission check, retry budget, input
    guardrails, HITL deferral, execution, and output guardrails).

    Args:
        agent: The current agent.
        tool_calls: List of tool calls from the LLM.
        ctx_wrapper: The run context wrapper.
        hooks: Lifecycle hooks.
        config: Run configuration.
        tool_failure_counts: Per-tool failure counter for retry budget enforcement.
        model: litellm model identifier for token counting (used by
            ``apply_result_limits``).
        messages: Current conversation history (used for
            ``ExecutionAwareToolContext``).
        turn: Current agent loop turn number.
        parallel: Whether to execute tool calls concurrently via
            ``asyncio.gather()``.  Defaults to ``False`` (sequential).

    Returns:
        Tuple of (tool_results, deferred_requests).
        If deferred_requests is not None, execution was interrupted for approval.
    """
    from troopai.adk.tools.deferred_tool import (
        DeferredToolCall,
        DeferredToolRequests,
    )
    from troopai.adk.types.output import FunctionToolCallResult

    results: list[FunctionToolCallResult] = []
    deferred_approvals: list[DeferredToolCall] = []

    # Validate max_parallel_tools whenever the executor runs — including
    # single-call batches and the sequential path — so a misconfigured cap
    # never passes silently.
    cap = config.max_parallel_tools
    if cap is not None and cap <= 0:
        raise ValueError(f"RunConfig.max_parallel_tools must be a positive integer, got {cap}")

    _check_tool_batch_limits(config, ctx_wrapper, len(tool_calls))

    if parallel and len(tool_calls) > 1:
        ctx_wrapper.usage.tool_calls += len(tool_calls)
        # Optional semaphore bounding concurrency. None means unbounded
        # gather (all tools in the batch start simultaneously).
        semaphore: asyncio.Semaphore | None = asyncio.Semaphore(cap) if cap is not None else None

        async def _run_with_cap(tc: LLMResponseFunctionToolCall) -> FunctionToolCallResult | DeferredToolCall:
            call = _execute_single_tool_call(
                agent=agent,
                tool_call=tc,
                ctx_wrapper=ctx_wrapper,
                hooks=hooks,
                config=config,
                tool_failure_counts=tool_failure_counts,
                model=model,
                messages=messages,
                turn=turn,
            )
            if semaphore is not None:
                async with semaphore:
                    return await call
            return await call

        # Use return_exceptions=True so that a single failing coroutine
        # does not leave sibling tasks running as untracked background
        # work (ghost hook calls, phantom audit entries).  After
        # collecting all outcomes we re-raise the first exception found.
        raw_outcomes = await asyncio.gather(
            *(_run_with_cap(tc) for tc in tool_calls),
            return_exceptions=True,
        )
        first_exc: BaseException | None = None
        for outcome in raw_outcomes:
            if isinstance(outcome, BaseException):
                if first_exc is None:
                    first_exc = outcome
            elif isinstance(outcome, FunctionToolCallResult):
                results.append(outcome)
            elif isinstance(outcome, DeferredToolCall):
                deferred_approvals.append(outcome)
        if first_exc is not None:
            raise first_exc
    else:
        for tool_call in tool_calls:
            ctx_wrapper.usage.tool_calls += 1
            outcome = await _execute_single_tool_call(
                agent=agent,
                tool_call=tool_call,
                ctx_wrapper=ctx_wrapper,
                hooks=hooks,
                config=config,
                tool_failure_counts=tool_failure_counts,
                model=model,
                messages=messages,
                turn=turn,
            )

            if isinstance(outcome, FunctionToolCallResult):
                results.append(outcome)
            elif isinstance(outcome, DeferredToolCall):
                deferred_approvals.append(outcome)

    # Build deferred requests if any
    deferred = None
    if len(deferred_approvals) > 0:
        deferred = DeferredToolRequests(
            approvals=deferred_approvals,
        )

    return results, deferred


# ---------------------------------------------------------------------------
# Streaming wrapper
# ---------------------------------------------------------------------------


async def execute_tool_calls_streamed(
    agent: Agent,
    tool_calls: list[LLMResponseFunctionToolCall],
    ctx_wrapper: RunContext[TContext],
    hooks: RunHooks[TContext],
    config: RunConfig,
    result: RunResultStreaming,
    tool_failure_counts: FunctionToolFailureCounts | None = None,
    model: str = DEFAULT_MODEL,
    messages: list[LLMInputContentItem] | None = None,
    turn: int = 0,
) -> tuple[list[FunctionToolCallResult], DeferredToolRequests | None]:
    """Execute tool calls with streaming events and HITL support.

    Uses the same ``_execute_single_tool_call()`` as the non-streaming
    path, adding stream events at TOOL_CALLED, TOOL_OUTPUT, and
    TOOL_APPROVAL_REQUESTED.  Full feature parity: input guardrails,
    output guardrails, and ``ExecutionAwareToolContext`` all work.

    Args:
        agent: The current agent.
        tool_calls: List of tool calls from the LLM.
        ctx_wrapper: The run context wrapper.
        hooks: Lifecycle hooks.
        config: Run configuration.
        result: The streaming result to emit events to.
        tool_failure_counts: Per-tool failure counter for retry budget enforcement.
        model: litellm model identifier for token counting.
        messages: Current conversation history (for ``ExecutionAwareToolContext``).
        turn: Current agent loop turn number.

    Returns:
        Tuple of (tool_results, deferred_requests).
        If deferred_requests is not None, execution was interrupted for approval.
    """
    from troopai.adk.run.stream import CancelMode, RunItemStreamEvent, RunItemType
    from troopai.adk.tools.deferred_tool import (
        DeferredToolCall,
        DeferredToolRequests,
    )
    from troopai.adk.types.output import FunctionToolCallResult

    results: list[FunctionToolCallResult] = []
    deferred_approvals: list[DeferredToolCall] = []

    _check_tool_batch_limits(config, ctx_wrapper, len(tool_calls))

    # Publish the streaming result as the active stream sink so the
    # drain helper inside the middleware terminal can forward
    # streaming-tool partial events to consumers without threading a
    # parameter through every call. Reset on exit so non-streaming
    # paths that follow this scope see the default ``None`` sink.
    sink_token = _TOOL_STREAM_SINK.set(result)
    try:
        for tool_call in tool_calls:
            # Honor an immediate cancel between tool calls: let whatever
            # finished become part of the record, but do not start anything
            # new. `after_turn` is intentionally not checked here — its
            # contract is "finish the current tool batch, then stop".
            if result.cancel_mode == CancelMode.IMMEDIATE:
                logger.info(
                    "execute_tool_calls_streamed: IMMEDIATE cancel observed between tool calls (agent=%s, remaining=%d)",
                    agent.name,
                    len(tool_calls) - len(results) - len(deferred_approvals),
                )
                break

            call_id = tool_call.call_id
            tool_name = tool_call.name
            raw_args = tool_call.arguments or "{}"
            # Mirror the non-streaming path: malformed JSON degrades to
            # an empty dict for the emitted event instead of crashing
            # the streamed run.
            try:
                tool_input = json.loads(raw_args)
            except json.JSONDecodeError:
                logger.warning(
                    "Tool %r produced malformed JSON arguments %r — treating as empty",
                    tool_name,
                    raw_args,
                )
                tool_input = {}

            # Emit tool called event
            await result.put_event(
                RunItemStreamEvent(
                    name=RunItemType.TOOL_CALLED,
                    item={"name": tool_name, "input": tool_input},
                )
            )

            ctx_wrapper.usage.tool_calls += 1
            outcome = await _execute_single_tool_call(
                agent=agent,
                tool_call=tool_call,
                ctx_wrapper=ctx_wrapper,
                hooks=hooks,
                config=config,
                tool_failure_counts=tool_failure_counts,
                model=model,
                messages=messages,
                turn=turn,
            )

            if isinstance(outcome, FunctionToolCallResult):
                results.append(outcome)

                # Emit tool output event
                await result.put_event(
                    RunItemStreamEvent(
                        name=RunItemType.TOOL_OUTPUT,
                        item={"name": tool_name, "output": outcome.output},
                    )
                )
            elif isinstance(outcome, DeferredToolCall):
                deferred_approvals.append(outcome)

                # Emit approval requested event
                event_item: dict[str, Any] = {
                    "name": tool_name,
                    "input": tool_input,
                    "tool_call_id": call_id,
                }
                # Include nested agent info if present
                if outcome.metadata is not None and outcome.metadata.nested_agent:
                    event_item["nested_agent"] = outcome.metadata.nested_agent_name

                await result.put_event(
                    RunItemStreamEvent(
                        name=RunItemType.TOOL_APPROVAL_REQUESTED,
                        item=event_item,
                    )
                )
    finally:
        _TOOL_STREAM_SINK.reset(sink_token)

    # Build deferred requests if any
    deferred = None
    if len(deferred_approvals) > 0:
        deferred = DeferredToolRequests(
            approvals=deferred_approvals,
        )

    return results, deferred


# ---------------------------------------------------------------------------
# Tool use behavior check
# ---------------------------------------------------------------------------


async def check_tool_use_behavior(
    agent: Agent,
    tool_results: list[FunctionToolCallResult],
    tool_calls: list[LLMResponseFunctionToolCall],
    ctx_wrapper: RunContext,
) -> Any | None:
    """Check if tool results should become final output based on agent.tool_use_behavior.

    Returns the final output value if tools should short-circuit, or None
    to continue the normal loop (send results back to LLM).
    """
    from troopai.adk.types.tools.tool_use_behavior import (
        FunctionToolResult,
        StopAtTools,
    )

    behavior = agent.tool_use_behavior

    # Check per-tool return_direct before agent-level behavior
    from troopai.adk.run.llm_calls import resolve_function_tool

    for tc, tr in zip(tool_calls, tool_results, strict=False):
        tool_obj = await resolve_function_tool(agent, tc.name, ctx_wrapper)
        if tool_obj is not None and tool_obj.return_direct:
            return tr.output

    if behavior == "run_llm_again":
        return None

    if behavior == "stop_on_first_tool":
        if len(tool_results) > 0:
            return tool_results[0].output
        return None

    if isinstance(behavior, StopAtTools):
        for tc, tr in zip(tool_calls, tool_results, strict=False):
            if tc.name in behavior.stop_at_tool_names:
                return tr.output
        return None

    if callable(behavior):
        enriched = [
            FunctionToolResult(
                name=tc.name,
                call_id=tc.call_id,
                output=tr.output,
            )
            for tc, tr in zip(tool_calls, tool_results, strict=False)
        ]
        result = behavior(ctx_wrapper, enriched)
        if isawaitable(result):
            result = await result
        return result.final_output if result.is_final_output else None

    # ``ToolUseBehavior`` is a closed union (two Literals, StopAtTools,
    # callable). ``assert_never`` is the typed exhaustiveness check: if the
    # union gains a variant that isn't handled above, mypy reports a type
    # error here, and at runtime the call raises ``AssertionError``.
    assert_never(behavior)
