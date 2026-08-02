"""Handoff execution — deterministic and LLM-orchestrated strategies.

Handles handoff preparation (strategy application, history collapse,
summarization, budget enforcement) extracted from ``Runner``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from troopai.adk.handoffs.handoff_collapse_mode import HandoffCollapseMode
from troopai.adk.handoffs.handoff_strategy import HandoffStrategy
from troopai.adk.run.config import DEFAULT_MODEL
from troopai.adk.tools.token_budget import TokenBudget
from troopai.adk.tracing import handoff_span
from troopai.adk.types.items import RunItem
from troopai.adk.types.items.items import ItemHelpers

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.handoffs import Handoff, HandoffInputData, HandoffTarget
    from troopai.adk.hooks.hooks import RunHooks
    from troopai.adk.llms.llm import LLM
    from troopai.adk.run.context import RunContext, TContext
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.input.llm_input_easy_message import LLMInputEasyMessage
    from troopai.adk.types.responses.llm_response import LLMResponseFunctionToolCall

logger = logging.getLogger(__name__)


async def execute_deterministic_handoff(
    from_agent: Agent,
    target: HandoffTarget,
    intent: Any,
    context_msgs: tuple[RunItem, ...],
    output_msgs: tuple[RunItem, ...],
    context: RunContext[TContext],
    ctx_wrapper: RunContext[TContext],
    hooks: RunHooks[TContext],
    tracing_enabled: bool = False,
    metrics_enabled: bool = False,
) -> tuple[Agent, HandoffInputData]:
    """Execute a deterministic handoff.

    Delegates to target.invoke() for execution logic,
    handles lifecycle hooks at runner level.

    Args:
        from_agent: The agent handing off.
        target: The HandoffTarget with target agent and config.
        intent: The Intent that triggered this handoff.
        context_msgs: Messages before the current agent's turn.
        output_msgs: Messages generated during the current agent's turn.
        context: The run context.
        ctx_wrapper: The context wrapper.
        hooks: Lifecycle hooks.
        tracing_enabled: Whether span tracing is enabled.
        metrics_enabled: Whether metric instruments are enabled.

    Returns:
        Tuple of (target agent, filtered HandoffInputData).
    """
    target_agent_name = getattr(
        getattr(target, "target", None),
        "name",
        None,
    ) or getattr(target, "agent_name", None)
    with handoff_span(
        from_agent=from_agent.name,
        to_agent=target_agent_name,
        disabled=not (tracing_enabled or metrics_enabled),
    ):
        # Execute handoff via target's invoke method
        new_agent, handoff_data = await target.invoke(
            intent=intent,
            context=context_msgs,
            output=output_msgs,
            run_context=context,
        )

        # Call lifecycle hooks (runner responsibility)
        await hooks.on_handoff(ctx_wrapper, from_agent, new_agent)
        if new_agent.hooks is not None:
            await new_agent.hooks.on_handoff(ctx_wrapper, new_agent, from_agent)

    return new_agent, handoff_data


async def prepare_handoff_input(
    target: HandoffTarget,
    handoff_data: HandoffInputData,
    llm: LLM | None = None,
    model: str = DEFAULT_MODEL,
    context: RunContext[Any] | None = None,
) -> list[LLMInputContentItem]:
    """Prepare input for the new agent based on filtered HandoffInputData.

    The input_filter has already been applied to handoff_data by target.invoke().
    This method applies the HandoffConfig.strategy on top of the filtered data.

    Uses ``data.forwarded`` if set by an input_filter, otherwise falls back
    to ``data.messages`` (context + output combined).

    Items are converted to Layer 1 params via ``run_items_to_params()``
    before returning.

    Args:
        target: The HandoffTarget with config.
        handoff_data: The filtered HandoffInputData from target.invoke().
        llm: The ``LLM`` instance for the summarization call (required
            only when ``strategy=SUMMARY``; ``None`` otherwise).
            Typically resolved via :func:`resolve_compaction_llm`.
        model: Model identifier for token counting in the summarization
            call (used only when ``strategy=SUMMARY``).

    Returns:
        Messages list prepared for the new agent.
    """
    config = target.config

    # Use forwarded if set by filter, otherwise full messages
    source = handoff_data.forwarded if handoff_data.forwarded is not None else handoff_data.messages
    items = list(source)

    match config.strategy:
        case HandoffStrategy.FULL:
            result_items = items
        case HandoffStrategy.LAST_N:
            # HandoffConfig.__post_init__ enforces window is a positive int
            # when strategy=LAST_N, so the fallback is unreachable.
            if config.window is None:
                raise ValueError("LAST_N requires window; HandoffConfig.__post_init__ should have rejected this")
            window = config.window
            result_items = items[-window:] if len(items) > window else items
        case HandoffStrategy.INTENT_ONLY:
            intent_msg: LLMInputEasyMessage = {"role": "user", "content": str(handoff_data.intent)}
            return [intent_msg]
        case HandoffStrategy.SUMMARY:
            if llm is None:
                raise ValueError(
                    "HandoffStrategy.SUMMARY requires an LLM instance; "
                    "pass llm=resolve_compaction_llm(agent, config) to "
                    "prepare_handoff_input(...).",
                )
            # Summary needs Layer 1 params for the compactor
            messages = ItemHelpers.run_items_to_params(items)
            result = await _summarize_for_handoff(messages, llm, model, context=context)
            if isinstance(config.collapse, HandoffCollapseMode) and config.collapse != HandoffCollapseMode.OFF:
                result = _collapse_history(result, mode=config.collapse)
            return _rewrite_trailing_assistant(result)

    # Convert items to Layer 1 params
    result = ItemHelpers.run_items_to_params(result_items)

    if isinstance(config.collapse, HandoffCollapseMode) and config.collapse != HandoffCollapseMode.OFF:
        result = _collapse_history(result, mode=config.collapse)

    return _rewrite_trailing_assistant(result)


async def execute_llm_handoff(
    from_agent: Agent,
    target: Handoff,
    tool_call: LLMResponseFunctionToolCall,
    context_msgs: tuple[RunItem, ...],
    output_msgs: tuple[RunItem, ...],
    context: RunContext[TContext],
    ctx_wrapper: RunContext[TContext],
    hooks: RunHooks[TContext],
    tracing_enabled: bool = False,
    metrics_enabled: bool = False,
) -> tuple[Agent, HandoffInputData]:
    """Execute an LLM-orchestrated handoff.

    Delegates to target.invoke() for execution logic,
    handles lifecycle hooks at runner level.

    Args:
        from_agent: The agent handing off.
        target: The Handoff with target agent and config.
        tool_call: The tool call from the LLM response.
        context_msgs: Messages before the current agent's turn.
        output_msgs: Messages generated during the current agent's turn.
        context: The run context.
        ctx_wrapper: The context wrapper.
        hooks: Lifecycle hooks.
        tracing_enabled: Whether span tracing is enabled.
        metrics_enabled: Whether metric instruments are enabled.

    Returns:
        Tuple of (target agent, filtered HandoffInputData).
    """
    # Parse tool call arguments
    tool_args = tool_call.arguments or "{}"

    target_agent_name = getattr(
        getattr(target, "target", None),
        "name",
        None,
    ) or getattr(target, "agent_name", None)
    with handoff_span(
        from_agent=from_agent.name,
        to_agent=target_agent_name,
        disabled=not (tracing_enabled or metrics_enabled),
    ):
        # Invoke target (builds HandoffInputData, applies filter, calls callback)
        new_agent, handoff_data = await target.invoke(
            tool_args=tool_args,
            context=context_msgs,
            output=output_msgs,
            run_context=context,
        )

        # Call lifecycle hooks (runner responsibility)
        await hooks.on_handoff(ctx_wrapper, from_agent, new_agent)
        if new_agent.hooks is not None:
            await new_agent.hooks.on_handoff(ctx_wrapper, new_agent, from_agent)

    return new_agent, handoff_data


async def prepare_handoff_input_from_data(
    target: Handoff,
    handoff_data: HandoffInputData,
    llm: LLM | None = None,
    model: str = DEFAULT_MODEL,
    context: RunContext[Any] | None = None,
) -> list[LLMInputContentItem]:
    """Prepare input for the new agent based on LLM handoff data.

    Uses the same HandoffConfig.strategy logic as prepare_handoff_input().

    Uses ``data.forwarded`` if set by an input_filter, otherwise falls back
    to ``data.messages`` (context + output combined).

    Items are converted to Layer 1 params via ``run_items_to_params()``
    before returning.

    Args:
        target: The Handoff with config.
        handoff_data: The filtered HandoffInputData from target.invoke().
        llm: The ``LLM`` instance for the summarization call (required
            only when ``strategy=SUMMARY``; ``None`` otherwise).
            Typically resolved via :func:`resolve_compaction_llm`.
        model: Model identifier for token counting in the summarization
            call (used only when ``strategy=SUMMARY``).

    Returns:
        Messages list prepared for the new agent.
    """
    config = target.config

    # Use forwarded if set by filter, otherwise full messages
    source = handoff_data.forwarded if handoff_data.forwarded is not None else handoff_data.messages
    items = list(source)

    match config.strategy:
        case HandoffStrategy.FULL:
            result_items = items
        case HandoffStrategy.LAST_N:
            # HandoffConfig.__post_init__ enforces window is a positive int
            # when strategy=LAST_N, so the fallback is unreachable.
            if config.window is None:
                raise ValueError("LAST_N requires window; HandoffConfig.__post_init__ should have rejected this")
            window = config.window
            result_items = items[-window:] if len(items) > window else items
        case HandoffStrategy.INTENT_ONLY:
            intent_msg_2: LLMInputEasyMessage = {"role": "user", "content": str(handoff_data.intent)}
            return [intent_msg_2]
        case HandoffStrategy.SUMMARY:
            if llm is None:
                raise ValueError(
                    "HandoffStrategy.SUMMARY requires an LLM instance; "
                    "pass llm=resolve_compaction_llm(agent, config) to "
                    "prepare_handoff_input_from_data(...).",
                )
            messages = ItemHelpers.run_items_to_params(items)
            result = await _summarize_for_handoff(messages, llm, model, context=context)
            if isinstance(config.collapse, HandoffCollapseMode) and config.collapse != HandoffCollapseMode.OFF:
                result = _collapse_history(result, mode=config.collapse)
            return _rewrite_trailing_assistant(result)

    result = ItemHelpers.run_items_to_params(result_items)

    if isinstance(config.collapse, HandoffCollapseMode) and config.collapse != HandoffCollapseMode.OFF:
        result = _collapse_history(result, mode=config.collapse)

    return _rewrite_trailing_assistant(result)


def _rewrite_trailing_assistant(
    messages: list[LLMInputContentItem],
) -> list[LLMInputContentItem]:
    """Finalize a forwarded history so it is safe for the target agent.

    Two repairs, in order:

    1. Drop unpaired tool-call / tool-result params via
       :func:`_drop_orphan_tool_calls` — the source agent's
       ``transfer_to_<name>`` handoff call enters the forwarded slice
       without its synthetic result, which strict providers (Anthropic)
       reject as a ``tool_use`` with no ``tool_result``.
    2. Rewrite trailing assistant messages to user role. After a handoff
       the last message(s) may be assistant messages from the *source*
       agent (e.g. the triage agent's structured Intent output). From the
       *target* agent's perspective this is input, not its own prior
       output. Some models (e.g. Claude Opus 4.6) reject conversations
       ending with assistant role ("does not support assistant message
       prefill"). Content is preserved; only the role changes.

    Args:
        messages: The prepared message list for the new agent.

    Returns:
        A list with unpaired tool-call params removed and trailing
        assistant messages rewritten to user role.
    """
    messages = _drop_orphan_tool_calls(messages)
    # Walk backwards, converting assistant messages until we hit a non-assistant.
    # Spread-merge a closed TypedDict from a wider union loses the Required-key
    # constraint, so we launder the result through ``Any`` — the role/content
    # invariants are preserved at runtime and re-validated on the next LLM call.
    i = len(messages) - 1
    while i >= 0 and messages[i].get("role") == "assistant":
        rewritten: Any = {**messages[i], "role": "user"}
        messages[i] = rewritten
        logger.debug(
            "Rewrote trailing assistant message at index %d to user role for handoff compatibility",
            i,
        )
        i -= 1
    return messages


def _drop_orphan_tool_calls(
    messages: list[LLMInputContentItem],
) -> list[LLMInputContentItem]:
    """Drop unpaired tool-call / tool-result params from a forwarded list.

    Strict providers (Anthropic) reject any ``tool_use`` not immediately
    followed by its ``tool_result``; an orphaned ``function_call_output``
    is likewise invalid. Two construction sites produce orphans in a
    forwarded handoff history:

    1. **Handoff slice.** A handoff forwards a temporal slice of the
       source agent's history to the target. That slice contains the
       ``transfer_to_<name>`` tool-call param but NOT its matching
       ``function_call_output``: the synthetic "Transferred to ..."
       result is appended to the *source* history after the slice is
       taken (see ``resolve_handoff_step`` — the post-invoke append is
       deliberate so a rejected handoff can emit a rejection result
       instead). The orphan is also semantically meaningless to the
       target — it is the source agent's tool, not the target's.
    2. **Budget truncation.** ``apply_handoff_budget`` evicts oldest
       messages FIFO with no pairing awareness, so a cut between a
       paired ``function_call`` and its ``function_call_output`` leaves
       either side orphaned.

    Drop every ``function_call`` whose ``call_id`` has no matching
    ``function_call_output`` AND every ``function_call_output`` whose
    ``call_id`` has no matching ``function_call``. Properly paired tool
    calls the source agent actually ran (both halves present) are
    preserved. The function is idempotent (a second pass over a cleaned
    list drops nothing) and order-preserving.

    Args:
        messages: The prepared Layer 1 param list for the target agent.

    Returns:
        The list with unpaired tool-call / tool-result params removed
        (order preserved).
    """
    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for m in messages:
        kind = m.get("type")
        cid = m.get("call_id")
        if not (isinstance(cid, str) and len(cid) > 0):
            continue
        if kind == "function_call":
            call_ids.add(cid)
        elif kind == "function_call_output":
            result_ids.add(cid)

    kept: list[LLMInputContentItem] = []
    dropped = 0
    for m in messages:
        kind = m.get("type")
        cid = m.get("call_id")
        # Mirror the pairing-set guard above: an item without a valid
        # non-empty ``call_id`` was never registered in either set, so it
        # cannot be classified as an orphan by id. Keep it — otherwise a
        # legitimately paired exchange whose ids are both empty/missing is
        # deleted wholesale (both halves fail the ``not in`` check against
        # the sets they were skipped from).
        has_valid_cid = isinstance(cid, str) and len(cid) > 0
        if kind == "function_call" and has_valid_cid and cid not in result_ids:
            dropped += 1
            continue
        if kind == "function_call_output" and has_valid_cid and cid not in call_ids:
            dropped += 1
            continue
        kept.append(m)

    if dropped > 0:
        logger.debug(
            "Dropped %d unpaired tool-call/result param(s) from forwarded handoff history",
            dropped,
        )
    return kept


def _content_to_str(content: str | list[Any]) -> str:
    """Flatten a message ``content`` field to readable transcript text.

    Layer 1 message content is either a plain string (``LLMInputEasyMessage``)
    or a list of typed content-part dicts (assistant ``LLMResponseMessageParam``
    text/refusal parts, multimodal ``input_text``/``image``/``audio`` parts).
    Interpolating a list directly into an f-string yields Python repr noise
    (``[{'type': 'output_text', ...}]``), so list content is joined from its
    textual parts instead.

    Args:
        content: A message ``content`` value (string or content-part list).

    Returns:
        The string content unchanged, or the joined ``text`` of every
        textual part in the list. A non-empty list with no textual part
        renders as ``"[non-text content]"``.
    """
    if isinstance(content, str):
        return content
    parts = [
        p["text"]
        for p in content
        if isinstance(p, dict)
        and p.get("type") in ("output_text", "text", "input_text")
        and isinstance(p.get("text"), str)
    ]
    if len(parts) > 0:
        return " ".join(parts)
    return "[non-text content]" if len(content) > 0 else ""


def _collapse_history(
    messages: list[LLMInputContentItem],
    mode: HandoffCollapseMode = HandoffCollapseMode.SYSTEM_MESSAGE,
) -> list[LLMInputContentItem]:
    """Collapse a list of messages into a single user message.

    Produces a compact transcript where each message is formatted as
    ``[role]: content``. Messages without content are skipped.

    Args:
        messages: The conversation messages to collapse.
        mode: Any non-``OFF`` value collapses the history. The collapsed
            transcript is always emitted as a ``user`` message, even for
            ``SYSTEM_MESSAGE``: after a handoff the target agent's own
            system prompt is injected at index 0, and a leading collapsed
            *system* message would be overwritten by that injection —
            silently discarding the entire transferred history. A
            user-role block is preceded by the injected system prompt
            instead, so the history survives as the target's prior
            context. ``OFF`` never reaches this function and is rejected
            explicitly.

    Returns:
        A single-element list with the collapsed transcript as a user
        message, or the original messages if none had content.
    """
    if mode is HandoffCollapseMode.OFF:
        raise ValueError("_collapse_history called with mode=OFF; caller should skip the collapse step.")
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, (str, list)) or len(content) == 0:
            continue
        parts.append(f"[{msg.get('role', 'unknown')}]: {_content_to_str(content)}")
    collapsed_content = "\n".join(parts)
    if len(collapsed_content) == 0:
        return messages
    # Always a user message. A leading collapsed *system* message is
    # overwritten by the target's own system prompt (injected at index 0
    # right after this step), which would discard the transferred history.
    collapsed_msg: LLMInputEasyMessage = {
        "role": "user",
        "content": f"Previous conversation:\n{collapsed_content}",
    }
    return [collapsed_msg]


async def _summarize_for_handoff(
    messages: list[LLMInputContentItem],
    llm: LLM,
    model: str,
    context: RunContext[Any] | None = None,
) -> list[LLMInputContentItem]:
    """Summarize conversation history for handoff via ContextCompactor.

    Uses the context compaction layer to produce a concise summary of the
    conversation, then wraps it as a single user message for the new agent.
    Routes through :meth:`LLM.acomplete` so the summarization tokens land
    in :attr:`RunContext.usage` (when ``context`` is provided) and
    ``Agent.middleware.llms`` sees the call.

    Args:
        messages: The conversation messages to summarize.
        llm: The ``LLM`` instance to invoke for summarization.
        model: Model identifier for token counting (litellm tokenizer
            is name-based).
        context: Optional ``RunContext``. When provided, the
            summarization call's usage is accumulated into
            ``context.usage`` so framework-wide usage limits include it.

    Returns:
        A single-element list with the summary as a user message.
        Falls back to the original messages if summarization fails
        or the conversation is too short.
    """
    from troopai.adk.context.compaction import ContextCompactor
    from troopai.adk.context.context_config import CompactionConfig

    config = CompactionConfig(
        enabled=True,
        preserve_recent_items=0,
    )

    try:
        result = await ContextCompactor.compact(
            messages=messages,
            llm=llm,
            model_name=model,
            config=config,
        )
        if context is not None and result.usage is not None:
            context.usage = context.usage + result.usage
        if result.summary:
            summary_msg: LLMInputEasyMessage = {"role": "user", "content": result.summary}
            return [summary_msg]
    except Exception:
        # Fail open: the caller falls back to original messages so the run
        # does not crash. Log at ERROR (with traceback) so auth/quota/rate-
        # limit failures surface in operator alerting rather than being
        # silently buried at WARNING level.
        logger.exception(
            "Handoff summarisation failed; using original messages",
        )

    # Fallback to original messages if summarization fails
    return messages


async def apply_handoff_budget(
    messages: list[LLMInputContentItem],
    handoff: Handoff | HandoffTarget,
    model: str,
) -> list[LLMInputContentItem]:
    """Apply context budget to messages before handoff via truncation.

    When ``handoff.config.budget`` is set and the prepared messages
    exceed it, drops oldest non-system messages until under budget.
    The system message (first message with ``role == "system"``) is
    always preserved. The iteration count is bounded by the initial
    body length (NASA R2 — explicit upper bound on the loop).

    This is **free**: no LLM call is made. Earlier revisions invoked
    :class:`ContextCompactor` (summarisation), which made a hidden
    LLM call that bypassed ``RunContext.usage`` and middleware. The
    truncation path keeps the budget cap useful for input-token
    economy without introducing an unobservable cost surface — the
    developer never opts OUT of an LLM call by setting a bounded
    budget. Developers who want summarisation can set
    ``HandoffConfig.strategy = HandoffStrategy.SUMMARY`` explicitly.

    Accepts either an LLM-orchestrated ``Handoff`` or a code-orchestrated
    ``HandoffTarget``; both expose ``.config.budget`` and the truncation
    logic only reads that field plus a display name for the log line.

    Args:
        messages: The messages prepared by strategy/filter.
        handoff: The ``Handoff`` or ``HandoffTarget`` whose
            ``config.budget`` is checked.
        model: Model identifier for token counting (litellm tokenizer
            is name-based).

    Returns:
        The original messages if within budget, or a truncated copy
        with oldest non-system messages dropped until under budget.
        FIFO eviction has no tool-call/result pairing awareness, so the
        truncated result is passed through
        :func:`_drop_orphan_tool_calls` to remove any pair split by the
        cut (a ``tool_use`` with no ``tool_result``, or vice versa)
        before it reaches a strict provider.
    """
    from troopai.adk.context.token_counter import TokenCounter

    budget_obj = handoff.config.budget
    if budget_obj is None:
        return messages
    # ``HandoffConfig.__post_init__`` coerces bare-int budgets to
    # ``TokenBudget``, so the runtime type here is always
    # ``TokenBudget | None``. The field annotation still accepts
    # ``int`` for ergonomic construction, so widen via isinstance
    # and surface any other shape as a framework-invariant violation
    # rather than silently coercing (which previously masked
    # post-construction mutation via ``object.__setattr__``).
    if not isinstance(budget_obj, TokenBudget):
        raise TypeError(
            f"HandoffConfig.budget reached apply_handoff_budget with type "
            f"{type(budget_obj).__name__}; expected TokenBudget. The "
            "runtime invariant established in HandoffConfig.__post_init__ "
            "has been violated."
        )
    budget: int = budget_obj.max_tokens
    drop_policy: str = budget_obj.drop_policy

    current_tokens: int = TokenCounter.count_messages(messages, model)
    if current_tokens <= budget:
        return messages

    # ``Handoff`` exposes ``get_name()`` (the LLM-facing tool name);
    # ``HandoffTarget`` has no name accessor but both expose
    # ``.target`` (the destination ``Agent``). Resolve a display name
    # for the log line via isinstance narrowing rather than
    # ``getattr`` reflection. The local import is required because
    # ``Handoff`` is only TYPE_CHECKING-imported at module top.
    from troopai.adk.handoffs.handoff import Handoff

    name = handoff.get_name() if isinstance(handoff, Handoff) else handoff.target.name

    logger.info(
        "Handoff '%s' budget exceeded: %d tokens > %d budget, truncating",
        name,
        current_tokens,
        budget,
    )

    system_msg: LLMInputContentItem | None = None
    body: list[LLMInputContentItem] = list(messages)
    # ``preserve_system`` peels the leading system message off so it
    # survives the drop loop; ``oldest_first`` leaves it in body so
    # FIFO eviction can drop it too.
    if drop_policy == "preserve_system" and body and body[0].get("role") == "system":
        system_msg = body[0]
        body = body[1:]

    # NASA R2: explicit loop bound. The body can shrink by at most
    # its initial length; if token counting misbehaves and never drops
    # below budget we still exit cleanly.
    #
    # Floor: never drop below the newest message. The loop stops while
    # ``len(body) > 1`` so ``body[-1]`` (the newest forwarded message) is
    # always retained — even if it alone exceeds the budget. An empty
    # forwarded history is a hard provider error (Anthropic rejects a
    # zero-message request), which is strictly worse than a single
    # over-budget message; the non-empty invariant wins over the soft cap.
    max_drops = len(body)
    dropped = 0
    while (
        dropped < max_drops
        and len(body) > 1
        and TokenCounter.count_messages(
            ([system_msg] if system_msg is not None else []) + body,
            model,
        )
        > budget
    ):
        body.pop(0)
        dropped += 1

    if dropped > 0:
        logger.info(
            "Handoff truncated: dropped %d oldest message(s) to fit %d token budget",
            dropped,
            budget,
        )

    # FIFO eviction above has no tool-call/result pairing awareness; a
    # cut between a paired call and its result leaves an orphan strict
    # providers reject. Re-apply the invariant repair post-truncation.
    if system_msg is not None:
        return _drop_orphan_tool_calls([system_msg, *body])
    return _drop_orphan_tool_calls(body)
