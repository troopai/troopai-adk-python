"""Shared-context preparation — builds the per-turn input message list.

Given the live :class:`~troopai.adk.swarms.state.SwarmState` and the
agent about to take the next turn, :func:`prepare_turn_input` returns a
list of Layer 1 :class:`~troopai.adk.types.input.LLMInputContentItem`
suitable for passing straight to the runner loop. Each
:class:`~troopai.adk.swarms.shared_context_strategy.SharedContextStrategy`
controls what the next agent sees:

- ``SCOPED`` (default) — the agent sees only its own per-agent scratch
  plus the explicit ``SwarmHandoff.message`` payload if the last yield
  was a handoff addressed to it. No hidden cross-agent broadcast —
  every cross-agent item is developer-supplied.
- ``LAST_N`` — the last N items from the full shared history. ``N``
  comes from :attr:`SharedContextConfig.window`.
- ``SUMMARIZED`` — full shared history compacted via
  :class:`~troopai.adk.context.ContextCompactor` to fit
  :attr:`SharedContextConfig.budget` tokens. Reuses the same compactor
  that powers run-level context management.
- ``FULL_BROADCAST`` — the entire shared history (AutoGen parity).
  Explicit opt-in only — the default is ``SCOPED`` for cost-control
  reasons (broadcast-all is the single biggest footgun in AutoGen's
  default swarm).

The returned list is Layer 1 only. Conversion to Layer 3 items happens
on the runner's side when assembling ``RunResult.new_items``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from troopai.adk.swarms.config import SharedContextConfig
from troopai.adk.swarms.shared_context_strategy import SharedContextStrategy
from troopai.adk.swarms.yield_signal import SwarmHandoff
from troopai.adk.types.items.items import ItemHelpers

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.llms.llm import LLM
    from troopai.adk.run.context import RunContext
    from troopai.adk.swarms.state import SwarmState
    from troopai.adk.swarms.yield_signal import SwarmYieldSignal
    from troopai.adk.types.input import LLMInputContentItem, LLMInputEasyMessage
    from troopai.adk.types.items.items import RunItem


logger = logging.getLogger(__name__)


async def prepare_turn_input(
    state: SwarmState,
    next_agent: Agent,
    last_yield: SwarmYieldSignal | None,
    config: SharedContextConfig,
    compaction_llm: LLM | None = None,
    compaction_model: str | None = None,
    context: RunContext[Any] | None = None,
) -> list[LLMInputContentItem]:
    """Build the Layer 1 message list the next agent's turn receives.

    Args:
        state: The live swarm state.
        next_agent: The agent about to take the turn.
        last_yield: The last yield signal (``SwarmHandoff`` or
            ``SwarmDone`` or ``None``). Used by ``SCOPED`` to inject
            the handoff message when the yield targets this agent.
        config: Strategy + parameters. Validated by
            :class:`~troopai.adk.swarms.config.SharedContextConfig.__post_init__`.

    Returns:
        Ordered list of Layer 1 input items for the runner.

    Raises:
        ValueError: If ``config.strategy`` is an unknown value — should
            not happen because ``SharedContextConfig`` is validated at
            construction time, but fail loudly rather than silently
            returning an empty list.
    """
    strategy = config.strategy

    if strategy == SharedContextStrategy.SCOPED:
        return _prepare_scoped(
            state,
            next_agent,
            last_yield,
            max_handoff_message_chars=config.max_handoff_message_chars,
        )

    # Cross-agent strategies build the turn from ``shared_history``, which only
    # ever holds the items each member *produced* — never the run's opening
    # prompt. Prepend the recorded prompt so a turn-2+ agent still sees the
    # question it is meant to answer.
    initial_params = list(ItemHelpers.run_items_to_params(state.initial_input_items))

    if strategy == SharedContextStrategy.LAST_N:
        if config.window is None:
            raise ValueError("SharedContextStrategy.LAST_N requires SharedContextConfig.window to be set.")
        return _prepend_initial_input(_prepare_last_n(state, window=config.window), initial_params)

    if strategy == SharedContextStrategy.SUMMARIZED:
        if config.budget is None:
            raise ValueError("SharedContextStrategy.SUMMARIZED requires SharedContextConfig.budget to be set.")
        if compaction_llm is None or compaction_model is None:
            raise ValueError(
                "SharedContextStrategy.SUMMARIZED requires compaction_llm "
                "and compaction_model arguments to prepare_turn_input(); "
                "the runner threads these via resolve_compaction_llm(...) "
                "and resolve_model_name(...).",
            )
        summarized = await _prepare_summarized(
            state,
            budget=config.budget,
            llm=compaction_llm,
            model=compaction_model,
            context=context,
        )
        return _prepend_initial_input(summarized, initial_params)

    if strategy == SharedContextStrategy.FULL_BROADCAST:
        return _prepend_initial_input(_prepare_full_broadcast(state), initial_params)

    raise ValueError(f"Unknown SharedContextStrategy: {strategy!r}. Add a branch to prepare_turn_input().")


def _prepend_initial_input(
    body: list[LLMInputContentItem],
    initial: list[LLMInputContentItem],
) -> list[LLMInputContentItem]:
    """Prepend the recorded opening prompt ahead of a broadcast body.

    Inserts ``initial`` after a leading ``system`` message when the body
    carries one (so system stays first), else at the very front. A no-op
    when ``initial`` is empty (the common case before turn 1 records it).
    """
    if len(initial) == 0:
        return body
    if len(body) > 0 and body[0].get("role") == "system":
        return [body[0], *initial, *body[1:]]
    return [*initial, *body]


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------


def _prepare_scoped(
    state: SwarmState,
    next_agent: Agent,
    last_yield: SwarmYieldSignal | None,
    *,
    max_handoff_message_chars: int | None,
) -> list[LLMInputContentItem]:
    """SCOPED — agent sees only its own scratch + handoff message.

    The default strategy. Gives each agent a clean view: its prior
    turns (if any) plus the explicit handoff payload. No cross-agent
    leakage. This is the production-safe default because:

    - Context stays small (cost).
    - No surprising context injection — every item the agent sees is
      developer-supplied or its own prior output.
    - Agents can be reasoned about in isolation.

    Behaviour:

    1. Start with the per-agent scratch (empty on first turn for this
       agent).
    2. If the last yield was a ``SwarmHandoff`` targeting this agent,
       append its ``message`` as a user-role message. When
       ``max_handoff_message_chars`` is set and the message exceeds
       that length, the message is truncated and a warning logged.
    3. Convert to Layer 1 params.
    """
    scratch = state.per_agent_scratch.get(next_agent.name, [])
    params: list[LLMInputContentItem] = list(ItemHelpers.run_items_to_params(scratch))

    if isinstance(last_yield, SwarmHandoff) and last_yield.target == next_agent.name and len(last_yield.message) > 0:
        message = last_yield.message
        if max_handoff_message_chars is not None and len(message) > max_handoff_message_chars:
            logger.warning(
                "SwarmHandoff.message truncated: %d chars > max_handoff_message_chars=%d "
                "(source=%s, target=%s). Bypasses FunctionTool.max_result_tokens and "
                "HandoffConfig.budget because it injects before the next turn's LLM call.",
                len(message),
                max_handoff_message_chars,
                state.current_agent_name,
                last_yield.target,
            )
            message = message[:max_handoff_message_chars]
        handoff_msg: LLMInputEasyMessage = {
            "role": "user",
            "content": message,
        }
        # Persist the delivered handoff message into the target's scratch so a
        # later revisit still carries the question that prompted this agent's
        # answers — scratch otherwise accumulates only the agent's own outputs.
        # Idempotent: a re-prepared turn (resume) whose scratch already ends
        # with this exact message neither re-delivers nor re-persists it (both
        # ``params`` and scratch already carry it via the scratch read above).
        if not _scratch_tail_matches(scratch, message):
            params.append(handoff_msg)
            persisted = state.per_agent_scratch.setdefault(next_agent.name, [])
            persisted.extend(ItemHelpers.messages_to_run_items([handoff_msg]))

    return params


def _scratch_tail_matches(scratch: list[RunItem], message: str) -> bool:
    """True when the last scratch item is a user message equal to ``message``.

    Guards the SCOPED handoff-persist against double-delivery when a parked
    turn is re-prepared on resume with the same ``last_yield``.
    """
    if len(scratch) == 0:
        return False
    tail = scratch[-1].to_param()
    return isinstance(tail, dict) and tail.get("role") == "user" and tail.get("content") == message


def _prepare_last_n(
    state: SwarmState,
    *,
    window: int,
) -> list[LLMInputContentItem]:
    """LAST_N — bounded window into the full shared history.

    Keeps the last ``window`` items from the shared history. Cheap,
    deterministic, no summarization call. Good middle ground when
    ``SCOPED`` loses too much context but ``FULL_BROADCAST`` blows the
    token budget.
    """
    history = state.shared_history
    tail = history[-window:] if len(history) > window else list(history)
    return list(ItemHelpers.run_items_to_params(tail))


async def _prepare_summarized(
    state: SwarmState,
    *,
    budget: int,
    llm: LLM,
    model: str,
    context: RunContext[Any] | None = None,
) -> list[LLMInputContentItem]:
    """SUMMARIZED — full history compacted to fit ``budget`` tokens.

    Reuses :class:`~troopai.adk.context.compaction.ContextCompactor` so
    summarization behaves identically to run-level context compaction.
    When the current history is already under ``budget`` the compactor
    returns the preserved messages unchanged (no extra LLM call). The
    summarization call routes through :meth:`LLM.acomplete`, so its
    usage lands on :attr:`RunContext.usage` and middleware sees it.

    The compactor is imported lazily to keep the swarms module
    importable in environments that don't install the
    context-management dependencies.
    """
    from troopai.adk.context.compaction import ContextCompactor
    from troopai.adk.context.context_config import CompactionConfig
    from troopai.adk.context.token_counter import TokenCounter

    history_params = list(ItemHelpers.run_items_to_params(state.shared_history))

    # Gate on the budget before summarizing. ``ContextCompactor.compact`` does
    # not read ``trigger_tokens`` — it summarizes whenever the body exceeds
    # ``preserve_recent_items`` — so passing ``trigger_tokens=budget`` alone
    # fires an LLM summarization call every turn. Skip the call while the
    # history still fits the budget (the cost-conservative behavior the budget
    # is meant to provide).
    if TokenCounter.count_messages(history_params, model) <= budget:
        return history_params

    compaction_config = CompactionConfig(trigger_tokens=budget)
    try:
        result = await ContextCompactor.compact(
            history_params,
            llm=llm,
            model_name=model,
            config=compaction_config,
        )
    except Exception:
        logger.exception(
            "_prepare_summarized: compaction LLM call failed — "
            "the configured token budget of %d is NOT enforced; "
            "falling back to LAST_N truncation to stay within budget.",
            budget,
        )
        # LAST_N truncation: approximate the budget in items using a
        # conservative 200-token-per-item estimate so the fallback stays
        # well within the budget rather than silently sending over-budget
        # uncompacted history (which would violate the cost-conservative
        # defaults invariant).
        _approx_items = max(1, budget // 200)
        return _prepare_last_n(state, window=_approx_items)

    # Accumulate compaction usage into RunContext.usage so the
    # framework ledger is complete (see CompactionResult docstring).
    if context is not None and result.usage is not None:
        context.usage = context.usage + result.usage

    if result.items_compacted == 0:
        return history_params

    preserve = compaction_config.preserve_recent_items
    system_msg: LLMInputContentItem | None = None
    body: list[LLMInputContentItem] = list(history_params)
    if len(body) > 0 and body[0].get("role") == "system":
        system_msg = body[0]
        body = body[1:]
    preserved = body[-preserve:] if 0 < preserve < len(body) else ([] if preserve == 0 else body)

    return ContextCompactor.build_compacted_messages(
        result.summary,
        preserved,
        system_msg,
    )


def _prepare_full_broadcast(
    state: SwarmState,
) -> list[LLMInputContentItem]:
    """FULL_BROADCAST — every item, every agent (AutoGen parity).

    Explicit opt-in only. This is the default in AutoGen's ``Swarm``
    and is the single biggest cost footgun in that framework — running
    a swarm of five agents for ten turns on broadcast-all can blow
    through a provider's context window in one run. Use only when you
    *know* the swarm is small and the turns are few.
    """
    return list(ItemHelpers.run_items_to_params(state.shared_history))
