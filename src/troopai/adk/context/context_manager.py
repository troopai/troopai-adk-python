"""Context manager orchestrating all context management strategies.

The :class:`ContextManager` is the single entry-point called by the
Runner before each LLM call.  It applies context editing first
(cheapest), then checks whether compaction is needed, and returns the
managed message list.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .compaction import ContextCompactor
from .context_config import CompactionConfig, ContextManagementConfig, TokenUsage
from .context_editing import ContextEditor
from .token_counter import TokenCounter

if TYPE_CHECKING:
    from troopai.adk.budgets import TenantBudget
    from troopai.adk.llms.llm import LLM
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext
    from troopai.adk.types.input import LLMInputContentItem

logger = logging.getLogger(__name__)


def effective_compaction_config(
    config: CompactionConfig,
    budget: TenantBudget | None,
    run_cost: float,
    threshold: float,
) -> CompactionConfig:
    """Return a budget-pressure-adjusted compaction config (or the original).

    No-op (returns ``config`` unchanged) unless ``config.cost_aware`` is set,
    a per-run dollar cap exists, and utilization (run_cost / cap) >=
    ``threshold``. Under pressure: halve ``trigger_tokens`` (floored at 1)
    and drop ``preserve_recent_items`` to at most 1, so subsequent turns send
    fewer input tokens as the run approaches its budget.

    Args:
        config: The compaction configuration to evaluate.
        budget: Optional per-run budget cap; if ``None`` or
            ``dollars_per_run`` is unset, no pressure adjustment occurs.
        run_cost: Accumulated cost (USD) for the current run.
        threshold: Utilization fraction (0.0–1.0) above which pressure
            adjustment is applied.

    Returns:
        The original ``config`` when no pressure applies, or a modified
        copy with tightened compaction parameters under budget pressure.
    """
    if not config.cost_aware or budget is None or budget.dollars_per_run is None:
        return config
    utilization = run_cost / budget.dollars_per_run
    if utilization < threshold:
        return config
    return config.model_copy(
        update={
            "trigger_tokens": max(1, config.trigger_tokens // 2),
            "preserve_recent_items": min(config.preserve_recent_items, 1),
        }
    )


class ContextManager:
    """Orchestrates context management strategies for a conversation.

    Instantiate once per conversation (or per ``Runner.run`` call) and
    call :meth:`prepare_messages` before every LLM invocation.
    """

    _PRESSURE_MARKER = "_context_pressure_feedback"
    """Marker key on injected pressure feedback messages."""

    def __init__(self, config: ContextManagementConfig) -> None:
        """Initialize the ContextManager.

        Args:
            config: The :class:`ContextManagementConfig` controlling
                compaction, editing, and token budgets.
        """
        self.config = config
        self._compaction_count: int = 0
        self._total_tokens_compacted: int = 0
        self._force_tool: str | None = None
        # Effective budget = max_context_tokens - tool_overhead
        self._effective_budget: int = max(1, config.max_context_tokens - config.tool_overhead)

    async def prepare_messages(
        self,
        messages: list[LLMInputContentItem],
        llm: LLM,
        model: str,
        run_config: RunConfig | None = None,
        context: RunContext[Any] | None = None,
    ) -> list[LLMInputContentItem]:
        """Apply context management strategies before an LLM call.

        Order of operations:

        1. **Context editing** — clear old tool results and/or thinking
           blocks (fast, no LLM call).
        2. **Token check** — count tokens in the (possibly edited)
           messages.
        3. **Compaction** — if over the trigger threshold, summarise
           older messages via an LLM call.
        4. **Truncation** — hard enforcement: drop oldest messages if
           still over the effective budget (opt-in).
        5. **Orphaned tool cleanup** — drop tool-result messages whose
           matching tool call was removed by compaction/truncation
           (always runs; Anthropic and Gemini reject orphans).
        6. **Pressure feedback** — inject a developer message so the
           LLM can see its budget status (opt-in).
        7. **Forced tool signal** — when ``forced_tool`` is configured
           and pressure exceeds the warning threshold, set the
           force-tool signal the Runner reads after this call (opt-in).

        Args:
            messages: The current conversation messages (including system
                message).
            llm: The ``LLM`` instance to call when compaction fires
                (typically resolved via :func:`resolve_compaction_llm`
                so ``RunConfig.compaction_llm`` overrides the agent's
                primary LLM).
            model: Model identifier used for token counting (the litellm
                tokenizer interface is name-based even when the LLM call
                itself is provider-agnostic).
            run_config: Optional ``RunConfig`` forwarded to the compactor
                and used to read ``tenant_budget`` for cost-aware pressure
                adjustment.
            context: Optional ``RunContext`` used to accumulate compaction
                token usage into the run's usage ledger and to read the
                current run cost for budget-pressure calculations.

        Returns:
            The managed message list (may be shorter than the input).
        """
        managed = list(messages)

        # --- 1. Context editing -------------------------------------------
        editing = self.config.editing

        if editing.clear_tool_results:
            current_tokens = TokenCounter.count_messages(managed, model)
            if current_tokens >= editing.tool_result_trigger_tokens:
                managed = ContextEditor.clear_tool_results(
                    managed,
                    keep=editing.tool_results_to_keep,
                    exclude_tools=editing.exclude_tools or None,
                )

        if editing.clear_thinking_blocks:
            managed = ContextEditor.clear_thinking_blocks(
                managed,
                keep_turns=editing.thinking_turns_to_keep,
            )

        # --- 2. Token budget warning --------------------------------------
        current_tokens = TokenCounter.count_messages(managed, model)
        warning_threshold = int(self._effective_budget * self.config.token_budget_warning_threshold)
        if current_tokens >= warning_threshold:
            logger.warning(
                "Context at %.0f%% capacity (%d / %d tokens)",
                (current_tokens / self._effective_budget) * 100,
                current_tokens,
                self._effective_budget,
            )

        # --- 3. Compaction ------------------------------------------------
        compaction_cfg = effective_compaction_config(
            self.config.compaction,
            run_config.tenant_budget if run_config is not None else None,
            context.cost_usd if context is not None else 0.0,
            self.config.token_budget_warning_threshold,
        )
        if self.should_compact(managed, model, compaction=compaction_cfg):
            result = await ContextCompactor.compact(
                managed,
                llm=llm,
                model_name=model,
                config=compaction_cfg,
                run_config=run_config,
            )

            # Compaction LLM tokens MUST land in RunContext.usage so
            # framework-wide ``LLMUsageLimits`` and cost reporting
            # include them. The LLM ABC routing alone makes the call
            # observable to middleware; usage accumulation is the
            # separate concern that closes the ledger.
            if context is not None and result.usage is not None:
                context.usage = context.usage + result.usage

            if result.items_compacted > 0:
                self._compaction_count += 1
                # Clamp to 0: when the LLM summary is larger than the original
                # (e.g. verbose model output), the delta would go negative and
                # permanently disable the total_token_budget guard in
                # should_compact (the >= check would never be True again).
                self._total_tokens_compacted += max(0, result.original_token_count - result.compacted_token_count)

                # Separate system message from body for rebuild.
                system_msg: LLMInputContentItem | None = None
                body = list(managed)
                if body and body[0].get("role") == "system":
                    system_msg = body[0]
                    body = body[1:]

                preserve = compaction_cfg.preserve_recent_items
                # Mirror ContextCompactor.compact's split: ``preserve == 0``
                # preserves nothing (entire body was summarised), so a ``-0``
                # slice that would keep the whole body is handled explicitly.
                if preserve == 0:
                    preserved = []
                elif 0 < preserve < len(body):
                    preserved = body[-preserve:]
                else:
                    preserved = body

                managed = ContextCompactor.build_compacted_messages(
                    result.summary,
                    preserved,
                    system_msg,
                )

                logger.info(
                    "Compaction #%d: %d -> %d tokens (%d messages summarised)",
                    self._compaction_count,
                    result.original_token_count,
                    result.compacted_token_count,
                    result.items_compacted,
                )

        # --- 4. Truncation (hard enforcement) ----------------------------
        if self.config.truncation:
            current_tokens = TokenCounter.count_messages(managed, model)
            if current_tokens > self._effective_budget:
                managed = self._truncate(managed, model)

        # --- 5. Remove orphaned tool results ──────────────────────────────
        # Both compaction and truncation can break tool_call/tool_result
        # pairing.  Anthropic and Gemini reject orphaned tool results.
        managed = ContextEditor.remove_orphaned_tool_results(managed)

        # --- 6. Pressure feedback (LLM-visible warning) ------------------
        # Capture pre-feedback token count so step 7 is not self-triggering:
        # the injected developer message would inflate the count and fire the
        # forced-tool signal every turn near threshold.
        pre_feedback_tokens = TokenCounter.count_messages(managed, model)
        if self.config.pressure_feedback:
            managed = self._apply_pressure_feedback(managed, model)

        # --- 7. Forced tool signal ----------------------------------------
        self._force_tool = None
        if self.config.forced_tool is not None:
            # Use the pre-feedback count: the injected developer message is
            # not conversation content and must not inflate the pressure check.
            warning_threshold = int(self._effective_budget * self.config.token_budget_warning_threshold)
            if pre_feedback_tokens >= warning_threshold:
                self._force_tool = self.config.forced_tool
                logger.info(
                    "Context pressure: forcing tool '%s' on next LLM call",
                    self._force_tool,
                )

        return managed

    @property
    def force_tool(self) -> str | None:
        """Tool name to force on the next LLM call, or ``None``.

        Set by :meth:`prepare_messages` when context pressure exceeds
        the warning threshold and ``forced_tool`` is configured.
        The Runner reads this after ``prepare_messages()`` and calls
        :meth:`consume_force_tool` to prevent re-triggering on
        subsequent loop iterations within the same turn.
        """
        return self._force_tool

    def consume_force_tool(self) -> None:
        """Clear the forced tool signal after the Runner consumes it.

        Prevents the forced tool from re-triggering on every loop
        iteration within the same agent turn (which would cause a
        compaction spiral).  The signal is re-evaluated on the next
        turn via :meth:`prepare_messages`.
        """
        self._force_tool = None

    def _truncate(
        self,
        messages: list[LLMInputContentItem],
        model: str,
    ) -> list[LLMInputContentItem]:
        """Drop oldest non-system messages until under budget.

        Preserves the system message (first message if role == "system")
        and removes from the front of the body (oldest first).

        Orphaned tool-result cleanup is handled separately by
        :meth:`ContextEditor.remove_orphaned_tool_results` (step 5).

        Args:
            messages: The conversation messages to truncate.
            model: Model identifier for token counting.

        Returns:
            A new message list within the effective token budget.
        """
        max_tokens = self._effective_budget

        system_msg_t: LLMInputContentItem | None = None
        body: list[LLMInputContentItem] = list(messages)
        if body and body[0].get("role") == "system":
            system_msg_t = body[0]
            body = body[1:]

        dropped = 0
        while (
            body
            and TokenCounter.count_messages(
                ([system_msg_t] if system_msg_t is not None else []) + body,
                model,
            )
            > max_tokens
        ):
            body.pop(0)
            dropped += 1

        if dropped > 0:
            logger.warning(
                "Truncation: dropped %d oldest message(s) to fit %d token budget",
                dropped,
                max_tokens,
            )

        if system_msg_t is not None:
            return [system_msg_t, *body]
        return body

    def _apply_pressure_feedback(
        self,
        messages: list[LLMInputContentItem],
        model: str,
    ) -> list[LLMInputContentItem]:
        """Inject or remove a pressure feedback message based on capacity.

        When context exceeds the warning threshold, injects a developer
        message so the LLM can see its budget status.  Removes the
        message when capacity drops below the threshold.

        Args:
            messages: The current conversation messages.
            model: Model identifier for token counting.

        Returns:
            A new message list with the pressure feedback message injected
            or removed as appropriate.
        """
        # Remove any existing pressure feedback message first
        managed: list[LLMInputContentItem] = [m for m in messages if not m.get(self._PRESSURE_MARKER)]

        current_tokens = TokenCounter.count_messages(managed, model)
        warning_threshold = int(self._effective_budget * self.config.token_budget_warning_threshold)

        if current_tokens >= warning_threshold:
            pct = int((current_tokens / self._effective_budget) * 100)
            # ``_PRESSURE_MARKER`` is an out-of-band marker not declared on any
            # member of the ``LLMInputContentItem`` union; widen to ``Any`` so
            # the extra key survives ``insert`` without a per-site ignore.
            feedback_msg: Any = {
                "role": "developer",
                "content": (
                    f"[Context budget: {pct}% used ({current_tokens:,} / "
                    f"{self._effective_budget:,} tokens). "
                    f"Use manage_context to free space or save_note to "
                    f"preserve key information before old messages are lost.]"
                ),
                self._PRESSURE_MARKER: True,
            }
            # Insert after system message, before conversation body
            if managed and managed[0].get("role") == "system":
                managed.insert(1, feedback_msg)
            else:
                managed.insert(0, feedback_msg)
            logger.debug("Pressure feedback injected: %d%% capacity", pct)

        return managed

    def should_compact(
        self,
        messages: list[LLMInputContentItem],
        model: str,
        compaction: CompactionConfig | None = None,
    ) -> bool:
        """Check whether compaction should be triggered.

        Returns ``True`` when compaction is enabled and the current token
        count exceeds :attr:`CompactionConfig.trigger_tokens`.

        Args:
            messages: The current conversation messages.
            model: Model identifier for token counting.
            compaction: Compaction config to evaluate; defaults to
                ``self.config.compaction`` when ``None``.

        Returns:
            ``True`` if compaction should fire, ``False`` otherwise.
        """
        cfg = compaction if compaction is not None else self.config.compaction
        if not cfg.enabled:
            return False

        # Respect total budget cap.
        if cfg.total_token_budget is not None and self._total_tokens_compacted >= cfg.total_token_budget:
            return False

        current = TokenCounter.count_messages(messages, model)
        return current >= cfg.trigger_tokens

    def get_token_usage(self, messages: list[LLMInputContentItem], model: str) -> TokenUsage:
        """Return token usage statistics for the current context.

        Args:
            messages: The current conversation messages.
            model: Model identifier for token counting.

        Returns:
            A :class:`TokenUsage` TypedDict with keys ``used``, ``max``,
            ``remaining``, ``utilisation``, and ``compaction_count``.
        """
        used = TokenCounter.count_messages(messages, model)
        max_ctx = self._effective_budget
        return TokenUsage(
            used=used,
            max=max_ctx,
            remaining=max_ctx - used,
            utilisation=used / max_ctx if max_ctx > 0 else 0.0,
            compaction_count=self._compaction_count,
        )
