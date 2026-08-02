"""Configuration models for context management.

These Pydantic models control how conversation context is managed —
compaction (LLM summarization), selective editing (clearing old tool
results / thinking blocks), overall token budgets, and prompt cache
optimization strategies.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypedDict

from pydantic import BaseModel, Field


class TokenUsage(TypedDict):
    """Token usage statistics returned by :meth:`ContextManager.get_token_usage`.

    Attributes:
        used: Number of tokens currently in the managed context.
        max: Effective token budget (``max_context_tokens - tool_overhead``).
        remaining: ``max - used`` (may be negative when over-budget).
        utilisation: ``used / max``, clamped to 0.0 when ``max == 0``.
        compaction_count: How many compaction rounds have fired this session.
    """

    used: int
    max: int
    remaining: int
    utilisation: float
    compaction_count: int


class CacheStrategy(StrEnum):
    """Strategy for prompt cache optimization of the tool list.

    Controls how disabled/exhausted tools are handled in the tool
    definitions sent to the LLM, trading off between minimal payload
    and cache prefix stability.

    Attributes:
        NONE: Remove disabled tools entirely (default). Smaller payload
            but invalidates prompt cache when tool list changes.
        STABLE: Keep all tools in the list, marking disabled ones as
            ``[UNAVAILABLE]``. Preserves cache prefix across turns,
            yielding up to 90% discount on cached input tokens.
    """

    NONE = "none"
    """Remove disabled tools from the tool list entirely."""

    STABLE = "stable"
    """Keep disabled tools with ``[UNAVAILABLE]`` description prefix.

    The tool schema stays identical — only the description changes —
    so the cache prefix (tools + system + early messages) is preserved
    across turns even when tool availability fluctuates.
    """


class CompactionConfig(BaseModel):
    """Configuration for context compaction (summarization).

    When the conversation exceeds ``trigger_tokens``, older messages are
    summarized by an LLM call and replaced with a compact summary.

    The LLM used for the summarization call is resolved from
    :attr:`RunConfig.compaction_llm` (explicit override) or, when unset,
    falls back to the agent's resolved ``LLM``. This keeps
    ``CompactionConfig`` provider-agnostic — no model name or ``LLM``
    instance lives on the context-management config itself.

    Attributes:
        enabled: Whether compaction is active.
        trigger_tokens: Token threshold that triggers compaction.
        instructions: Custom prompt prepended to the summarization
            request.  Falls back to a sensible default.
        preserve_recent_items: Number of most-recent conversation turns
            to keep verbatim (never summarized).
        total_token_budget: Optional hard cap on accumulated tokens
            across all compaction rounds.
        cost_aware: Tighten compaction under per-run budget pressure
            (opt-in, default ``False``).
    """

    enabled: bool = False
    """Whether compaction is active. Default is False (compaction disabled)."""

    trigger_tokens: int = Field(default=150_000, ge=1)
    """Token threshold that triggers compaction (e.g., when conversation exceeds this many tokens).
    Note: this is a soft trigger; actual compaction may occur later depending on the conversation flow and token counting.
    Default is 150,000 tokens, which is a common threshold for large-context models,
    but should be tuned based on the specific model and use case.
    """

    instructions: str | None = None
    """Custom prompt prepended to the summarization request.  Falls back to a sensible default if None."""

    preserve_recent_items: int = Field(default=2, ge=0)
    """Number of most-recent conversation turns to keep verbatim (never summarized)."""

    total_token_budget: int | None = None
    """Optional hard cap on accumulated tokens across all compaction rounds (e.g., to prevent runaway compaction).
    If set, the compactor will track the total number of tokens that have been compacted across all rounds,
    and will stop performing compaction once this total exceeds the specified budget.
    This can help prevent excessive loss of information due to repeated compaction in long-running conversations.
    Default is None (no hard cap on total compacted tokens).
    """

    cost_aware: bool = False
    """When ``True`` and a per-run ``TenantBudget`` is active, compaction
    tightens (compacts earlier, preserves fewer recent items) as the run
    nears its budget — shedding input tokens, the dominant cost driver. The
    pressure threshold reuses ``token_budget_warning_threshold``.
    Default ``False`` (opt-in)."""


class ContextEditingConfig(BaseModel):
    """Configuration for selective context clearing.

    Allows clearing old tool call results and thinking blocks to reclaim
    tokens without full compaction.

    Attributes:
        clear_tool_results: Whether to replace old tool-result content
            with a placeholder.
        tool_result_trigger_tokens: Token count above which tool-result
            clearing kicks in.
        tool_results_to_keep: Number of most-recent tool results to
            preserve.
        exclude_tools: Tool names whose results should never be cleared.
        clear_thinking_blocks: Whether to strip old ``thinking`` content
            blocks.
        thinking_turns_to_keep: Number of most-recent assistant turns
            whose thinking blocks are preserved.
    """

    clear_tool_results: bool = False
    """Whether to replace old tool-result content with a placeholder.

    Defaults to ``False`` so the developer never opts out of a
    behavioural change to LLM inputs they didn't choose. When opted in
    (``True``), old tool results past
    ``tool_result_trigger_tokens`` are replaced with a placeholder.
    Observation masking performs as well as LLM summarization
    (JetBrains NeurIPS 2025) at zero additional LLM cost — but the
    developer MUST opt in.

    **Security consideration:** with ``clear_tool_results=False`` (the
    default), sensitive content returned by a tool (API keys, session
    tokens, PII) persists verbatim in every subsequent LLM call until
    the run ends — clearing is not a security control. Tools that
    handle secrets, credentials, or PII SHOULD set a tight
    ``FunctionTool.max_result_tokens`` to bound the per-tool result,
    AND/OR set ``clear_tool_results=True`` explicitly for sessions
    handling sensitive data.
    """

    tool_result_trigger_tokens: int = Field(default=100_000, ge=1)
    """Token count above which tool-result clearing kicks in."""

    tool_results_to_keep: int = 3
    """Number of most-recent tool results to preserve (never cleared)."""

    exclude_tools: list[str] = Field(default_factory=list)
    """Tool names whose results should never be cleared (even when clear_tool_results is True)."""

    clear_thinking_blocks: bool = False
    """Whether to strip old ``thinking`` content blocks (e.g., those generated by a chain-of-thought process)."""

    thinking_turns_to_keep: int = Field(default=1, ge=0)
    """Number of most-recent assistant turns whose thinking blocks are preserved (never cleared)."""


class ContextManagementConfig(BaseModel):
    """Top-level context management configuration.

    Bundles compaction and editing settings together with global token
    budget thresholds.

    Attributes:
        compaction: Compaction (summarization) settings.
        editing: Selective context-editing settings (clearing old tool
            results, clearing thinking blocks).
        max_context_tokens: The model's context window size used for
            budget calculations.
        tool_overhead: Estimated token cost of tool definitions sent with
            each LLM call, reserved from the effective message budget.
        token_budget_warning_threshold: Fraction of ``max_context_tokens``
            at which a warning is emitted (0.0–1.0).
        truncation: When ``True``, drop oldest non-system messages after
            editing and compaction if the context still exceeds the
            effective budget. Last-resort data loss; opt-in only.
        pressure_feedback: When ``True``, inject a developer message
            when context usage exceeds the warning threshold so the LLM
            can see its budget status and act proactively.
        forced_tool: Tool name to force on the next LLM call when context
            pressure exceeds the warning threshold. ``None`` disables the
            signal.
        cache_strategy: Strategy for handling disabled/exhausted tools in
            the tool list to optimize prompt cache stability.
    """

    compaction: CompactionConfig = Field(default_factory=CompactionConfig)
    """Compaction (summarization) settings."""

    editing: ContextEditingConfig = Field(default_factory=ContextEditingConfig)
    """Selective context-editing settings (e.g., clearing old tool results)."""

    max_context_tokens: int = 200_000
    """Maximum context tokens for budget calculations (typically the model's context window size)."""

    tool_overhead: int = 0
    """Estimated token cost of tool definitions sent with each LLM call.

    Tool schemas are counted as input tokens by the API but are not
    part of the message list that ``prepare_messages()`` manages.
    Set this to reserve space for tool definitions so the total input
    (messages + tools) stays within the model's context window.

    Example: 6 tool schemas ≈ 1,500 tokens → set ``tool_overhead=1500``.
    The effective message budget becomes ``max_context_tokens - tool_overhead``.
    """

    token_budget_warning_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    """Fraction of max_context_tokens at which a warning is emitted (0.0–1.0)."""

    truncation: bool = False
    """Hard enforcement of ``max_context_tokens``.

    When ``True``, if the context still exceeds the effective budget
    (``max_context_tokens - tool_overhead``) after editing and
    compaction, the oldest non-system messages are dropped until the
    budget is met.  This is a last-resort measure — data is lost, but
    the API call won't be rejected.

    Equivalent to OpenAI's ``truncation: "auto"`` but implemented
    locally (provider-agnostic).
    """

    pressure_feedback: bool = False
    """Inject a developer message when context usage exceeds the warning threshold.

    When ``True``, a brief budget status message is added to the
    conversation so the LLM can *see* it's near the limit and
    proactively use ``manage_context`` or ``save_note``.  Only
    injected when capacity exceeds ``token_budget_warning_threshold``.
    Removed when capacity drops below the threshold.
    """

    forced_tool: str | None = None
    """Force a specific tool call when context pressure exceeds the warning threshold.

    When set to a tool name (e.g. ``"manage_context"``), the Runner
    overrides ``tool_choice`` to force that tool on the next LLM call
    whenever context usage exceeds ``token_budget_warning_threshold``.
    After the forced call, the Runner calls
    ``ContextManager.consume_force_tool()`` to clear the signal so the
    LLM resumes normal tool selection on the next call.

    Works independently of ``pressure_feedback`` — can be used alone
    or together.
    """

    cache_strategy: CacheStrategy = CacheStrategy.NONE
    """Strategy for prompt cache optimization of the tool list.

    - ``NONE`` (default): Disabled tools are removed from the tool
      list, keeping the payload small but invalidating the prompt
      cache when the list changes.
    - ``STABLE``: Disabled tools remain in the list with an
      ``[UNAVAILABLE]`` description prefix, preserving the cache
      prefix for up to 90% savings on cached input tokens.
    """
