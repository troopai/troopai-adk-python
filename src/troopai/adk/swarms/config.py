"""Swarm-level configuration — budgets, hard guards, timeouts.

``SwarmConfig`` holds the knobs that belong to the swarm as a whole,
not to any individual member agent. It deliberately reuses existing
levers where possible:

- ``RunConfig.max_total_turns`` — cross-agent turn counter (already shipped)
- ``FunctionTool.max_result_tokens`` — per-tool result cap
- ``HandoffConfig.budget`` — per-inter-agent transfer cap

``SwarmConfig`` adds the swarm-wide dimensions: handoff count and
cumulative token budget.

``per_turn_timeout`` and ``retry_on_throttle`` are intentionally
*not* fields here: the driver does not enforce them, and shipping
silent-no-op config would mislead callers. Any such field is added
only together with the enforcement code that honours it.

See ``docs/swarms/cost_optimization.md`` for how the cost levers
interact.
"""

from __future__ import annotations

from dataclasses import dataclass

from troopai.adk.swarms.shared_context_strategy import SharedContextStrategy


@dataclass(frozen=True)
class SharedContextConfig:
    """Bundles a ``SharedContextStrategy`` with its parameters.

    Attributes:
        strategy: Which strategy to apply. Default ``SCOPED``.
        window: Number of items kept when strategy is ``LAST_N``.
            Ignored otherwise.
        budget: Token cap when strategy is ``SUMMARIZED`` — older
            history is compacted until the total fits. Ignored otherwise.
        max_handoff_message_chars: Optional cap on the number of
            characters of a :class:`~troopai.adk.swarms.yield_signal.SwarmHandoff`
            message that are injected into the target agent's turn.
            When ``None`` (default), no truncation. When set, the
            ``SCOPED`` strategy truncates any longer handoff message to
            this length and logs a :func:`logging.Logger.warning`.
            Characters (not tokens) because a character bound is
            deterministic across providers and cheap (``len(str)``); a
            token bound would need a provider-specific tokenizer.
            Bypasses :attr:`FunctionTool.max_result_tokens`,
            :attr:`HandoffConfig.budget`, and
            :attr:`SwarmConfig.max_total_tokens` because the handoff
            message is injected *before* the next turn's LLM call —
            the other three caps measure LLM-observable tokens, while
            this one measures the raw string pre-injection.

            **Production guidance**: the default ``None`` keeps the
            framework from silently capping developer-supplied
            messages, but production deployments SHOULD set an
            explicit cap (e.g. ``32_768`` chars ≈ 8K tokens) to bound
            the DoS-via-handoff surface.

            **Scope**: this cap applies per-handoff-message, not
            per-turn-cumulative. A pathological agent that emits
            multiple handoffs within a single turn can inject
            N × cap characters over N handoffs. The turn-cumulative
            bound comes from :attr:`SwarmConfig.max_total_tokens`
            (measured post-LLM) plus the existing
            :attr:`SwarmConfig.max_handoffs` agent-switch counter.
    """

    strategy: SharedContextStrategy = SharedContextStrategy.SCOPED
    """Default: SCOPED — each agent sees only its own scratch plus the explicit handoff payload (no cross-agent broadcast)."""

    window: int | None = None
    """Item count for ``LAST_N``. MUST be > 0 when strategy is ``LAST_N``."""

    budget: int | None = None
    """Token cap for ``SUMMARIZED``. MUST be > 0 when strategy is ``SUMMARIZED``."""

    max_handoff_message_chars: int | None = None
    """Optional character-count cap on handoff-message injection. ``None`` = no cap."""

    def __post_init__(self) -> None:
        """Validate strategy/parameter coupling.

        Lives here rather than on :class:`SwarmConfig` so the check
        runs at the earliest construction site — a caller building a
        bare ``SharedContextConfig`` still gets the validation without
        having to wrap it in a ``SwarmConfig``.
        """
        if self.strategy == SharedContextStrategy.LAST_N and (self.window is None or self.window <= 0):
            raise ValueError(f"window must be a positive integer when strategy is LAST_N, got {self.window}.")
        if self.strategy == SharedContextStrategy.SUMMARIZED and (self.budget is None or self.budget <= 0):
            raise ValueError(f"budget must be a positive integer when strategy is SUMMARIZED, got {self.budget}.")
        if self.max_handoff_message_chars is not None and self.max_handoff_message_chars <= 0:
            raise ValueError(f"max_handoff_message_chars must be > 0 or None, got {self.max_handoff_message_chars}.")


@dataclass(frozen=True)
class SwarmConfig:
    """Swarm-level execution configuration.

    Attributes:
        max_handoffs: Hard cap on the number of agent switches in a
            run. Independent of the ``TerminationCondition`` — trips
            even if the condition has not fired. Default 20.
        max_total_tokens: Cumulative input+output token budget across
            all member agents. ``None`` disables the swarm-level cap
            (per-agent ``LLMUsageLimits`` still applies). Tracked
            against ``SwarmState.cumulative_usage.total_tokens``.
        shared_context: Strategy + parameters controlling what each
            member sees on its turn. Default strategy ``SCOPED``.
    """

    max_handoffs: int = 20
    """Hard cap on agent switches. Trips via ``StopReason(kind="max_handoffs")``."""

    max_total_tokens: int | None = None
    """Cumulative token budget across the whole swarm run. ``None`` = no cap."""

    shared_context: SharedContextConfig = SharedContextConfig()
    """Shared-context strategy + parameters. Default strategy ``SCOPED``."""

    def __post_init__(self) -> None:
        """Validate guard values are in legal ranges.

        ``shared_context`` validation lives on
        :class:`SharedContextConfig` itself — this method only checks
        the swarm-level budgets.
        """
        if self.max_handoffs <= 0:
            raise ValueError(
                f"max_handoffs must be > 0, got {self.max_handoffs}. "
                "Use a large positive integer (e.g. 1000) instead of disabling."
            )
        if self.max_total_tokens is not None and self.max_total_tokens <= 0:
            raise ValueError(f"max_total_tokens must be > 0 or None, got {self.max_total_tokens}.")
