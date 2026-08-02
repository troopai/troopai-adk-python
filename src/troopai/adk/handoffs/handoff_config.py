from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NoReturn

from troopai.adk.exceptions import HandoffRejection
from troopai.adk.handoffs.handoff_collapse_mode import HandoffCollapseMode
from troopai.adk.handoffs.handoff_strategy import HandoffStrategy
from troopai.adk.tools.token_budget import TokenBudget

logger = logging.getLogger(__name__)

HandoffErrorPolicy = Literal["halt", "reject_with_message"]
"""How ``Handoff`` handles exceptions from its ``input_filter`` or
``on_handoff`` callback.

- ``"halt"`` (default): propagate the exception and halt the run.
- ``"reject_with_message"``: convert the exception to a tool-result
  message and let the LLM see it. The handoff is NOT taken; the LLM
  can react on its next turn (retry the same handoff, pick a
  different one, or respond directly).
"""


@dataclass(frozen=True)
class HandoffConfig:
    """Configuration for how conversation context transfers during handoff.

    Controls what history the target agent receives and how much of it,
    plus how filter / callback failures are surfaced.

    Attributes:
        strategy: Which messages to include (full, last_n, intent_only, summary).
        window: Number of messages for the ``last_n`` strategy.
        budget: Maximum tokens of history to transfer. Default ``20_000``.
            When the selected messages exceed this cap, oldest non-system
            messages are dropped (truncation, no LLM call). Set to
            ``None`` to disable the cap; use
            ``strategy=HandoffStrategy.SUMMARY`` to opt INTO LLM
            summarisation explicitly.
        collapse: How to collapse transferred history. Accepts a
            ``HandoffCollapseMode`` (``OFF`` / ``SYSTEM_MESSAGE`` /
            ``USER_MESSAGE``) or a bare ``bool`` (``True`` →
            ``SYSTEM_MESSAGE``, ``False`` → ``OFF``). Default ``OFF``.
            Reduces token count at the cost of message-level fidelity.
        on_error: Policy for ``input_filter`` and ``on_handoff``
            callback exceptions. Default ``"halt"`` — propagate and
            halt. ``"reject_with_message"`` converts the exception to
            a tool-result message the LLM sees. Does NOT govern
            Pydantic ``input_type`` validation errors — those always
            surface to the LLM (the LLM made the bad tool call).
        error_message_builder: Optional builder transforming the
            caught exception into the tool-result message text.
            Default formatter: ``f"Handoff failed in <kind> callback: {type(exc).__name__}: {exc}"``
            (``<kind>`` = ``"input_filter"`` or ``"on_handoff"``).
    """

    strategy: HandoffStrategy = HandoffStrategy.FULL
    """The conversation context to include during handoff.

    - ``HandoffStrategy.FULL``: Include the entire conversation history.
    - ``HandoffStrategy.LAST_N``: Include the last N messages (requires ``window``).
    - ``HandoffStrategy.INTENT_ONLY``: Include only the detected intent.
    - ``HandoffStrategy.SUMMARY``: Include a summarized version of the conversation.
    """

    window: int | None = None
    """Number of messages to include when using the ``last_n`` strategy."""

    budget: TokenBudget | int | None = 20_000
    """Maximum tokens of history to transfer to the target agent.

    Accepts a :class:`TokenBudget` (exposes the drop-policy knob) or
    a bare ``int`` (normalized in ``__post_init__`` to
    ``TokenBudget(max_tokens=<int>, drop_policy="preserve_system")``).
    ``None`` disables the cap.

    Applied **after** the strategy (full, last_n, etc.) has selected
    messages. When the selected messages exceed this budget, messages
    are dropped per the budget's ``drop_policy`` until under budget.
    **No LLM call is made** — truncation is free.

    Default is ``20_000`` — cost-conservative for the input-token
    budget without introducing a hidden LLM call. Set to ``None`` to
    opt OUT of the cap.

    Developers who want summarisation (paid LLM call) on the
    overflow MUST opt in explicitly by setting
    ``strategy=HandoffStrategy.SUMMARY``, which routes through
    :meth:`LLM.acomplete` and accumulates usage in
    :attr:`RunContext.usage`.
    """

    collapse: HandoffCollapseMode | bool = HandoffCollapseMode.OFF
    """How to collapse transferred history.

    Accepts a :class:`HandoffCollapseMode` (``OFF`` / ``SYSTEM_MESSAGE`` /
    ``USER_MESSAGE``) or a bare ``bool`` (normalized in
    ``__post_init__``: ``True`` → ``SYSTEM_MESSAGE``,
    ``False`` → ``OFF``).

    When non-``OFF``, the Runner wraps the transferred conversation
    history in a single message of the corresponding role instead of
    replaying individual messages. Reduces token count at the cost
    of message-level fidelity.
    """

    on_error: HandoffErrorPolicy = "halt"
    """Policy for ``input_filter`` and ``on_handoff`` callback exceptions.

    - ``"halt"`` (default, cost-conservative): the original exception
      propagates and halts the run. Surfaces user-callback bugs loudly
      and avoids the silent extra-LLM-turn cost of the alternative.
    - ``"reject_with_message"``: the exception is caught, converted
      to a tool-result message via ``error_message_builder``
      (defaults to ``f"Handoff failed in <kind> callback: {type(exc).__name__}: {exc}"``),
      and emitted to the LLM. The handoff is NOT taken; the LLM can
      react on its next turn (retry, pick a different handoff, or
      respond directly).

    Does NOT govern Pydantic ``input_type`` validation errors — those
    always surface to the LLM with the offending args (the LLM made
    the bad tool call, not user callback code).
    """

    error_message_builder: Callable[[Exception], str] | None = None
    """Optional builder transforming a caught exception into the LLM-visible
    tool-result message when ``on_error="reject_with_message"``.

    When ``None``, the default message is::

        f"Handoff failed in <kind> callback: {type(exc).__name__}: {exc}"

    where ``<kind>`` is ``"input_filter"`` or ``"on_handoff"``. A custom
    builder receives only the exception.

    Useful for redacting sensitive details from internal exceptions
    before showing them to the model.
    """

    def __post_init__(self) -> None:
        """Normalize scalar inputs to the typed forms.

        ``budget=<int>`` → ``TokenBudget(max_tokens=<int>, drop_policy="preserve_system")``.
        ``collapse=True`` → ``HandoffCollapseMode.SYSTEM_MESSAGE``;
        ``collapse=False`` → ``HandoffCollapseMode.OFF``.

        Frozen dataclass, so writes use ``object.__setattr__``.
        """
        # ``bool`` is a subclass of ``int`` — reject bool explicitly
        # before the int branch so ``HandoffConfig(budget=True)``
        # surfaces as a clear TypeError instead of silently coercing
        # to ``TokenBudget(max_tokens=1)``.
        if isinstance(self.budget, bool):
            raise TypeError(
                f"HandoffConfig.budget cannot be bool; got {self.budget!r}. Use TokenBudget(max_tokens=...) or an int."
            )
        if isinstance(self.budget, int):
            object.__setattr__(
                self,
                "budget",
                TokenBudget(max_tokens=self.budget, drop_policy="preserve_system"),
            )
        elif self.budget is not None and not isinstance(self.budget, TokenBudget):
            # Reject non-int / non-TokenBudget / non-None — including
            # ``Decimal`` and ``Fraction`` — at construction time so
            # downstream readers don't silently truncate.
            raise TypeError(
                f"HandoffConfig.budget must be TokenBudget, int, or None; got {type(self.budget).__name__}."
            )

        if isinstance(self.collapse, bool):
            mode = HandoffCollapseMode.SYSTEM_MESSAGE if self.collapse else HandoffCollapseMode.OFF
            object.__setattr__(self, "collapse", mode)

        # LAST_N requires an explicit window. Without this guard the executor
        # silently falls back to a hardcoded default window — a hidden,
        # non-cost-conservative token cost the developer never opted into.
        if self.strategy == HandoffStrategy.LAST_N and self.window is None:
            raise ValueError(
                "HandoffConfig(strategy=HandoffStrategy.LAST_N) requires an explicit "
                "`window` (number of messages to keep); got window=None."
            )
        if self.window is not None and self.window <= 0:
            raise ValueError(f"HandoffConfig.window must be positive, got {self.window}.")


def _default_callback_error_message(callback_kind: Literal["input_filter", "on_handoff"], exc: Exception) -> str:
    """Default message when ``on_error='reject_with_message'`` and no builder is set."""
    return f"Handoff failed in {callback_kind} callback: {type(exc).__name__}: {exc}"


def apply_callback_error_policy(
    *,
    name: str,
    config: HandoffConfig,
    exc: Exception,
    callback_kind: Literal["input_filter", "on_handoff"],
) -> NoReturn:
    """Apply ``config.on_error`` to an exception from a handoff user callback.

    Shared by both handoff paths — ``Handoff.invoke`` (LLM-orchestrated) and
    ``HandoffTarget.invoke`` (code-orchestrated) — so a raising ``input_filter``
    / ``on_handoff`` callback honors the same ``on_error`` policy regardless of
    which path drove the handoff.

    Always raises: re-raises the original exception under ``"halt"`` (or an
    explicit ``HandoffRejection``), or raises ``HandoffRejection`` carrying the
    builder-formatted message under ``"reject_with_message"``.

    Args:
        name: Identifier for the handoff (target agent name) used in logs and
            the rejection message.
        config: The handoff's :class:`HandoffConfig` carrying ``on_error`` +
            ``error_message_builder``.
        exc: The exception the user callback raised.
        callback_kind: ``"input_filter"`` or ``"on_handoff"``.

    Raises:
        HandoffRejection: when ``config.on_error == "reject_with_message"`` (or
            the callback raised ``HandoffRejection`` directly).
        Exception: the original exception when ``config.on_error == "halt"``.
    """
    # A callback that explicitly raised HandoffRejection opted into the
    # rejection path regardless of policy — pass it through unchanged.
    if isinstance(exc, HandoffRejection):
        raise exc
    if config.on_error != "reject_with_message":
        raise exc
    builder = config.error_message_builder
    if builder is not None:
        try:
            message = builder(exc)
        except Exception as builder_exc:
            logger.error(
                "Handoff '%s' error_message_builder raised %s; using default formatter.",
                name,
                type(builder_exc).__name__,
            )
            message = _default_callback_error_message(callback_kind, exc)
    else:
        message = _default_callback_error_message(callback_kind, exc)
    logger.info(
        "Handoff '%s' %s raised %s; rejecting with message.",
        name,
        callback_kind,
        type(exc).__name__,
    )
    raise HandoffRejection(name, message, cause=exc) from exc
