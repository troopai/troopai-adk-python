"""Context directives — LLM-driven context management commands.

Provides directive types that tools can emit to request context mutations.
The Runner applies these before the next LLM call via :func:`apply_directives`.

The LLM strategizes (emits directives), the Runner executes (applies them).
Directives operate on the *working view* only — the Session (permanent
record) is never modified.

Directive types:

- :class:`DropDirective` — drop old messages, keeping the N most recent
- :class:`CompactDirective` — summarize old messages via LLM, keeping
  the N most recent

See ``docs/tools/jit_context_aware.md`` for usage patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from troopai.adk.context.compaction import ContextCompactor
from troopai.adk.context.context_config import CompactionConfig
from troopai.adk.context.context_editing import ContextEditor

if TYPE_CHECKING:
    from troopai.adk.llms.llm import LLM
    from troopai.adk.run.config import RunConfig
    from troopai.adk.run.context import RunContext
    from troopai.adk.types.input import LLMInputContentItem

logger = logging.getLogger(__name__)


# =====================================================================
# Directive types
# =====================================================================


@dataclass(frozen=True)
class DropDirective:
    """Drop old messages, keeping the N most recent.

    The system message is always preserved.  Messages are counted
    from the end (most recent), excluding the system message.

    Attributes:
        preserve: Number of most-recent non-system messages to keep.
    """

    preserve: int


@dataclass(frozen=True)
class CompactDirective:
    """Summarize old messages via LLM, keeping the N most recent.

    Uses :class:`ContextCompactor` to generate a summary of older
    messages, then rebuilds the message list with
    ``[system, summary, ...preserved]``.

    Attributes:
        preserve: Number of most-recent non-system messages to keep.
    """

    preserve: int


ContextDirective = DropDirective | CompactDirective
"""Union of all directive types that tools can emit."""


# =====================================================================
# DirectiveStore — pending directive container
# =====================================================================


class DirectiveStore:
    """Holds pending directives written by tools, consumed by the Runner.

    Thread-safe for single-threaded async (Python event loop).
    Tools call :meth:`add` during execution; the Runner calls
    :meth:`consume` before the next LLM call.
    """

    def __init__(self) -> None:
        self._pending: list[ContextDirective] = []

    def add(self, directive: ContextDirective) -> None:
        """Add a directive to the pending queue."""
        self._pending.append(directive)
        logger.debug("Directive queued: %s", directive)

    def consume(self) -> list[ContextDirective]:
        """Return and clear all pending directives."""
        directives = list(self._pending)
        self._pending.clear()
        return directives

    @property
    def count(self) -> int:
        """Number of pending directives."""
        return len(self._pending)


# =====================================================================
# apply_directives — standalone processing function
# =====================================================================


async def apply_directives(
    messages: list[LLMInputContentItem],
    store: DirectiveStore,
    llm: LLM,
    model: str,
    run_config: RunConfig | None = None,
    context: RunContext[Any] | None = None,
) -> list[LLMInputContentItem]:
    """Consume and apply pending directives to the message list.

    Called by the Runner **before** the ContextManager pipeline.
    Works independently of ``ContextManagementConfig``.

    Directive processing order:

    1. All directives are consumed from the store.
    2. ``DropDirective`` is applied first (cheapest — no LLM call).
    3. ``CompactDirective`` is applied second (invokes LLM summarization).

    The system message (first message with ``role == "system"``) is
    always preserved regardless of directive parameters.

    Args:
        messages: Current conversation messages (including system message).
        store: The :class:`DirectiveStore` to consume directives from.
        llm: The ``LLM`` instance to invoke for ``CompactDirective``
            (typically resolved via :func:`resolve_compaction_llm`).
        model: Model identifier for token counting (litellm tokenizer
            is name-based).
        run_config: Optional ``RunConfig`` forwarded to the compactor.
        context: Optional ``RunContext`` used to accumulate compaction
            token usage into the run's usage ledger.

    Returns:
        The modified message list (may be shorter than the input).
    """
    directives = store.consume()
    if len(directives) == 0:
        return messages

    managed = list(messages)

    # Separate system message from body
    system_msg: LLMInputContentItem | None = None
    if managed and managed[0].get("role") == "system":
        system_msg = managed[0]
        body = managed[1:]
    else:
        body = managed

    # Step 1: apply drop directives (cheapest first)
    for directive in directives:
        if isinstance(directive, DropDirective):
            preserve = max(1, directive.preserve)
            if preserve < len(body):
                dropped = len(body) - preserve
                body = body[-preserve:]
                logger.info(
                    "DropDirective applied: dropped %d messages, kept %d",
                    dropped,
                    preserve,
                )
            else:
                logger.debug(
                    "DropDirective skipped: preserve=%d >= message count=%d",
                    preserve,
                    len(body),
                )

    # Step 2: apply compact directives (invokes LLM).
    # Each directive processes the already-compacted body so that multiple
    # CompactDirectives in the same batch all take effect (mirroring the
    # "apply all" behaviour of DropDirective above).
    for directive in directives:
        if isinstance(directive, CompactDirective):
            preserve = max(1, directive.preserve)
            if preserve >= len(body):
                logger.debug(
                    "CompactDirective skipped: preserve=%d >= message count=%d",
                    preserve,
                    len(body),
                )
                continue

            # Build temporary message list for compaction
            temp_messages = ([system_msg] if system_msg is not None else []) + body

            # Create a temporary CompactionConfig with the directive's preserve value
            compact_config = CompactionConfig(
                enabled=True,
                preserve_recent_items=preserve,
            )

            result = await ContextCompactor.compact(
                temp_messages,
                llm=llm,
                model_name=model,
                config=compact_config,
                run_config=run_config,
            )

            # Accumulate compaction usage into RunContext.usage so the
            # framework's usage ledger is complete (see CompactionResult
            # docstring).
            if context is not None and result.usage is not None:
                context.usage = context.usage + result.usage

            if result.items_compacted > 0:
                preserved_body = body[-preserve:] if preserve < len(body) else body
                rebuilt = ContextCompactor.build_compacted_messages(
                    result.summary,
                    preserved_body,
                    system_msg,
                )
                logger.info(
                    "CompactDirective applied: %d -> %d tokens (%d messages summarized)",
                    result.original_token_count,
                    result.compacted_token_count,
                    result.items_compacted,
                )
                # Re-split for subsequent directives in this batch.
                if rebuilt and rebuilt[0].get("role") == "system":
                    system_msg = rebuilt[0]
                    body = rebuilt[1:]
                else:
                    system_msg = None
                    body = rebuilt

    # Reassemble with system message, dropping any tool results orphaned by a
    # DropDirective recency-slice (its matching function_call may have been
    # evicted, which Anthropic/Gemini reject). prepare_messages runs the same
    # cleanup, but it is skipped entirely when no ContextManagementConfig is
    # set — so apply_directives must guarantee it on every path.
    assembled = [system_msg, *body] if system_msg is not None else body
    return ContextEditor.remove_orphaned_tool_results(assembled)
