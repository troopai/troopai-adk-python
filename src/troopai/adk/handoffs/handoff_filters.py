import logging
from typing import Any

from troopai.adk.handoffs import HandoffInputData, HandoffInputFilter
from troopai.adk.types.items import (
    HandoffOutputItem,
    RunItem,
    SystemItem,
    ToolCallItem,
    ToolCallOutputItem,
)

logger = logging.getLogger(__name__)


def source_messages(data: HandoffInputData) -> tuple[RunItem, ...]:
    """Get the effective message source for filtering.

    Returns ``forwarded`` if already set by a prior filter (e.g. in a
    ``compose()`` chain), otherwise ``messages`` (context + output).
    """
    return data.forwarded if data.forwarded is not None else data.messages


def remove_tool_calls(data: HandoffInputData) -> HandoffInputData:
    """Remove all tool call and tool result messages.

    Strips ``ToolCallItem``, ``ToolCallOutputItem``, and ``HandoffOutputItem``
    instances.  For raw dict items, strips messages with
    ``role="tool"`` or ``role="function"``, and removes ``tool_calls``
    from assistant messages.

    Args:
        data: The handoff input data to filter.

    Returns:
        New HandoffInputData with tool messages removed via ``forwarded``.
    """
    # `source` is typed as tuple[RunItem, ...] but runtime may include raw
    # dict history (see TestHandoffFilters.test_filter_uses_messages_property).
    # Widen to Any so the defensive dict dispatch below type-checks.
    source: tuple[Any, ...] = source_messages(data)
    filtered: list[Any] = []
    for msg in source:
        # Typed item checks
        if isinstance(msg, (ToolCallItem, ToolCallOutputItem, HandoffOutputItem)):
            continue

        # Raw dict fallback
        if isinstance(msg, dict):
            role = msg.get("role", "")
            if role in ("tool", "function"):
                continue
            if role == "assistant" and "tool_calls" in msg:
                msg = {k: v for k, v in msg.items() if k != "tool_calls"}

        filtered.append(msg)

    return data.clone(forwarded=tuple(filtered))


def remove_system_messages(data: HandoffInputData) -> HandoffInputData:
    """Remove all system messages.

    Strips ``SystemItem`` instances and raw dicts with ``role="system"``.

    Args:
        data: The handoff input data to filter.

    Returns:
        New HandoffInputData with system messages removed via ``forwarded``.
    """
    # See `remove_tool_calls` above for why the source is re-typed as Any.
    source: tuple[Any, ...] = source_messages(data)
    filtered: list[Any] = []
    for msg in source:
        if isinstance(msg, SystemItem):
            continue
        if isinstance(msg, dict) and msg.get("role") == "system":
            continue
        filtered.append(msg)

    return data.clone(forwarded=tuple(filtered))


def keep_last_n(n: int) -> HandoffInputFilter:
    """Create a filter that keeps only the last N items.

    Factory function that returns a configured filter.

    Args:
        n: Number of trailing items to keep. ``0`` forwards nothing;
            negative values raise :class:`ValueError`.

    Returns:
        A HandoffInputFilter function.

    Example::

        Handoff(target=target, input_filter=keep_last_n(4))
    """
    if n < 0:
        raise ValueError(f"keep_last_n requires n >= 0, got {n}.")

    def _filter(data: HandoffInputData) -> HandoffInputData:
        source = source_messages(data)
        items = list(source)
        # items[-0:] == items[0:] == ALL items, so n == 0 must be handled
        # explicitly to forward nothing rather than the whole history.
        kept: list[RunItem] = items[-n:] if n > 0 else []
        kept = _drop_orphan_run_items(kept)
        return data.clone(forwarded=tuple(kept))

    return _filter


def forward_intent(data: HandoffInputData) -> HandoffInputData:
    """Append the classified Intent as a user message.

    Injects the structured Intent data (e.g., order_id, reason) so the
    specialist agent can use the classification without re-extracting
    it from the raw user message.

    Use with code-orchestrated handoffs (HandoffRoute) where the triage
    agent outputs a structured Intent.

    Args:
        data: The handoff input data to filter.

    Returns:
        New HandoffInputData with Intent appended as user message via ``forwarded``.

    Example::

        from troopai.adk.handoffs.handoff_filters import forward_intent

        HandoffRoute("triage")
            .when(RefundIntent).to(refunds_agent, input_filter=forward_intent)
    """
    from troopai.adk.types.input import LLMInputEasyMessage
    from troopai.adk.types.items import UserItem

    source = source_messages(data)
    intent = data.intent
    if hasattr(intent, "model_dump_json"):
        content = f"Classified intent: {intent.model_dump_json()}"
    else:
        content = f"Classified intent: {intent}"

    history = list(source)
    history.append(UserItem(raw=LLMInputEasyMessage(role="user", content=content)))
    return data.clone(forwarded=tuple(history))


def intent_only(data: HandoffInputData) -> HandoffInputData:
    """Remove all messages, keeping only the intent.

    The most aggressive filter — the target agent sees only the
    classified intent, no conversation history.

    Args:
        data: The handoff input data to filter.

    Returns:
        New HandoffInputData with empty forwarded.
    """
    return data.clone(forwarded=())


def _drop_orphan_run_items(items: list[RunItem]) -> list[RunItem]:
    """Remove unpaired ToolCallItem / ToolCallOutputItem pairs from a RunItem list.

    When ``keep_last_n`` slices the history at a boundary that splits a paired
    ToolCallItem and its ToolCallOutputItem, the orphaned half must be removed to
    avoid strict-provider rejections downstream.  Both halves are dropped when
    either is missing from the slice.

    Args:
        items: The sliced RunItem list to clean.

    Returns:
        The same list with any unpaired ToolCallItem / ToolCallOutputItem removed
        (order preserved).
    """
    call_ids: set[str] = set()
    result_ids: set[str] = set()
    for item in items:
        if isinstance(item, ToolCallItem):
            cid = item.raw.call_id
            if isinstance(cid, str) and len(cid) > 0:
                call_ids.add(cid)
        elif isinstance(item, ToolCallOutputItem):
            cid = item.raw.call_id
            if isinstance(cid, str) and len(cid) > 0:
                result_ids.add(cid)

    kept: list[RunItem] = []
    dropped = 0
    for item in items:
        if isinstance(item, ToolCallItem):
            cid = item.raw.call_id
            # Mirror the set-building guard (lines above) so that
            # empty-string call_ids — which were never added to
            # result_ids — are not incorrectly classified as orphans.
            if isinstance(cid, str) and len(cid) > 0 and cid not in result_ids:
                dropped += 1
                continue
        elif isinstance(item, ToolCallOutputItem):
            cid = item.raw.call_id
            if isinstance(cid, str) and len(cid) > 0 and cid not in call_ids:
                dropped += 1
                continue
        kept.append(item)

    if dropped > 0:
        logger.warning(
            "keep_last_n: dropped %d orphaned ToolCallItem/ToolCallOutputItem(s) "
            "at the slice boundary to avoid unpaired tool-call provider errors.",
            dropped,
        )
    return kept


def compose(*filters: HandoffInputFilter) -> HandoffInputFilter:
    """Chain multiple filters into a single pipeline.

    Filters are applied left-to-right: the output of each filter
    becomes the input of the next. Each filter in the chain can
    read ``forwarded`` set by the previous filter.

    Args:
        *filters: Two or more HandoffInputFilter functions.

    Returns:
        A single HandoffInputFilter that applies all filters in order.

    Example::

        strict = compose(remove_tool_calls, remove_system_messages, keep_last_n(6))
        Handoff(target=target, input_filter=strict)
    """

    def pipeline(data: HandoffInputData) -> HandoffInputData:
        for f in filters:
            data = f(data)
        return data

    return pipeline
