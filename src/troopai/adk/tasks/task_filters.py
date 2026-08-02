"""Built-in :data:`TaskInputFilter` callables for common forwarding patterns.

Mirrors :mod:`troopai.adk.handoffs.handoff_filters` for the
task-dependency surface. Compose these via :func:`compose` to chain
multiple transforms; write a custom callable when you need richer
control over what flows into a downstream task.

Every filter is a pure function: ``TaskInputData → TaskInputData``
with ``forwarded`` set. Filters never mutate ``task_id``, ``output``,
or ``items`` — those audit fields are read-only by contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from troopai.adk.tasks.task_input_data import TaskInputData
from troopai.adk.types.items.items import MessageOutputItem, UserItem

if TYPE_CHECKING:
    from troopai.adk.tasks.task import TaskInputFilter


def forward_final_output(data: TaskInputData) -> TaskInputData:
    """Forward only the upstream task's ``final_output`` as a user message.

    The most common pattern: the downstream task receives just the
    upstream's final answer, rendered as a single :class:`UserItem`.
    Discards intermediate tool calls, reasoning, and message-output
    chunks.

    Args:
        data: The :class:`TaskInputData` supplied by the runner for
            one upstream task's completion.

    Returns:
        A clone of ``data`` with :attr:`TaskInputData.forwarded` set
        to a single-element tuple containing the ``final_output``
        rendered as a :class:`UserItem`, or an empty tuple when
        ``final_output`` is ``None``.
    """
    if data.output.final_output is None:
        return data.clone(forwarded=())
    text = str(data.output.final_output)
    item = UserItem(raw={"role": "user", "content": text})
    return data.clone(forwarded=(item,))


def forward_new_items(data: TaskInputData) -> TaskInputData:
    """Forward the upstream's entire :attr:`TaskInputData.items` stream.

    The downstream task sees the full upstream conversation —
    system / user / assistant / tool messages — as prepended input.
    Use when context fidelity matters more than token cost.

    Args:
        data: The :class:`TaskInputData` supplied by the runner for
            one upstream task's completion.

    Returns:
        A clone of ``data`` with :attr:`TaskInputData.forwarded` set
        to the full ``items`` tuple.
    """
    return data.clone(forwarded=data.items)


def forward_messages_only(data: TaskInputData) -> TaskInputData:
    """Forward only the assistant message-output items from the upstream run.

    Strips system / user / tool-call / tool-result items, keeping
    just the agent's final answer chunks. Cheaper than
    :func:`forward_new_items` when the downstream agent doesn't need
    to see tool internals.

    Args:
        data: The :class:`TaskInputData` supplied by the runner for
            one upstream task's completion.

    Returns:
        A clone of ``data`` with :attr:`TaskInputData.forwarded` set
        to the subset of ``items`` that are
        :class:`~troopai.adk.types.items.items.MessageOutputItem`
        instances.
    """
    kept = tuple(item for item in data.items if isinstance(item, MessageOutputItem))
    return data.clone(forwarded=kept)


def keep_last_n(n: int) -> TaskInputFilter:
    """Return a filter that forwards only the last ``n`` upstream items.

    Useful for bounding the prepended prompt when the upstream
    produced a long conversation.

    Args:
        n: Number of trailing items to keep. Negative values raise
            :class:`ValueError`.

    Returns:
        A :data:`TaskInputFilter` that slices the upstream's items.
    """
    if n < 0:
        raise ValueError(f"keep_last_n requires n >= 0, got {n}.")

    def _filter(data: TaskInputData) -> TaskInputData:
        return data.clone(forwarded=tuple(data.items[-n:]) if n > 0 else ())

    return _filter


def compose(*filters: TaskInputFilter) -> TaskInputFilter:
    """Chain multiple filters into a single pipeline.

    Each filter receives the output of the previous one — so a
    composed filter sees ``forwarded`` already set by the earlier
    stage. The result is the final stage's output.

    Args:
        *filters: One or more :data:`TaskInputFilter` instances. An
            empty tuple is acceptable: the result is a passthrough
            that returns the input :class:`TaskInputData` unchanged,
            leaving ``forwarded`` at its incoming value (``None`` when
            called by the runner, so nothing flows downstream).

    Returns:
        A single :data:`TaskInputFilter` running the stages in order.
    """

    def _filter(data: TaskInputData) -> TaskInputData:
        current = data
        for stage in filters:
            current = stage(current)
        return current

    return _filter
