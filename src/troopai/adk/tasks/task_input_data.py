"""TaskInputData — upstream task slices forwarded to a downstream task.

Mirrors :class:`troopai.adk.handoffs.handoff_input_data.HandoffInputData`
for the task-dependency surface. When an upstream is wrapped in a
:class:`TaskDependency` whose :attr:`TaskDependency.input_filter` is set,
the runner calls it once PER upstream (in ``depends_on`` declaration
order) with the upstream's completion snapshot. The filter sets
:attr:`forwarded` on the returned data; the runner concatenates the
forwarded items across all upstreams and prepends them to the downstream
task's description (converted to ``LLMInputContentItem`` via
``RunItem.to_param``, not plain text).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.tasks.task_output import TaskOutput
    from troopai.adk.types.items import RunItem

_SENTINEL: object = object()
"""Sentinel for 'keep existing value' in :meth:`TaskInputData.clone`."""


@dataclass(frozen=True)
class TaskInputData:
    """Upstream-task slices used to prepare a downstream task's input.

    The filter receives one of these per upstream, optionally mutates
    :attr:`forwarded` via :meth:`clone`, and returns the result.

    Attributes:
        task_id: Stable ``task_id`` of the upstream task that produced
            this data.
        output: The :class:`TaskOutput` from the upstream task's
            completed run. Filters can read ``final_output``,
            ``usage``, ``metadata``, ``skipped``, ``error``,
            ``task_name``, etc.
        items: Full :class:`RunItem` stream produced during the
            upstream task's run — the agent's conversation in order:
            system / user / assistant / tool. Same audit shape as
            ``HandoffInputData.context + .output``.
        forwarded: Filtered subset to flow into the downstream task's
            input. When ``None`` (the default), nothing flows from
            this upstream. When set, the runner converts each item to
            a Layer-1 ``LLMInputContentItem`` via
            :meth:`RunItem.to_param` and prepends the resulting
            messages BEFORE the message(s) derived from the downstream
            task's :attr:`Task.description` — items are NOT rendered
            to plain text.
    """

    task_id: str
    """Stable upstream task identifier."""

    output: TaskOutput
    """Completed :class:`TaskOutput` from the upstream task."""

    items: tuple[RunItem, ...]
    """Full conversation produced during the upstream task's run."""

    forwarded: tuple[RunItem, ...] | None = None
    """Filtered subset to prepend; ``None`` ⇒ this upstream's output
    is not forwarded into the downstream task."""

    def clone(
        self,
        *,
        forwarded: tuple[RunItem, ...] | None | object = _SENTINEL,
    ) -> TaskInputData:
        """Return a copy with :attr:`forwarded` optionally replaced.

        Filters call ``data.clone(forwarded=...)`` to set the filtered
        subset without mutating the audit fields (``task_id``,
        ``output``, ``items``). Only :attr:`forwarded` can be changed;
        other fields are read-only via this method to enforce the
        audit-trail contract.

        Args:
            forwarded: The :class:`RunItem` subset to flow into the
                downstream task's input. Pass ``None`` to forward
                nothing. Omit the argument entirely to keep the
                current :attr:`forwarded` value.

        Returns:
            A new :class:`TaskInputData` with :attr:`forwarded` updated
            (when provided) and all other fields carried over from
            ``self``.
        """
        if forwarded is _SENTINEL:
            return dataclasses.replace(self)
        return dataclasses.replace(self, forwarded=forwarded)  # type: ignore[arg-type]
