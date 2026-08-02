from __future__ import annotations

from dataclasses import dataclass, replace as dataclass_replace
from types import EllipsisType
from typing import Any

from troopai.adk.types.items import RunItem


@dataclass(frozen=True)
class HandoffInputData:
    """Input payload for a handoff transfer.

    Bundles the trigger intent with the message history (context and
    output) to be processed by the target agent.

    Attributes:
        intent: What triggered the handoff. For code-orchestrated handoffs
            this is an Intent model from the HandoffRoute; for LLM-orchestrated
            handoffs it is a validated Pydantic model (when ``input_type`` is
            set) or a raw tool-args string.
        context: Messages that existed before the current agent's turn started.
            For the first agent this is ``[system, user]``; for subsequent
            agents after handoff this is the filtered history from the prior agent.
        output: Messages generated during the current agent's turn — LLM
            responses, tool calls, tool results, and the handoff trigger itself.
            Empty tuple for code-orchestrated handoffs that fire immediately
            after classification.
        forwarded: Filtered messages to forward to the next agent. When
            ``None``, the handoff executor uses ``context + output``. Set
            by an ``input_filter`` to decouple what the next agent sees from
            the full audit trail.
    """

    intent: Any
    """What triggered the handoff.

    Code-orch: Intent model from the HandoffRoute.
    LLM-orch: validated Pydantic model (when input_type is set)
    or raw tool args string.
    """

    context: tuple[RunItem, ...]
    """Messages that existed BEFORE the current agent's turn started.

    For the first agent, this is [system, user]. For subsequent agents
    after handoff, this is the filtered/prepared history from the prior agent.
    """

    output: tuple[RunItem, ...]
    """Messages generated DURING the current agent's turn — LLM responses,
    tool calls, tool results, and the handoff trigger itself. Empty tuple
    for code-orch handoffs that fire immediately after classification.
    """

    forwarded: tuple[RunItem, ...] | None = None
    """Filtered messages to forward to the next agent. When None, the
    handoff executor uses context + output. Set by input_filter to
    decouple what the next agent sees from the full audit trail.
    """

    @property
    def messages(self) -> tuple[RunItem, ...]:
        """Full message list: ``context + output`` (the complete pre-filter view)."""
        return self.context + self.output

    def clone(
        self,
        *,
        forwarded: tuple[RunItem, ...] | None | EllipsisType = ...,
    ) -> HandoffInputData:
        """Return a copy with the ``forwarded`` view replaced.

        Input filters MUST only replace ``forwarded`` — the ``intent``,
        ``context``, and ``output`` fields form the audit trail and must
        not be modified by filters.

        Args:
            forwarded: The new forwarded message slice for the target agent.
                Pass ``None`` to clear the forwarded view (executor falls back
                to ``context + output``). Omitting the argument copies the
                original value unchanged.

        Returns:
            A new HandoffInputData with ``forwarded`` replaced and all other
            fields (``intent``, ``context``, ``output``) copied from the original.
        """
        if isinstance(forwarded, EllipsisType):
            return dataclass_replace(self)
        return dataclass_replace(self, forwarded=forwarded)
