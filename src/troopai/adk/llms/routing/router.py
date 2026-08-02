"""Model-routing abstraction: ordered candidates + escalation predicate."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from troopai.adk.llms.llm import LLM
    from troopai.adk.types.input import LLMInputContentItem
    from troopai.adk.types.responses.llm_response import LLMResponse


@dataclass(frozen=True)
class RoutedModel:
    """A routing candidate: a framework ``LLM`` instance + its model name.

    Attributes:
        llm: The LLM instance to call for this candidate.
        model: The model name (for cost estimation, spans, and ordering).
    """

    llm: LLM
    """The LLM instance to call for this candidate."""

    model: str
    """The model name (for cost estimation, spans, and ordering)."""


@dataclass(frozen=True)
class RoutingContext:
    """Inputs a router may use to order/select candidates.

    Attributes:
        messages: The Layer-1 messages about to be sent.
        tenant_id: Active tenant, if any.
        run_cost: USD spent so far in this run.
    """

    messages: list[LLMInputContentItem]
    """The Layer-1 messages about to be sent."""

    tenant_id: str | None
    """Active tenant, if any."""

    run_cost: float
    """USD spent so far in this run."""


class LLMRouter(ABC):
    """Returns an ordered candidate list; the loop tries each in turn.

    Cheap-first-with-fallback: candidates are ordered cheapest-acceptable
    first; the runner escalates to the next on failure (exception,
    output-schema validation failure, or :meth:`should_escalate`).
    """

    @abstractmethod
    def candidates(self, ctx: RoutingContext) -> Sequence[RoutedModel]:
        """Return an ordered list of candidates to try (index 0 first).

        Args:
            ctx: Routing context carrying messages, tenant, and run cost.

        Returns:
            Ordered sequence of candidates.  The runner tries them in
            order, escalating on provider exception, output-schema
            validation failure, or :meth:`should_escalate` returning
            ``True``.
        """

    def should_escalate(self, response: LLMResponse | None) -> bool:
        """Return ``True`` to escalate based on the response content.

        Default: never escalate on content (only exceptions / schema
        failures drive escalation). Override to wrap guardrail/quality checks.

        Args:
            response: The completed ``LLMResponse``, or ``None`` when the
                candidate raised before producing one.

        Returns:
            ``True`` to abandon the current candidate and try the next.
        """
        del response
        return False
