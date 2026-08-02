from __future__ import annotations

from abc import ABC
from typing import Literal

from pydantic import BaseModel as PydanticModel, Field


class Intent(PydanticModel, ABC):
    """
    Base class for all routing intents.

    Intents represent classified user requests that determine routing.
    The LLM outputs an Intent, and the routing DSL maps it to an agent.

    Example:
        class Refund(Intent):
            kind: Literal["refund"] = "refund"
            order_id: Optional[str] = None
            reason: Optional[str] = None
    """

    kind: str = Field(..., description="Intent type discriminator for routing")


class Respond(PydanticModel):
    """
    Special output indicating the agent wants to respond directly.

    When an agent outputs Respond instead of an Intent, no handoff occurs.
    This allows agents to either respond OR route based on the situation.

    Example:
        SupportOutput = Respond | Refund | Billing | Escalate

        # LLM can output:
        # - Respond(message="Here's your answer") → No handoff, respond directly
        # - Refund(order_id="123") → Hand off to refunds agent
    """

    kind: Literal["respond"] = "respond"

    message: str = Field(..., description="The response message to return to the user")


__all__ = ["Intent", "Respond"]
