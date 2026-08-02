"""Structured-routing swarm: triage -> specialist, zero LLM routing tokens.

The :class:`StructuredRoutingPolicy` wraps an existing
:class:`~troopai.adk.handoffs.HandoffRoute`. The active agent's
``output_schema`` produces an :class:`Intent` union; the policy reads
the parsed intent off ``SwarmState.last_structured_output`` and
dispatches via the ``.when(IntentType).to(agent)`` DSL.

This is **the TroopAI differentiator**:

- **Zero LLM routing tokens.** The routing signal is the structured
  output itself, not a ``transfer_to_<name>`` tool call. Same
  efficiency advantage as code-orchestrated handoffs — now applied
  to swarm cycles.
- **Full type safety.** The union of Intent types is expressed in
  Python; the LLM emits one of them; routing dispatches on the
  concrete type.
- **Reuses the existing handoff DSL.** No new routing grammar. The
  same ``.when(Refund).to(refunds_agent)`` chain that code-
  orchestrated handoffs already use.

Run with ``ANTHROPIC_API_KEY`` (or any litellm-supported key) set in
the environment.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
from typing import Literal, Union

from pydantic import Field

from troopai.adk.agents import Agent
from troopai.adk.handoffs import HandoffRoute
from troopai.adk.run.config import DEFAULT_MODEL
from troopai.adk.run.runner import Runner
from troopai.adk.swarms import (
    ExplicitDoneTermination,
    MaxTurnsTermination,
    Swarm,
    SwarmConfig,
    prompt_with_swarm_instructions,
)
from troopai.adk.types.intents import Intent, Respond

logger = logging.getLogger(__name__)


# -- Intent schema -----------------------------------------------------


class FlightIntent(Intent):
    """User wants flight information."""

    kind: Literal["flight"] = "flight"
    origin: str | None = Field(None, description="Origin city or airport.")
    destination: str | None = Field(None, description="Destination city or airport.")


class HotelIntent(Intent):
    """User wants hotel recommendations."""

    kind: Literal["hotel"] = "hotel"
    city: str | None = Field(None, description="City or area.")
    nights: int | None = Field(None, description="Number of nights.")


TravelTriageOutput = Union[FlightIntent, HotelIntent, Respond]


# -- Specialist agents -------------------------------------------------

flight_specialist = Agent(
    name="flight_specialist",
    system_prompt=prompt_with_swarm_instructions(
        "You recommend flights. Give 1-2 concrete options with realistic "
        "times and prices. When you're done, call `swarm_done` with a "
        "1-line summary."
    ),
)

hotel_specialist = Agent(
    name="hotel_specialist",
    system_prompt=prompt_with_swarm_instructions(
        "You recommend hotels. Give 1-2 concrete options with rough "
        "price per night. When you're done, call `swarm_done` with a "
        "1-line summary."
    ),
)


# -- Triage agent (speaks the Intent schema) ---------------------------

triage = Agent(
    name="travel_triage",
    system_prompt=(
        "You are a travel triage agent. Classify the user's request into "
        "a FlightIntent, HotelIntent, or Respond. The FlightIntent and "
        "HotelIntent variants extract the salient fields; Respond is a "
        "direct answer when neither applies."
    ),
    output_schema=TravelTriageOutput,
)


# -- Swarm assembly ----------------------------------------------------

travel_swarm: Swarm = (
    Swarm.new("travel", description="triage → flight | hotel")
    .members(triage, flight_specialist, hotel_specialist)
    .entry("travel_triage")
    .routed(
        HandoffRoute("travel_swarm")
        .when(FlightIntent)
        .to(flight_specialist)
        .when(HotelIntent)
        .to(hotel_specialist)
        .otherwise(triage)
    )
    .terminate_on(ExplicitDoneTermination() | MaxTurnsTermination(6))
    .with_config(SwarmConfig(max_total_tokens=20_000))
    .compile()
)


# -- Driver ------------------------------------------------------------


async def main() -> None:
    logger.info("=" * 64)
    logger.info("Structured-routing swarm (triage -> flight | hotel)")
    logger.info("=" * 64)

    result = (
        await Runner.configure()
        .model(DEFAULT_MODEL)
        .swarm(travel_swarm)
        .verbose()
        .arun("I need a hotel in Tokyo for 3 nights.")
    )

    logger.info("Stopped because: %s", result.stop_reason)
    logger.info("Final output: %s", result.final_output)
    logger.info("Last agent: %s", result.last_agent.name if result.last_agent is not None else "<none>")
    logger.info("Total turns: %d", result.total_turns)
    logger.info(
        "Routing cost: 0 tokens (StructuredRoutingPolicy dispatches on output schema, not via transfer_to_* tools)"
    )


if __name__ == "__main__":
    asyncio.run(main())
