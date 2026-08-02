"""Router flow example — conditional dispatch via ``@flow_router``.

Demonstrates how a ``@flow_router`` method dispatches to one of several
``@flow_listen("label")`` methods based on its return value, with no
auto-routing from bare string returns on plain ``@flow_listen`` steps.

Run: ``python examples/flows/router_flow.py``
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from troopai.adk import Flow, Runner, flow_listen, flow_router, flow_start

logger = logging.getLogger(__name__)


class TriageState(BaseModel):
    """Typed shared state for the triage flow."""

    severity: int = 0
    """Synthetic incident severity score, set by ``intake``."""

    route_taken: str = ""
    """The label emitted by the flow_router, populated by the chosen branch."""

    handled_by: str = ""
    """Which downstream handler ran."""


class TriageFlow(Flow[TriageState]):
    """Synthetic incident triage flow with three branches."""

    @flow_start
    async def intake(self) -> None:
        """Set up the synthetic incident; assigns a severity score."""
        self.state.severity = 7
        logger.info("intake: severity=%d", self.state.severity)

    @flow_router(intake)
    async def classify(self) -> str:
        """Return one of three labels based on severity.

        Each label is matched by a separate ``@flow_listen("...")`` method
        below. The framework dispatches on the literal string this
        method returns — no source-code introspection.
        """
        if self.state.severity >= 8:
            return "critical"
        if self.state.severity >= 5:
            return "high"
        return "low"

    @flow_listen("critical")
    async def page_oncall(self) -> None:
        """Branch for critical incidents."""
        self.state.route_taken = "critical"
        self.state.handled_by = "oncall_pager"
        logger.info("page_oncall fired")

    @flow_listen("high")
    async def file_ticket(self) -> None:
        """Branch for high-severity incidents."""
        self.state.route_taken = "high"
        self.state.handled_by = "ticket_system"
        logger.info("file_ticket fired")

    @flow_listen("low")
    async def log_only(self) -> None:
        """Branch for low-severity incidents."""
        self.state.route_taken = "low"
        self.state.handled_by = "log_archive"
        logger.info("log_only fired")


async def main() -> None:
    flow = TriageFlow(TriageState)
    result = await Runner.arun_flow(flow)
    logger.info("status=%s", result.status)
    logger.info("route_taken=%s", result.final_state.route_taken)
    logger.info("handled_by=%s", result.final_state.handled_by)
    logger.info("completed_steps=%s", result.completed_steps)


if __name__ == "__main__":
    asyncio.run(main())
