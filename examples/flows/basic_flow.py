"""Basic Flow example — decorator-driven multi-step orchestration.

Demonstrates every core Flow primitive in one runnable script:

- ``@flow_start`` — multiple entry points firing in parallel
- ``@flow_listen(method_ref)`` — listener on a single step
- ``@flow_listen(method_a | method_b)`` — OR gate (fluent operator)
- ``@flow_listen(method_a & method_b)`` — AND gate (fluent operator)
- ``@flow_router`` — conditional dispatch returning a route label
- ``@flow_listen("label")`` — listener on a router's returned label
- Typed Pydantic state shared across steps via ``self.state``
- :class:`FlowCheckpoint` round-trip (capture + resume)

The example uses synthetic step bodies (no LLM calls) so it runs
without credentials. For a runnable example with real agents, see
``examples/flows/research_with_agents.py``.

Run:

    python examples/flows/basic_flow.py
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from troopai.adk import (
    Flow,
    FlowCheckpoint,
    FlowConfig,
    Runner,
    flow_listen,
    flow_router,
    flow_start,
)

logger = logging.getLogger(__name__)


class ResearchState(BaseModel):
    """Typed shared state mutated by every step.

    The developer constructs this and passes it to the Flow; the
    framework NEVER auto-instantiates state from the generic parameter.
    """

    topic: str = ""
    """Research topic, set by the first start method."""

    fact: str = ""
    """Fact gathered by the parallel research step."""

    quote: str = ""
    """Quote gathered by the parallel quote step."""

    summary: str = ""
    """Synthesized summary after both fact and quote are in."""

    classification: str = ""
    """Final classification label produced by the router."""


class ResearchFlow(Flow[ResearchState]):
    """Synthetic research flow demonstrating every Flow primitive."""

    @flow_start
    async def init_topic(self) -> None:
        """Seed the topic. Fires in parallel with :meth:`init_session`."""
        self.state.topic = "decentralized AI orchestration"
        logger.info("init_topic: topic=%s", self.state.topic)

    @flow_start
    async def init_session(self) -> None:
        """Set up a synthetic session. Fires in parallel with :meth:`init_topic`."""
        logger.info("init_session: session ready")

    @flow_listen(init_topic)
    async def gather_fact(self) -> None:
        """Listener triggered by :meth:`init_topic` completion."""
        await asyncio.sleep(0)  # Yield to demonstrate async
        self.state.fact = f"{self.state.topic} reduces single-vendor lock-in."
        logger.info("gather_fact: fact=%s", self.state.fact)

    @flow_listen(init_topic)
    async def gather_quote(self) -> None:
        """Second listener on :meth:`init_topic` — fires in parallel with :meth:`gather_fact`."""
        await asyncio.sleep(0)
        self.state.quote = f"'{self.state.topic} is the next frontier.'"
        logger.info("gather_quote: quote=%s", self.state.quote)

    @flow_listen(gather_fact & gather_quote)
    async def synthesize(self) -> None:
        """AND gate — fires ONCE after BOTH gather_fact and gather_quote complete.

        Built fluently via the ``&`` operator on :class:`FlowStep`
        wrappers. The framework tracks arrivals by name and fires
        ``synthesize`` exactly once when the gate is satisfied.
        """
        self.state.summary = f"{self.state.fact} {self.state.quote}"
        logger.info("synthesize: summary=%s", self.state.summary)

    @flow_router(synthesize)
    async def classify(self) -> str:
        """Router method — returns a non-empty string label.

        Downstream ``@flow_listen("label")`` methods react to the returned
        label. The framework dispatches on the literal string the
        router returns — there is no source-code introspection of
        return statements.
        """
        if len(self.state.summary) > 50:
            return "comprehensive"
        return "brief"

    @flow_listen("comprehensive")
    async def handle_comprehensive(self) -> None:
        """Listener on the ``"comprehensive"`` route label."""
        self.state.classification = "COMPREHENSIVE"
        logger.info("handle_comprehensive fired")

    @flow_listen("brief")
    async def handle_brief(self) -> None:
        """Listener on the ``"brief"`` route label."""
        self.state.classification = "BRIEF"
        logger.info("handle_brief fired")


async def demo_basic_run() -> None:
    """Run the ResearchFlow once and inspect the result."""
    flow = ResearchFlow(ResearchState)
    result = await Runner.arun_flow(flow, config=FlowConfig(max_steps=50))
    logger.info("=== basic run ===")
    logger.info("status=%s", result.status)
    logger.info("completed_steps=%s", result.completed_steps)
    logger.info("final_state.summary=%s", result.final_state.summary)
    logger.info("final_state.classification=%s", result.final_state.classification)


async def demo_streaming() -> None:
    """Stream events from a ResearchFlow run."""
    flow = ResearchFlow(ResearchState)
    streaming = Runner.arun_flow_streamed(flow)
    logger.info("=== streaming run ===")
    async for event in streaming.stream_events():
        logger.info("event: %s", event.type)
    logger.info("final status: %s", streaming.status)


async def demo_checkpoint_roundtrip() -> None:
    """Capture a checkpoint, serialize, rehydrate, and verify equality."""
    flow = ResearchFlow(ResearchState)
    result = await Runner.arun_flow(flow)

    # Capture a checkpoint manually using the executor's final state.
    # (In a real HITL scenario, FlowCheckpoint.capture would be called
    # at a step boundary from a hook.)
    checkpoint = FlowCheckpoint(
        flow_id=result.flow_id,
        completed_steps=result.completed_steps,
        pending_steps=(),
        and_gate_arrivals={},
        consumed_gates=(),
        state_data=result.final_state.model_dump_json(),
    )
    serialized = checkpoint.to_json()
    rehydrated = FlowCheckpoint.from_json(serialized)
    assert rehydrated.flow_id == checkpoint.flow_id
    assert rehydrated.completed_steps == checkpoint.completed_steps
    logger.info("=== checkpoint round-trip ===")
    logger.info("flow_id preserved: %s", rehydrated.flow_id)
    logger.info("completed_steps preserved: %s", rehydrated.completed_steps)

    # Rehydrate state and confirm
    state_back = ResearchState.model_validate_json(rehydrated.state_data)
    assert state_back.summary == flow.state.summary
    logger.info("state.summary rehydrated correctly: %s", state_back.summary)


async def main() -> None:
    """Run all three demos."""
    await demo_basic_run()
    await demo_streaming()
    await demo_checkpoint_roundtrip()


if __name__ == "__main__":
    asyncio.run(main())
