"""Example: fan spans out to two backends at once via ``MultiTracer``.

Pairs the OTel bridge (production observability) with an in-memory
recorder (instant assertions / local inspection) in one ``set_tracer``
call. Every span travels to both — the recorder gives a short inline
summary at the end of the run; the OTel bridge ships to whatever OTLP
endpoint is configured (defaults to the no-op provider if no endpoint
is wired).

Run::

    python examples/tracing/multi_tracer.py

To also ship to a real OTLP collector::

    docker run -d --name jaeger -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 \\
        python examples/tracing/multi_tracer.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging
import os
from typing import Any

from troopai.adk.agents.agent import Agent
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.tracing import MultiTracer, Span, set_tracer
from troopai.adk.tracing.otel import setup_otel
from troopai.adk.types.tracing import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    SpanData,
)
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


class Recorder:
    """In-memory recording tracer — useful for tests and local demos."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, SpanData]] = []

    def _track(self, kind: str, data: SpanData) -> Span[Any]:
        self.spans.append((kind, data))
        return Span(data)

    def agent_span(self, data: AgentSpanData) -> Span[AgentSpanData]:
        return self._track("agent", data)

    def function_span(self, data: FunctionSpanData) -> Span[FunctionSpanData]:
        return self._track("function", data)

    def generation_span(self, data: GenerationSpanData) -> Span[GenerationSpanData]:
        return self._track("generation", data)

    def response_span(self, data: ResponseSpanData) -> Span[ResponseSpanData]:
        return self._track("response", data)

    def handoff_span(self, data: HandoffSpanData) -> Span[HandoffSpanData]:
        return self._track("handoff", data)

    def guardrail_span(self, data: GuardrailSpanData) -> Span[GuardrailSpanData]:
        return self._track("guardrail", data)

    def custom_span(self, data: CustomSpanData) -> Span[CustomSpanData]:
        return self._track("custom", data)


async def _noop_invoker(_ctx: object, _raw_args: str) -> str:
    return "ok"


def build_agent() -> Agent:
    ping = FunctionTool(
        name="ping",
        description="Ping the service.",
        schema={"type": "object", "properties": {}, "required": []},
        on_invoke=_noop_invoker,
    )
    return Agent(
        name="multi-tracer-demo",
        system_prompt="You are a demo agent that pings.",
        tools=[ping],
    )


async def main() -> None:
    otel_tracer = setup_otel(
        endpoint=os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"),
        service_name="multi-tracer-demo",
        console=False,
    )
    recorder = Recorder()

    set_tracer(MultiTracer([otel_tracer, recorder]))

    config = RunConfig(
        tracing_enabled=True,
        tracing_metadata={"demo": "multi_tracer"},
        verbose=VerboseConfig(),
    )

    agent = build_agent()
    result = await Runner.arun(agent, "ping please", max_turns=3, run_config=config)
    logger.info("final_output=%s", result.final_output)

    logger.info("recorded span kinds: %s", [k for k, _ in recorder.spans])
    logger.info("recorded span count: %d", len(recorder.spans))


if __name__ == "__main__":
    asyncio.run(main())
