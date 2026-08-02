"""Example: OpenTelemetry bridge printing spans to stdout.

Demonstrates the simplest opt-in path:

1. ``pip install 'troopai-adk-python[otel]'``
2. ``setup_otel(console=True)`` — prints every finished span to
   stdout via OTel's ``ConsoleSpanExporter``.
3. ``set_tracer(tracer)`` — installs the bridge.
4. ``RunConfig(tracing_enabled=True)`` — flips the per-call-site
   ``disabled`` flag off so spans actually reach the tracer.

Run with ``ANTHROPIC_API_KEY`` (or any litellm-supported provider key)
set in the environment or in a ``.env`` file::

    python examples/tracing/otel_console.py

You should see JSON-shaped spans printed for the outer ``agent.*``,
the inner ``llm.generation``, and the tool ``tool.*`` calls.
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import asyncio
import logging

from troopai.adk.agents.agent import Agent
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.otel import setup_otel
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


async def _echo_invoker(_ctx: object, raw_args: str) -> str:
    return f"echoed: {raw_args}"


def build_agent() -> Agent:
    echo = FunctionTool(
        name="echo",
        description="Echo back the input.",
        schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        on_invoke=_echo_invoker,
    )
    return Agent(
        name="otel-console-demo",
        system_prompt="You are a minimal demo agent.",
        tools=[echo],
    )


async def main() -> None:
    tracer = setup_otel(service_name="otel-console-demo", console=True)
    set_tracer(tracer)

    config = RunConfig(
        tracing_enabled=True,
        tracing_metadata={"demo": "otel_console"},
        verbose=VerboseConfig(),
    )

    agent = build_agent()
    result = await Runner.arun(agent, "say hello", max_turns=2, run_config=config)
    logger.info("final_output=%s", result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
