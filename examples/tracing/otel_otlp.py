"""Example: ship spans to an OTLP collector (Jaeger / Datadog / etc.).

Start a local Jaeger all-in-one::

    docker run -d --name jaeger \\
        -p 16686:16686 -p 4317:4317 \\
        jaegertracing/all-in-one

Then run this example — the default endpoint points at
``http://localhost:4317`` (override via ``OTEL_EXPORTER_OTLP_ENDPOINT``
or the ``--endpoint`` arg)::

    python examples/tracing/otel_otlp.py
    # or
    OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317 \\
        python examples/tracing/otel_otlp.py

Open http://localhost:16686 and pick the ``otel-otlp-demo`` service to
inspect the trace tree.

The span tree shape for a two-turn run with one tool call is::

    agent.otel-otlp-demo
    ├── llm.generation      (turn 1)
    ├── tool.echo
    └── llm.generation      (turn 2)
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

import argparse
import asyncio
import logging
import os
import socket
from urllib.parse import urlparse

from troopai.adk.agents.agent import Agent
from troopai.adk.run import RunConfig, Runner
from troopai.adk.tools.function_tool import FunctionTool
from troopai.adk.tracing import set_tracer
from troopai.adk.tracing.otel import setup_otel
from troopai.adk.verbose import VerboseConfig

logger = logging.getLogger(__name__)


def _endpoint_reachable(endpoint: str, timeout: float = 2.0) -> bool:
    """Probe ``endpoint`` with a short TCP connect before enabling
    the exporter. Without this the OTLP gRPC exporter would retry for
    many seconds against a missing collector and the example would
    appear to hang — a poor demo UX.
    """
    parsed = urlparse(endpoint)
    host = parsed.hostname or "localhost"
    port = parsed.port or 4317
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


async def _lookup_invoker(_ctx: object, raw_args: str) -> str:
    return f"lookup result for: {raw_args}"


def build_agent() -> Agent:
    lookup = FunctionTool(
        name="lookup",
        description="Look up a value by key.",
        schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
        on_invoke=_lookup_invoker,
    )
    return Agent(
        name="otel-otlp-demo",
        system_prompt="You are a demo agent that uses the lookup tool.",
        tools=[lookup],
    )


async def main(endpoint: str | None) -> None:
    resolved_endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    if not _endpoint_reachable(resolved_endpoint):
        logger.warning(
            "OTLP endpoint %s is not reachable — start a collector (see docstring "
            "for the Jaeger one-liner) and re-run. Exiting cleanly without shipping spans.",
            resolved_endpoint,
        )
        return

    logger.info("shipping spans to %s", resolved_endpoint)

    tracer = setup_otel(
        endpoint=resolved_endpoint,
        service_name="otel-otlp-demo",
        console=False,
    )
    set_tracer(tracer)

    config = RunConfig(
        tracing_enabled=True,
        tracing_metadata={"demo": "otel_otlp", "endpoint": resolved_endpoint},
        verbose=VerboseConfig(),
    )

    agent = build_agent()
    result = await Runner.arun(
        agent,
        "look up the value of 'price'",
        max_turns=3,
        run_config=config,
    )
    logger.info("final_output=%s", result.final_output)
    logger.info("trace shipped — inspect it in your OTLP backend (Jaeger UI: http://localhost:16686)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default=None,
        help="OTLP endpoint (overrides OTEL_EXPORTER_OTLP_ENDPOINT).",
    )
    args = parser.parse_args()
    asyncio.run(main(args.endpoint))
