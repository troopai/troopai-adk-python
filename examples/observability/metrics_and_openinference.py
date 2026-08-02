"""OTel span tracing + metrics in a MultiTracer with OpenInference convention.

Demonstrates:
- setup_otel with TracingConvention.OPENINFERENCE for Phoenix/Arize-compatible spans
- setup_metrics for OTel metric instruments (agent duration, token counts, tool calls)
- composing both into a MultiTracer via set_tracer
- RunConfig with tracing_enabled=True and metrics_enabled=True

Prerequisites:
    pip install "troopai-adk-python[otel]"
    # Optional: start a local Phoenix collector
    #   pip install arize-phoenix
    #   python -m phoenix.server.main

Run with:
    python examples/observability/metrics_and_openinference.py
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Step 1 — import guards (opentelemetry is an optional extra)
# ---------------------------------------------------------------------------

try:
    from troopai.adk.tracing import MetricsTracer, setup_metrics, setup_otel
except ImportError as _exc:
    raise SystemExit("opentelemetry not installed. Run: pip install 'troopai-adk-python[otel]'") from _exc

# ---------------------------------------------------------------------------
# Step 2 — define the agent (Agent = config, not execution)
# ---------------------------------------------------------------------------

from troopai.adk.agents import Agent
from troopai.adk.llms import LiteLLM

_llm = LiteLLM(model="gpt-4o-mini")
_agent = Agent(
    name="classifier",
    system_prompt="Classify the user's message as positive, neutral, or negative sentiment.",
    llm=_llm,
)

# ---------------------------------------------------------------------------
# Step 3 — compose an OTel span tracer + a MetricsTracer in a MultiTracer
# ---------------------------------------------------------------------------

from troopai.adk.tracing import MultiTracer, TracingConvention, set_tracer

# setup_otel wires a TracerProvider + BatchSpanProcessor. The
# TracingConvention.OPENINFERENCE selection emits OpenInference span-kind and
# attribute keys (openinference.span.kind, input.value, output.value,
# llm.token_count.*) that Phoenix/Arize dashboards read natively.
#
# endpoint=None reads OTEL_EXPORTER_OTLP_ENDPOINT from the environment; set
# it to your Phoenix OTLP endpoint (e.g. "http://localhost:4317") or any
# other OTLP-compatible collector.
_otel_tracer = setup_otel(
    service_name="troopai-classifier",
    convention=TracingConvention.OPENINFERENCE,
    console=True,  # also print finished spans to stdout for local inspection
)

# setup_metrics wires a MeterProvider + PeriodicExportingMetricReader.
# Metrics are independent of span export: tracing_enabled gates spans,
# metrics_enabled gates metric instruments. Both can be active simultaneously.
_metrics_tracer: MetricsTracer = setup_metrics(service_name="troopai-classifier")

# MultiTracer fans every span factory call out to both inner tracers.
# The OTelTracer ships spans to the configured collector; the MetricsTracer
# reads typed SpanData at finish() and records OTel metric instruments.
set_tracer(MultiTracer([_otel_tracer, _metrics_tracer]))

logger.info("Tracer configured: OTel(OpenInference) + Metrics in MultiTracer")

# ---------------------------------------------------------------------------
# Step 4 — run the agent with tracing and metrics enabled
# ---------------------------------------------------------------------------

from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.verbose import VerboseConfig


async def _run() -> None:
    """Execute a single agent turn with observability fully enabled."""
    run_config = RunConfig(
        tracing_enabled=True,  # emit OTel spans via the MultiTracer
        metrics_enabled=True,  # record OTel metric instruments via MetricsTracer
        verbose=VerboseConfig(),
    )

    prompt = "I absolutely love how easy this API is to use!"
    logger.info("Running agent with prompt: %r", prompt)

    result = await Runner.arun(_agent, prompt, run_config=run_config)

    output = result.final_output if isinstance(result.final_output, str) else str(result.final_output)
    logger.info("Agent output: %r", output)
    logger.info(
        "Usage — input_tokens=%s  output_tokens=%s",
        result.context.usage.input_tokens,
        result.context.usage.output_tokens,
    )


if __name__ == "__main__":
    asyncio.run(_run())
