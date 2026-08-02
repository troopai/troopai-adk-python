"""Per-tenant cost tracking with OTel spans, metrics, and status readback.

Demonstrates:
- RunConfig.tenant_id threading tenant identity to spans, metrics, and status records
- RunContext.cost_usd reading the running USD cost after a run
- AgentStatusStore + StatusTrackingHooks recording per-run cost in SQLite
- store.get_status(agent_name, tenant_id=...) returning per-tenant cumulative cost
- MultiTracer composing an OTelTracer (OpenInference) with a MetricsTracer

Prerequisites:
    pip install "troopai-adk-python[otel]"
    export OPENAI_API_KEY="sk-..."   # or any LiteLLM-supported model key

Run with:
    python examples/observability/tenant_and_cost.py
"""

from __future__ import annotations

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

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
# Step 2 — define the agent
# ---------------------------------------------------------------------------

from troopai.adk.agents import Agent
from troopai.adk.llms import LiteLLM

_llm = LiteLLM(model="gpt-4o-mini")
_agent = Agent(
    name="sentiment",
    system_prompt="Classify the user's message as positive, neutral, or negative.",
    llm=_llm,
)

# ---------------------------------------------------------------------------
# Step 3 — compose OTel span tracer + MetricsTracer in a MultiTracer
# ---------------------------------------------------------------------------

from troopai.adk.tracing import MultiTracer, TracingConvention, set_tracer

_otel_tracer = setup_otel(
    service_name="troopai-tenant-demo",
    convention=TracingConvention.OPENINFERENCE,
    console=True,
)

_metrics_tracer: MetricsTracer = setup_metrics(service_name="troopai-tenant-demo")

# Both tracers receive every span; the OTel tracer ships spans to the
# configured collector and the MetricsTracer records OTel instruments.
set_tracer(MultiTracer([_otel_tracer, _metrics_tracer]))

logger.info("Tracer configured: OTel(OpenInference) + Metrics in MultiTracer")

# ---------------------------------------------------------------------------
# Step 4 — set up status tracking (records each run in SQLite, per-tenant)
# ---------------------------------------------------------------------------

from troopai.adk.status import AgentStatusStore, StatusTrackingHooks

_store = AgentStatusStore(path=":memory:")
_hooks: StatusTrackingHooks = StatusTrackingHooks(store=_store)

# ---------------------------------------------------------------------------
# Step 5 — run the agent with tenant_id set and read back per-tenant cost
# ---------------------------------------------------------------------------

from troopai.adk.run.config import RunConfig
from troopai.adk.run.runner import Runner
from troopai.adk.verbose import VerboseConfig


async def _run() -> None:
    """Execute a single tenant-tagged run and log the resulting cost figures."""
    run_config = RunConfig(
        tenant_id="acme",
        tracing_enabled=True,
        metrics_enabled=True,
        verbose=VerboseConfig(),
    )

    prompt = "I absolutely love this product — it changed my workflow!"
    logger.info("Running agent for tenant 'acme' with prompt: %r", prompt)

    result = await Runner.arun(
        _agent,
        prompt,
        run_config=run_config,
        hooks=_hooks,
    )

    output = result.final_output if isinstance(result.final_output, str) else str(result.final_output)
    logger.info("Agent output: %r", output)

    # RunContext.cost_usd: running total for this invocation (best-effort;
    # 0.0 when the resolved model has no price entry in litellm).
    logger.info(
        "Live run cost — cost_usd=%.6f  tenant_id=%s",
        result.context.cost_usd,
        result.context.tenant_id,
    )

    # AgentStatusStore aggregates per-run records in SQLite.
    # Passing tenant_id restricts the query to runs tagged "acme" only.
    status = await _store.get_status(_agent.name, tenant_id="acme")
    logger.info(
        "Per-tenant cumulative — total_runs=%d  total_cost_usd=%.6f",
        status.total_runs,
        status.total_cost_usd,
    )

    await _store.close()


if __name__ == "__main__":
    asyncio.run(_run())
