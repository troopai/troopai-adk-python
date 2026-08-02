"""OTel metrics for the framework: MetricsTracer + setup."""

from troopai.adk.tracing.metrics.instruments import Instruments
from troopai.adk.tracing.metrics.setup import setup_metrics
from troopai.adk.tracing.metrics.tracer import MetricsTracer

__all__ = ["Instruments", "MetricsTracer", "setup_metrics"]
