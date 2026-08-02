"""Tracing types for the TroopAI Agents ADK.

Typed span-data dataclasses used by the framework-level tracer. Each
span kind has a dedicated ``@dataclass(frozen=True)`` that derives from
:class:`SpanData` and implements ``export()`` — producing a JSON-safe
``dict[str, Any]`` snapshot for observability backends.

Span data objects are provider-agnostic. The ``Tracer`` protocol in
``troopai.adk.tracing`` consumes them without knowing which exporter
will persist the result.
"""

from troopai.adk.types.tracing.convention import TracingConvention
from troopai.adk.types.tracing.span_data import (
    AgentSpanData,
    AnySpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GraphNodeSpanData,
    GraphSpanData,
    GraphSuperstepSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    ResponseSpanData,
    SandboxSpanData,
    SpanData,
    SwarmSpanData,
    SwarmTurnSpanData,
)

__all__ = [
    "AgentSpanData",
    "AnySpanData",
    "CustomSpanData",
    "FunctionSpanData",
    "GenerationSpanData",
    "GraphNodeSpanData",
    "GraphSpanData",
    "GraphSuperstepSpanData",
    "GuardrailSpanData",
    "HandoffSpanData",
    "ResponseSpanData",
    "SandboxSpanData",
    "SpanData",
    "SwarmSpanData",
    "SwarmTurnSpanData",
    "TracingConvention",
]
