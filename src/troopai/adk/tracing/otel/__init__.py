"""OpenTelemetry bridge for the TroopAI tracing layer.

Opt-in integration: install the ``otel`` extra with
``pip install 'troopai-adk-python[otel]'``. If the OpenTelemetry packages are
not importable, constructing :class:`OTelTracer` or calling
:func:`setup_otel` raises
:class:`~troopai.adk.exceptions.TracingDependencyError` with the install
command.

See ``docs/tracing/otel.md`` for the full walkthrough.
"""

from __future__ import annotations

from troopai.adk.tracing.otel.otel_span import OTelSpan
from troopai.adk.tracing.otel.otel_tracer import OTelTracer
from troopai.adk.tracing.otel.setup import setup_otel, setup_otel_from_env

__all__ = [
    "OTelSpan",
    "OTelTracer",
    "setup_otel",
    "setup_otel_from_env",
]
