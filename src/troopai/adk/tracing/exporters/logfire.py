"""Pydantic Logfire exporter — OTLP span + metric export with a write token.

Docs: https://logfire.pydantic.dev/docs/how-to-guides/alternative-clients/
"""

from __future__ import annotations

from troopai.adk.tracing.otel.otel_tracer import OTelTracer
from troopai.adk.tracing.otel.setup import setup_otel


def logfire_headers(*, token: str) -> dict[str, str]:
    """Build Logfire OTLP headers from a write token.

    Args:
        token: Logfire write token (non-empty). Load from the environment.

    Returns:
        Header dict suitable for the ``headers`` argument of
        :func:`~troopai.adk.tracing.otel.setup.setup_otel`.

    Raises:
        ValueError: When ``token`` is empty.
    """
    if len(token) == 0:
        raise ValueError("logfire token must be non-empty")
    # Logfire OTLP ingestion takes the write token as the RAW Authorization
    # value (no "Bearer" prefix) — per the alternative-clients guide linked
    # in the module docstring. Do not add a "Bearer " prefix.
    return {"Authorization": token}


def setup_logfire(*, token: str, endpoint: str | None = None, service_name: str = "troopai-adk-python") -> OTelTracer:
    """Return an :class:`OTelTracer` exporting spans to Logfire's OTLP endpoint.

    Args:
        token: Logfire write token (non-empty). Load from the environment.
        endpoint: Logfire OTLP endpoint; ``None`` uses the OTel environment
            default.
        service_name: Value for the ``service.name`` resource attribute.

    Returns:
        A configured :class:`OTelTracer` pointing at Logfire.

    Raises:
        TracingDependencyError: When the ``opentelemetry`` packages are
            not installed (``pip install 'troopai-adk-python[otel]'``).
        ValueError: When ``token`` is empty.
    """
    return setup_otel(endpoint=endpoint, service_name=service_name, headers=logfire_headers(token=token))
