"""LangSmith exporter — OTLP ingestion endpoint with API-key headers.

Docs: https://docs.smith.langchain.com/observability/how_to_guides/trace_with_opentelemetry
"""

from __future__ import annotations

from troopai.adk.tracing.otel.otel_tracer import OTelTracer
from troopai.adk.tracing.otel.setup import setup_otel


def langsmith_headers(*, api_key: str, project: str | None = None) -> dict[str, str]:
    """Build LangSmith OTLP headers.

    Args:
        api_key: LangSmith API key (non-empty). Load from the environment.
        project: Optional LangSmith project name; added as
            ``Langsmith-Project`` when provided (non-empty).

    Returns:
        Header dict suitable for the ``headers`` argument of
        :func:`~troopai.adk.tracing.otel.setup.setup_otel`.

    Raises:
        ValueError: When ``api_key`` is empty.
        ValueError: When ``project`` is provided but empty.
    """
    if len(api_key) == 0:
        raise ValueError("langsmith api_key must be non-empty")
    if project is not None and len(project) == 0:
        raise ValueError("langsmith project must be non-empty when provided")
    headers = {"x-api-key": api_key}
    if project is not None:
        headers["Langsmith-Project"] = project
    return headers


def setup_langsmith(
    *, api_key: str, project: str | None = None, endpoint: str | None = None, service_name: str = "troopai-adk-python"
) -> OTelTracer:
    """Return an :class:`OTelTracer` exporting spans to LangSmith's OTLP endpoint.

    Args:
        api_key: LangSmith API key (non-empty). Load from the environment.
        project: Optional LangSmith project name; omit to use the
            default project.
        endpoint: LangSmith OTLP endpoint; ``None`` uses the OTel
            environment default.
        service_name: Value for the ``service.name`` resource attribute.

    Returns:
        A configured :class:`OTelTracer` pointing at LangSmith.

    Raises:
        TracingDependencyError: When the ``opentelemetry`` packages are
            not installed (``pip install 'troopai-adk-python[otel]'``).
        ValueError: When ``api_key`` is empty or ``project`` is empty.
    """
    return setup_otel(
        endpoint=endpoint, service_name=service_name, headers=langsmith_headers(api_key=api_key, project=project)
    )
