"""Operational health routes for the serving layer.

``GET /healthz`` (liveness — the process is up) and ``GET /readyz``
(readiness — dependencies are reachable) let Kubernetes probes, Cloud
Run startup checks, and load-balancer health checks observe the service.

Both routes are **opt-in**: :func:`health_routes` is mounted only when
the caller asks for it, so the framework never serves an endpoint the
developer did not request.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from starlette.responses import JSONResponse
from starlette.routing import Route

if TYPE_CHECKING:
    from starlette.requests import Request

logger = logging.getLogger(__name__)

ReadinessProbe = Callable[[], Awaitable[bool]]
"""Async predicate returning ``True`` when the service can accept traffic."""

DEFAULT_LIVENESS_PATH = "/healthz"
DEFAULT_READINESS_PATH = "/readyz"


def health_routes(
    *,
    liveness_path: str = DEFAULT_LIVENESS_PATH,
    readiness_path: str = DEFAULT_READINESS_PATH,
    readiness_probe: ReadinessProbe | None = None,
) -> list[Route]:
    """Build the liveness and readiness routes.

    Args:
        liveness_path: Path for the liveness route (process-up check).
        readiness_path: Path for the readiness route (dependency check).
        readiness_probe: Optional async predicate. When supplied, the
            readiness route returns ``200`` while it yields ``True`` and
            ``503`` once it yields ``False``. When ``None``, readiness
            always reports ready.

    Returns:
        The liveness and readiness :class:`starlette.routing.Route` objects.

    Raises:
        ValueError: If either path is empty.
    """
    if len(liveness_path) == 0:
        raise ValueError("liveness_path must be a non-empty path.")
    if len(readiness_path) == 0:
        raise ValueError("readiness_path must be a non-empty path.")

    # Starlette invokes route handlers with the ASGI Request positionally;
    # the parameter is part of the route contract even when unread.
    async def liveness(request: Request) -> JSONResponse:  # noqa: ARG001
        return JSONResponse({"status": "alive"})

    async def readiness(request: Request) -> JSONResponse:  # noqa: ARG001
        if readiness_probe is None:
            return JSONResponse({"status": "ready"})
        ready = await readiness_probe()
        if ready:
            return JSONResponse({"status": "ready"})
        return JSONResponse({"status": "not_ready"}, status_code=503)

    return [
        Route(liveness_path, liveness, methods=["GET"]),
        Route(readiness_path, readiness, methods=["GET"]),
    ]
