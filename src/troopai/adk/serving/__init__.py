"""HTTP serving layer for the TroopAI ADK.

This package turns a local :class:`~troopai.adk.agents.agent.Agent`
into an ASGI app the developer's own runtime serves. It is the single
place that maps a :class:`~troopai.adk.run.runner.Runner` call onto
HTTP, exposing three **opt-in** surfaces composed by :func:`build_app`:

* a plain-REST surface — ``POST /run`` (collect → JSON) and
  ``POST /run_sse`` (Server-Sent Events) for generic HTTP clients;
* health routes — ``GET /healthz`` (liveness) and ``GET /readyz``
  (readiness) for container probes and load-balancer checks;
* the A2A JSON-RPC + discovery routes (when an ``A2AServer`` is passed),
  delegated to :mod:`troopai.adk.a2a`.

Responses serialize the framework's provider-agnostic layers only
(Layer 1 / Layer 3); the provider wire format never crosses the HTTP
boundary.

The Starlette + sse-starlette stack is an optional extra. Install with::

    pip install 'troopai-adk-python[server]'

When the extra is missing, every public name in this module is ``None``;
downstream code can branch on ``build_app is None`` to skip serving
wiring gracefully. The mechanism mirrors :mod:`troopai.adk.a2a`.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import build_app
    from .health import ReadinessProbe, health_routes
    from .rest import SessionFactory, rest_routes
    from .serializers import (
        run_result_to_dict,
        stream_event_to_dict,
        streaming_result_to_dict,
        usage_to_dict,
    )
else:
    try:
        from .app import build_app
        from .health import ReadinessProbe, health_routes
        from .rest import SessionFactory, rest_routes
        from .serializers import (
            run_result_to_dict,
            stream_event_to_dict,
            streaming_result_to_dict,
            usage_to_dict,
        )
    except ImportError as _exc:
        # Only swallow "the server stack is not installed". Any other
        # ImportError (a typo in our modules, a transitive dep failing)
        # MUST surface rather than be masked as "extra missing".
        # ``ModuleNotFoundError.name`` holds the exact top-level module the
        # interpreter could not find.
        if getattr(_exc, "name", None) not in {"starlette", "sse_starlette"}:
            raise
        build_app = None
        health_routes = None
        rest_routes = None
        ReadinessProbe = None
        SessionFactory = None
        run_result_to_dict = None
        stream_event_to_dict = None
        streaming_result_to_dict = None
        usage_to_dict = None

__all__ = [
    "ReadinessProbe",
    "SessionFactory",
    "build_app",
    "health_routes",
    "rest_routes",
    "run_result_to_dict",
    "stream_event_to_dict",
    "streaming_result_to_dict",
    "usage_to_dict",
]
