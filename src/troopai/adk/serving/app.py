"""Compose the enabled HTTP surfaces into one ASGI app.

:func:`build_app` mounts the plain-REST surface, the A2A surface, and/or
the health routes onto a single :class:`starlette.applications.Starlette`
app. Every surface is **off by default**: the caller enables exactly the
ones it wants, so the framework never serves a route the developer did
not ask for. The caller's own ASGI runtime (uvicorn, hypercorn, granian)
serves the returned app — the framework does not own the runtime.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from starlette.applications import Starlette

from troopai.adk.serving.health import ReadinessProbe, health_routes
from troopai.adk.serving.rest import SessionFactory, rest_routes

if TYPE_CHECKING:
    from starlette.routing import BaseRoute

    from troopai.adk.a2a.server import A2AServer
    from troopai.adk.agents.agent import Agent
    from troopai.adk.run.config import RunConfig

logger = logging.getLogger(__name__)


def build_app(
    agent: Agent[Any],
    *,
    rest: bool = False,
    a2a_server: A2AServer | None = None,
    health: bool = False,
    max_turns: int | None = None,
    run_config: RunConfig | None = None,
    session_factory: SessionFactory | None = None,
    allow_client_max_turns_above_server_limit: bool = False,
    readiness_probe: ReadinessProbe | None = None,
) -> Starlette:
    """Build a Starlette app exposing the enabled surfaces.

    Args:
        agent: The agent the REST surface runs. (The A2A surface binds
            its own agent through ``a2a_server``.)
        rest: Mount ``POST /run`` and ``POST /run_sse``.
        a2a_server: An :class:`A2AServer` config to mount the A2A
            JSON-RPC + discovery routes; ``None`` leaves A2A off.
            Requires the ``a2a`` extra.
        health: Mount ``GET /healthz`` and ``GET /readyz``.
        max_turns: Default per-request agent-loop budget for the REST
            surface; ``None`` defers to the framework default.
        run_config: Optional :class:`RunConfig` applied to every REST run.
        session_factory: Optional ``(user_id, session_id) -> SessionStore``
            used when a REST request carries a ``session`` block.
        allow_client_max_turns_above_server_limit: When ``False`` (default),
            REST ``max_turns`` is a server-enforced ceiling. Set ``True`` only
            when clients may deliberately request larger turn budgets.
        readiness_probe: Optional async predicate backing ``GET /readyz``.

    Returns:
        A :class:`starlette.applications.Starlette` app for the caller's
        ASGI runtime to serve.

    Raises:
        ValueError: If no surface is enabled.
    """
    routes: list[BaseRoute] = []
    if health:
        routes.extend(health_routes(readiness_probe=readiness_probe))
    if rest:
        routes.extend(
            rest_routes(
                agent,
                max_turns=max_turns,
                run_config=run_config,
                session_factory=session_factory,
                allow_client_max_turns_above_server_limit=allow_client_max_turns_above_server_limit,
            )
        )
    if a2a_server is not None:
        routes.extend(_a2a_routes(a2a_server))
    if len(routes) == 0:
        raise ValueError("build_app needs at least one surface: set rest=True, health=True, or pass a2a_server.")
    logger.debug("serving app built with %d route(s)", len(routes))
    return Starlette(routes=routes)


def _a2a_routes(a2a_server: A2AServer) -> list[BaseRoute]:
    """Extract the A2A JSON-RPC + discovery routes for mounting.

    Imports the A2A factory lazily so the ``a2a`` extra is required only
    when an ``a2a_server`` is actually supplied.

    Args:
        a2a_server: An :class:`A2AServer` config object.

    Returns:
        The route objects from the A2A Starlette app.

    Raises:
        RuntimeError: If the ``a2a`` extra is not installed.
    """
    from troopai.adk.a2a import build_starlette_app

    if build_starlette_app is None:
        raise RuntimeError("a2a_server was provided but the 'a2a' extra is not installed.")
    app = build_starlette_app(a2a_server)
    return list(app.routes)
