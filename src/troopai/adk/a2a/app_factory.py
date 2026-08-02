"""``build_starlette_app`` — turn an :class:`A2AServer` config into an ASGI app.

This is the factory that composes:

* :class:`A2AExecutor` (server-side bridge to ``Runner.arun``)
* :class:`a2a.server.tasks.InMemoryTaskStore` (or the caller's
  :class:`TaskStore`)
* :class:`a2a.server.request_handlers.DefaultRequestHandler` (a2a-sdk's
  request-handling glue)
* :class:`starlette.applications.Starlette` (the ASGI app)

into a runnable Starlette app the developer's own ASGI server (uvicorn,
hypercorn, granian) serves.

Usage::

    from troopai.adk.a2a import A2AServer, build_starlette_app
    from a2a.types import AgentCard, AgentCapabilities, AgentInterface

    server = A2AServer(
        agent=my_local_agent,
        agent_card=AgentCard(
            name="my-agent",
            description="Helps with X.",
            version="1.0.0",
            supported_interfaces=[
                AgentInterface(
                    url="https://my.example.com",
                    protocol_binding="JSONRPC",
                ),
            ],
            capabilities=AgentCapabilities(streaming=True),
        ),
    )
    app = build_starlette_app(server)

    # Serve via your own ASGI runtime:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)

The factory is a pure function (same input → same output) and does
not start any background tasks. Lifecycle is the caller's job.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

try:
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes import (
        create_agent_card_routes,
        create_jsonrpc_routes,
    )
    from a2a.server.tasks import InMemoryTaskStore
    from starlette.applications import Starlette
except ImportError as ie:
    _extra_owned = ("a2a", "starlette", "sse_starlette")
    if ie.name is not None and not any(ie.name == m or ie.name.startswith(m + ".") for m in _extra_owned):
        # A transitive dependency of an installed a2a-sdk failed — surface
        # the real error instead of mislabeling it "extra not installed".
        raise
    raise ImportError(
        "Please install the 'a2a' extra to use A2A protocol support. Run: pip install 'troopai-adk-python[a2a]'",
        # name="a2a" lets optional-extra guards (e.g. the graphs adapter's
        # A2A fallthrough) recognize the missing extra and degrade gracefully.
        name="a2a",
    ) from ie

from troopai.adk.a2a.executor import A2AExecutor

if TYPE_CHECKING:
    from troopai.adk.a2a.server import A2AServer

logger = logging.getLogger(__name__)


def build_starlette_app(server: A2AServer) -> Starlette:
    """Build a Starlette ASGI app from an :class:`A2AServer` config.

    The returned app exposes:

    * ``GET /.well-known/agent-card.json`` — the public AgentCard,
      published verbatim from ``server.agent_card``. This is the
      discovery URL every A2A client reads before sending a task.
    * ``POST {server.rpc_url}`` (default ``/``) — JSON-RPC dispatcher
      handling ``send_message`` (blocking and streaming),
      ``get_task``, ``cancel_task``, ``list_tasks``, and the push
      notification config endpoints. SSE responses are emitted by the
      a2a-sdk's :class:`EventSourceResponse` for streaming methods.

    When ``server.task_store is None``, the factory installs an
    :class:`InMemoryTaskStore`. **Production callers MUST provide a
    persistent ``task_store``** — the in-memory store loses every
    task on server restart, which breaks the
    :class:`A2AContinuationToken` resume contract for any task that
    outlives the process.

    ``server.executor_task_store`` is forwarded to the
    :class:`A2AExecutor` for its own task-snapshot persistence. When
    ``None`` (default) the executor uses an in-memory store; pass a
    :class:`~troopai.adk.a2a.task_store.SQLiteTaskStore` for
    executor-level restart recovery via its ``recover_on_startup``.

    Args:
        server: The frozen :class:`A2AServer` config to materialise.

    Returns:
        A :class:`starlette.applications.Starlette` ASGI app, ready
        for the caller's ASGI runtime to serve.
    """
    task_store = server.task_store if server.task_store is not None else InMemoryTaskStore()
    if server.task_store is None:
        logger.warning(
            "A2AServer for agent %r built with the default InMemoryTaskStore; "
            "tasks will be lost on process restart. Pass a persistent "
            "TaskStore (e.g. a2a.server.tasks.DatabaseTaskStore) for "
            "production deployments where A2AContinuationToken-based "
            "resumption must survive restarts.",
            server.agent.name,
        )

    executor = A2AExecutor(
        agent=server.agent,
        max_turns=server.max_turns,
        run_config=server.run_config,
        task_store=server.executor_task_store,
    )

    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store,
        agent_card=server.agent_card,
        extended_agent_card=server.extended_agent_card,
        extended_card_modifier=server.extended_card_modifier,
    )

    routes = [
        *create_agent_card_routes(
            server.agent_card,
            card_modifier=server.card_modifier,
        ),
        *create_jsonrpc_routes(
            handler,
            rpc_url=server.rpc_url,
            enable_v0_3_compat=server.compat_earlier_protocol,
        ),
    ]
    return Starlette(routes=routes)


__all__ = ["build_starlette_app"]
