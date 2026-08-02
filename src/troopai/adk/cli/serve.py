"""``troopai serve`` — expose a local Agent over HTTP.

Serves a single Agent through the framework's serving layer
(:func:`troopai.adk.serving.build_app`) under uvicorn. By default it
exposes the plain-REST surface (``POST /run``, ``POST /run_sse``) and the
health routes (``GET /healthz``, ``GET /readyz``); passing ``--card``
additionally publishes the A2A JSON-RPC + discovery surface.

This is the one command that owns an ASGI runtime — the framework itself
never imports uvicorn. Everything stays behind the ``serve`` extra.

For containers, bind all interfaces and read the platform port::

    troopai serve --agent app:agent --host 0.0.0.0 --port "$PORT"

State stores: ``--task-db`` / ``--session-db`` use SQLite (single replica);
``--task-dsn`` / ``--session-dsn`` use Postgres so A2A tasks and REST
sessions are shared across replicas in a horizontally-scaled deployment.
"""

from __future__ import annotations

import contextlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import click

from troopai.adk.cli.errors import framework_errors
from troopai.adk.cli.loading import load_env_file, primary_executable, resolve_target
from troopai.adk.cli.options import target_options

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from a2a.types import AgentCard
    from starlette.applications import Starlette

    from troopai.adk.a2a.server import A2AServer
    from troopai.adk.agents.agent import Agent
    from troopai.adk.session.multi_sessions import MultiSessions
    from troopai.adk.types.session.store import SessionStore

logger = logging.getLogger(__name__)


class _DurableTaskStore(Protocol):
    """A durable A2A task store recovered on the serving event loop.

    Both the SQLite and Postgres task stores satisfy this: their restart
    recovery must run on the loop that serves requests so any connection
    pool it opens is reused there, not stranded on a bootstrap loop.
    """

    async def recover_on_startup(self) -> int:
        """Mark tasks a prior process left non-terminal as FAILED.

        Returns:
            The number of tasks transitioned to FAILED.
        """
        ...


SERVE_INSTALL_HINT = "troopai serve requires the serve extra. Install with: pip install 'troopai-adk-python[serve]'"
A2A_INSTALL_HINT = "the A2A surface requires the a2a extra. Install with: pip install 'troopai-adk-python[a2a]'"
A2A_PG_HINT = "--task-dsn requires the a2a-postgres extra. Install with: pip install 'troopai-adk-python[a2a-postgres]'"
SESSION_PG_HINT = "--session-dsn requires the session-postgres extra. Install with: pip install 'troopai-adk-python[session-postgres]'"


@click.command(name="serve")
@target_options
@click.option(
    "--card",
    "card_path",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Developer-authored A2A AgentCard JSON. Enables the A2A surface (served at /.well-known/agent-card.json).",
)
@click.option("--rest/--no-rest", default=True, show_default=True, help="Expose POST /run and POST /run_sse.")
@click.option("--health/--no-health", default=True, show_default=True, help="Expose GET /healthz and GET /readyz.")
@click.option("--a2a/--no-a2a", "a2a", default=False, show_default=True, help="Expose the A2A surface (needs --card).")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address. Use 0.0.0.0 inside a container.")
@click.option("--port", default=8000, show_default=True, type=int, help="Bind port.")
@click.option(
    "--max-turns",
    type=int,
    default=None,
    help="Per-request agent loop turn limit (framework default when omitted).",
)
@click.option(
    "--task-db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite file for durable A2A task storage (single replica). In-memory when omitted.",
)
@click.option(
    "--task-dsn",
    default=None,
    help="Postgres DSN for a shared, durable A2A task store (multi-replica). Excludes --task-db.",
)
@click.option(
    "--session-db",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="SQLite file backing REST sessions (single replica). Excludes --session-dsn.",
)
@click.option(
    "--session-dsn",
    default=None,
    help="Postgres DSN for shared REST sessions across replicas (multi-replica). Excludes --session-db.",
)
@click.option(
    "--env-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Load KEY=VALUE pairs from this file into the environment (never auto-discovered).",
)
@framework_errors
def serve(
    config: Path | None,
    agent_ref: str | None,
    card_path: Path | None,
    rest: bool,
    health: bool,
    a2a: bool,
    host: str,
    port: int,
    max_turns: int | None,
    task_db: Path | None,
    task_dsn: str | None,
    session_db: Path | None,
    session_dsn: str | None,
    env_file: Path | None,
) -> None:
    """Serve a single Agent over HTTP (REST + health by default; A2A with --card)."""
    if env_file is not None:
        load_env_file(env_file)

    want_a2a = a2a or card_path is not None
    if not rest and not health and not want_a2a:
        raise click.UsageError("Nothing to serve: enable at least one of --rest, --health, or --a2a (with --card).")
    if task_db is not None and task_dsn is not None:
        raise click.UsageError("--task-db and --task-dsn are mutually exclusive; choose one A2A task store.")
    if session_db is not None and session_dsn is not None:
        raise click.UsageError("--session-db and --session-dsn are mutually exclusive; choose one session store.")

    from troopai.adk.serving import build_app

    if build_app is None:
        raise click.UsageError(SERVE_INSTALL_HINT)
    try:
        import uvicorn
    except ImportError as exc:
        raise click.UsageError(SERVE_INSTALL_HINT) from exc

    executable = primary_executable(resolve_target(config, agent_ref))
    from troopai.adk.agents.agent import Agent

    if not isinstance(executable, Agent):
        raise click.UsageError("serve exposes a single agent; swarms and graphs have no HTTP server form.")

    a2a_server: A2AServer | None = None
    executor_store: _DurableTaskStore | None = None
    if want_a2a:
        if card_path is None:
            raise click.UsageError("--a2a requires --card (the developer-authored AgentCard JSON).")
        executor_store = _open_task_store(task_db, task_dsn)
        a2a_server = _build_a2a_server(executable, card_path, max_turns, executor_store)

    manager = _build_session_manager(executable, session_db, session_dsn)
    session_factory = _session_factory(manager) if manager is not None else None

    app = build_app(
        executable,
        rest=rest,
        health=health,
        a2a_server=a2a_server,
        max_turns=max_turns,
        session_factory=session_factory,
    )
    _install_lifespan(app, executor_store, manager)
    _echo_endpoints(executable, host, port, rest=rest, health=health, a2a=want_a2a)
    uvicorn.run(app, host=host, port=port)


def _build_a2a_server(
    agent: Agent,
    card_path: Path,
    max_turns: int | None,
    executor_store: _DurableTaskStore | None,
) -> A2AServer:
    """Build the :class:`A2AServer` config around an opened task store.

    Args:
        agent: The local agent to expose.
        card_path: Path to the developer-authored AgentCard JSON.
        max_turns: Per-task agent-loop budget, or ``None`` for the default.
        executor_store: The durable task store the executor persists to,
            or ``None`` for the in-memory default.

    Returns:
        A frozen :class:`A2AServer` config the serving app mounts.

    Raises:
        click.UsageError: If the ``a2a`` extra is not installed.
    """
    from troopai.adk.a2a import A2AServer

    if A2AServer is None:
        raise click.UsageError(A2A_INSTALL_HINT)
    card = _load_card(card_path)
    if max_turns is not None:
        return A2AServer(agent=agent, agent_card=card, executor_task_store=executor_store, max_turns=max_turns)
    return A2AServer(agent=agent, agent_card=card, executor_task_store=executor_store)


def _install_lifespan(
    app: Starlette,
    executor_store: _DurableTaskStore | None,
    manager: MultiSessions | None,
) -> None:
    """Attach a lifespan so durable stores live on the serving event loop.

    The A2A task store's restart recovery and the session manager's
    connection pool must run on the same loop uvicorn serves requests on.
    Doing that work in the ASGI lifespan (which uvicorn drives) keeps
    every pool bound to the serving loop, instead of a short-lived
    bootstrap loop that is closed before the first request — and closes
    the pool on that same loop at shutdown.

    Args:
        app: The built serving app to attach the lifespan to.
        executor_store: The durable A2A task store to recover on startup,
            or ``None`` for the in-memory default.
        manager: The REST session manager to close on shutdown, or
            ``None`` when sessions are off.
    """

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncIterator[None]:
        if executor_store is not None:
            recovered = await executor_store.recover_on_startup()
            logger.debug("A2A task store recovery marked %d task(s) FAILED", recovered)
        try:
            yield
        finally:
            if manager is not None:
                await manager.close()

    app.router.lifespan_context = lifespan


def _open_task_store(task_db: Path | None, task_dsn: str | None) -> _DurableTaskStore | None:
    """Construct a durable A2A task store, or ``None`` for in-memory.

    The store's restart recovery is deferred to the serving lifespan
    (:func:`_install_lifespan`), so any connection pool it opens is bound
    to the loop that serves requests rather than a bootstrap loop that is
    closed before the first request.

    Args:
        task_db: SQLite file path, or ``None``.
        task_dsn: Postgres DSN, or ``None``.

    Returns:
        The durable task store, or ``None`` when neither is set.

    Raises:
        click.UsageError: If ``--task-dsn`` is used without the
            ``a2a-postgres`` extra.
    """
    if task_dsn is not None:
        try:
            from troopai.adk.a2a.postgres_task_store import PostgresTaskStore
        except ImportError as exc:
            raise click.UsageError(A2A_PG_HINT) from exc
        logger.debug("shared A2A task store configured (Postgres)")
        return PostgresTaskStore(task_dsn)
    if task_db is not None:
        from troopai.adk.a2a.task_store import SQLiteTaskStore
        from troopai.adk.databases.connections.sqlite import SQLiteDatabaseConnection

        logger.debug("durable A2A task store configured at %s", task_db)
        return SQLiteTaskStore(SQLiteDatabaseConnection(path=task_db))
    return None


def _build_session_manager(
    agent: Agent,
    session_db: Path | None,
    session_dsn: str | None,
) -> MultiSessions | None:
    """Build the REST session manager, or ``None`` when sessions are off.

    Args:
        agent: The served agent (its name scopes the session app-name).
        session_db: SQLite file path, or ``None``.
        session_dsn: Postgres DSN, or ``None``.

    Returns:
        A multi-session manager, or ``None`` when neither store is set.

    Raises:
        click.UsageError: If ``--session-dsn`` is used without the
            ``session-postgres`` extra.
    """
    if session_dsn is not None:
        try:
            from troopai.adk.session.postgres_multi_sessions import PostgresMultiSessions
        except ImportError as exc:
            raise click.UsageError(SESSION_PG_HINT) from exc
        logger.debug("shared REST session store opened (Postgres)")
        return PostgresMultiSessions(session_dsn, app_name=agent.name)
    if session_db is not None:
        from troopai.adk.session.sqlite_multi_sessions import SQLiteMultiSessions

        logger.debug("REST session store opened at %s", session_db)
        return SQLiteMultiSessions(path=session_db, app_name=agent.name)
    return None


def _session_factory(manager: MultiSessions) -> Callable[[str, str], Awaitable[SessionStore]]:
    """Wrap a session manager as the REST surface's per-request factory.

    Args:
        manager: The multi-session manager backing the REST surface.

    Returns:
        An async ``(user_id, session_id) -> SessionStore`` factory.
    """

    async def factory(user_id: str, session_id: str) -> SessionStore:
        return await manager.get_or_create(session_id, user_id=user_id)

    return factory


def _echo_endpoints(agent: Agent, host: str, port: int, *, rest: bool, health: bool, a2a: bool) -> None:
    """Print the served base URL and the enabled routes.

    Args:
        agent: The agent being served (its name labels the output).
        host: Bind address.
        port: Bind port.
        rest: Whether the REST surface is enabled.
        health: Whether the health routes are enabled.
        a2a: Whether the A2A surface is enabled.
    """
    base = f"http://{host}:{port}"
    click.echo(f"Serving agent {agent.name!r} on {base}")
    if rest:
        click.echo(f"  REST    POST {base}/run, POST {base}/run_sse (SSE)")
    if health:
        click.echo(f"  Health  GET {base}/healthz, GET {base}/readyz")
    if a2a:
        click.echo(f"  A2A     POST {base}/ (JSON-RPC), {base}/.well-known/agent-card.json")


def _load_card(card_path: Path) -> AgentCard:
    """Parse and validate the developer-authored AgentCard JSON.

    ``AgentCard`` is a protobuf message, so the JSON parses through
    ``google.protobuf.json_format`` (camelCase field names, strict on
    unknown fields).

    Args:
        card_path: Path to the AgentCard JSON file.

    Returns:
        The parsed :class:`a2a.types.AgentCard`.

    Raises:
        click.UsageError: If the file is not valid AgentCard JSON.
    """
    from a2a.types import AgentCard
    from google.protobuf import json_format

    try:
        data = json.loads(card_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise click.UsageError(f"Invalid JSON in agent card {str(card_path)!r}: {exc}") from exc
    try:
        return json_format.ParseDict(data, AgentCard())
    except json_format.ParseError as exc:
        raise click.UsageError(f"Invalid agent card in {str(card_path)!r}: {exc}") from exc
