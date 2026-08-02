"""``troopai sessions`` — inspect and prune SQLite session stores.

Operates on the same stores ``troopai run --session-db`` and
``troopai chat --session-db`` write. The store is app-scoped, so every
subcommand takes the ``--app-name`` the sessions were written under
(the agent name, a swarm's entry-agent name, or a graph id).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click

if TYPE_CHECKING:
    from collections.abc import Callable

    from troopai.adk.session.session_event import SessionEvent
    from troopai.adk.session.sqlite_multi_sessions import SessionInfo

logger = logging.getLogger(__name__)

CONTENT_PREVIEW_CHARS = 200


def store_options[F: "Callable[..., object]"](f: F) -> F:
    """Attach the store-addressing options every subcommand needs."""
    f = click.option(
        "--app-name",
        required=True,
        help="Application scope the sessions were written under (the run/chat target name).",
    )(f)
    f = click.option(
        "--db",
        required=True,
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        help="SQLite session store file.",
    )(f)
    return f


@click.group(name="sessions")
def sessions() -> None:
    """Inspect and prune the session stores run/chat write."""


@sessions.command(name="list")
@store_options
@click.option("--user-id", default=None, help="Filter by user; omit to list every user's sessions.")
def sessions_list(db: Path, app_name: str, user_id: str | None) -> None:
    """List sessions in the store, oldest first."""
    infos = asyncio.run(_list_infos(db, app_name, user_id))
    logger.debug("listed %d sessions from %s", len(infos), db)
    if len(infos) == 0:
        click.echo("no sessions")
        return
    for info in infos:
        click.echo(f"{info.session_id}\tuser={info.user_id}\tcreated={info.created_at}\tupdated={info.updated_at}")


@sessions.command(name="show")
@store_options
@click.option("--id", "session_id", required=True, help="Session id to render.")
@click.option("--user-id", default="default", show_default=True, help="User scope of the session.")
@click.option("--limit", type=int, default=None, help="Maximum number of events to render (all when omitted).")
@click.option(
    "--output",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Render format on stdout.",
)
def sessions_show(db: Path, app_name: str, session_id: str, user_id: str, limit: int | None, output: str) -> None:
    """Render a session's conversation events."""
    events = asyncio.run(_load_events(db, app_name, session_id, user_id, limit))
    if output == "json":
        payload = [{"id": e.id, "author": e.author, "timestamp": e.timestamp, "content": e.content} for e in events]
        click.echo(json.dumps(payload, default=str))
        return
    if len(events) == 0:
        click.echo("(empty session)")
        return
    for event in events:
        click.echo(f"[{event.author}] {_preview(event)}")


@sessions.command(name="delete")
@store_options
@click.option("--id", "session_id", required=True, help="Session id to delete.")
@click.option("--user-id", default="default", show_default=True, help="User scope of the session.")
@click.option("--yes", is_flag=True, default=False, help="Skip the confirmation prompt.")
def sessions_delete(db: Path, app_name: str, session_id: str, user_id: str, yes: bool) -> None:
    """Delete one session and all its messages."""
    if not yes:
        click.confirm(f"Delete session {session_id!r} (user {user_id!r}) from {db}?", abort=True)
    deleted = asyncio.run(_delete(db, app_name, session_id, user_id))
    if not deleted:
        raise click.UsageError(f"No session {session_id!r} for user {user_id!r} in {db} (app {app_name!r}).")
    click.echo(f"deleted {session_id}")


def _preview(event: SessionEvent) -> str:
    """One-line, length-bounded rendering of an event's content."""
    rendered = json.dumps(event.content, default=str)
    if len(rendered) > CONTENT_PREVIEW_CHARS:
        return rendered[:CONTENT_PREVIEW_CHARS] + "…"
    return rendered


async def _list_infos(db: Path, app_name: str, user_id: str | None) -> list[SessionInfo]:
    """List session metadata, closing the manager deterministically."""
    from troopai.adk.session.sqlite_multi_sessions import SQLiteMultiSessions

    manager = SQLiteMultiSessions(path=db, app_name=app_name)
    try:
        return await manager.list(user_id=user_id)
    finally:
        await manager.close()


async def _load_events(db: Path, app_name: str, session_id: str, user_id: str, limit: int | None) -> list[Any]:
    """Load a session's events or fail with a guiding usage error."""
    from troopai.adk.session.sqlite_multi_sessions import SQLiteMultiSessions

    manager = SQLiteMultiSessions(path=db, app_name=app_name)
    try:
        session = await manager.get(session_id, user_id=user_id)
        if session is None:
            raise click.UsageError(f"No session {session_id!r} for user {user_id!r} in {db} (app {app_name!r}).")
        return await session.get(limit)
    finally:
        await manager.close()


async def _delete(db: Path, app_name: str, session_id: str, user_id: str) -> bool:
    """Delete a session; ``True`` when it existed."""
    from troopai.adk.session.sqlite_multi_sessions import SQLiteMultiSessions

    manager = SQLiteMultiSessions(path=db, app_name=app_name)
    try:
        return await manager.delete(session_id, user_id=user_id)
    finally:
        await manager.close()
