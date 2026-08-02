"""``troopai new`` — scaffold a runnable agent project.

Generates a config (with a ``$schema`` pointer to a schema file written
beside it), a sibling tools module the config references, an env-file
example, and a README — a project ``troopai validate --resolve`` passes
with no edits.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import click

from troopai.adk.cli.templates import render_project

logger = logging.getLogger(__name__)

NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")


@click.command(name="new")
@click.argument("name")
@click.option(
    "--kind",
    type=click.Choice(["agent", "topology"]),
    default="agent",
    show_default=True,
    help="Scaffold a single agent or a multi-agent topology.",
)
@click.option(
    "--dir",
    "parent_dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("."),
    help="Parent directory to create the project in.",
)
def new(name: str, kind: str, parent_dir: Path) -> None:
    """Scaffold a new project named NAME in its own directory."""
    if NAME_PATTERN.fullmatch(name) is None:
        raise click.UsageError("NAME must match [a-z][a-z0-9_]* — it becomes the directory and agent name.")
    target_dir = parent_dir / name
    if target_dir.exists():
        if not target_dir.is_dir():
            raise click.UsageError(f"Path {str(target_dir)!r} already exists and is not a directory.")
        if any(target_dir.iterdir()):
            raise click.UsageError(f"Directory {str(target_dir)!r} already exists and is not empty.")
    target_dir.mkdir(parents=True, exist_ok=True)

    files = render_project(kind, name)
    for filename, content in files.items():
        (target_dir / filename).write_text(content, encoding="utf-8")
        click.echo(str(target_dir / filename))

    logger.debug("scaffolded %s project at %s", kind, target_dir)
    config_name = "topology.json" if kind == "topology" else "agent.json"
    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  troopai validate {target_dir / config_name}")
    click.echo(f'  troopai run {target_dir / config_name} "hello"')
