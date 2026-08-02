"""Command-line interface for the TroopAI ADK.

The ``troopai`` console script (also runnable as ``python -m
troopai.adk.cli``) drives agents built with this ADK from the terminal:
running and chatting with agents declared in JSON/YAML config files or
referenced as Python objects, validating configs against the published
schemas, scaffolding new agent projects, inspecting session stores, and
serving an agent over the A2A protocol.

The CLI is a consumer of the ADK's public API only — it loads through
:func:`~troopai.adk.config.load_agent` / ``load_topology`` or a dotted
reference, executes through :class:`~troopai.adk.run.runner.Runner`, and
persists through the session manager. Command results are written to
stdout (pipeable); diagnostics go through :mod:`logging`. Every
cost-affecting behavior (sessions, verbose rendering, tracing, env-file
loading) is off until its flag is passed.
"""

from __future__ import annotations

import click

from troopai.adk import __version__, setup_logging


@click.group(name="troopai")
@click.version_option(version=__version__, prog_name="troopai")
def main() -> None:
    """TroopAI ADK — run, chat with, validate, scaffold, and serve agents."""
    setup_logging()


# Command registrations live below the group so each command module can
# import the group's siblings (options, errors, loading) without cycles.
from troopai.adk.cli.chat import chat
from troopai.adk.cli.deploy import deploy
from troopai.adk.cli.new import new
from troopai.adk.cli.run import run
from troopai.adk.cli.schema import schema
from troopai.adk.cli.serve import serve
from troopai.adk.cli.sessions import sessions
from troopai.adk.cli.validate import validate

main.add_command(chat)
main.add_command(deploy)
main.add_command(new)
main.add_command(run)
main.add_command(schema)
main.add_command(serve)
main.add_command(sessions)
main.add_command(validate)
