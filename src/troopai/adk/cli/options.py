"""Shared click parameter decorators used across ``troopai`` commands.

Each decorator attaches one cohesive flag set so command signatures stay
declarative and consistent. Flags whose absence must leave framework
defaults untouched (``--model``, ``--max-turns``) default to ``None`` —
commands only forward a value the user actually passed, so the CLI never
restates (and never drifts from) a framework default.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import click


def target_options[F: Callable[..., object]](f: F) -> F:
    """Attach the run-target parameters: a CONFIG path or ``--agent`` ref.

    ``CONFIG`` is an optional positional path to a ``.json`` / ``.yaml`` /
    ``.yml`` agent or topology file. ``--agent`` references a Python object
    (``module:variable``) instead. Commands require exactly one of the two
    via :func:`troopai.adk.cli.loading.resolve_target`.
    """
    f = click.option(
        "--agent",
        "agent_ref",
        metavar="MODULE:VAR",
        default=None,
        help="Dotted reference to an Agent/Swarm/Graph object, e.g. 'my_pkg.agents:support'.",
    )(f)
    # No eager exists=True here: with --agent, a lone positional is the
    # prompt, not a config path — commands reconcile via
    # ``loading.reconcile_positionals`` and existence is checked at load.
    f = click.argument(
        "config",
        required=False,
        type=click.Path(path_type=Path),
    )(f)
    return f


def run_options[F: Callable[..., object]](f: F) -> F:
    """Attach execution flags shared by ``run`` and ``chat``.

    Every flag is opt-in; omitted flags forward nothing, leaving framework
    defaults in charge.
    """
    f = click.option(
        "--env-file",
        type=click.Path(exists=True, dir_okay=False, path_type=Path),
        default=None,
        help="Load KEY=VALUE pairs from this file into the environment (never auto-discovered).",
    )(f)
    f = click.option(
        "--trace",
        is_flag=True,
        default=False,
        help="Enable framework tracing with a console span exporter (requires the [otel] extra).",
    )(f)
    f = click.option(
        "--verbose",
        is_flag=True,
        default=False,
        help="Render the run verbosely (Rich when the [verbose] extra is installed, ANSI otherwise).",
    )(f)
    f = click.option(
        "--max-turns",
        type=int,
        default=None,
        help="Per-agent loop turn limit (framework default when omitted).",
    )(f)
    f = click.option(
        "--model",
        default=None,
        help="Override the model for this invocation.",
    )(f)
    return f


def session_options[F: Callable[..., object]](f: F) -> F:
    """Attach session-persistence flags shared by ``run`` and ``chat``.

    Persistence activates only when ``--session-db`` is passed; the id and
    user default so a single flag is enough to opt in.
    """
    f = click.option(
        "--user-id",
        default="default",
        show_default=True,
        help="User scope for the session.",
    )(f)
    f = click.option(
        "--session-id",
        default="default",
        show_default=True,
        help="Session id to create or resume.",
    )(f)
    f = click.option(
        "--session-db",
        type=click.Path(dir_okay=False, path_type=Path),
        default=None,
        help="SQLite file for session persistence; omit to keep the conversation in memory.",
    )(f)
    return f


def output_option[F: Callable[..., object]](f: F) -> F:
    """Attach ``--output text|json`` for commands with a machine-readable mode."""
    f = click.option(
        "--output",
        type=click.Choice(["text", "json"]),
        default="text",
        show_default=True,
        help="Result format on stdout.",
    )(f)
    return f
