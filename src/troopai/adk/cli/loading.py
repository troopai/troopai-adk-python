"""Resolve CLI invocation targets into runnable framework objects.

Commands accept either a config file path or a dotted ``module:var``
reference; this module owns the classification, resolution, and dispatch
rules they share, plus the explicit ``--env-file`` loader. Dynamic import
goes through the config resolver — the framework's single sanctioned
dotted-reference boundary — never a bespoke ``importlib`` call.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import click

if TYPE_CHECKING:
    from troopai.adk.agents.agent import Agent
    from troopai.adk.config.topology import AgentTopology
    from troopai.adk.graphs import Graph
    from troopai.adk.swarms import Swarm

logger = logging.getLogger(__name__)

ConfigKind = Literal["agent", "topology"]

type RunnableTarget = Agent | Swarm | Graph | AgentTopology


def detect_config_kind(data: Mapping[str, object]) -> ConfigKind:
    """Classify a parsed config document.

    A file is classified as a topology when it contains an ``"agents"``
    key whose value is a mapping (dict).  A bare ``"agents"`` key whose
    value is not a dict (or whose presence is accidental in an agent
    config) falls through to ``"agent"`` so that ``load_agent`` produces
    a schema-accurate error message rather than a confusing
    ``"Invalid topology config"`` error.

    Args:
        data: The root mapping of a parsed JSON/YAML config file.

    Returns:
        ``"topology"`` when the root declares an ``agents`` mapping,
        else ``"agent"``.
    """
    agents_value = data.get("agents")
    return "topology" if isinstance(agents_value, Mapping) else "agent"


def resolve_target(config: Path | None, agent_ref: str | None) -> RunnableTarget:
    """Turn a CLI target — config path or dotted reference — into an object.

    Args:
        config: Path to a ``.json`` / ``.yaml`` / ``.yml`` agent or topology
            file, when the command received one.
        agent_ref: A ``module:var`` (or ``module.var``) reference to an
            ``Agent`` / ``Swarm`` / ``Graph`` object, when ``--agent`` was
            passed. Resolved with the current working directory importable,
            mirroring how config-relative references resolve.

    Returns:
        The loaded ``Agent``, ``Swarm``, ``Graph``, or ``AgentTopology``.

    Raises:
        click.UsageError: If both or neither input is provided, or the
            reference resolves to a non-runnable object.
        ConfigParseError: If the config file fails parsing or validation.
        ConfigResolutionError: If a dotted reference cannot be imported.
    """
    if config is not None and agent_ref is not None:
        raise click.UsageError("Pass either a CONFIG file or --agent, not both.")
    if config is not None:
        return _load_config_target(config)
    if agent_ref is not None:
        return _load_object_target(agent_ref)
    raise click.UsageError("Pass a CONFIG file or --agent MODULE:VAR.")


def primary_executable(target: RunnableTarget) -> Agent | Swarm | Graph:
    """Pick the executable a command should drive for ``target``.

    Args:
        target: The object ``resolve_target`` produced.

    Returns:
        The object itself for ``Agent`` / ``Swarm`` / ``Graph``; for an
        ``AgentTopology``, deterministic precedence — its ``graph`` if
        declared, else its ``swarm``, else its ``entry`` agent.

    Raises:
        click.UsageError: If a topology declares none of graph, swarm, or
            entry.
    """
    from troopai.adk.config.topology import AgentTopology

    if not isinstance(target, AgentTopology):
        return target
    if target.graph is not None:
        return target.graph
    if target.swarm is not None:
        return target.swarm
    if target.entry is not None:
        return target.agents[target.entry]
    raise click.UsageError("Topology declares no graph, swarm, or entry agent; add one to make it runnable.")


def load_env_file(path: Path) -> None:
    """Load ``KEY=VALUE`` pairs from ``path`` into the environment.

    Only runs when the user passes ``--env-file`` — never auto-discovered.
    Existing environment variables win (``setdefault``); blank lines and
    ``#`` comments are skipped; surrounding single/double quotes on values
    are stripped.

    Args:
        path: The env file to read (UTF-8).

    Raises:
        click.UsageError: For a line without ``=`` or with an empty key,
            naming the offending line number.
    """
    text = path.read_text(encoding="utf-8")
    loaded = 0
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        if "=" not in line:
            raise click.UsageError(f"{path}:{lineno}: expected KEY=VALUE, got {raw_line!r}")
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(key) == 0:
            raise click.UsageError(f"{path}:{lineno}: empty key before '='")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
        loaded += 1
    logger.debug("Loaded %d env entries from %s", loaded, path)


def reconcile_positionals(
    config: Path | None, agent_ref: str | None, prompt: str | None
) -> tuple[Path | None, str | None]:
    """Disambiguate the two trailing positionals of run-style commands.

    ``troopai run CONFIG PROMPT`` and ``troopai run --agent REF PROMPT`` share
    the same positional slots, so with ``--agent`` a lone positional binds to
    ``CONFIG`` even though the user meant the prompt.

    Args:
        config: The first positional, as parsed.
        agent_ref: The ``--agent`` value, if passed.
        prompt: The second positional, as parsed.

    Returns:
        The ``(config, prompt)`` pair with that case re-interpreted.
    """
    if agent_ref is not None and prompt is None and config is not None:
        return None, str(config)
    return config, prompt


def _load_config_target(config: Path) -> RunnableTarget:
    """Load a config file as an ``Agent`` or ``AgentTopology`` by kind."""
    from troopai.adk.config.loader import load_agent, read_config_document
    from troopai.adk.config.topology import load_topology

    if not config.is_file():
        raise click.UsageError(f"Config file {str(config)!r} does not exist.")
    data = read_config_document(config)
    if detect_config_kind(data) == "topology":
        return load_topology(config, document=data)
    return load_agent(config, document=data)


def _load_object_target(agent_ref: str) -> Agent | Swarm | Graph:
    """Resolve ``--agent module:var`` to a runnable framework object."""
    from troopai.adk.agents.agent import Agent
    from troopai.adk.config.resolver import importable_dir, resolve_dotted_spec
    from troopai.adk.graphs import Graph
    from troopai.adk.swarms import Swarm

    with importable_dir(Path.cwd()):
        obj = resolve_dotted_spec(agent_ref)
    if isinstance(obj, (Agent, Swarm, Graph)):
        return obj
    raise click.UsageError(
        f"--agent {agent_ref!r} resolved to {type(obj).__name__}; expected an Agent, Swarm, or Graph."
    )
