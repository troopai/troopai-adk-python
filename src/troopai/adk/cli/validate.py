"""``troopai validate`` — check a config file without spending a token.

Schema validation alone is side-effect-free. ``--resolve`` additionally
assembles the config, so dotted references (tools, guardrails, output
schemas, providers) actually import — the same path ``troopai run`` takes,
minus execution.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click

from troopai.adk.cli.errors import framework_errors
from troopai.adk.cli.loading import ConfigKind, detect_config_kind

logger = logging.getLogger(__name__)


@click.command(name="validate")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--kind",
    type=click.Choice(["agent", "topology"]),
    default=None,
    help="Override document-kind detection (a root 'agents' key means topology).",
)
@click.option(
    "--resolve",
    "resolve_refs",
    is_flag=True,
    default=False,
    help="Also assemble the config so dotted references import.",
)
@framework_errors
def validate(config: Path, kind: ConfigKind | None, resolve_refs: bool) -> None:
    """Validate an agent or topology CONFIG file against the published schema."""
    from troopai.adk.config.loader import read_config_document

    data = read_config_document(config)
    effective_kind: ConfigKind = kind if kind is not None else detect_config_kind(data)
    logger.debug("validating %s as %s", config, effective_kind)

    summary = _assemble(config, effective_kind) if resolve_refs else _schema_check(config, data, effective_kind)
    click.echo(f"{summary}: OK")


def _schema_check(path: Path, data: dict[str, Any], kind: ConfigKind) -> str:
    """Validate ``data`` against the right schema model; describe the doc."""
    from pydantic import ValidationError

    from troopai.adk.exceptions import ConfigParseError

    if kind == "topology":
        from troopai.adk.types.config.topology_config import TopologyConfig

        try:
            topology_config = TopologyConfig.model_validate(data)
        except ValidationError as exc:
            raise ConfigParseError(f"Invalid topology config in {str(path)!r}: {exc}") from exc
        names = ", ".join(sorted(topology_config.agents))
        return f"topology ({len(topology_config.agents)} agents: {names})"

    from troopai.adk.types.config.agent_config import AgentConfig

    try:
        agent_config = AgentConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigParseError(f"Invalid agent config in {str(path)!r}: {exc}") from exc
    return f"agent {agent_config.name!r}"


def _assemble(path: Path, kind: ConfigKind) -> str:
    """Build the config for real so every dotted reference imports."""
    if kind == "topology":
        from troopai.adk.config.topology import load_topology

        topology = load_topology(path)
        names = ", ".join(sorted(topology.agents))
        return f"topology ({len(topology.agents)} agents: {names}) resolved"

    from troopai.adk.config.loader import load_agent

    agent = load_agent(path)
    return f"agent {agent.name!r} resolved"
