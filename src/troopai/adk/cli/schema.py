"""``troopai schema`` — print the published config JSON Schemas.

Exposes the same generated schemas that ship inside the package, so
editors and CI can validate config files without locating the installed
package directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command(name="schema")
@click.argument("kind", type=click.Choice(["agent", "node", "topology"]), default="agent")
@click.option(
    "--out",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    default=None,
    help="Write the schema to this file instead of stdout.",
)
def schema(kind: str, out: Path | None) -> None:
    """Print the JSON Schema for a config KIND (agent, node, or topology)."""
    from troopai.adk.config.schema import (
        dump_agent_config_schema,
        dump_agent_node_config_schema,
        dump_topology_config_schema,
    )

    dumpers = {
        "agent": dump_agent_config_schema,
        "node": dump_agent_node_config_schema,
        "topology": dump_topology_config_schema,
    }
    logger.debug("dumping %s config schema", kind)
    document = json.dumps(dumpers[kind](), indent=2)
    if out is not None:
        out.write_text(document + "\n", encoding="utf-8")
        click.echo(str(out))
    else:
        click.echo(document)
