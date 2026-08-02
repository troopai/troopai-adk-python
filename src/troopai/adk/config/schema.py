"""Generate the JSON Schema for the agent config format.

The Pydantic models are the single source of truth. This module derives a
JSON Schema from them and writes it next to the models so it can be published
for editor and CI validation (the artifact a config file's ``$schema`` key
points at). A test asserts the committed file stays in sync with the models.

Regenerate the committed schema with::

    python -m troopai.adk.config.schema
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from troopai.adk.types.config.agent_config import AgentConfig
from troopai.adk.types.config.topology_config import AgentNodeConfig, TopologyConfig

logger = logging.getLogger(__name__)

_SCHEMA_DIR: Path = Path(__file__).resolve().parent.parent / "types" / "config"

AGENT_CONFIG_SCHEMA_PATH: Path = _SCHEMA_DIR / "agent_config.schema.json"
"""Filesystem path of the committed JSON Schema generated from ``AgentConfig``."""

AGENT_NODE_CONFIG_SCHEMA_PATH: Path = _SCHEMA_DIR / "agent_node_config.schema.json"
"""Filesystem path of the committed JSON Schema generated from ``AgentNodeConfig``.

This is the schema a standalone sub-agent file (one referenced by a topology's
``config_path``) points its ``$schema`` at — an ``AgentConfig`` plus the
``handoffs`` field, so a file-sourced member that declares handoffs by name
still validates in an editor."""

TOPOLOGY_CONFIG_SCHEMA_PATH: Path = _SCHEMA_DIR / "topology_config.schema.json"
"""Filesystem path of the committed JSON Schema generated from ``TopologyConfig``."""


def dump_agent_config_schema() -> dict[str, Any]:
    """Return the JSON Schema for ``AgentConfig``, generated from the model.

    Returns:
        The JSON Schema as a plain dict, with field aliases (so ``$schema``
        appears under its alias rather than ``schema_ref``).
    """
    return AgentConfig.model_json_schema(by_alias=True)


def dump_agent_node_config_schema() -> dict[str, Any]:
    """Return the JSON Schema for ``AgentNodeConfig``, generated from the model.

    Returns:
        The JSON Schema as a plain dict, with field aliases. This is the schema
        for a standalone sub-agent file (``AgentConfig`` plus ``handoffs``).
    """
    return AgentNodeConfig.model_json_schema(by_alias=True)


def dump_topology_config_schema() -> dict[str, Any]:
    """Return the JSON Schema for ``TopologyConfig``, generated from the model.

    Returns:
        The JSON Schema as a plain dict, with field aliases.
    """
    return TopologyConfig.model_json_schema(by_alias=True)


def write_config_schemas() -> None:
    """Regenerate the committed JSON Schema files from the models."""
    AGENT_CONFIG_SCHEMA_PATH.write_text(json.dumps(dump_agent_config_schema(), indent=2) + "\n", encoding="utf-8")
    AGENT_NODE_CONFIG_SCHEMA_PATH.write_text(
        json.dumps(dump_agent_node_config_schema(), indent=2) + "\n", encoding="utf-8"
    )
    TOPOLOGY_CONFIG_SCHEMA_PATH.write_text(json.dumps(dump_topology_config_schema(), indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote config JSON Schemas to %s", str(_SCHEMA_DIR))


if __name__ == "__main__":
    write_config_schemas()
