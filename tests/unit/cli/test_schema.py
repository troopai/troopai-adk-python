"""Tests for ``troopai schema``."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from troopai.adk.cli import main
from troopai.adk.config.schema import (
    dump_agent_config_schema,
    dump_agent_node_config_schema,
    dump_topology_config_schema,
)


def test_agent_schema_matches_dump() -> None:
    result = CliRunner().invoke(main, ["schema", "agent"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == dump_agent_config_schema()


def test_node_schema_matches_dump() -> None:
    result = CliRunner().invoke(main, ["schema", "node"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == dump_agent_node_config_schema()


def test_topology_schema_matches_dump() -> None:
    result = CliRunner().invoke(main, ["schema", "topology"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == dump_topology_config_schema()


def test_out_writes_file(tmp_path: Path) -> None:
    target = tmp_path / "agent.schema.json"
    result = CliRunner().invoke(main, ["schema", "agent", "--out", str(target)])
    assert result.exit_code == 0, result.output
    assert json.loads(target.read_text(encoding="utf-8")) == dump_agent_config_schema()
    assert str(target) in result.output
