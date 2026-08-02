"""Tests for ``troopai validate``."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from troopai.adk.cli import main


def _write_config(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_agent_config_ok(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path / "agent.json", {"name": "helper", "system_prompt": "Be brief."})
    result = CliRunner().invoke(main, ["validate", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "helper" in result.output
    assert "OK" in result.output


def test_topology_config_ok(tmp_path: Path) -> None:
    payload = {
        "agents": {
            "triage": {"name": "triage", "system_prompt": "Route requests.", "handoffs": ["expert"]},
            "expert": {"name": "expert", "system_prompt": "Answer in depth."},
        },
        "entry": "triage",
    }
    cfg = _write_config(tmp_path / "topology.json", payload)
    result = CliRunner().invoke(main, ["validate", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "topology" in result.output
    assert "expert" in result.output


def test_typo_field_fails_with_guidance(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path / "agent.json", {"name": "helper", "system_promp": "typo"})
    result = CliRunner().invoke(main, ["validate", str(cfg)])
    assert result.exit_code == 2
    assert "system_promp" in result.output


def test_unsupported_extension_fails(tmp_path: Path) -> None:
    cfg = tmp_path / "agent.toml"
    cfg.write_text("name = 'x'", encoding="utf-8")
    result = CliRunner().invoke(main, ["validate", str(cfg)])
    assert result.exit_code == 2
    assert "Unsupported config extension" in result.output


def test_resolve_imports_sibling_tools_module(tmp_path: Path) -> None:
    (tmp_path / "cli_validate_fixture_tools.py").write_text(
        textwrap.dedent(
            '''
            from troopai.adk.tools import function_tool


            @function_tool
            def shout(text: str) -> str:
                """Uppercase the text.

                Args:
                    text: The text to uppercase.
                """
                return text.upper()
            '''
        ),
        encoding="utf-8",
    )
    cfg = _write_config(
        tmp_path / "agent.json",
        {
            "name": "shouter",
            "system_prompt": "Shout everything.",
            "tools": ["cli_validate_fixture_tools.shout"],
        },
    )
    result = CliRunner().invoke(main, ["validate", "--resolve", str(cfg)])
    assert result.exit_code == 0, result.output
    assert "resolved" in result.output


def test_resolve_reports_missing_tool_ref(tmp_path: Path) -> None:
    cfg = _write_config(
        tmp_path / "agent.json",
        {"name": "broken", "system_prompt": "x", "tools": ["nowhere_module.missing"]},
    )
    result = CliRunner().invoke(main, ["validate", "--resolve", str(cfg)])
    assert result.exit_code == 2
    assert "nowhere_module" in result.output


def test_kind_override_agent_on_topology_doc_fails(tmp_path: Path) -> None:
    payload = {"agents": {"a": {"name": "a", "system_prompt": "x"}}}
    cfg = _write_config(tmp_path / "topology.json", payload)
    result = CliRunner().invoke(main, ["validate", "--kind", "agent", str(cfg)])
    assert result.exit_code == 2
    assert "agents" in result.output


def test_kind_override_topology_on_agent_doc_fails(tmp_path: Path) -> None:
    cfg = _write_config(tmp_path / "agent.json", {"name": "solo", "system_prompt": "x"})
    result = CliRunner().invoke(main, ["validate", "--kind", "topology", str(cfg)])
    assert result.exit_code == 2
