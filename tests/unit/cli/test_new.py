"""Tests for ``troopai new`` — every scaffold must validate with no edits."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from troopai.adk.cli import main


def test_agent_scaffold_resolves(tmp_path: Path) -> None:
    result = CliRunner().invoke(main, ["new", "demo_agent", "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    project = tmp_path / "demo_agent"
    for expected in ("agent.json", "demo_agent_tools.py", "agent_config.schema.json", ".env.example", "README.md"):
        assert (project / expected).is_file(), expected

    check = CliRunner().invoke(main, ["validate", "--resolve", str(project / "agent.json")])
    assert check.exit_code == 0, check.output


def test_topology_scaffold_validates(tmp_path: Path) -> None:
    result = CliRunner().invoke(main, ["new", "demo_team", "--kind", "topology", "--dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    project = tmp_path / "demo_team"
    assert (project / "topology.json").is_file()
    assert (project / "topology_config.schema.json").is_file()

    check = CliRunner().invoke(main, ["validate", "--resolve", str(project / "topology.json")])
    assert check.exit_code == 0, check.output


def test_invalid_name_rejected(tmp_path: Path) -> None:
    result = CliRunner().invoke(main, ["new", "Bad-Name", "--dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "NAME" in result.output


def test_existing_nonempty_directory_rejected(tmp_path: Path) -> None:
    taken = tmp_path / "demo_agent"
    taken.mkdir()
    (taken / "keep.txt").write_text("precious", encoding="utf-8")
    result = CliRunner().invoke(main, ["new", "demo_agent", "--dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "not empty" in result.output
    assert (taken / "keep.txt").read_text(encoding="utf-8") == "precious"


def test_existing_file_at_target_path_rejected(tmp_path: Path) -> None:
    # A regular file where the project directory would go must raise a clean
    # UsageError, not a NotADirectoryError from iterdir().
    clash = tmp_path / "demo_agent"
    clash.write_text("i am a file", encoding="utf-8")
    result = CliRunner().invoke(main, ["new", "demo_agent", "--dir", str(tmp_path)])
    assert result.exit_code == 2
    assert "not a directory" in result.output
    assert clash.read_text(encoding="utf-8") == "i am a file"
