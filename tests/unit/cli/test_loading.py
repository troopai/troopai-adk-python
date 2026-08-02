"""Tests for the CLI target-loading seam."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from typing import Any

import click
import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.cli.loading import (
    detect_config_kind,
    load_env_file,
    primary_executable,
    reconcile_positionals,
    resolve_target,
)
from troopai.adk.config.topology import AgentTopology, load_topology
from troopai.adk.graphs import Graph
from troopai.adk.swarms import Swarm

AGENT_NODE: dict[str, Any] = {"name": "author", "system_prompt": "Write."}
REVIEWER_NODE: dict[str, Any] = {"name": "reviewer", "system_prompt": "Review."}


def _write_topology(tmp_path: Path, extra: dict[str, Any]) -> Path:
    payload: dict[str, Any] = {"agents": {"author": AGENT_NODE, "reviewer": REVIEWER_NODE}, **extra}
    path = tmp_path / "topology.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ── detect_config_kind ───────────────────────────────────────────────


def test_detect_kind() -> None:
    assert detect_config_kind({"agents": {}}) == "topology"
    assert detect_config_kind({"name": "x"}) == "agent"


def test_detect_kind_agents_dict_value_required() -> None:
    """'agents' key with a non-dict value should NOT classify as topology.

    An agent config that mistakenly includes a root-level 'agents' key
    with a non-dict value (e.g. a list or a string) would previously be
    misclassified as a topology, producing a confusing 'Invalid topology
    config' error.  The discriminating check requires the value to be a
    Mapping.
    """
    # Non-dict values for 'agents' must classify as 'agent'
    assert detect_config_kind({"name": "x", "agents": "some-string"}) == "agent"
    assert detect_config_kind({"name": "x", "agents": ["a", "b"]}) == "agent"
    assert detect_config_kind({"name": "x", "agents": None}) == "agent"
    assert detect_config_kind({"name": "x", "agents": 42}) == "agent"
    # A dict value (the real topology pattern) still classifies as 'topology'
    assert detect_config_kind({"agents": {"author": {"name": "author"}}}) == "topology"


# ── resolve_target ───────────────────────────────────────────────────


def test_both_inputs_rejected(tmp_path: Path) -> None:
    cfg = tmp_path / "a.json"
    cfg.write_text("{}", encoding="utf-8")
    with pytest.raises(click.UsageError, match="not both"):
        resolve_target(cfg, "mod:var")


def test_neither_input_rejected() -> None:
    with pytest.raises(click.UsageError, match="CONFIG file or --agent"):
        resolve_target(None, None)


def test_agent_ref_resolves_from_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "cli_loading_fixture_agents.py").write_text(
        textwrap.dedent(
            """
            from troopai.adk.agents.agent import Agent

            support = Agent(name="support", system_prompt="Help politely.")
            not_runnable = "just a string"
            """
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    target = resolve_target(None, "cli_loading_fixture_agents:support")
    assert isinstance(target, Agent)
    assert target.name == "support"


def test_agent_ref_to_non_runnable_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "cli_loading_fixture_bad.py").write_text("thing = 42\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(click.UsageError, match="int"):
        resolve_target(None, "cli_loading_fixture_bad:thing")


def test_config_path_loads_agent(tmp_path: Path) -> None:
    cfg = tmp_path / "agent.json"
    cfg.write_text(json.dumps({"name": "solo", "system_prompt": "x"}), encoding="utf-8")
    target = resolve_target(cfg, None)
    assert isinstance(target, Agent)


def test_config_path_loads_topology(tmp_path: Path) -> None:
    cfg = _write_topology(tmp_path, {"entry": "author"})
    target = resolve_target(cfg, None)
    assert isinstance(target, AgentTopology)


def test_config_path_reads_file_exactly_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Kind detection and loading share one parse — no re-read between them.

    A second disk read would both waste IO and open a window where the file
    changes between the kind decision and the load.
    """
    from troopai.adk.config import loader as loader_module, topology as topology_module

    cfg = _write_topology(tmp_path, {"entry": "author"})
    calls: list[Path] = []
    real_read = loader_module.read_config_document

    def counting_read(path: Path) -> dict[str, Any]:
        calls.append(path)
        return real_read(path)

    # topology.py binds the name at import time, so patch both namespaces.
    monkeypatch.setattr(loader_module, "read_config_document", counting_read)
    monkeypatch.setattr(topology_module, "read_config_document", counting_read)
    target = resolve_target(cfg, None)
    assert isinstance(target, AgentTopology)
    assert len(calls) == 1, f"config parsed {len(calls)} times; expected exactly one read"


# ── primary_executable ───────────────────────────────────────────────


def test_agent_passes_through() -> None:
    agent = Agent(name="a", system_prompt="x")
    assert primary_executable(agent) is agent


def test_topology_graph_wins(tmp_path: Path) -> None:
    cfg = _write_topology(
        tmp_path,
        {
            "entry": "author",
            "swarm": {
                "members": ["author", "reviewer"],
                "entry": "author",
                "policy": {"type": "round_robin"},
                "termination": {"type": "max_turns", "limit": 2},
            },
            "graph": {
                "id": "flow",
                "nodes": {"author": {"agent": "author"}, "reviewer": {"agent": "reviewer"}},
                "edges": [{"from": "author", "to": "reviewer"}],
                "entry": "author",
                "terminals": ["reviewer"],
            },
        },
    )
    topology = load_topology(cfg)
    assert isinstance(primary_executable(topology), Graph)


def test_topology_swarm_beats_entry(tmp_path: Path) -> None:
    cfg = _write_topology(
        tmp_path,
        {
            "entry": "author",
            "swarm": {
                "members": ["author", "reviewer"],
                "entry": "author",
                "policy": {"type": "round_robin"},
                "termination": {"type": "max_turns", "limit": 2},
            },
        },
    )
    topology = load_topology(cfg)
    assert isinstance(primary_executable(topology), Swarm)


def test_topology_entry_agent(tmp_path: Path) -> None:
    cfg = _write_topology(tmp_path, {"entry": "reviewer"})
    topology = load_topology(cfg)
    executable = primary_executable(topology)
    assert isinstance(executable, Agent)
    assert executable.name == "reviewer"


def test_topology_without_runnable_declaration(tmp_path: Path) -> None:
    cfg = _write_topology(tmp_path, {})
    topology = load_topology(cfg)
    with pytest.raises(click.UsageError, match="entry"):
        primary_executable(topology)


# ── load_env_file ────────────────────────────────────────────────────


def test_env_file_sets_and_strips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLI_ENV_NEW", raising=False)
    env = tmp_path / "vars.env"
    env.write_text('# comment\n\nCLI_ENV_NEW="quoted value"\n', encoding="utf-8")
    load_env_file(env)
    assert os.environ.pop("CLI_ENV_NEW") == "quoted value"


def test_env_file_never_overwrites(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLI_ENV_KEEP", "original")
    env = tmp_path / "vars.env"
    env.write_text("CLI_ENV_KEEP=changed\n", encoding="utf-8")
    load_env_file(env)
    assert os.environ["CLI_ENV_KEEP"] == "original"


def test_env_file_malformed_line(tmp_path: Path) -> None:
    env = tmp_path / "vars.env"
    env.write_text("JUSTAKEY\n", encoding="utf-8")
    with pytest.raises(click.UsageError, match=":1:"):
        load_env_file(env)


# ── reconcile_positionals ────────────────────────────────────────────


def test_reconcile_lone_positional_with_agent_becomes_prompt() -> None:
    assert reconcile_positionals(Path("hello"), "mod:var", None) == (None, "hello")


def test_reconcile_keeps_config_and_prompt_without_agent() -> None:
    assert reconcile_positionals(Path("a.json"), None, "hi") == (Path("a.json"), "hi")


def test_reconcile_keeps_both_positionals_with_agent() -> None:
    # Both positionals + --agent stays as-is, so resolve_target rejects it.
    assert reconcile_positionals(Path("a.json"), "mod:var", "hi") == (Path("a.json"), "hi")


def test_reconcile_passes_through_when_nothing_set() -> None:
    assert reconcile_positionals(None, "mod:var", None) == (None, None)


def test_env_file_empty_key_rejected(tmp_path: Path) -> None:
    env = tmp_path / "vars.env"
    env.write_text("=VALUE\n", encoding="utf-8")
    with pytest.raises(click.UsageError, match="empty key"):
        load_env_file(env)


def test_env_file_strips_single_quotes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLI_ENV_SINGLE", raising=False)
    env = tmp_path / "vars.env"
    env.write_text("CLI_ENV_SINGLE='single quoted'\n", encoding="utf-8")
    load_env_file(env)
    assert os.environ.pop("CLI_ENV_SINGLE") == "single quoted"
