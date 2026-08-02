"""Tests for multi-agent topology loading (handoffs).

A topology file has an ``agents`` map (name -> node config) where each node
may declare ``handoffs`` by name. Loading is two-pass: build every agent as a
stub, then wire handoff targets by name. This wires A<->B cycles because
``Agent`` is mutable and validates no handoff targets at construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from troopai.adk.config import build_topology, load_topology
from troopai.adk.exceptions import ConfigParseError, ConfigResolutionError
from troopai.adk.handoffs import Handoff
from troopai.adk.tools import FunctionTool
from troopai.adk.types.config import TopologyConfig


def _topo(data: dict[str, object]):
    return build_topology(TopologyConfig.model_validate(data))


class TestHandoffWiring:
    def test_one_way_handoff_bare_string(self) -> None:
        topo = _topo(
            {
                "agents": {
                    "triage": {"name": "triage", "system_prompt": "p", "handoffs": ["refunds"]},
                    "refunds": {"name": "refunds", "system_prompt": "p"},
                }
            }
        )
        assert topo.agents["triage"].handoffs is not None
        assert topo.agents["triage"].handoffs[0] is topo.agents["refunds"]

    def test_mutual_cycle(self) -> None:
        topo = _topo(
            {
                "agents": {
                    "a": {"name": "a", "system_prompt": "p", "handoffs": ["b"]},
                    "b": {"name": "b", "system_prompt": "p", "handoffs": ["a"]},
                }
            }
        )
        assert topo.agents["a"].handoffs[0] is topo.agents["b"]
        assert topo.agents["b"].handoffs[0] is topo.agents["a"]

    def test_object_handoff_with_description(self) -> None:
        topo = _topo(
            {
                "agents": {
                    "a": {
                        "name": "a",
                        "system_prompt": "p",
                        "handoffs": [{"target": "b", "description": "go to b"}],
                    },
                    "b": {"name": "b", "system_prompt": "p"},
                }
            }
        )
        handoff = topo.agents["a"].handoffs[0]
        assert isinstance(handoff, Handoff)
        assert handoff.target is topo.agents["b"]
        assert handoff.description == "go to b"

    def test_missing_target_raises(self) -> None:
        with pytest.raises(ConfigResolutionError):
            _topo({"agents": {"a": {"name": "a", "system_prompt": "p", "handoffs": ["ghost"]}}})

    def test_no_handoffs_stays_none(self) -> None:
        topo = _topo({"agents": {"a": {"name": "a", "system_prompt": "p"}}})
        assert topo.agents["a"].handoffs is None

    def test_entry_preserved(self) -> None:
        topo = _topo({"agents": {"a": {"name": "a", "system_prompt": "p"}}, "entry": "a"})
        assert topo.entry == "a"

    def test_entry_not_in_agents_raises(self) -> None:
        with pytest.raises(ConfigResolutionError):
            _topo({"agents": {"a": {"name": "a", "system_prompt": "p"}}, "entry": "ghost"})

    def test_schema_pointer_tolerated(self) -> None:
        topo = _topo(
            {
                "$schema": "https://example.com/topology.schema.json",
                "agents": {"a": {"name": "a", "system_prompt": "p"}},
            }
        )
        assert topo.agents["a"].name == "a"


class TestLoadTopologyFile:
    def test_loads_from_file(self, tmp_path: Path) -> None:
        path = tmp_path / "topology.json"
        path.write_text(
            json.dumps(
                {
                    "agents": {
                        "a": {"name": "a", "system_prompt": "p", "handoffs": ["b"]},
                        "b": {"name": "b", "system_prompt": "p"},
                    },
                    "entry": "a",
                }
            )
        )
        topo = load_topology(path)
        assert set(topo.agents) == {"a", "b"}
        assert topo.agents["a"].handoffs[0] is topo.agents["b"]
        assert topo.entry == "a"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_topology(tmp_path / "nope.json")

    def test_invalid_json_raises_parse_error(self, tmp_path: Path) -> None:
        path = tmp_path / "t.json"
        path.write_text("{ not json ")
        with pytest.raises(ConfigParseError):
            load_topology(path)

    def test_validation_failure_raises_parse_error(self, tmp_path: Path) -> None:
        path = tmp_path / "t.json"
        path.write_text(json.dumps({"entry": "a"}))  # missing required 'agents'
        with pytest.raises(ConfigParseError):
            load_topology(path)


class TestConfigPathSubAgents:
    """A topology ``agents`` entry may point at a standalone agent file."""

    def _write(self, path: Path, data: dict[str, object]) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_config_path_agents_load_and_wire(self, tmp_path: Path) -> None:
        # A file-sourced member may itself declare handoffs by name.
        self._write(tmp_path / "triage.json", {"name": "triage", "system_prompt": "route", "handoffs": ["spanish"]})
        self._write(tmp_path / "spanish.json", {"name": "spanish", "system_prompt": "es"})
        topo_path = tmp_path / "topology.json"
        self._write(
            topo_path,
            {
                "agents": {"triage": {"config_path": "triage.json"}, "spanish": {"config_path": "spanish.json"}},
                "entry": "triage",
            },
        )
        topo = load_topology(topo_path)
        assert set(topo.agents) == {"triage", "spanish"}
        assert topo.agents["triage"].handoffs[0] is topo.agents["spanish"]
        assert topo.entry == "triage"

    def test_config_path_mixed_with_inline(self, tmp_path: Path) -> None:
        self._write(tmp_path / "spanish.json", {"name": "spanish", "system_prompt": "es"})
        topo_path = tmp_path / "topology.json"
        self._write(
            topo_path,
            {
                "agents": {
                    "triage": {"name": "triage", "system_prompt": "route", "handoffs": ["spanish"]},
                    "spanish": {"config_path": "spanish.json"},
                }
            },
        )
        topo = load_topology(topo_path)
        assert set(topo.agents) == {"triage", "spanish"}
        assert topo.agents["triage"].handoffs[0] is topo.agents["spanish"]

    def test_config_path_missing_file_raises(self, tmp_path: Path) -> None:
        topo_path = tmp_path / "topology.json"
        self._write(topo_path, {"agents": {"a": {"config_path": "nope.json"}}})
        with pytest.raises(FileNotFoundError):
            load_topology(topo_path)

    def test_config_path_invalid_agent_file_raises(self, tmp_path: Path) -> None:
        self._write(tmp_path / "bad.json", {"system_prompt": "missing name"})
        topo_path = tmp_path / "topology.json"
        self._write(topo_path, {"agents": {"a": {"config_path": "bad.json"}}})
        with pytest.raises(ConfigParseError, match="config_path"):
            load_topology(topo_path)

    def test_config_path_without_base_dir_raises(self) -> None:
        # Built from an in-memory model (no file) — a config_path has nothing
        # to resolve against.
        config = TopologyConfig.model_validate({"agents": {"a": {"config_path": "x.json"}}})
        with pytest.raises(ConfigResolutionError, match="config_path"):
            build_topology(config)

    def test_config_path_absolute(self, tmp_path: Path) -> None:
        # An absolute config_path is used as-is (base_dir is not prepended).
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        self._write(agents_dir / "worker.json", {"name": "worker", "system_prompt": "p"})
        topo_dir = tmp_path / "topo"
        topo_dir.mkdir()
        topo_path = topo_dir / "topology.json"
        self._write(topo_path, {"agents": {"worker": {"config_path": str((agents_dir / "worker.json").resolve())}}})

        topo = load_topology(topo_path)

        assert set(topo.agents) == {"worker"}

    def test_config_path_yaml_target(self, tmp_path: Path) -> None:
        # A config_path may point at a YAML agent file (same read path).
        (tmp_path / "worker.yaml").write_text("name: worker\nsystem_prompt: p\n", encoding="utf-8")
        topo_path = tmp_path / "topology.json"
        self._write(topo_path, {"agents": {"worker": {"config_path": "worker.yaml"}}})

        topo = load_topology(topo_path)

        assert set(topo.agents) == {"worker"}

    def test_build_topology_base_dir_keeps_graph_predicate_importable(self, tmp_path: Path) -> None:
        # build_topology(base_dir=...) called directly must keep base_dir
        # importable through graph assembly, so an edge `when` predicate that
        # names a sibling module resolves (not just the agents' tool refs).
        (tmp_path / "pred.py").write_text("def always(result: object) -> bool:\n    return True\n", encoding="utf-8")
        config = TopologyConfig.model_validate(
            {
                "agents": {"a": {"name": "a", "system_prompt": "p"}, "b": {"name": "b", "system_prompt": "p"}},
                "graph": {
                    "id": "g",
                    "nodes": {"na": {"agent": "a"}, "nb": {"agent": "b"}},
                    "edges": [{"from": "na", "to": "nb", "when": "pred.always"}],
                    "entry": "na",
                    "terminals": ["nb"],
                },
            }
        )

        topo = build_topology(config, base_dir=tmp_path)

        assert topo.graph is not None

    def test_config_path_child_resolves_own_sibling_tool(self, tmp_path: Path) -> None:
        # A child in a *different* directory resolves its own sibling tool
        # module by bare name (relative to the child file, not the topology).
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "subtool.py").write_text(
            'from troopai.adk.tools import function_tool\n\n\n@function_tool\ndef ping() -> str:\n    return "pong"\n',
            encoding="utf-8",
        )
        self._write(sub / "worker.json", {"name": "worker", "system_prompt": "p", "tools": ["subtool.ping"]})
        topo_path = tmp_path / "topology.json"
        self._write(topo_path, {"agents": {"worker": {"config_path": "sub/worker.json"}}})

        topo = load_topology(topo_path)

        assert len(topo.agents["worker"].tools) == 1
        tool = topo.agents["worker"].tools[0]
        assert isinstance(tool, FunctionTool)
        assert tool.name == "ping"


def test_topology_with_both_swarm_and_graph_builds_both() -> None:
    from troopai.adk.config import build_topology
    from troopai.adk.types.config import TopologyConfig

    topo = build_topology(
        TopologyConfig.model_validate(
            {
                "agents": {"a": {"name": "a", "system_prompt": "p"}, "b": {"name": "b", "system_prompt": "p"}},
                "swarm": {"members": ["a", "b"], "entry": "a", "termination": {"type": "explicit_done"}},
                "graph": {
                    "id": "g",
                    "nodes": {"na": {"agent": "a"}, "nb": {"agent": "b"}},
                    "edges": [{"from": "na", "to": "nb"}],
                    "entry": "na",
                    "terminals": ["nb"],
                },
            }
        )
    )
    assert topo.swarm is not None
    assert topo.graph is not None
