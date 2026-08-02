"""Tests for declarative graph assembly.

A topology's optional ``graph`` section declares nodes (each referencing a
local agent), edges, an entry, and terminals. It maps onto ``GraphBuilder``.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from troopai.adk.config import build_topology
from troopai.adk.exceptions import ConfigParseError, ConfigResolutionError
from troopai.adk.graphs import Graph
from troopai.adk.types.config import TopologyConfig

_PREDICATE_REF = "tests.unit.config.sample_symbols:always_true"
_NON_CALLABLE_REF = "tests.unit.config.sample_symbols:NOT_A_TOOL"

_AGENTS = {
    "triage": {"name": "triage", "system_prompt": "Triage."},
    "writer": {"name": "writer", "system_prompt": "Write."},
}


def _topo(data: dict[str, object]):
    return build_topology(TopologyConfig.model_validate(data))


class TestGraphAssembly:
    def test_linear_graph(self) -> None:
        topo = _topo(
            {
                "agents": _AGENTS,
                "graph": {
                    "id": "pipeline",
                    "nodes": {"triage": {"agent": "triage"}, "writer": {"agent": "writer"}},
                    "edges": [{"from": "triage", "to": "writer"}],
                    "entry": "triage",
                    "terminals": ["writer"],
                },
            }
        )
        assert isinstance(topo.graph, Graph)
        assert topo.graph.id == "pipeline"
        assert {node.id for node in topo.graph.nodes} == {"triage", "writer"}
        assert topo.graph.entry == "triage"

    def test_node_with_merge_and_join(self) -> None:
        topo = _topo(
            {
                "agents": _AGENTS,
                "graph": {
                    "id": "g",
                    "nodes": {
                        "triage": {"agent": "triage"},
                        "writer": {"agent": "writer", "merge": "concat_text", "join": "or"},
                    },
                    "edges": [{"from": "triage", "to": "writer"}],
                    "entry": "triage",
                    "terminals": ["writer"],
                },
            }
        )
        assert isinstance(topo.graph, Graph)

    def test_unknown_agent_in_node_raises(self) -> None:
        with pytest.raises(ConfigResolutionError):
            _topo(
                {
                    "agents": _AGENTS,
                    "graph": {
                        "id": "g",
                        "nodes": {"a": {"agent": "ghost"}},
                        "edges": [],
                        "entry": "a",
                        "terminals": ["a"],
                    },
                }
            )

    def test_unknown_node_in_edge_raises(self) -> None:
        with pytest.raises(ConfigResolutionError):
            _topo(
                {
                    "agents": _AGENTS,
                    "graph": {
                        "id": "g",
                        "nodes": {"triage": {"agent": "triage"}},
                        "edges": [{"from": "triage", "to": "ghost"}],
                        "entry": "triage",
                        "terminals": ["triage"],
                    },
                }
            )

    def test_conditional_edge_with_predicate_ref(self) -> None:
        topo = _topo(
            {
                "agents": _AGENTS,
                "graph": {
                    "id": "g",
                    "nodes": {"triage": {"agent": "triage"}, "writer": {"agent": "writer"}},
                    "edges": [{"from": "triage", "to": "writer", "when": _PREDICATE_REF}],
                    "entry": "triage",
                    "terminals": ["writer"],
                },
            }
        )
        assert isinstance(topo.graph, Graph)

    def test_non_callable_when_ref_raises(self) -> None:
        with pytest.raises(ConfigResolutionError):
            _topo(
                {
                    "agents": _AGENTS,
                    "graph": {
                        "id": "g",
                        "nodes": {"triage": {"agent": "triage"}, "writer": {"agent": "writer"}},
                        "edges": [{"from": "triage", "to": "writer", "when": _NON_CALLABLE_REF}],
                        "entry": "triage",
                        "terminals": ["writer"],
                    },
                }
            )

    def test_entry_node_not_in_nodes_raises(self) -> None:
        with pytest.raises(ConfigResolutionError):
            _topo(
                {
                    "agents": _AGENTS,
                    "graph": {
                        "id": "g",
                        "nodes": {"triage": {"agent": "triage"}},
                        "edges": [],
                        "entry": "ghost_node",
                        "terminals": ["triage"],
                    },
                }
            )

    def test_empty_terminals_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TopologyConfig.model_validate(
                {
                    "agents": _AGENTS,
                    "graph": {
                        "id": "g",
                        "nodes": {"triage": {"agent": "triage"}},
                        "edges": [],
                        "entry": "triage",
                        "terminals": [],
                    },
                }
            )

    def test_entry_with_incoming_edge_raises_parse_error(self) -> None:
        # entry "a" has an incoming edge b->a; GraphBuilder.compile rejects it.
        with pytest.raises(ConfigParseError):
            _topo(
                {
                    "agents": _AGENTS,
                    "graph": {
                        "id": "g",
                        "nodes": {"triage": {"agent": "triage"}, "writer": {"agent": "writer"}},
                        "edges": [{"from": "triage", "to": "writer"}, {"from": "writer", "to": "triage"}],
                        "entry": "triage",
                        "terminals": ["writer"],
                    },
                }
            )
