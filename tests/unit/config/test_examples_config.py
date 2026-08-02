"""The example config files under examples/config/ must load + assemble."""

from __future__ import annotations

from pathlib import Path

from troopai.adk.config import load_topology
from troopai.adk.handoffs import Handoff

_EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "config"


def test_swarm_example_loads() -> None:
    topo = load_topology(_EXAMPLES / "swarm.json")
    assert topo.swarm is not None


def test_graph_example_loads() -> None:
    topo = load_topology(_EXAMPLES / "graph.json")
    assert topo.graph is not None


def test_topology_example_loads_per_file_agents() -> None:
    # topology.json sources both agents from their own files via config_path.
    topo = load_topology(_EXAMPLES / "topology.json")
    assert set(topo.agents) == {"triage", "spanish"}
    assert topo.entry == "triage"
    handoffs = topo.agents["triage"].handoffs
    assert handoffs is not None
    assert topo.agents["spanish"] in [h.target if isinstance(h, Handoff) else h for h in handoffs]
