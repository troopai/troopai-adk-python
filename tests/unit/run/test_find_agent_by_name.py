"""Tests for find_agent_by_name recursive search."""

from troopai.adk.agents.agent import Agent
from troopai.adk.handoffs.handoff import Handoff
from troopai.adk.run.llm_calls import find_agent_by_name


def _agent(name: str, handoffs=None) -> Agent:
    """Construct a minimal real Agent — prompts are required by __post_init__."""
    return Agent(name=name, system_prompt="p", handoffs=handoffs)


def _handoff(agent: Agent) -> Handoff:
    """Wrap the agent in a real Handoff so isinstance narrowing succeeds."""
    return Handoff(target=agent)


class TestFindAgentByName:
    def test_root_agent_matches(self) -> None:
        root = _agent("root")
        assert find_agent_by_name(root, "root") is root

    def test_direct_child(self) -> None:
        child = _agent("child")
        root = _agent("root", handoffs=[child])
        assert find_agent_by_name(root, "child") is child

    def test_direct_child_via_handoff_object(self) -> None:
        child = _agent("child")
        root = _agent("root", handoffs=[_handoff(child)])
        assert find_agent_by_name(root, "child") is child

    def test_deep_chain(self) -> None:
        """A → B → C: find_agent_by_name should find C."""
        c = _agent("C")
        b = _agent("B", handoffs=[c])
        a = _agent("A", handoffs=[b])
        assert find_agent_by_name(a, "C") is c

    def test_three_levels_deep(self) -> None:
        """A → B → C → D"""
        d = _agent("D")
        c = _agent("C", handoffs=[d])
        b = _agent("B", handoffs=[c])
        a = _agent("A", handoffs=[b])
        assert find_agent_by_name(a, "D") is d

    def test_wide_tree(self) -> None:
        """Root has multiple children, target is in second subtree."""
        target = _agent("target")
        branch1 = _agent("branch1", handoffs=[_agent("leaf1")])
        branch2 = _agent("branch2", handoffs=[target])
        root = _agent("root", handoffs=[branch1, branch2])
        assert find_agent_by_name(root, "target") is target

    def test_not_found(self) -> None:
        root = _agent("root", handoffs=[_agent("child")])
        assert find_agent_by_name(root, "nonexistent") is None

    def test_no_handoffs(self) -> None:
        root = _agent("root")
        assert find_agent_by_name(root, "other") is None

    def test_none_handoffs(self) -> None:
        root = _agent("root", handoffs=None)
        assert find_agent_by_name(root, "other") is None

    def test_cycle_does_not_infinite_loop(self) -> None:
        """Agents with circular handoffs should not cause infinite recursion."""
        a = _agent("A")
        b = _agent("B")
        a.handoffs = [b]
        b.handoffs = [a]
        # Should terminate and not find "C"
        assert find_agent_by_name(a, "C") is None
        # Should find B
        assert find_agent_by_name(a, "B") is b
