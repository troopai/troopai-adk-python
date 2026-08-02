"""Tests for declarative swarm assembly.

A topology's optional ``swarm`` section names its members and entry (agent
names in the ``agents`` map), a policy, a (possibly composed) termination
condition, and config budgets.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from troopai.adk.config import build_topology
from troopai.adk.exceptions import ConfigResolutionError
from troopai.adk.swarms import (
    AndTermination,
    ExplicitDoneTermination,
    HandoffToTermination,
    LLMHandoffPolicy,
    MaxTurnsTermination,
    OrTermination,
    RoundRobinPolicy,
    Swarm,
)
from troopai.adk.types.config import TopologyConfig
from troopai.adk.types.config.swarm_config import (
    AndTerminationRef,
    ExplicitDoneTerminationRef,
    HandoffToTerminationRef,
    MaxTurnsTerminationRef,
    OrTerminationRef,
    TerminationRef,
)

_TERM: TypeAdapter[object] = TypeAdapter(TerminationRef)


class TestTerminationUnion:
    def test_max_turns(self) -> None:
        t = _TERM.validate_python({"type": "max_turns", "limit": 5})
        assert isinstance(t, MaxTurnsTerminationRef)
        assert t.limit == 5

    def test_max_turns_requires_limit(self) -> None:
        with pytest.raises(ValidationError):
            _TERM.validate_python({"type": "max_turns"})

    def test_max_turns_limit_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            _TERM.validate_python({"type": "max_turns", "limit": 0})

    def test_explicit_done(self) -> None:
        t = _TERM.validate_python({"type": "explicit_done"})
        assert isinstance(t, ExplicitDoneTerminationRef)

    def test_handoff_to(self) -> None:
        t = _TERM.validate_python({"type": "handoff_to", "target": "reviewer"})
        assert isinstance(t, HandoffToTerminationRef)
        assert t.target == "reviewer"

    def test_handoff_to_empty_target_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _TERM.validate_python({"type": "handoff_to", "target": ""})

    def test_or_requires_two_conditions(self) -> None:
        with pytest.raises(ValidationError):
            _TERM.validate_python({"type": "or", "conditions": [{"type": "explicit_done"}]})

    def test_or_nested(self) -> None:
        t = _TERM.validate_python(
            {"type": "or", "conditions": [{"type": "max_turns", "limit": 3}, {"type": "explicit_done"}]}
        )
        assert isinstance(t, OrTerminationRef)
        assert len(t.conditions) == 2

    def test_and_nested(self) -> None:
        t = _TERM.validate_python(
            {"type": "and", "conditions": [{"type": "max_turns", "limit": 3}, {"type": "handoff_to", "target": "x"}]}
        )
        assert isinstance(t, AndTerminationRef)

    def test_unknown_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _TERM.validate_python({"type": "forever"})


_AGENTS = {
    "author": {"name": "author", "system_prompt": "Write."},
    "reviewer": {"name": "reviewer", "system_prompt": "Review."},
}


def _topo(data: dict[str, object]):
    return build_topology(TopologyConfig.model_validate(data))


class TestSwarmAssembly:
    def test_round_robin_max_turns(self) -> None:
        topo = _topo(
            {
                "agents": _AGENTS,
                "swarm": {
                    "members": ["author", "reviewer"],
                    "entry": "author",
                    "policy": {"type": "round_robin"},
                    "termination": {"type": "max_turns", "limit": 6},
                },
            }
        )
        assert isinstance(topo.swarm, Swarm)
        assert isinstance(topo.swarm.policy, RoundRobinPolicy)
        assert isinstance(topo.swarm.termination, MaxTurnsTermination)
        assert topo.swarm.termination.limit == 6
        assert topo.swarm.entry is topo.agents["author"]
        assert {m.name for m in topo.swarm.members} == {"author", "reviewer"}

    def test_policy_defaults_llm_handoff(self) -> None:
        topo = _topo(
            {
                "agents": _AGENTS,
                "swarm": {
                    "members": ["author", "reviewer"],
                    "entry": "author",
                    "termination": {"type": "explicit_done"},
                },
            }
        )
        assert isinstance(topo.swarm.policy, LLMHandoffPolicy)

    def test_termination_and_composition(self) -> None:
        topo = _topo(
            {
                "agents": _AGENTS,
                "swarm": {
                    "members": ["author", "reviewer"],
                    "entry": "author",
                    "termination": {
                        "type": "and",
                        "conditions": [{"type": "explicit_done"}, {"type": "max_turns", "limit": 10}],
                    },
                },
            }
        )
        assert isinstance(topo.swarm.termination, AndTermination)

    def test_termination_handoff_to(self) -> None:
        topo = _topo(
            {
                "agents": _AGENTS,
                "swarm": {
                    "members": ["author", "reviewer"],
                    "entry": "author",
                    "termination": {"type": "handoff_to", "target": "user"},
                },
            }
        )
        assert isinstance(topo.swarm.termination, HandoffToTermination)

    def test_termination_or_composition(self) -> None:
        topo = _topo(
            {
                "agents": _AGENTS,
                "swarm": {
                    "members": ["author", "reviewer"],
                    "entry": "author",
                    "termination": {
                        "type": "or",
                        "conditions": [{"type": "explicit_done"}, {"type": "max_turns", "limit": 10}],
                    },
                },
            }
        )
        assert isinstance(topo.swarm.termination, OrTermination)
        assert isinstance(topo.swarm.termination.left, ExplicitDoneTermination)
        assert isinstance(topo.swarm.termination.right, MaxTurnsTermination)

    def test_config_budgets(self) -> None:
        topo = _topo(
            {
                "agents": _AGENTS,
                "swarm": {
                    "members": ["author", "reviewer"],
                    "entry": "author",
                    "termination": {"type": "explicit_done"},
                    "config": {"max_handoffs": 5, "max_total_tokens": 1000},
                },
            }
        )
        assert topo.swarm.config.max_handoffs == 5
        assert topo.swarm.config.max_total_tokens == 1000

    def test_unknown_member_raises(self) -> None:
        with pytest.raises(ConfigResolutionError):
            _topo(
                {
                    "agents": _AGENTS,
                    "swarm": {
                        "members": ["author", "ghost"],
                        "entry": "author",
                        "termination": {"type": "explicit_done"},
                    },
                }
            )

    def test_entry_not_in_members_raises(self) -> None:
        with pytest.raises(ConfigResolutionError):
            _topo(
                {
                    "agents": _AGENTS,
                    "swarm": {"members": ["author"], "entry": "reviewer", "termination": {"type": "explicit_done"}},
                }
            )


class TestSwarmConstraints:
    def _swarm(self, **overrides: object) -> dict[str, object]:
        swarm: dict[str, object] = {
            "members": ["author", "reviewer"],
            "entry": "author",
            "termination": {"type": "explicit_done"},
        }
        swarm.update(overrides)
        return {"agents": _AGENTS, "swarm": swarm}

    def test_or_with_single_condition_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TopologyConfig.model_validate(
                self._swarm(termination={"type": "or", "conditions": [{"type": "explicit_done"}]})
            )

    def test_empty_members_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TopologyConfig.model_validate(self._swarm(members=[]))

    def test_non_positive_max_handoffs_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TopologyConfig.model_validate(self._swarm(config={"max_handoffs": 0}))
