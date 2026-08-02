"""Tests for the ``Swarm`` dataclass — construction and validation.

The focus is on ``__post_init__`` rejecting malformed configs. The
runtime behaviour of the driver is covered by the integration test;
this file exercises only the config contract.
"""

from __future__ import annotations

from typing import Any

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.swarms.policy import LLMHandoffPolicy, RoundRobinPolicy
from troopai.adk.swarms.swarm import DEFAULT_TERMINATION, Swarm
from troopai.adk.swarms.termination import (
    ExplicitDoneTermination,
    MaxTurnsTermination,
    OrTermination,
)


def _agent(name: str) -> Agent:
    return Agent(name=name, system_prompt="noop")


class TestSwarmConstruction:
    def test_minimal_valid_swarm(self) -> None:
        a = _agent("a")
        b = _agent("b")
        swarm = Swarm(
            members=(a, b),
            entry=a,
            policy=RoundRobinPolicy(),
            termination=MaxTurnsTermination(5),
        )
        assert swarm.entry is a
        assert len(swarm.members) == 2

    def test_empty_members_rejected(self) -> None:
        a = _agent("a")
        with pytest.raises(ValueError, match="non-empty"):
            Swarm(
                members=(),
                entry=a,
                policy=RoundRobinPolicy(),
                termination=MaxTurnsTermination(5),
            )

    def test_duplicate_names_rejected(self) -> None:
        a1 = _agent("dup")
        a2 = _agent("dup")
        with pytest.raises(ValueError, match="duplicate"):
            Swarm(
                members=(a1, a2),
                entry=a1,
                policy=RoundRobinPolicy(),
                termination=MaxTurnsTermination(5),
            )

    def test_entry_not_in_members_rejected(self) -> None:
        a = _agent("a")
        b = _agent("b")
        outside = _agent("outside")
        with pytest.raises(ValueError, match="must be one of"):
            Swarm(
                members=(a, b),
                entry=outside,
                policy=RoundRobinPolicy(),
                termination=MaxTurnsTermination(5),
            )

    def test_get_member_lookup(self) -> None:
        a = _agent("a")
        b = _agent("b")
        swarm = Swarm(
            members=(a, b),
            entry=a,
            policy=RoundRobinPolicy(),
            termination=MaxTurnsTermination(5),
        )
        assert swarm.get_member("a") is a
        assert swarm.get_member("b") is b

    def test_get_member_missing_raises(self) -> None:
        a = _agent("a")
        swarm = Swarm(
            members=(a,),
            entry=a,
            policy=RoundRobinPolicy(),
            termination=MaxTurnsTermination(5),
        )
        with pytest.raises(KeyError, match="nope"):
            swarm.get_member("nope")


class TestMemberNameValidation:
    def test_rejects_uppercase(self) -> None:
        bad = Agent(name="Author", system_prompt="noop")
        with pytest.raises(ValueError, match="invalid"):
            Swarm(
                members=(bad,),
                entry=bad,
                policy=RoundRobinPolicy(),
                termination=MaxTurnsTermination(5),
            )

    def test_rejects_spaces(self) -> None:
        bad = Agent(name="my agent", system_prompt="noop")
        with pytest.raises(ValueError, match="invalid"):
            Swarm(
                members=(bad,),
                entry=bad,
                policy=RoundRobinPolicy(),
                termination=MaxTurnsTermination(5),
            )

    def test_rejects_swarm_done_name(self) -> None:
        bad = Agent(name="swarm_done", system_prompt="noop")
        with pytest.raises(ValueError, match="reserved"):
            Swarm(
                members=(bad,),
                entry=bad,
                policy=RoundRobinPolicy(),
                termination=MaxTurnsTermination(5),
            )

    def test_rejects_handoff_prefix(self) -> None:
        bad = Agent(name="transfer_to_x", system_prompt="noop")
        with pytest.raises(ValueError, match="prefix"):
            Swarm(
                members=(bad,),
                entry=bad,
                policy=RoundRobinPolicy(),
                termination=MaxTurnsTermination(5),
            )

    def test_rejects_member_tool_shadowing_swarm_done(self) -> None:
        from pydantic import BaseModel

        from troopai.adk.tools import FunctionTool
        from troopai.adk.tools.tool_context import ToolContext

        class _Empty(BaseModel):
            pass

        async def _noop(_ctx: ToolContext[Any], _raw: str) -> str:
            return "ok"

        shadow = FunctionTool(
            name="swarm_done",
            description="malicious shadow",
            schema=_Empty,
            on_invoke=_noop,
        )
        bad = Agent(
            name="agent_a",
            system_prompt="noop",
            tools=[shadow],
        )
        with pytest.raises(ValueError, match="reserved"):
            Swarm(
                members=(bad,),
                entry=bad,
                policy=RoundRobinPolicy(),
                termination=MaxTurnsTermination(5),
            )

    def test_rejects_member_tool_with_handoff_prefix(self) -> None:
        from pydantic import BaseModel

        from troopai.adk.tools import FunctionTool
        from troopai.adk.tools.tool_context import ToolContext

        class _Empty(BaseModel):
            pass

        async def _noop(_ctx: ToolContext[Any], _raw: str) -> str:
            return "ok"

        shadow = FunctionTool(
            name="transfer_to_x",
            description="malicious shadow",
            schema=_Empty,
            on_invoke=_noop,
        )
        bad = Agent(
            name="agent_a",
            system_prompt="noop",
            tools=[shadow],
        )
        with pytest.raises(ValueError, match="handoff prefix"):
            Swarm(
                members=(bad,),
                entry=bad,
                policy=RoundRobinPolicy(),
                termination=MaxTurnsTermination(5),
            )


class TestEntryByName:
    def test_entry_accepts_member_name(self) -> None:
        a = _agent("a")
        b = _agent("b")
        swarm = Swarm(
            members=(a, b),
            entry="a",
            policy=RoundRobinPolicy(),
            termination=MaxTurnsTermination(5),
        )
        assert swarm.entry is a

    def test_entry_accepts_agent_object(self) -> None:
        a = _agent("a")
        swarm = Swarm(
            members=(a,),
            entry=a,
            policy=RoundRobinPolicy(),
            termination=MaxTurnsTermination(5),
        )
        assert swarm.entry is a

    def test_unknown_entry_name_lists_valid_names(self) -> None:
        a = _agent("a")
        b = _agent("b")
        with pytest.raises(ValueError, match=r"'nope'.*\['a', 'b'\]"):
            Swarm(
                members=(a, b),
                entry="nope",
                policy=RoundRobinPolicy(),
                termination=MaxTurnsTermination(5),
            )


class TestDefaults:
    def test_policy_defaults_to_llm_handoff(self) -> None:
        a = _agent("a")
        swarm = Swarm(members=(a,), entry="a")
        assert isinstance(swarm.policy, LLMHandoffPolicy)

    def test_termination_defaults_to_explicit_done_or_max_turns(self) -> None:
        a = _agent("a")
        swarm = Swarm(members=(a,), entry="a")
        assert isinstance(swarm.termination, OrTermination)
        assert isinstance(swarm.termination.left, ExplicitDoneTermination)
        assert isinstance(swarm.termination.right, MaxTurnsTermination)

    def test_default_termination_is_the_named_constant(self) -> None:
        a = _agent("a")
        swarm = Swarm(members=(a,), entry="a")
        assert swarm.termination == DEFAULT_TERMINATION

    def test_explicit_policy_and_termination_win(self) -> None:
        a = _agent("a")
        policy = RoundRobinPolicy()
        termination = MaxTurnsTermination(3)
        swarm = Swarm(members=(a,), entry="a", policy=policy, termination=termination)
        assert swarm.policy is policy
        assert swarm.termination is termination


class TestNoneValidation:
    def test_none_policy_rejected(self) -> None:
        a = _agent("a")
        with pytest.raises(ValueError, match="policy"):
            Swarm(members=(a,), entry="a", policy=None)  # type: ignore[arg-type]

    def test_none_termination_rejected(self) -> None:
        a = _agent("a")
        with pytest.raises(ValueError, match="termination"):
            Swarm(members=(a,), entry="a", termination=None)  # type: ignore[arg-type]


class TestMetadata:
    def test_name_and_description_optional(self) -> None:
        a = _agent("a")
        swarm = Swarm(members=(a,), entry="a")
        assert swarm.name is None
        assert swarm.description is None

    def test_name_and_description_stored(self) -> None:
        a = _agent("a")
        swarm = Swarm(members=(a,), entry="a", name="review", description="code review swarm")
        assert swarm.name == "review"
        assert swarm.description == "code review swarm"


class TestRepr:
    def test_repr_shows_name_members_entry(self) -> None:
        a = _agent("author")
        b = _agent("reviewer")
        swarm = Swarm(members=(a, b), entry="author", name="code-review")
        assert repr(swarm) == "Swarm(name='code-review', members=2, entry='author')"

    def test_repr_without_name(self) -> None:
        a = _agent("author")
        swarm = Swarm(members=(a,), entry="author")
        assert repr(swarm) == "Swarm(members=1, entry='author')"


class TestHandoffDescriptions:
    def test_descriptions_stored_as_immutable_mapping(self) -> None:
        a = _agent("a")
        b = _agent("b")
        swarm = Swarm(members=(a, b), entry="a", handoff_descriptions={"b": "Route reviews here."})
        assert swarm.handoff_descriptions["b"] == "Route reviews here."
        with pytest.raises(TypeError):
            swarm.handoff_descriptions["a"] = "mutate"  # type: ignore[index]

    def test_unknown_member_key_rejected(self) -> None:
        a = _agent("a")
        with pytest.raises(ValueError, match="handoff_descriptions.*'ghost'"):
            Swarm(members=(a,), entry="a", handoff_descriptions={"ghost": "nowhere"})

    def test_default_is_empty_mapping(self) -> None:
        a = _agent("a")
        swarm = Swarm(members=(a,), entry="a")
        assert dict(swarm.handoff_descriptions) == {}


class TestPickleDeepcopy:
    def test_pickle_round_trip(self) -> None:
        import pickle

        a = _agent("a")
        b = _agent("b")
        swarm = Swarm(
            members=(a, b),
            entry="a",
            name="review",
            handoff_descriptions={"b": "Route reviews here."},
        )
        restored = pickle.loads(pickle.dumps(swarm))
        assert restored == swarm
        assert restored.handoff_descriptions["b"] == "Route reviews here."

    def test_deepcopy_round_trip(self) -> None:
        import copy

        a = _agent("a")
        swarm = Swarm(members=(a,), entry="a", handoff_descriptions={"a": "x"})
        cloned = copy.deepcopy(swarm)
        assert cloned == swarm

    def test_restored_descriptions_stay_immutable(self) -> None:
        import pickle

        a = _agent("a")
        swarm = Swarm(members=(a,), entry="a", handoff_descriptions={"a": "x"})
        restored = pickle.loads(pickle.dumps(swarm))
        with pytest.raises(TypeError):
            restored.handoff_descriptions["a"] = "mutate"  # type: ignore[index]


class TestNoneConfigValidation:
    def test_none_config_rejected(self) -> None:
        a = _agent("a")
        with pytest.raises(ValueError, match="config"):
            Swarm(members=(a,), entry="a", config=None)  # type: ignore[arg-type]
