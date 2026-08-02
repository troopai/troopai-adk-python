"""Feature 4: Typed interrupts on results.

Tests that:
- StructuredInterrupts.from_interrupts classifies correctly.
- GraphRunResult has structured_interrupts with correct defaults.
- GraphRunResultStreaming has structured_interrupts with correct defaults.
- StructuredInterrupts.by_node gives direct node-id lookup.
- Additive — existing interrupts field is unchanged.
"""

from __future__ import annotations

from troopai.adk.graphs.interrupt import (
    Interrupt,
    NestedAgentInterrupt,
    NestedGraphInterrupt,
)
from troopai.adk.graphs.result import GraphRunResult, GraphRunResultStreaming, GraphRunStatus, StructuredInterrupts


class TestStructuredInterruptsFromInterrupts:
    def test_empty_tuple_produces_empty_categories(self) -> None:
        si = StructuredInterrupts.from_interrupts(())
        assert si.generic == ()
        assert si.nested_agent == ()
        assert si.nested_graph == ()
        assert si.by_node == {}

    def test_plain_interrupt_goes_to_generic(self) -> None:
        iv = Interrupt(node_id="n1", question="decide")
        si = StructuredInterrupts.from_interrupts((iv,))
        assert si.generic == (iv,)
        assert si.nested_agent == ()
        assert si.nested_graph == ()
        assert si.by_node == {"n1": iv}

    def test_nested_agent_interrupt_classified(self) -> None:
        iv = NestedAgentInterrupt(
            node_id="na",
            question="approve?",
            agent_name="agent-x",
            tool_call_ids=("tc1",),
        )
        si = StructuredInterrupts.from_interrupts((iv,))
        assert si.nested_agent == (iv,)
        assert si.generic == ()
        assert si.nested_graph == ()
        assert si.by_node == {"na": iv}

    def test_nested_graph_interrupt_classified(self) -> None:
        iv = NestedGraphInterrupt(node_id="ng", question="inner graph paused")
        si = StructuredInterrupts.from_interrupts((iv,))
        assert si.nested_graph == (iv,)
        assert si.generic == ()
        assert si.nested_agent == ()
        assert si.by_node == {"ng": iv}

    def test_mixed_interrupts_each_in_correct_bucket(self) -> None:
        generic_iv = Interrupt(node_id="g1", question="generic")
        agent_iv = NestedAgentInterrupt(
            node_id="a1",
            question="approve",
            agent_name="agent",
            tool_call_ids=("tc",),
        )
        graph_iv = NestedGraphInterrupt(node_id="gn1", question="inner")

        si = StructuredInterrupts.from_interrupts((generic_iv, agent_iv, graph_iv))
        assert si.generic == (generic_iv,)
        assert si.nested_agent == (agent_iv,)
        assert si.nested_graph == (graph_iv,)
        assert si.by_node == {"g1": generic_iv, "a1": agent_iv, "gn1": graph_iv}

    def test_by_node_last_entry_wins_for_same_node_id(self) -> None:
        iv1 = Interrupt(node_id="dup", question="first")
        iv2 = Interrupt(node_id="dup", question="second")
        si = StructuredInterrupts.from_interrupts((iv1, iv2))
        # Both land in generic; by_node keeps the last
        assert si.by_node["dup"] is iv2

    def test_kind_field_on_generic_interrupt(self) -> None:
        """A plain Interrupt with a custom kind still goes to generic bucket."""
        iv = Interrupt(node_id="n", question="q", kind="custom_kind")
        si = StructuredInterrupts.from_interrupts((iv,))
        assert si.generic == (iv,)


class TestGraphRunResultStructuredInterrupts:
    def test_default_structured_interrupts_is_empty(self) -> None:
        result = GraphRunResult(
            final_output=None,
            status=GraphRunStatus.COMPLETED,
            user_prompt="",
        )
        si = result.structured_interrupts
        assert si.generic == ()
        assert si.nested_agent == ()
        assert si.nested_graph == ()
        assert si.by_node == {}

    def test_interrupts_field_unchanged(self) -> None:
        """Adding structured_interrupts must not change the interrupts field."""
        iv = Interrupt(node_id="n", question="q")
        result = GraphRunResult(
            final_output=None,
            status=GraphRunStatus.INTERRUPTED,
            user_prompt="",
            interrupts=(iv,),
            structured_interrupts=StructuredInterrupts.from_interrupts((iv,)),
        )
        assert result.interrupts == (iv,)
        assert result.structured_interrupts.generic == (iv,)

    def test_structured_interrupts_populated_correctly(self) -> None:
        iv = Interrupt(node_id="x", question="decide", kind="route_choice")
        si = StructuredInterrupts.from_interrupts((iv,))
        result = GraphRunResult(
            final_output=None,
            status=GraphRunStatus.INTERRUPTED,
            user_prompt="",
            interrupts=(iv,),
            structured_interrupts=si,
        )
        assert result.structured_interrupts is si
        assert "x" in result.structured_interrupts.by_node


class TestGraphRunResultStreamingStructuredInterrupts:
    def test_default_structured_interrupts_is_empty(self) -> None:
        r = GraphRunResultStreaming()
        si = r.structured_interrupts
        assert si.generic == ()
        assert si.nested_agent == ()
        assert si.by_node == {}

    def test_assign_structured_interrupts(self) -> None:
        r = GraphRunResultStreaming()
        iv = Interrupt(node_id="n", question="q")
        r.interrupts = (iv,)
        r.structured_interrupts = StructuredInterrupts.from_interrupts((iv,))
        assert r.structured_interrupts.generic == (iv,)
        assert r.interrupts == (iv,)
