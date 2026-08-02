"""Unit tests for swarms-side HITL primitives.

Covers ``request_human_input_in_swarm`` consume/raise dispatch and
the :class:`RunContext` swarm-resume-reply slot semantics (key
presence, ``None`` reply validity, double-consume detection).
"""

from __future__ import annotations

import pytest

from troopai.adk.graphs.interrupt import Interrupt, InterruptException
from troopai.adk.run.context import RunContext
from troopai.adk.swarms.interrupt import request_human_input_in_swarm


class TestRunContextSwarmReplySlot:
    def test_unseeded_context_reports_no_reply(self) -> None:
        ctx: RunContext[None] = RunContext.make(None)
        assert ctx.has_swarm_resume_reply() is False

    def test_seeded_string_round_trips(self) -> None:
        ctx: RunContext[None] = RunContext.make(None)
        ctx.seed_swarm_resume_reply("approved")
        assert ctx.has_swarm_resume_reply() is True
        assert ctx.consume_swarm_resume_reply() == "approved"
        assert ctx.has_swarm_resume_reply() is False

    def test_explicit_none_reply_is_distinguishable_from_unseeded(self) -> None:
        """Abstain answer (``None``) must be a valid reply, not 'unseeded'."""
        ctx: RunContext[None] = RunContext.make(None)
        ctx.seed_swarm_resume_reply(None)
        assert ctx.has_swarm_resume_reply() is True
        assert ctx.consume_swarm_resume_reply() is None
        assert ctx.has_swarm_resume_reply() is False

    def test_double_consume_raises_lookup_error(self) -> None:
        ctx: RunContext[None] = RunContext.make(None)
        ctx.seed_swarm_resume_reply({"decision": "yes"})
        ctx.consume_swarm_resume_reply()
        with pytest.raises(LookupError, match="no reply seeded"):
            ctx.consume_swarm_resume_reply()

    def test_clear_after_seed_returns_to_unseeded(self) -> None:
        ctx: RunContext[None] = RunContext.make(None)
        ctx.seed_swarm_resume_reply("queued")
        ctx.clear_swarm_resume_reply()
        assert ctx.has_swarm_resume_reply() is False

    def test_seed_overwrites_prior_unconsumed_reply(self) -> None:
        ctx: RunContext[None] = RunContext.make(None)
        ctx.seed_swarm_resume_reply("first")
        ctx.seed_swarm_resume_reply("second")
        assert ctx.consume_swarm_resume_reply() == "second"


class TestRequestHumanInputInSwarm:
    def test_raises_when_no_reply_seeded(self) -> None:
        ctx: RunContext[None] = RunContext.make(None)
        with pytest.raises(InterruptException) as exc_info:
            request_human_input_in_swarm(
                ctx,
                "approver",
                "Approve this action?",
                kind="tool_approval",
                metadata={"tool_call_id": "c1"},
            )
        interrupt = exc_info.value.interrupt
        assert interrupt.node_id == "approver"
        assert interrupt.question == "Approve this action?"
        assert interrupt.kind == "tool_approval"
        assert interrupt.metadata == {"tool_call_id": "c1"}

    def test_returns_seeded_reply_and_clears_slot(self) -> None:
        ctx: RunContext[None] = RunContext.make(None)
        ctx.seed_swarm_resume_reply("approved")
        result = request_human_input_in_swarm(
            ctx,
            "approver",
            "Approve this action?",
        )
        assert result == "approved"
        assert ctx.has_swarm_resume_reply() is False

    def test_returns_explicit_none_reply(self) -> None:
        """``None`` is a valid abstain answer — must not re-raise."""
        ctx: RunContext[None] = RunContext.make(None)
        ctx.seed_swarm_resume_reply(None)
        result = request_human_input_in_swarm(ctx, "approver", "Decide?")
        assert result is None
        assert ctx.has_swarm_resume_reply() is False

    def test_default_kind_is_generic(self) -> None:
        ctx: RunContext[None] = RunContext.make(None)
        with pytest.raises(InterruptException) as exc_info:
            request_human_input_in_swarm(ctx, "m", "q?")
        assert exc_info.value.interrupt.kind == "generic"

    def test_interrupt_is_concrete_interrupt_not_subclass(self) -> None:
        """HITL-pure parks a plain Interrupt — no agent snapshot."""
        ctx: RunContext[None] = RunContext.make(None)
        with pytest.raises(InterruptException) as exc_info:
            request_human_input_in_swarm(ctx, "m", "q?")
        assert type(exc_info.value.interrupt) is Interrupt


# ---------------------------------------------------------------------------
# Regression: metadata must be a named dict param, not **kwargs (#MED)
# ---------------------------------------------------------------------------


class TestRequestHumanInputInSwarmMetadata:
    """Regression: the old signature used ``**metadata: Any`` which was the
    rejected kwargs-spread pattern. The fix changes to an explicit
    ``metadata: dict[str, Any] | None = None`` parameter."""

    def test_metadata_dict_forwarded_onto_interrupt(self) -> None:
        """Passing metadata as a dict stores it on the interrupt payload."""
        ctx: RunContext[None] = RunContext.make(None)
        with pytest.raises(InterruptException) as exc_info:
            request_human_input_in_swarm(
                ctx,
                "m",
                "Approve?",
                metadata={"tool_call_id": "t1", "agent": "x"},
            )
        assert exc_info.value.interrupt.metadata == {"tool_call_id": "t1", "agent": "x"}

    def test_none_metadata_becomes_empty_dict(self) -> None:
        """``metadata=None`` must produce an empty dict on the interrupt, not None."""
        ctx: RunContext[None] = RunContext.make(None)
        with pytest.raises(InterruptException) as exc_info:
            request_human_input_in_swarm(ctx, "m", "q?", metadata=None)
        assert exc_info.value.interrupt.metadata == {}

    def test_omitted_metadata_becomes_empty_dict(self) -> None:
        """Omitting the metadata parameter must produce an empty dict on the interrupt."""
        ctx: RunContext[None] = RunContext.make(None)
        with pytest.raises(InterruptException) as exc_info:
            request_human_input_in_swarm(ctx, "m", "q?")
        assert exc_info.value.interrupt.metadata == {}
