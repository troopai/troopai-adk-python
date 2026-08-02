"""Unit tests for the agent-middleware module.

Covers:

- ``compose_agent_middleware`` chain order (outer-to-inner, then unwind)
- Empty-list zero-overhead identity
- Messages mutation propagating to the inner terminal
- Outcome transformation propagating back through outer middleware
- ``AgentMiddlewareTermination`` short-circuiting the chain
- Exception propagation
- ``AgentLoggingMiddleware`` log records via ``caplog``
- ``AgentMetricsMiddleware`` recorder calls
- ``AgentBlockOutcome`` discriminator shape

The hook-ordering and handoff-re-entry contracts that depend on the
real ``run_agent_loop`` refactor are exercised in the integration test
at ``tests/integration/test_all_middleware_layers.py`` because they
need a working LLM mock and the full Runner stack.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.run.agent_middleware import (
    AgentBlockOutcome,
    AgentLoggingMiddleware,
    AgentMetricsMiddleware,
    AgentMetricsRecorder,
    AgentMiddleware,
    AgentMiddlewareTermination,
    compose_agent_middleware,
)
from troopai.adk.run.context import RunContext
from troopai.adk.types.input import LLMInputContentItem


def _agent(name: str = "X") -> Agent:
    return Agent(name=name, system_prompt="t")


def _ctx() -> RunContext[Any]:
    return RunContext.make(None)


def _final_outcome() -> AgentBlockOutcome:
    """An outcome the unit tests can mutate via dataclasses.replace."""
    return AgentBlockOutcome(
        kind="final",
        result=None,  # tests assert on field changes, not RunResult
        turn=1,
        total_turns_consumed=1,
    )


class _OrderRecorder:
    """Helper middleware that records its own pre/post tags."""

    def __init__(self, label: str, order: list[str]) -> None:
        self.label = label
        self.order = order

    async def __call__(
        self,
        agent: Agent,
        messages: list[LLMInputContentItem],
        context: RunContext[Any],
        next: Any,
    ) -> AgentBlockOutcome:
        self.order.append(f"+{self.label}")
        outcome = await next(agent, messages)
        self.order.append(f"-{self.label}")
        return outcome


class TestChainComposition:
    async def test_outer_to_inner_then_unwind(self) -> None:
        order: list[str] = []
        agent = _agent()
        ctx = _ctx()

        async def terminal(ag: Agent, msgs: list[LLMInputContentItem]) -> AgentBlockOutcome:
            order.append("=terminal")
            return _final_outcome()

        chain = compose_agent_middleware(
            [
                _OrderRecorder("A", order),
                _OrderRecorder("B", order),
                _OrderRecorder("C", order),
            ],
            terminal,
            context=ctx,
        )
        await chain(agent, [])
        assert order == ["+A", "+B", "+C", "=terminal", "-C", "-B", "-A"]

    async def test_empty_list_returns_terminal_unchanged(self) -> None:
        ctx = _ctx()

        async def terminal(ag: Agent, msgs: list[LLMInputContentItem]) -> AgentBlockOutcome:
            return _final_outcome()

        chain = compose_agent_middleware([], terminal, context=ctx)
        # Empty middleware list MUST hand back the exact terminal —
        # zero-overhead identity (mirrors compose_tool_middleware).
        assert chain is terminal


class TestMessagesMutation:
    async def test_middleware_messages_mutation_visible_to_terminal(self) -> None:
        ctx = _ctx()
        seen_by_terminal: list[int] = []

        class Prepend:
            async def __call__(self, agent, messages, context, next):  # type: ignore[no-untyped-def]
                # Add a synthetic system message before forwarding.
                messages = [{"role": "system", "content": "injected"}, *messages]
                return await next(agent, messages)

        async def terminal(ag: Agent, msgs: list[LLMInputContentItem]) -> AgentBlockOutcome:
            seen_by_terminal.append(len(msgs))
            return _final_outcome()

        chain = compose_agent_middleware([Prepend()], terminal, context=ctx)
        await chain(_agent(), [{"role": "user", "content": "hello"}])
        assert seen_by_terminal == [2]


class TestOutcomeTransformation:
    async def test_middleware_can_replace_outcome(self) -> None:
        ctx = _ctx()

        class TagTurnsConsumed:
            async def __call__(self, agent, messages, context, next):  # type: ignore[no-untyped-def]
                outcome = await next(agent, messages)
                # Mutate via dataclasses.replace because AgentBlockOutcome is frozen.
                import dataclasses as _d

                return _d.replace(outcome, turn=outcome.turn + 100)

        async def terminal(ag, msgs):  # type: ignore[no-untyped-def]
            return AgentBlockOutcome(kind="final", turn=1, total_turns_consumed=1)

        chain = compose_agent_middleware([TagTurnsConsumed()], terminal, context=ctx)
        result = await chain(_agent(), [])
        assert result.turn == 101


class TestTermination:
    async def test_inner_termination_returns_through_outer_middleware(self) -> None:
        """``compose_agent_middleware`` catches the termination at the
        raising middleware's frame and converts it into a normal
        return. Outer middleware therefore sees the carried outcome
        as a regular ``await next(…)`` result and runs its post-call
        code (e.g. metrics ``record_outcome(success=True)``).
        """
        ctx = _ctx()
        order: list[str] = []

        class ShortCircuit:
            async def __call__(self, agent, messages, context, next):  # type: ignore[no-untyped-def]
                raise AgentMiddlewareTermination(_final_outcome())

        async def terminal(ag, msgs):  # type: ignore[no-untyped-def]
            order.append("=terminal")  # Never reached
            return _final_outcome()

        chain = compose_agent_middleware(
            [_OrderRecorder("A", order), ShortCircuit()],
            terminal,
            context=ctx,
        )
        outcome = await chain(_agent(), [])
        assert outcome.kind == "final"
        # Outer "+A" fired AND "-A" fired (chain unwound via normal
        # return because the termination was caught at ShortCircuit's
        # frame). Terminal was NOT called.
        assert order == ["+A", "-A"]


class TestExceptionPropagation:
    async def test_runtime_error_from_terminal_propagates(self) -> None:
        ctx = _ctx()

        async def terminal(ag, msgs):  # type: ignore[no-untyped-def]
            raise RuntimeError("boom")

        order: list[str] = []
        chain = compose_agent_middleware([_OrderRecorder("A", order)], terminal, context=ctx)
        with pytest.raises(RuntimeError, match="boom"):
            await chain(_agent(), [])
        # "+A" fired; "-A" did NOT (exception unwound through it).
        assert order == ["+A"]


class TestAgentBlockOutcomeShape:
    def test_final_outcome_carries_result_field(self) -> None:
        outcome = AgentBlockOutcome(kind="final", turn=1, total_turns_consumed=1)
        assert outcome.kind == "final"
        assert outcome.result is None
        assert outcome.handoff_target is None

    def test_handoff_outcome_carries_target_and_messages(self) -> None:
        target = _agent("Beta")
        outcome = AgentBlockOutcome(
            kind="handoff",
            handoff_target=target,
            next_messages=[{"role": "user", "content": "go"}],
            next_context_end=1,
            turn=1,
            total_turns_consumed=1,
        )
        assert outcome.kind == "handoff"
        assert outcome.handoff_target is target
        assert outcome.next_messages is not None
        assert len(outcome.next_messages) == 1
        assert outcome.next_context_end == 1


class TestAgentLoggingMiddleware:
    async def test_start_and_end_records_emitted(self, caplog: pytest.LogCaptureFixture) -> None:
        ctx = _ctx()

        async def terminal(ag, msgs):  # type: ignore[no-untyped-def]
            return _final_outcome()

        with caplog.at_level(logging.INFO, logger="troopai.adk.run.agent_middleware"):
            chain = compose_agent_middleware([AgentLoggingMiddleware()], terminal, context=ctx)
            await chain(_agent("Alpha"), [])

        records = [r.message for r in caplog.records if "agent" in r.message.lower()]
        starting = [r for r in records if "starting" in r]
        completed = [r for r in records if "completed" in r]
        assert len(starting) == 1
        assert "Alpha" in starting[0]
        assert len(completed) == 1
        assert "Alpha" in completed[0]


class TestAgentMetricsMiddleware:
    async def test_recorder_called_on_success(self) -> None:
        ctx = _ctx()
        durations: list[tuple[str, float]] = []
        outcomes: list[tuple[str, bool]] = []

        class Sink:
            def record_duration(self, agent_name: str, duration_seconds: float) -> None:
                durations.append((agent_name, duration_seconds))

            def record_outcome(self, agent_name: str, *, success: bool) -> None:
                outcomes.append((agent_name, success))

        sink = Sink()
        # Sanity: the sink conforms to the Protocol structurally.
        assert isinstance(sink, AgentMetricsRecorder)

        async def terminal(ag, msgs):  # type: ignore[no-untyped-def]
            return _final_outcome()

        chain = compose_agent_middleware([AgentMetricsMiddleware(recorder=sink)], terminal, context=ctx)
        await chain(_agent("Beta"), [])
        assert len(durations) == 1
        assert durations[0][0] == "Beta"
        assert durations[0][1] >= 0.0
        assert outcomes == [("Beta", True)]

    async def test_recorder_called_on_failure(self) -> None:
        ctx = _ctx()
        outcomes: list[tuple[str, bool]] = []

        class Sink:
            def record_duration(self, agent_name: str, duration_seconds: float) -> None:
                pass

            def record_outcome(self, agent_name: str, *, success: bool) -> None:
                outcomes.append((agent_name, success))

        async def terminal(ag, msgs):  # type: ignore[no-untyped-def]
            raise RuntimeError("bang")

        chain = compose_agent_middleware([AgentMetricsMiddleware(recorder=Sink())], terminal, context=ctx)
        with pytest.raises(RuntimeError):
            await chain(_agent("Gamma"), [])
        assert outcomes == [("Gamma", False)]


class TestProtocolStructuralConformance:
    def test_logging_middleware_satisfies_protocol(self) -> None:
        # ``runtime_checkable`` Protocols allow ``isinstance`` against
        # the structural shape — ``__call__`` accepting the right
        # arity. AgentMetricsMiddleware and AgentLoggingMiddleware
        # both have async ``__call__``s so the structural check passes.
        assert isinstance(AgentLoggingMiddleware(), AgentMiddleware)

    def test_metrics_middleware_satisfies_protocol(self) -> None:
        class _NoopSink:
            def record_duration(self, agent_name: str, duration_seconds: float) -> None:
                pass

            def record_outcome(self, agent_name: str, *, success: bool) -> None:
                pass

        assert isinstance(AgentMetricsMiddleware(recorder=_NoopSink()), AgentMiddleware)
