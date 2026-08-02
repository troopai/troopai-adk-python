"""``AgentExecutable.stream_async`` lifts a streamed HITL deferral into
``InterruptException(NestedAgentInterrupt)`` by inspecting the
``RunResultStreaming`` AFTER ``stream_events()`` exhausts.

The streaming runner's interruption arm absorbs the
``AgentToolDeferral`` raised by the inner agent loop and stores the
deferral payload on the result (``deferred_requests`` + ``state``)
instead of re-raising. The bridge therefore cannot rely on the
iteration boundary catching anything — the lift happens post-iteration
by synthesising the deferral payload and routing through the same
``_lift_deferral_to_interrupt`` helper ``invoke`` uses.

The reserved metadata keys ``__interrupt_node_id__`` and
``__nested_agent_snapshots__`` are the BSP loop's side-channel; the
producer side lives in the BSP loop, the consumer side is exercised
here on the adapter.
"""

from __future__ import annotations

from typing import Any

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs.adapters import AgentExecutable
from troopai.adk.graphs.interrupt import InterruptException, NestedAgentInterrupt
from troopai.adk.orchestration.executable import ExecutableInput
from troopai.adk.run.context import RunContext
from troopai.adk.run.state import RunState
from troopai.adk.tools.deferred_tool import DeferredToolCall, DeferredToolRequests
from troopai.adk.types.tokens.llm_usage import LLMUsage


class _DeferredStreamed:
    """Stand-in for ``RunResultStreaming`` returned by ``Runner.arun(stream=True)``
    when the inner agent loop hits a HITL deferral.

    The streaming runner's interruption arm absorbs the
    ``AgentToolDeferral`` raised inside the loop and sets
    ``deferred_requests`` + ``state`` on the streaming result — it
    does NOT re-raise. ``stream_events()`` exhausts cleanly. The
    ``AgentExecutable.stream_async`` bridge inspects the result AFTER
    iteration and lifts the deferral to ``InterruptException``.
    """

    def __init__(
        self,
        *,
        events: list[Any],
        deferred_requests: DeferredToolRequests,
        state: RunState | None,
        agent_name: str = "planner",
    ) -> None:
        self._events = events
        self.deferred_requests = deferred_requests
        self.state = state
        self.current_agent = type("_FakeAgent", (), {"name": agent_name})()
        self.final_output: Any = None
        self.new_items: list[Any] = []
        self.context = type("_FakeRC", (), {"usage": LLMUsage()})()

    async def stream_events(self):  # type: ignore[no-untyped-def]
        # Async generator: yield events then exit cleanly. The deferral
        # is already on self.deferred_requests / self.state, mirroring
        # the production shape where the loop's interruption arm
        # populates those fields and does not re-raise.
        for ev in self._events:
            yield ev


def _deferral_requests_with(tool_call_ids: tuple[str, ...]) -> DeferredToolRequests:
    return DeferredToolRequests(
        approvals=[
            DeferredToolCall(tool_call_id=cid, tool_name="t", tool_arguments={}, raw_arguments="{}")
            for cid in tool_call_ids
        ],
    )


async def test_stream_async_lifts_deferred_requests_to_nested_interrupt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the streamed run exhausts cleanly with ``deferred_requests``
    populated, the bridge synthesises the deferral and raises
    ``InterruptException(NestedAgentInterrupt)`` AFTER iteration —
    inner events that fired surface to the consumer, the terminal
    result event is suppressed, and the snapshot is deposited."""
    deferred_requests = _deferral_requests_with(("c1",))
    fake = _DeferredStreamed(
        events=[{"type": "demo"}, {"type": "demo2"}],
        deferred_requests=deferred_requests,
        state=RunState(current_agent_name="planner", turn_count=4),
    )

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        return fake

    from troopai.adk.run import runner as runner_mod

    monkeypatch.setattr(runner_mod.Runner, "arun", classmethod(fake_arun))

    snapshots: dict[str, RunState] = {}
    executable: AgentExecutable[Any] = AgentExecutable(agent=Agent(name="planner", system_prompt="x"))
    input_ = ExecutableInput(
        content=[],
        metadata={
            "__interrupt_node_id__": "n",
            "__nested_agent_snapshots__": snapshots,
        },
    )
    context: RunContext[Any] = RunContext(context=None, usage=LLMUsage())

    yielded: list[dict[str, Any]] = []
    with pytest.raises(InterruptException) as exc_info:
        async for ev in executable.stream_async(input_, context, config=...):  # type: ignore[arg-type]
            # config is unused on this path; the fake streamed result
            # carries the deferral on its own fields.
            yielded.append(ev)

    # The two inner events emitted before exhaustion surface to the consumer.
    assert yielded == [
        {"type": "agent_event", "event": {"type": "demo"}},
        {"type": "agent_event", "event": {"type": "demo2"}},
    ]
    # The terminal `{"type": "result", ...}` event was suppressed by the lift.
    assert all(ev.get("type") != "result" for ev in yielded)
    assert isinstance(exc_info.value.interrupt, NestedAgentInterrupt)
    assert exc_info.value.interrupt.node_id == "n"
    assert exc_info.value.interrupt.tool_call_ids == ("c1",)
    assert snapshots["n"].turn_count == 4


async def test_stream_async_raises_when_snapshots_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same fail-fast contract as invoke: missing
    ``__nested_agent_snapshots__`` on the side-channel is a producer-side
    programmer error and must raise ``RuntimeError`` rather than silently
    dropping the snapshot. Exercises the precondition check inside
    ``_lift_deferral_to_interrupt`` from the streaming path."""
    fake = _DeferredStreamed(
        events=[],
        deferred_requests=_deferral_requests_with(("c1",)),
        state=RunState(current_agent_name="planner", turn_count=4),
    )

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        return fake

    from troopai.adk.run import runner as runner_mod

    monkeypatch.setattr(runner_mod.Runner, "arun", classmethod(fake_arun))

    executable: AgentExecutable[Any] = AgentExecutable(agent=Agent(name="planner", system_prompt="x"))
    # No __nested_agent_snapshots__ in metadata.
    input_ = ExecutableInput(content=[], metadata={"__interrupt_node_id__": "n"})
    context: RunContext[Any] = RunContext(context=None, usage=LLMUsage())

    with pytest.raises(RuntimeError, match="__nested_agent_snapshots__"):
        async for _ev in executable.stream_async(input_, context, config=...):  # type: ignore[arg-type]
            # config is unused on the failure path.
            pass


async def test_stream_async_raises_when_node_id_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing ``__interrupt_node_id__`` is the same producer-side
    programmer error as in the non-streaming path: the bridge must
    refuse rather than lift to an interrupt whose node_id is empty
    (which would later trip the GraphState cross-reference check with
    no breadcrumb to the bridge). Streaming parity with the invoke
    tests."""
    fake = _DeferredStreamed(
        events=[],
        deferred_requests=_deferral_requests_with(("c1",)),
        state=RunState(current_agent_name="planner", turn_count=4),
    )

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        return fake

    from troopai.adk.run import runner as runner_mod

    monkeypatch.setattr(runner_mod.Runner, "arun", classmethod(fake_arun))

    executable: AgentExecutable[Any] = AgentExecutable(agent=Agent(name="planner", system_prompt="x"))
    # No __interrupt_node_id__ in metadata.
    input_ = ExecutableInput(content=[], metadata={"__nested_agent_snapshots__": {}})
    context: RunContext[Any] = RunContext(context=None, usage=LLMUsage())

    with pytest.raises(RuntimeError, match="__interrupt_node_id__"):
        async for _ev in executable.stream_async(input_, context, config=...):  # type: ignore[arg-type]
            # config is unused on the failure path.
            pass


async def test_stream_async_raises_when_state_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defensive shape: when ``deferred_requests`` is populated but
    ``state`` is None the bridge cannot synthesise a ``RunState`` to
    deposit and must raise a clear ``RuntimeError`` naming the agent.
    The streaming runner's interruption arm always sets both fields
    in lockstep — this guard catches a regression in that contract."""
    fake = _DeferredStreamed(
        events=[],
        deferred_requests=_deferral_requests_with(("c1",)),
        state=None,
    )

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        return fake

    from troopai.adk.run import runner as runner_mod

    monkeypatch.setattr(runner_mod.Runner, "arun", classmethod(fake_arun))

    executable: AgentExecutable[Any] = AgentExecutable(agent=Agent(name="planner", system_prompt="x"))
    input_ = ExecutableInput(
        content=[],
        metadata={
            "__interrupt_node_id__": "n",
            "__nested_agent_snapshots__": {},
        },
    )
    context: RunContext[Any] = RunContext(context=None, usage=LLMUsage())

    with pytest.raises(RuntimeError, match="state"):
        async for _ev in executable.stream_async(input_, context, config=...):  # type: ignore[arg-type]
            # config is unused on the failure path.
            pass


async def test_stream_async_emits_terminal_result_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the inner stream completes with no deferral, the
    post-iteration check is a no-op and the terminal
    ``{"type": "result", "result": NodeResult}`` event still fires."""

    class _SuccessStreamed:
        context = type("C", (), {"usage": LLMUsage()})()
        final_output = "ok"
        new_items: list[Any] = []
        current_agent = type("A", (), {"name": "planner"})()
        deferred_requests: Any = None
        state: Any = None

        async def stream_events(self):  # type: ignore[no-untyped-def]
            yield {"type": "interior"}

    async def fake_arun(cls: Any, *args: Any, **kwargs: Any) -> Any:
        return _SuccessStreamed()

    from troopai.adk.run import runner as runner_mod

    monkeypatch.setattr(runner_mod.Runner, "arun", classmethod(fake_arun))

    executable: AgentExecutable[Any] = AgentExecutable(agent=Agent(name="planner", system_prompt="x"))
    input_ = ExecutableInput(
        content=[],
        metadata={
            "__interrupt_node_id__": "n",
            "__nested_agent_snapshots__": {},
        },
    )
    context: RunContext[Any] = RunContext(context=None, usage=LLMUsage())

    yielded: list[dict[str, Any]] = []
    async for ev in executable.stream_async(input_, context, config=...):  # type: ignore[arg-type]
        # config is unused on this success path; the fake streamed result
        # carries the needed terminal fields directly.
        yielded.append(ev)

    # Two events: the interior event then the terminal result.
    assert len(yielded) == 2
    assert yielded[0] == {"type": "agent_event", "event": {"type": "interior"}}
    assert yielded[1]["type"] == "result"
    assert yielded[1]["result"].output == "ok"
