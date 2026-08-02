"""Tests for ``to_executable()`` type dispatch and ``CallableExecutable``.

Covers:
- ``to_executable`` returns the right adapter for each supported input.
- Arity detection for callables (0-arg, 1-arg, 2-arg text, 2-arg full-input).
- ``CallableExecutable.invoke`` smoke test: passthrough 1-arg callable.
"""

from __future__ import annotations

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.graphs.adapters import (
    AgentExecutable,
    CallableExecutable,
    SwarmExecutable,
    to_executable,
)
from troopai.adk.graphs.graph import Graph
from troopai.adk.orchestration.executable import ExecutableInput, NodeResult
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _agent(name: str = "test-agent") -> Agent:
    return Agent(name=name, system_prompt="noop")


def _minimal_swarm() -> Swarm:
    a = _agent("swarmmember")
    return Swarm(
        members=(a,),
        entry=a,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(1),
    )


def _noop_graph() -> Graph:
    def noop() -> str:
        return "noop"

    return Graph.new("inner").node("n", noop).entry("n").terminal("n").compile()


def _minimal_executable_input() -> ExecutableInput:
    """Construct an ``ExecutableInput`` with a user-role message."""
    # Layer 1 easy-message TypedDict
    msg = {"role": "user", "content": "hi"}
    return ExecutableInput(content=[msg])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# to_executable dispatch
# ---------------------------------------------------------------------------


class TestToExecutableDispatch:
    def test_agent_returns_agent_executable(self) -> None:
        a = _agent()
        result = to_executable(a)
        assert isinstance(result, AgentExecutable)
        assert result.agent is a

    def test_swarm_returns_swarm_executable(self) -> None:
        s = _minimal_swarm()
        result = to_executable(s)
        assert isinstance(result, SwarmExecutable)
        assert result.swarm is s

    def test_graph_returns_graph_itself(self) -> None:
        """``Graph`` implements ``Executable`` — returned as-is, no wrapping."""
        g = _noop_graph()
        result = to_executable(g)
        assert result is g

    def test_sync_callable_returns_callable_executable(self) -> None:
        def my_fn(text: str) -> str:
            return text.upper()

        result = to_executable(my_fn)
        assert isinstance(result, CallableExecutable)

    def test_lambda_returns_callable_executable(self) -> None:
        fn = lambda t: t  # noqa: E731
        result = to_executable(fn)
        assert isinstance(result, CallableExecutable)

    def test_unsupported_type_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Cannot wrap"):
            to_executable(42)  # type: ignore[arg-type]

    def test_unsupported_none_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="Cannot wrap"):
            to_executable(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Arity detection
# ---------------------------------------------------------------------------


class TestArityDetection:
    def test_zero_arg_callable(self) -> None:
        fn = lambda: "ok"  # noqa: E731
        exe = CallableExecutable(fn=fn)
        assert exe.arity == 0
        assert exe.passes_full_input is False

    def test_one_arg_callable(self) -> None:
        fn = lambda t: t  # noqa: E731
        exe = CallableExecutable(fn=fn)
        assert exe.arity == 1
        assert exe.passes_full_input is False

    def test_two_arg_text_and_context(self) -> None:
        def fn(text: str, context: object) -> str:
            del context
            return text

        exe = CallableExecutable(fn=fn)
        assert exe.arity == 2
        assert exe.passes_full_input is False

    def test_two_arg_full_input(self) -> None:
        """When first param is annotated as ``ExecutableInput``,
        ``passes_full_input`` must be ``True``."""

        async def fn(inp: ExecutableInput, context: object) -> str:
            del context
            return str(inp.from_node)

        exe = CallableExecutable(fn=fn)
        assert exe.arity == 2
        assert exe.passes_full_input is True

    def test_too_many_params_raises(self) -> None:
        def fn(a: str, b: str, c: str) -> str:
            del b, c
            return a

        with pytest.raises(ValueError, match="at most 2"):
            CallableExecutable(fn=fn)

    def test_non_callable_raises(self) -> None:
        with pytest.raises(ValueError, match="callable"):
            CallableExecutable(fn="not-callable")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# CallableExecutable.invoke smoke test
# ---------------------------------------------------------------------------


class TestCallableExecutableInvoke:
    @pytest.mark.asyncio
    async def test_passthrough_one_arg_callable(self) -> None:
        """1-arg callable receives the text from ExecutableInput and returns it."""

        def passthrough(text: str) -> str:
            return text

        exe = CallableExecutable(fn=passthrough)
        # Build an input whose content is a user message containing "hi"
        msg = {"role": "user", "content": "hi"}
        inp = ExecutableInput(content=[msg])  # type: ignore[list-item]
        result = await exe.invoke(inp, context=None, config=None)  # type: ignore[arg-type]
        assert isinstance(result, NodeResult)
        assert result.output == "hi"
        assert result.final_text == "hi"

    @pytest.mark.asyncio
    async def test_zero_arg_callable_returns_value(self) -> None:
        def producer() -> str:
            return "produced"

        exe = CallableExecutable(fn=producer)
        inp = _minimal_executable_input()
        result = await exe.invoke(inp, context=None, config=None)  # type: ignore[arg-type]
        assert result.output == "produced"

    @pytest.mark.asyncio
    async def test_async_callable_awaited(self) -> None:
        async def async_fn(text: str) -> str:
            return text + "!"

        exe = CallableExecutable(fn=async_fn)
        msg = {"role": "user", "content": "hello"}
        inp = ExecutableInput(content=[msg])  # type: ignore[list-item]
        result = await exe.invoke(inp, context=None, config=None)  # type: ignore[arg-type]
        assert result.output == "hello!"

    @pytest.mark.asyncio
    async def test_usage_is_zero_for_callable(self) -> None:
        exe = CallableExecutable(fn=lambda t: t)
        inp = _minimal_executable_input()
        result = await exe.invoke(inp, context=None, config=None)  # type: ignore[arg-type]
        assert result.usage.total_tokens == 0
        assert result.usage.requests == 0
