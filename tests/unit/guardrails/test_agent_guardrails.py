"""Tests for agent-level input/output guardrails.

Covers:
- Gap 1: Guardrail results tracked in RunResult/RunResultStreaming
- Gap 2: Global guardrails in RunConfig
- Gap 3: Output guardrails run on correct agent after handoffs
- Gap 4: True parallel-with-agent input guardrails
- Gap 5: Logging in guardrail executor
"""

import asyncio
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentGuardrails,
    AgentGuardrailSeverity,
    AgentGuardrailTimeoutInfo,
    AgentInputGuardrail,
    AgentInputGuardrailData,
    AgentInputGuardrailResult,
    AgentOutputGuardrail,
    AgentOutputGuardrailData,
    AgentOutputGuardrailResult,
    AgentTimeoutPolicy,
)
from troopai.adk.exceptions import (
    AgentInputGuardrailTripwireTriggered,
    AgentOutputGuardrailTripwireTriggered,
)
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.context import RunContext
from troopai.adk.run.guardrails_executor import (
    run_blocking_input_guardrails,
    run_input_guardrails,
    run_output_guardrails,
    run_parallel_input_guardrails,
)
from troopai.adk.run.runner import Runner

# ── Helpers ──────────────────────────────────────────────────


def _make_agent(
    name: str = "test_agent",
    input_guardrails: list | None = None,
    output_guardrails: list | None = None,
) -> Agent:
    """Create a minimal Agent for testing."""
    return Agent(
        name=name,
        system_prompt="test",
        guardrails=AgentGuardrails(
            input=input_guardrails or [],
            output=output_guardrails or [],
        ),
    )


def _passing_input_guardrail(name: str = "pass_guard", parallel: bool = False) -> AgentInputGuardrail:
    """Create a guardrail that always passes."""

    async def _fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
        return AgentGuardrailFunctionOutput(tripwire_triggered=False, output_info=f"{name}_ok")

    return AgentInputGuardrail(guardrail_function=_fn, name=name, run_in_parallel=parallel)


def _failing_input_guardrail(name: str = "fail_guard", parallel: bool = False) -> AgentInputGuardrail:
    """Create a guardrail that always trips."""

    async def _fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
        return AgentGuardrailFunctionOutput(tripwire_triggered=True, output_info=f"{name}_fail")

    return AgentInputGuardrail(guardrail_function=_fn, name=name, run_in_parallel=parallel)


def _passing_output_guardrail(name: str = "pass_out_guard") -> AgentOutputGuardrail:
    """Create an output guardrail that always passes."""

    async def _fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
        return AgentGuardrailFunctionOutput(tripwire_triggered=False, output_info=f"{name}_ok")

    return AgentOutputGuardrail(guardrail_function=_fn, name=name)


def _failing_output_guardrail(name: str = "fail_out_guard") -> AgentOutputGuardrail:
    """Create an output guardrail that always trips."""

    async def _fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
        return AgentGuardrailFunctionOutput(tripwire_triggered=True, output_info=f"{name}_fail")

    return AgentOutputGuardrail(guardrail_function=_fn, name=name)


def _make_context() -> RunContext:
    return RunContext(context=None)


def _make_hooks() -> RunHooks:
    return RunHooks()


# ── Gap 1: Results returned from guardrail executors ─────────


class TestGuardrailResultsReturned:
    """Guardrail executor functions return result lists."""

    @pytest.mark.asyncio
    async def test_blocking_input_returns_results(self) -> None:
        """run_blocking_input_guardrails returns list of AgentInputGuardrailResult."""
        g1 = _passing_input_guardrail("g1", parallel=False)
        g2 = _passing_input_guardrail("g2", parallel=False)
        agent = _make_agent(input_guardrails=[g1, g2])

        results = await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(results) == 2
        assert all(isinstance(r, AgentInputGuardrailResult) for r in results)
        assert results[0].guardrail is g1
        assert results[1].guardrail is g2

    @pytest.mark.asyncio
    async def test_parallel_input_returns_results(self) -> None:
        """run_parallel_input_guardrails returns list of AgentInputGuardrailResult."""
        g1 = _passing_input_guardrail("g1", parallel=True)
        g2 = _passing_input_guardrail("g2", parallel=True)
        agent = _make_agent(input_guardrails=[g1, g2])

        results = await run_parallel_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(results) == 2
        assert all(isinstance(r, AgentInputGuardrailResult) for r in results)

    @pytest.mark.asyncio
    async def test_convenience_input_returns_combined(self) -> None:
        """run_input_guardrails returns blocking + parallel results."""
        blocking = _passing_input_guardrail("blocking", parallel=False)
        parallel = _passing_input_guardrail("parallel", parallel=True)
        agent = _make_agent(input_guardrails=[blocking, parallel])

        results = await run_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(results) == 2
        assert results[0].guardrail is blocking
        assert results[1].guardrail is parallel

    @pytest.mark.asyncio
    async def test_output_returns_results(self) -> None:
        """run_output_guardrails returns list of AgentOutputGuardrailResult."""
        g1 = _passing_output_guardrail("g1")
        g2 = _passing_output_guardrail("g2")
        agent = _make_agent(output_guardrails=[g1, g2])

        results = await run_output_guardrails(agent, "some output", _make_context(), _make_hooks())

        assert len(results) == 2
        assert all(isinstance(r, AgentOutputGuardrailResult) for r in results)

    @pytest.mark.asyncio
    async def test_empty_guardrails_return_empty_list(self) -> None:
        """No guardrails configured returns empty list."""
        agent = _make_agent()

        input_results = await run_input_guardrails(agent, "hello", _make_context(), _make_hooks())
        output_results = await run_output_guardrails(agent, "output", _make_context(), _make_hooks())

        assert input_results == []
        assert output_results == []


# ── Gap 1d: all_results on exceptions ────────────────────────


class TestExceptionAllResults:
    """Tripwire exceptions carry all_results collected before trigger."""

    @pytest.mark.asyncio
    async def test_blocking_tripwire_has_all_results(self) -> None:
        """When blocking guardrail trips, exception carries prior results."""
        g1 = _passing_input_guardrail("g1", parallel=False)
        g2 = _failing_input_guardrail("g2", parallel=False)
        agent = _make_agent(input_guardrails=[g1, g2])

        with pytest.raises(AgentInputGuardrailTripwireTriggered) as exc_info:
            await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        exc = exc_info.value
        assert exc.all_results is not None
        assert len(exc.all_results) == 2
        # First result is the passing guardrail
        assert not exc.all_results[0].guardrail_output.tripwire_triggered
        # Second result is the tripping guardrail
        assert exc.all_results[1].guardrail_output.tripwire_triggered

    @pytest.mark.asyncio
    async def test_parallel_tripwire_has_all_results(self) -> None:
        """When parallel guardrail trips, exception carries results."""
        g1 = _failing_input_guardrail("g1", parallel=True)
        agent = _make_agent(input_guardrails=[g1])

        with pytest.raises(AgentInputGuardrailTripwireTriggered) as exc_info:
            await run_parallel_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        exc = exc_info.value
        assert exc.all_results is not None
        assert len(exc.all_results) == 1

    @pytest.mark.asyncio
    async def test_output_tripwire_has_all_results(self) -> None:
        """When output guardrail trips, exception carries all results."""
        g1 = _passing_output_guardrail("g1")
        g2 = _failing_output_guardrail("g2")
        agent = _make_agent(output_guardrails=[g1, g2])

        with pytest.raises(AgentOutputGuardrailTripwireTriggered) as exc_info:
            await run_output_guardrails(agent, "output", _make_context(), _make_hooks())

        exc = exc_info.value
        assert exc.all_results is not None
        # Note: output guardrails run in parallel, so the number of results
        # depends on gather ordering. At minimum, the tripping result is present.
        assert len(exc.all_results) >= 1

    @pytest.mark.asyncio
    async def test_exception_backward_compatible_without_all_results(self) -> None:
        """all_results defaults to None for backward compatibility."""
        guardrail = _failing_input_guardrail("g1")
        agent = _make_agent()
        output = AgentGuardrailFunctionOutput(tripwire_triggered=True)
        result = AgentInputGuardrailResult(guardrail=guardrail, agent=agent, guardrail_output=output)

        exc = AgentInputGuardrailTripwireTriggered(result)
        assert exc.all_results is None
        assert exc.guardrail_result is result


# ── Gap 2: Config guardrails merge with agent guardrails ─────


class TestConfigGuardrailsMerge:
    """Global guardrails from RunConfig merge with agent guardrails."""

    @pytest.mark.asyncio
    async def test_config_input_guardrails_run_first(self) -> None:
        """Config guardrails execute before agent guardrails."""
        execution_order = []

        async def _config_fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            execution_order.append("config")
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        async def _agent_fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            execution_order.append("agent")
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        config_guard = AgentInputGuardrail(
            guardrail_function=_config_fn,
            name="config_guard",
            run_in_parallel=False,
        )
        agent_guard = AgentInputGuardrail(
            guardrail_function=_agent_fn,
            name="agent_guard",
            run_in_parallel=False,
        )
        agent = _make_agent(input_guardrails=[agent_guard])

        results = await run_blocking_input_guardrails(
            agent,
            "hello",
            _make_context(),
            _make_hooks(),
            config_guardrails=[config_guard],
        )

        assert execution_order == ["config", "agent"]
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_config_output_guardrails_merge(self) -> None:
        """Config and agent output guardrails both run."""
        config_guard = _passing_output_guardrail("config_out")
        agent_guard = _passing_output_guardrail("agent_out")
        agent = _make_agent(output_guardrails=[agent_guard])

        results = await run_output_guardrails(
            agent,
            "output",
            _make_context(),
            _make_hooks(),
            config_guardrails=[config_guard],
        )

        assert len(results) == 2
        names = {r.guardrail.get_name() for r in results}
        assert names == {"config_out", "agent_out"}

    @pytest.mark.asyncio
    async def test_config_only_no_agent_guardrails(self) -> None:
        """Config guardrails run even when agent has none."""
        config_guard = _passing_input_guardrail("config_only", parallel=False)
        agent = _make_agent()  # No guardrails

        results = await run_blocking_input_guardrails(
            agent,
            "hello",
            _make_context(),
            _make_hooks(),
            config_guardrails=[config_guard],
        )

        assert len(results) == 1
        assert results[0].guardrail.get_name() == "config_only"

    @pytest.mark.asyncio
    async def test_none_config_guardrails_safe(self) -> None:
        """Passing None for config_guardrails is safe."""
        agent_guard = _passing_input_guardrail("agent", parallel=False)
        agent = _make_agent(input_guardrails=[agent_guard])

        results = await run_blocking_input_guardrails(
            agent,
            "hello",
            _make_context(),
            _make_hooks(),
            config_guardrails=None,
        )

        assert len(results) == 1


# ── Gap 4: Blocking vs parallel split ────────────────────────


class TestBlockingParallelSplit:
    """Blocking and parallel guardrails are correctly split."""

    @pytest.mark.asyncio
    async def test_blocking_only_runs_blocking(self) -> None:
        """run_blocking_input_guardrails ignores parallel guardrails."""
        blocking = _passing_input_guardrail("blocking", parallel=False)
        parallel = _passing_input_guardrail("parallel", parallel=True)
        agent = _make_agent(input_guardrails=[blocking, parallel])

        results = await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(results) == 1
        assert results[0].guardrail.get_name() == "blocking"

    @pytest.mark.asyncio
    async def test_parallel_only_runs_parallel(self) -> None:
        """run_parallel_input_guardrails ignores blocking guardrails."""
        blocking = _passing_input_guardrail("blocking", parallel=False)
        parallel = _passing_input_guardrail("parallel", parallel=True)
        agent = _make_agent(input_guardrails=[blocking, parallel])

        results = await run_parallel_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(results) == 1
        assert results[0].guardrail.get_name() == "parallel"

    @pytest.mark.asyncio
    async def test_parallel_guardrails_run_concurrently(self) -> None:
        """Parallel guardrails overlap in time (concurrency test)."""
        timestamps = {}

        async def _slow_guardrail(name: str):
            async def _fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
                timestamps[f"{name}_start"] = asyncio.get_event_loop().time()
                await asyncio.sleep(0.05)
                timestamps[f"{name}_end"] = asyncio.get_event_loop().time()
                return AgentGuardrailFunctionOutput(tripwire_triggered=False)

            return _fn

        g1_fn = await _slow_guardrail("g1")
        g2_fn = await _slow_guardrail("g2")

        g1 = AgentInputGuardrail(guardrail_function=g1_fn, name="g1", run_in_parallel=True)
        g2 = AgentInputGuardrail(guardrail_function=g2_fn, name="g2", run_in_parallel=True)
        agent = _make_agent(input_guardrails=[g1, g2])

        await run_parallel_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        # If they ran concurrently, g2 started before g1 finished
        assert timestamps["g2_start"] < timestamps["g1_end"]


# ── Gap 5: Logging ───────────────────────────────────────────


class TestGuardrailLogging:
    """Guardrail executor produces log output."""

    @pytest.mark.asyncio
    async def test_blocking_guardrail_logs(self, caplog) -> None:
        """Blocking guardrails log start and completion."""
        guard = _passing_input_guardrail("logged_guard", parallel=False)
        agent = _make_agent(input_guardrails=[guard])

        with caplog.at_level(logging.DEBUG, logger="troopai.adk.run.guardrails_executor"):
            await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        messages = [r.message for r in caplog.records]
        assert any("blocking input guardrails" in m for m in messages)
        assert any("logged_guard" in m for m in messages)

    @pytest.mark.asyncio
    async def test_tripwire_logs_at_info(self, caplog) -> None:
        """Tripwire trigger logs at INFO level."""
        guard = _failing_input_guardrail("fail_guard", parallel=False)
        agent = _make_agent(input_guardrails=[guard])

        with (
            caplog.at_level(logging.DEBUG, logger="troopai.adk.run.guardrails_executor"),
            pytest.raises(AgentInputGuardrailTripwireTriggered),
        ):
            await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        info_messages = [r.message for r in caplog.records if r.levelno >= logging.INFO]
        assert any("tripwire triggered" in m for m in info_messages)


# ── RunResult / RunResultStreaming fields ─────────────────────


class TestResultFields:
    """RunResult and RunResultStreaming have guardrail result fields."""

    def test_run_result_defaults_empty(self) -> None:
        """RunResult guardrail fields default to empty tuples."""
        from troopai.adk.types.run import RunResult

        result = RunResult(final_output="test", user_prompt="hello")
        assert result.guardrail_results.input == ()
        assert result.guardrail_results.output == ()
        assert isinstance(result.guardrail_results.input, tuple)
        assert isinstance(result.guardrail_results.output, tuple)

    def test_run_result_streaming_has_guardrail_fields(self) -> None:
        """RunResultStreaming has guardrail result fields with correct defaults."""
        import dataclasses

        from troopai.adk.run.stream import RunResultStreaming

        fields = {f.name: f for f in dataclasses.fields(RunResultStreaming)}
        assert "guardrail_results" in fields


# ── RunConfig guardrail fields ───────────────────────────────


class TestRunConfigGuardrails:
    """RunConfig has input_guardrails and output_guardrails fields."""

    def test_defaults_to_empty(self) -> None:
        """RunConfig guardrails default to empty AgentGuardrails."""
        from troopai.adk.run.config import RunConfig

        config = RunConfig()
        assert config.guardrails.input == []
        assert config.guardrails.output == []

    def test_set_guardrails(self) -> None:
        """RunConfig accepts guardrail lists via the AgentGuardrails config object."""
        from troopai.adk.run.config import RunConfig

        ig = _passing_input_guardrail("config_in")
        og = _passing_output_guardrail("config_out")
        config = RunConfig(guardrails=AgentGuardrails(input=[ig], output=[og]))

        assert len(config.guardrails.input) == 1
        assert len(config.guardrails.output) == 1


# ── RunnerProfile guardrail methods ──────────────────────────


class TestRunnerProfileGuardrails:
    """RunnerProfile supports run-scope input/output guardrails."""

    def test_profile_wires_guardrails_into_config(self) -> None:
        """Profile guardrail method sets guardrails on copied RunConfig."""
        ig = _passing_input_guardrail("profile_in")
        og = _passing_output_guardrail("profile_out")

        profile = Runner.configure().guardrails(input=[ig], output=[og])

        config = profile.run_config
        assert config.guardrails.input == [ig]
        assert config.guardrails.output == [og]

    def test_profile_methods_return_new_profile(self) -> None:
        """Profile methods return new immutable profiles."""
        profile = Runner.configure()

        result = profile.guardrails(input=[])
        assert result is not profile

        result = profile.guardrails(output=[])
        assert result is not profile


# ── Improvement 5: Immutability ──────────────────────────────


class TestResultImmutability:
    """RunResult guardrail results are immutable tuples."""

    def test_run_result_input_results_are_tuple(self) -> None:
        """input_guardrail_results on RunResult is a tuple."""
        from troopai.adk.types.run import RunResult

        result = RunResult(final_output="test", user_prompt="hello")
        assert isinstance(result.guardrail_results.input, tuple)

        # Assigning a tuple works
        result.guardrail_results.input = (MagicMock(),)
        assert len(result.guardrail_results.input) == 1

    def test_run_result_output_results_are_tuple(self) -> None:
        """output_guardrail_results on RunResult is a tuple."""
        from troopai.adk.types.run import RunResult

        result = RunResult(final_output="test", user_prompt="hello")
        assert isinstance(result.guardrail_results.output, tuple)


# ── Improvement 3: Result ordering ──────────────────────────


class TestResultOrdering:
    """Parallel guardrail results preserve input order."""

    @pytest.mark.asyncio
    async def test_parallel_results_match_input_order(self) -> None:
        """Results are returned in guardrail definition order, not completion order."""
        completion_order = []

        async def _slow_fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            await asyncio.sleep(0.05)
            completion_order.append("slow")
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        async def _fast_fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            completion_order.append("fast")
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        slow = AgentInputGuardrail(guardrail_function=_slow_fn, name="slow", run_in_parallel=True)
        fast = AgentInputGuardrail(guardrail_function=_fast_fn, name="fast", run_in_parallel=True)
        # slow is first in definition order, but fast finishes first
        agent = _make_agent(input_guardrails=[slow, fast])

        results = await run_parallel_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        # fast completed first...
        assert completion_order[0] == "fast"
        # ...but results preserve input (definition) order
        assert results[0].guardrail.get_name() == "slow"
        assert results[1].guardrail.get_name() == "fast"


# ── Enriched result types: agent + agent_output fields ────


class TestInputGuardrailResultAgent:
    """AgentInputGuardrailResult carries the agent whose input was checked."""

    @pytest.mark.asyncio
    async def test_blocking_result_has_agent(self) -> None:
        """Blocking input guardrail result includes the checked agent."""
        guard = _passing_input_guardrail("g1", parallel=False)
        agent = _make_agent(name="my_agent", input_guardrails=[guard])

        results = await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(results) == 1
        assert results[0].agent is agent
        assert results[0].agent.name == "my_agent"

    @pytest.mark.asyncio
    async def test_parallel_result_has_agent(self) -> None:
        """Parallel input guardrail result includes the checked agent."""
        guard = _passing_input_guardrail("g1", parallel=True)
        agent = _make_agent(name="parallel_agent", input_guardrails=[guard])

        results = await run_parallel_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(results) == 1
        assert results[0].agent is agent

    @pytest.mark.asyncio
    async def test_tripwire_result_has_agent(self) -> None:
        """Tripping input guardrail result includes the checked agent."""
        guard = _failing_input_guardrail("g1", parallel=False)
        agent = _make_agent(input_guardrails=[guard])

        with pytest.raises(AgentInputGuardrailTripwireTriggered) as exc_info:
            await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert exc_info.value.guardrail_result.agent is agent


class TestOutputGuardrailResultAgentAndOutput:
    """AgentOutputGuardrailResult carries agent and agent_output."""

    @pytest.mark.asyncio
    async def test_result_has_agent(self) -> None:
        """Output guardrail result includes the producing agent."""
        guard = _passing_output_guardrail("g1")
        agent = _make_agent(name="output_agent", output_guardrails=[guard])

        results = await run_output_guardrails(agent, "some output", _make_context(), _make_hooks())

        assert len(results) == 1
        assert results[0].agent is agent
        assert results[0].agent.name == "output_agent"

    @pytest.mark.asyncio
    async def test_result_has_agent_output(self) -> None:
        """Output guardrail result includes the validated agent output."""
        guard = _passing_output_guardrail("g1")
        agent = _make_agent(output_guardrails=[guard])
        agent_output = {"answer": "hello", "confidence": 0.95}

        results = await run_output_guardrails(agent, agent_output, _make_context(), _make_hooks())

        assert len(results) == 1
        assert results[0].agent_output is agent_output

    @pytest.mark.asyncio
    async def test_tripwire_result_has_agent_and_output(self) -> None:
        """Tripping output guardrail result includes agent and agent_output."""
        guard = _failing_output_guardrail("g1")
        agent = _make_agent(output_guardrails=[guard])
        agent_output = "bad content"

        with pytest.raises(AgentOutputGuardrailTripwireTriggered) as exc_info:
            await run_output_guardrails(agent, agent_output, _make_context(), _make_hooks())

        result = exc_info.value.guardrail_result
        assert result.agent is agent
        assert result.agent_output is agent_output

    @pytest.mark.asyncio
    async def test_multiple_output_guardrails_share_same_agent_output(self) -> None:
        """All output guardrail results reference the same agent output object."""
        g1 = _passing_output_guardrail("g1")
        g2 = _passing_output_guardrail("g2")
        agent = _make_agent(output_guardrails=[g1, g2])
        agent_output = ["item1", "item2"]

        results = await run_output_guardrails(agent, agent_output, _make_context(), _make_hooks())

        assert len(results) == 2
        assert results[0].agent_output is results[1].agent_output
        assert results[0].agent_output is agent_output


# ── Improvement A: Guardrail Severity ────────────────────────


class TestGuardrailSeverity:
    """AgentGuardrailSeverity controls halt behavior independently of tripwire_triggered."""

    @pytest.mark.asyncio
    async def test_severity_none_uses_tripwire(self) -> None:
        """When severity is None, tripwire_triggered controls halting."""
        guard = _failing_input_guardrail("g1", parallel=False)
        agent = _make_agent(input_guardrails=[guard])

        with pytest.raises(AgentInputGuardrailTripwireTriggered):
            await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

    @pytest.mark.asyncio
    async def test_severity_error_halts(self) -> None:
        """severity=ERROR halts even if tripwire_triggered=False."""

        async def _fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(
                tripwire_triggered=False,
                severity=AgentGuardrailSeverity.ERROR,
                output_info="severity override",
            )

        guard = AgentInputGuardrail(guardrail_function=_fn, name="error_guard", run_in_parallel=False)
        agent = _make_agent(input_guardrails=[guard])

        with pytest.raises(AgentInputGuardrailTripwireTriggered):
            await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

    @pytest.mark.asyncio
    async def test_severity_warning_does_not_halt(self) -> None:
        """severity=WARNING does not halt even if tripwire_triggered=True."""

        async def _fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(
                tripwire_triggered=True,
                severity=AgentGuardrailSeverity.WARNING,
                output_info="possible PII",
            )

        guard = AgentInputGuardrail(guardrail_function=_fn, name="warning_guard", run_in_parallel=False)
        agent = _make_agent(input_guardrails=[guard])

        results = await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(results) == 1
        assert results[0].guardrail_output.severity == AgentGuardrailSeverity.WARNING
        assert results[0].guardrail_output.tripwire_triggered is True

    @pytest.mark.asyncio
    async def test_severity_info_does_not_halt(self) -> None:
        """severity=INFO does not halt even if tripwire_triggered=True."""

        async def _fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(
                tripwire_triggered=True,
                severity=AgentGuardrailSeverity.INFO,
                output_info="low confidence",
            )

        guard = AgentInputGuardrail(guardrail_function=_fn, name="info_guard", run_in_parallel=False)
        agent = _make_agent(input_guardrails=[guard])

        results = await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(results) == 1
        assert results[0].guardrail_output.severity == AgentGuardrailSeverity.INFO

    @pytest.mark.asyncio
    async def test_severity_warning_on_output_guardrail(self) -> None:
        """severity=WARNING on output guardrail does not halt."""

        async def _fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(
                tripwire_triggered=True,
                severity=AgentGuardrailSeverity.WARNING,
                output_info="possible issue",
            )

        guard = AgentOutputGuardrail(guardrail_function=_fn, name="warn_out")
        agent = _make_agent(output_guardrails=[guard])

        results = await run_output_guardrails(agent, "output", _make_context(), _make_hooks())

        assert len(results) == 1
        assert results[0].guardrail_output.severity == AgentGuardrailSeverity.WARNING

    @pytest.mark.asyncio
    async def test_severity_error_on_output_guardrail_halts(self) -> None:
        """severity=ERROR on output guardrail halts."""

        async def _fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(
                severity=AgentGuardrailSeverity.ERROR,
            )

        guard = AgentOutputGuardrail(guardrail_function=_fn, name="error_out")
        agent = _make_agent(output_guardrails=[guard])

        with pytest.raises(AgentOutputGuardrailTripwireTriggered):
            await run_output_guardrails(agent, "output", _make_context(), _make_hooks())

    @pytest.mark.asyncio
    async def test_severity_warning_logged_at_warning(self, caplog) -> None:
        """severity=WARNING produces WARNING-level log."""

        async def _fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(
                tripwire_triggered=True,
                severity=AgentGuardrailSeverity.WARNING,
                output_info="audit info",
            )

        guard = AgentInputGuardrail(guardrail_function=_fn, name="warn_guard", run_in_parallel=False)
        agent = _make_agent(input_guardrails=[guard])

        with caplog.at_level(logging.DEBUG, logger="troopai.adk.run.guardrails_executor"):
            await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("warning" in m.lower() and "warn_guard" in m for m in warning_msgs)

    @pytest.mark.asyncio
    async def test_parallel_severity_warning_no_halt(self) -> None:
        """Parallel input guardrail with severity=WARNING does not halt."""

        async def _fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(
                tripwire_triggered=True,
                severity=AgentGuardrailSeverity.WARNING,
            )

        guard = AgentInputGuardrail(guardrail_function=_fn, name="par_warn", run_in_parallel=True)
        agent = _make_agent(input_guardrails=[guard])

        results = await run_parallel_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(results) == 1
        assert results[0].guardrail_output.severity == AgentGuardrailSeverity.WARNING

    @pytest.mark.asyncio
    async def test_mixed_severity_first_error_halts(self) -> None:
        """With multiple guardrails, first ERROR halts even if others are WARNING."""

        async def _warning_fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(
                tripwire_triggered=True,
                severity=AgentGuardrailSeverity.WARNING,
            )

        async def _error_fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(
                severity=AgentGuardrailSeverity.ERROR,
            )

        g1 = AgentInputGuardrail(guardrail_function=_warning_fn, name="warn", run_in_parallel=False)
        g2 = AgentInputGuardrail(guardrail_function=_error_fn, name="error", run_in_parallel=False)
        agent = _make_agent(input_guardrails=[g1, g2])

        with pytest.raises(AgentInputGuardrailTripwireTriggered) as exc_info:
            await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        # g1 (warning) passed, g2 (error) halted
        assert exc_info.value.all_results is not None
        assert len(exc_info.value.all_results) == 2
        assert exc_info.value.guardrail_result.guardrail.get_name() == "error"


# ── Improvement C: Guardrail Timeout ─────────────────────────


class TestGuardrailTimeout:
    """Guardrail timeout wraps execution in asyncio.wait_for."""

    @pytest.mark.asyncio
    async def test_no_timeout_runs_normally(self) -> None:
        """Without timeout, guardrail runs normally."""
        guard = _passing_input_guardrail("no_timeout", parallel=False)
        agent = _make_agent(input_guardrails=[guard])

        results = await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_timeout_fail_policy_trips(self) -> None:
        """Timeout with FAIL policy trips the guardrail."""

        async def _slow_fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            await asyncio.sleep(5.0)
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        guard = AgentInputGuardrail(
            guardrail_function=_slow_fn,
            name="slow_guard",
            run_in_parallel=False,
            timeout=0.05,
            timeout_policy=AgentTimeoutPolicy.FAIL,
        )
        agent = _make_agent(input_guardrails=[guard])

        with pytest.raises(AgentInputGuardrailTripwireTriggered) as exc_info:
            await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        result = exc_info.value.guardrail_result
        assert "timed out" in str(result.guardrail_output.output_info)

    @pytest.mark.asyncio
    async def test_timeout_pass_policy_continues(self) -> None:
        """Timeout with PASS policy continues without halting."""

        async def _slow_fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            await asyncio.sleep(5.0)
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        guard = AgentInputGuardrail(
            guardrail_function=_slow_fn,
            name="slow_pass",
            run_in_parallel=False,
            timeout=0.05,
            timeout_policy=AgentTimeoutPolicy.PASS,
        )
        agent = _make_agent(input_guardrails=[guard])

        results = await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(results) == 1
        assert "timed out" in str(results[0].guardrail_output.output_info)
        assert results[0].guardrail_output.tripwire_triggered is False

    @pytest.mark.asyncio
    async def test_timeout_on_output_guardrail(self) -> None:
        """Output guardrail respects timeout."""

        async def _slow_fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            await asyncio.sleep(5.0)
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        guard = AgentOutputGuardrail(
            guardrail_function=_slow_fn,
            name="slow_out",
            timeout=0.05,
            timeout_policy=AgentTimeoutPolicy.FAIL,
        )
        agent = _make_agent(output_guardrails=[guard])

        with pytest.raises(AgentOutputGuardrailTripwireTriggered):
            await run_output_guardrails(agent, "output", _make_context(), _make_hooks())

    @pytest.mark.asyncio
    async def test_timeout_parallel_guardrail(self) -> None:
        """Parallel input guardrail timeout with PASS policy continues."""

        async def _slow_fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            await asyncio.sleep(5.0)
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        guard = AgentInputGuardrail(
            guardrail_function=_slow_fn,
            name="par_slow",
            run_in_parallel=True,
            timeout=0.05,
            timeout_policy=AgentTimeoutPolicy.PASS,
        )
        agent = _make_agent(input_guardrails=[guard])

        results = await run_parallel_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(results) == 1
        assert results[0].guardrail_output.tripwire_triggered is False

    @pytest.mark.asyncio
    async def test_within_timeout_runs_normally(self) -> None:
        """Guardrail that finishes within timeout returns actual result."""

        async def _fast_fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(
                tripwire_triggered=False,
                output_info="fast_result",
            )

        guard = AgentInputGuardrail(
            guardrail_function=_fast_fn,
            name="fast_guard",
            run_in_parallel=False,
            timeout=5.0,
        )
        agent = _make_agent(input_guardrails=[guard])

        results = await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(results) == 1
        assert results[0].guardrail_output.output_info == "fast_result"

    @pytest.mark.asyncio
    async def test_timeout_logs_warning(self, caplog) -> None:
        """Timeout produces a WARNING-level log."""

        async def _slow_fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            await asyncio.sleep(5.0)
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        guard = AgentInputGuardrail(
            guardrail_function=_slow_fn,
            name="logged_timeout",
            run_in_parallel=False,
            timeout=0.05,
            timeout_policy=AgentTimeoutPolicy.PASS,
        )
        agent = _make_agent(input_guardrails=[guard])

        with caplog.at_level(logging.DEBUG, logger="troopai.adk.run.guardrails_executor"):
            await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("timed out" in m and "logged_timeout" in m for m in warning_msgs)

    @pytest.mark.asyncio
    async def test_on_timeout_callback_invoked(self) -> None:
        """on_timeout callback is invoked with AgentGuardrailTimeoutInfo on timeout."""
        captured_info: list[AgentGuardrailTimeoutInfo] = []

        async def _on_timeout(info: AgentGuardrailTimeoutInfo) -> None:
            captured_info.append(info)

        async def _slow_fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            await asyncio.sleep(5.0)
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        guard = AgentInputGuardrail(
            guardrail_function=_slow_fn,
            name="callback_guard",
            run_in_parallel=False,
            timeout=0.05,
            timeout_policy=AgentTimeoutPolicy.PASS,
            on_timeout=_on_timeout,
        )
        agent = _make_agent(name="cb_agent", input_guardrails=[guard])

        await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(captured_info) == 1
        info = captured_info[0]
        assert info.guardrail_name == "callback_guard"
        assert info.agent_name == "cb_agent"
        assert info.timeout == 0.05
        assert info.policy == AgentTimeoutPolicy.PASS

    @pytest.mark.asyncio
    async def test_on_timeout_callback_with_fail_policy(self) -> None:
        """on_timeout callback fires even when policy is FAIL."""
        captured: list[AgentGuardrailTimeoutInfo] = []

        async def _on_timeout(info: AgentGuardrailTimeoutInfo) -> None:
            captured.append(info)

        async def _slow_fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            await asyncio.sleep(5.0)
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        guard = AgentInputGuardrail(
            guardrail_function=_slow_fn,
            name="fail_cb",
            run_in_parallel=False,
            timeout=0.05,
            timeout_policy=AgentTimeoutPolicy.FAIL,
            on_timeout=_on_timeout,
        )
        agent = _make_agent(input_guardrails=[guard])

        with pytest.raises(AgentInputGuardrailTripwireTriggered):
            await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(captured) == 1
        assert captured[0].policy == AgentTimeoutPolicy.FAIL

    @pytest.mark.asyncio
    async def test_on_timeout_callback_on_output_guardrail(self) -> None:
        """on_timeout callback works on output guardrails too."""
        captured: list[AgentGuardrailTimeoutInfo] = []

        async def _on_timeout(info: AgentGuardrailTimeoutInfo) -> None:
            captured.append(info)

        async def _slow_fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            await asyncio.sleep(5.0)
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        guard = AgentOutputGuardrail(
            guardrail_function=_slow_fn,
            name="out_cb",
            timeout=0.05,
            timeout_policy=AgentTimeoutPolicy.PASS,
            on_timeout=_on_timeout,
        )
        agent = _make_agent(output_guardrails=[guard])

        results = await run_output_guardrails(agent, "output", _make_context(), _make_hooks())

        assert len(results) == 1
        assert len(captured) == 1
        assert captured[0].guardrail_name == "out_cb"

    @pytest.mark.asyncio
    async def test_no_callback_when_no_timeout(self) -> None:
        """on_timeout callback is NOT invoked when guardrail completes in time."""
        captured: list[AgentGuardrailTimeoutInfo] = []

        async def _on_timeout(info: AgentGuardrailTimeoutInfo) -> None:
            captured.append(info)

        async def _fast_fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        guard = AgentInputGuardrail(
            guardrail_function=_fast_fn,
            name="fast_cb",
            run_in_parallel=False,
            timeout=5.0,
            on_timeout=_on_timeout,
        )
        agent = _make_agent(input_guardrails=[guard])

        await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

        assert len(captured) == 0


# ── Improvement B: Output Guardrail Remediation ──────────────


class TestOutputGuardrailRemediation:
    """Output guardrail remediation retries via on_remediate callback."""

    @pytest.mark.asyncio
    async def test_remediation_retries_and_passes(self) -> None:
        """Remediation re-runs guardrails with new output; passes on retry."""
        call_count = 0

        async def _fn(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            nonlocal call_count
            call_count += 1
            if data.output == "bad":
                return AgentGuardrailFunctionOutput(tripwire_triggered=True, output_info="bad output")
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        guard = AgentOutputGuardrail(
            guardrail_function=_fn,
            name="remediate_guard",
            remediation="Fix your output.",
            max_retries=1,
        )
        agent = _make_agent(output_guardrails=[guard])

        async def on_remediate(msg: str) -> str:
            assert msg == "Fix your output."
            return "good"

        results = await run_output_guardrails(
            agent,
            "bad",
            _make_context(),
            _make_hooks(),
            on_remediate=on_remediate,
        )

        assert len(results) == 1
        assert results[0].guardrail_output.tripwire_triggered is False
        assert call_count == 2  # First call tripped, second passed

    @pytest.mark.asyncio
    async def test_remediation_exhausted_raises(self) -> None:
        """After max_retries exhausted, raises AgentOutputGuardrailTripwireTriggered."""

        async def _fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(tripwire_triggered=True)

        guard = AgentOutputGuardrail(
            guardrail_function=_fn,
            name="always_fail",
            remediation="Try again.",
            max_retries=2,
        )
        agent = _make_agent(output_guardrails=[guard])

        remediate_count = 0

        async def on_remediate(_msg: str) -> str:
            nonlocal remediate_count
            remediate_count += 1
            return f"attempt_{remediate_count}"

        with pytest.raises(AgentOutputGuardrailTripwireTriggered):
            await run_output_guardrails(
                agent,
                "bad",
                _make_context(),
                _make_hooks(),
                on_remediate=on_remediate,
            )

        assert remediate_count == 2  # Tried max_retries times

    @pytest.mark.asyncio
    async def test_no_remediation_raises_immediately(self) -> None:
        """Without remediation field, trips raise immediately."""
        guard = _failing_output_guardrail("no_remediation")
        agent = _make_agent(output_guardrails=[guard])

        with pytest.raises(AgentOutputGuardrailTripwireTriggered):
            await run_output_guardrails(agent, "output", _make_context(), _make_hooks())

    @pytest.mark.asyncio
    async def test_no_callback_raises_even_with_remediation(self) -> None:
        """With remediation but no on_remediate callback, trips raise."""

        async def _fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(tripwire_triggered=True)

        guard = AgentOutputGuardrail(
            guardrail_function=_fn,
            name="has_remediation",
            remediation="Fix it.",
            max_retries=3,
        )
        agent = _make_agent(output_guardrails=[guard])

        with pytest.raises(AgentOutputGuardrailTripwireTriggered):
            await run_output_guardrails(
                agent,
                "output",
                _make_context(),
                _make_hooks(),
                on_remediate=None,
            )

    @pytest.mark.asyncio
    async def test_remediation_logs(self, caplog) -> None:
        """Remediation attempts are logged."""
        call_count = 0

        async def _fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return AgentGuardrailFunctionOutput(tripwire_triggered=True)
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        guard = AgentOutputGuardrail(
            guardrail_function=_fn,
            name="logged_remediation",
            remediation="Please fix.",
            max_retries=1,
        )
        agent = _make_agent(output_guardrails=[guard])

        async def on_remediate(_msg: str) -> str:
            return "fixed"

        with caplog.at_level(logging.DEBUG, logger="troopai.adk.run.guardrails_executor"):
            await run_output_guardrails(
                agent,
                "bad",
                _make_context(),
                _make_hooks(),
                on_remediate=on_remediate,
            )

        info_msgs = [r.message for r in caplog.records if r.levelno >= logging.INFO]
        assert any("remediation" in m.lower() and "logged_remediation" in m for m in info_msgs)

    @pytest.mark.asyncio
    async def test_remediation_updates_agent_output_on_result(self) -> None:
        """After remediation, result.agent_output reflects the new output."""
        call_count = 0

        async def _fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return AgentGuardrailFunctionOutput(tripwire_triggered=True)
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        guard = AgentOutputGuardrail(
            guardrail_function=_fn,
            name="remediation_output",
            remediation="Fix it.",
            max_retries=1,
        )
        agent = _make_agent(output_guardrails=[guard])

        async def on_remediate(_msg: str) -> str:
            return "corrected_output"

        results = await run_output_guardrails(
            agent,
            "bad_output",
            _make_context(),
            _make_hooks(),
            on_remediate=on_remediate,
        )

        assert len(results) == 1
        assert results[0].agent_output == "corrected_output"

    @pytest.mark.asyncio
    async def test_remediation_via_decorator(self) -> None:
        """@agent_output_guardrail decorator supports remediation and max_retries."""
        from troopai.adk.agents.agent_guardrails import agent_output_guardrail

        call_count = 0

        @agent_output_guardrail(remediation="Remove PII.", max_retries=2)
        async def pii_check(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return AgentGuardrailFunctionOutput(tripwire_triggered=True)
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        assert pii_check.remediation == "Remove PII."
        assert pii_check.max_retries == 2


# ── Sync guardrail functions ─────────────────────────────────


class TestSyncGuardrailFunctions:
    """Sync (non-async) guardrail functions are supported."""

    @pytest.mark.asyncio
    async def test_sync_input_guardrail(self) -> None:
        """Sync input guardrail function works."""

        def _sync_fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(tripwire_triggered=False, output_info="sync_ok")

        guard = AgentInputGuardrail(guardrail_function=_sync_fn, name="sync_in", run_in_parallel=False)
        agent = _make_agent(input_guardrails=[guard])
        results = await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())
        assert len(results) == 1
        assert results[0].guardrail_output.output_info == "sync_ok"

    @pytest.mark.asyncio
    async def test_sync_output_guardrail(self) -> None:
        """Sync output guardrail function works."""

        def _sync_fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(tripwire_triggered=False, output_info="sync_out_ok")

        guard = AgentOutputGuardrail(guardrail_function=_sync_fn, name="sync_out")
        agent = _make_agent(output_guardrails=[guard])
        results = await run_output_guardrails(agent, "output", _make_context(), _make_hooks())
        assert len(results) == 1
        assert results[0].guardrail_output.output_info == "sync_out_ok"

    @pytest.mark.asyncio
    async def test_sync_input_guardrail_via_decorator(self) -> None:
        """@agent_input_guardrail decorator works with sync function."""
        from troopai.adk.agents.agent_guardrails import agent_input_guardrail

        @agent_input_guardrail
        def check_sync(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        assert check_sync.get_name() == "check_sync"

    @pytest.mark.asyncio
    async def test_sync_output_guardrail_via_decorator(self) -> None:
        """@agent_output_guardrail decorator works with sync function."""
        from troopai.adk.agents.agent_guardrails import agent_output_guardrail

        @agent_output_guardrail
        def check_sync_out(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        assert check_sync_out.get_name() == "check_sync_out"

    @pytest.mark.asyncio
    async def test_sync_tripping_guardrail(self) -> None:
        """Sync guardrail that trips raises correctly."""

        def _sync_fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(tripwire_triggered=True, output_info="sync_trip")

        guard = AgentInputGuardrail(guardrail_function=_sync_fn, name="sync_trip", run_in_parallel=False)
        agent = _make_agent(input_guardrails=[guard])
        with pytest.raises(AgentInputGuardrailTripwireTriggered):
            await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())


# ── Non-tripwire exception propagation ───────────────────────


class TestGuardrailNonTripwireException:
    """Guardrail functions that raise non-tripwire exceptions propagate correctly."""

    @pytest.mark.asyncio
    async def test_input_guardrail_exception_propagates(self) -> None:
        """Non-tripwire exception in input guardrail propagates."""

        async def _fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            raise ValueError("guardrail internal error")

        guard = AgentInputGuardrail(guardrail_function=_fn, name="error_guard", run_in_parallel=False)
        agent = _make_agent(input_guardrails=[guard])
        with pytest.raises(ValueError, match="guardrail internal error"):
            await run_blocking_input_guardrails(agent, "hello", _make_context(), _make_hooks())

    @pytest.mark.asyncio
    async def test_parallel_guardrail_exception_propagates(self) -> None:
        """Non-tripwire exception in parallel guardrail propagates."""

        async def _fn(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            raise RuntimeError("parallel error")

        guard = AgentInputGuardrail(guardrail_function=_fn, name="par_error", run_in_parallel=True)
        agent = _make_agent(input_guardrails=[guard])
        with pytest.raises(RuntimeError, match="parallel error"):
            await run_parallel_input_guardrails(agent, "hello", _make_context(), _make_hooks())

    @pytest.mark.asyncio
    async def test_output_guardrail_exception_propagates(self) -> None:
        """Non-tripwire exception in output guardrail propagates."""

        async def _fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            raise TypeError("output error")

        guard = AgentOutputGuardrail(guardrail_function=_fn, name="out_error")
        agent = _make_agent(output_guardrails=[guard])
        with pytest.raises(TypeError, match="output error"):
            await run_output_guardrails(agent, "output", _make_context(), _make_hooks())


# ── Hook invocation ──────────────────────────────────────────


class TestHookInvocation:
    """Guardrail executor invokes hooks at correct points."""

    @pytest.mark.asyncio
    async def test_input_hooks_called(self) -> None:
        """Input guardrail hooks are called with correct arguments."""
        hooks = RunHooks()
        hooks.on_input_guardrail_start = AsyncMock()
        hooks.on_input_guardrail_end = AsyncMock()

        guard = _passing_input_guardrail("hooked", parallel=False)
        agent = _make_agent(input_guardrails=[guard])
        ctx = _make_context()

        await run_blocking_input_guardrails(agent, "hello", ctx, hooks)

        hooks.on_input_guardrail_start.assert_called_once()
        hooks.on_input_guardrail_end.assert_called_once()
        # Verify args: context, agent, guardrail_name
        start_args = hooks.on_input_guardrail_start.call_args
        assert start_args[0][1] is agent
        assert start_args[0][2] == "hooked"

    @pytest.mark.asyncio
    async def test_output_hooks_called(self) -> None:
        """Output guardrail hooks are called with correct arguments."""
        hooks = RunHooks()
        hooks.on_output_guardrail_start = AsyncMock()
        hooks.on_output_guardrail_end = AsyncMock()

        guard = _passing_output_guardrail("out_hooked")
        agent = _make_agent(output_guardrails=[guard])
        ctx = _make_context()

        await run_output_guardrails(agent, "output", ctx, hooks)

        hooks.on_output_guardrail_start.assert_called_once()
        hooks.on_output_guardrail_end.assert_called_once()

    @pytest.mark.asyncio
    async def test_parallel_hooks_called_for_each(self) -> None:
        """Parallel guardrails invoke hooks for each guardrail."""
        hooks = RunHooks()
        hooks.on_input_guardrail_start = AsyncMock()
        hooks.on_input_guardrail_end = AsyncMock()

        g1 = _passing_input_guardrail("h1", parallel=True)
        g2 = _passing_input_guardrail("h2", parallel=True)
        agent = _make_agent(input_guardrails=[g1, g2])

        await run_parallel_input_guardrails(agent, "hello", _make_context(), hooks)

        assert hooks.on_input_guardrail_start.call_count == 2
        assert hooks.on_input_guardrail_end.call_count == 2


# ── Remediation edge cases ───────────────────────────────────


class TestRemediationEdgeCases:
    """Edge cases in output guardrail remediation."""

    @pytest.mark.asyncio
    async def test_remediation_max_retries_zero(self) -> None:
        """max_retries=0 means no remediation attempts."""

        async def _fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(tripwire_triggered=True)

        guard = AgentOutputGuardrail(
            guardrail_function=_fn,
            name="zero_retry",
            remediation="Fix it.",
            max_retries=0,
        )
        agent = _make_agent(output_guardrails=[guard])
        remediate_count = 0

        async def on_remediate(_msg: str) -> str:
            nonlocal remediate_count
            remediate_count += 1
            return "fixed"

        with pytest.raises(AgentOutputGuardrailTripwireTriggered):
            await run_output_guardrails(agent, "bad", _make_context(), _make_hooks(), on_remediate=on_remediate)
        assert remediate_count == 0  # Never called

    @pytest.mark.asyncio
    async def test_multiple_guardrails_only_one_has_remediation(self) -> None:
        """When multiple output guardrails exist, only the one with remediation retries."""

        async def _pass_fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        call_count = 0

        async def _fail_fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return AgentGuardrailFunctionOutput(tripwire_triggered=True)
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        g1 = AgentOutputGuardrail(guardrail_function=_pass_fn, name="always_pass")
        g2 = AgentOutputGuardrail(guardrail_function=_fail_fn, name="remediable", remediation="Fix.", max_retries=1)
        agent = _make_agent(output_guardrails=[g1, g2])

        async def on_remediate(_msg: str) -> str:
            return "corrected"

        results = await run_output_guardrails(agent, "bad", _make_context(), _make_hooks(), on_remediate=on_remediate)
        assert len(results) == 2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_on_remediate_exception_propagates(self) -> None:
        """Exception in on_remediate callback propagates."""

        async def _fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(tripwire_triggered=True)

        guard = AgentOutputGuardrail(guardrail_function=_fn, name="rem_error", remediation="Fix.", max_retries=1)
        agent = _make_agent(output_guardrails=[guard])

        async def on_remediate(_msg: str) -> str:
            raise ConnectionError("remediation service down")

        with pytest.raises(ConnectionError, match="remediation service down"):
            await run_output_guardrails(agent, "bad", _make_context(), _make_hooks(), on_remediate=on_remediate)


# ── Decorator new parameters ─────────────────────────────────


class TestDecoratorNewParams:
    """Decorators correctly wire all new parameters."""

    def test_input_decorator_timeout_params(self) -> None:
        """@agent_input_guardrail passes timeout params correctly."""
        from troopai.adk.agents.agent_guardrails import AgentTimeoutPolicy, agent_input_guardrail

        @agent_input_guardrail(timeout=3.0, timeout_policy=AgentTimeoutPolicy.PASS)
        async def timed(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        assert timed.timeout == 3.0
        assert timed.timeout_policy == AgentTimeoutPolicy.PASS
        assert timed.on_timeout is None

    def test_output_decorator_all_params(self) -> None:
        """@agent_output_guardrail passes all new params correctly."""
        from troopai.adk.agents.agent_guardrails import AgentTimeoutPolicy, agent_output_guardrail

        async def _cb(_info) -> None:
            pass

        @agent_output_guardrail(
            remediation="Fix it.", max_retries=3, timeout=2.0, timeout_policy=AgentTimeoutPolicy.FAIL, on_timeout=_cb
        )
        async def full(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        assert full.remediation == "Fix it."
        assert full.max_retries == 3
        assert full.timeout == 2.0
        assert full.timeout_policy == AgentTimeoutPolicy.FAIL
        assert full.on_timeout is _cb

    def test_input_decorator_with_on_timeout_callback(self) -> None:
        """@agent_input_guardrail passes on_timeout callback."""
        from troopai.adk.agents.agent_guardrails import agent_input_guardrail

        async def _my_cb(_info) -> None:
            pass

        @agent_input_guardrail(timeout=1.0, on_timeout=_my_cb)
        async def guarded(_data: AgentInputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        assert guarded.on_timeout is _my_cb
