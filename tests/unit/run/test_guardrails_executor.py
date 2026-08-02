"""Regression tests for run.guardrails_executor.

Focused on the interaction between output-guardrail timeouts and the
remediation retry path: a timeout under FAIL policy is an infrastructure
failure, not a content verdict, so it must NOT trigger agent re-runs via
``on_remediate`` (which would pay full LLM cost trying to "self-correct"
output that was never flagged as bad).
"""

import asyncio

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentGuardrails,
    AgentOutputGuardrail,
    AgentOutputGuardrailData,
    AgentTimeoutPolicy,
)
from troopai.adk.exceptions import AgentOutputGuardrailTripwireTriggered
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.context import RunContext
from troopai.adk.run.guardrails_executor import run_output_guardrails


def _make_agent(output_guardrails: list | None = None) -> Agent:
    return Agent(
        name="test_agent",
        system_prompt="test",
        guardrails=AgentGuardrails(output=output_guardrails or []),
    )


def _make_context() -> RunContext:
    return RunContext(context=None)


def _make_hooks() -> RunHooks:
    return RunHooks()


class TestTimeoutDoesNotTriggerRemediation:
    """An output-guardrail timeout (FAIL policy) must not re-run the agent."""

    @pytest.mark.asyncio
    async def test_timeout_fail_with_remediation_does_not_call_on_remediate(self) -> None:
        """A slow output guardrail that times out raises without remediating.

        Before the fix the synthetic FAIL output (tripwire_triggered=True,
        severity=None) was indistinguishable from a real content trip, so the
        remediation branch fired and called on_remediate up to max_retries
        times — re-running the agent for an infrastructure failure.
        """
        remediate_calls = 0

        async def _slow_fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            await asyncio.sleep(5.0)
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        guard = AgentOutputGuardrail(
            guardrail_function=_slow_fn,
            name="slow_out",
            remediation="Fix your output.",
            max_retries=3,
            timeout=0.05,
            timeout_policy=AgentTimeoutPolicy.FAIL,
        )
        agent = _make_agent(output_guardrails=[guard])

        async def on_remediate(_msg: str) -> str:
            nonlocal remediate_calls
            remediate_calls += 1
            return "corrected"

        with pytest.raises(AgentOutputGuardrailTripwireTriggered) as exc_info:
            await run_output_guardrails(
                agent,
                "original output",
                _make_context(),
                _make_hooks(),
                on_remediate=on_remediate,
            )

        # Remediation must NOT have fired for a timeout-induced halt.
        assert remediate_calls == 0
        # The halt is preserved and surfaces the timeout verdict.
        assert "timed out" in str(exc_info.value.guardrail_result.guardrail_output.output_info)

    @pytest.mark.asyncio
    async def test_timeout_pass_policy_still_passes_with_remediation_set(self) -> None:
        """PASS-policy timeout continues silently even when remediation is set."""
        remediate_calls = 0

        async def _slow_fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            await asyncio.sleep(5.0)
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        guard = AgentOutputGuardrail(
            guardrail_function=_slow_fn,
            name="slow_pass_out",
            remediation="Fix your output.",
            max_retries=3,
            timeout=0.05,
            timeout_policy=AgentTimeoutPolicy.PASS,
        )
        agent = _make_agent(output_guardrails=[guard])

        async def on_remediate(_msg: str) -> str:
            nonlocal remediate_calls
            remediate_calls += 1
            return "corrected"

        results = await run_output_guardrails(
            agent,
            "original output",
            _make_context(),
            _make_hooks(),
            on_remediate=on_remediate,
        )

        assert remediate_calls == 0
        assert len(results) == 1
        assert results[0].guardrail_output.tripwire_triggered is False

    @pytest.mark.asyncio
    async def test_genuine_content_trip_still_remediates(self) -> None:
        """A real content trip (not a timeout) still triggers remediation.

        Guards against over-correction: only timeout-induced halts skip the
        remediation path; genuine content verdicts must remain remediable.
        """
        call_count = 0
        remediate_calls = 0

        async def _fn(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            nonlocal call_count
            call_count += 1
            if data.output == "bad":
                return AgentGuardrailFunctionOutput(tripwire_triggered=True, output_info="bad output")
            return AgentGuardrailFunctionOutput(tripwire_triggered=False)

        # A generous timeout that never fires — the trip is a content verdict.
        guard = AgentOutputGuardrail(
            guardrail_function=_fn,
            name="content_guard",
            remediation="Fix your output.",
            max_retries=1,
            timeout=5.0,
            timeout_policy=AgentTimeoutPolicy.FAIL,
        )
        agent = _make_agent(output_guardrails=[guard])

        async def on_remediate(_msg: str) -> str:
            nonlocal remediate_calls
            remediate_calls += 1
            return "good"

        results = await run_output_guardrails(
            agent,
            "bad",
            _make_context(),
            _make_hooks(),
            on_remediate=on_remediate,
        )

        assert remediate_calls == 1
        assert call_count == 2  # First call tripped, second passed
        assert len(results) == 1
        assert results[0].guardrail_output.tripwire_triggered is False
