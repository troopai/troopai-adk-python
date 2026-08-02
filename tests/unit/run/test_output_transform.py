"""Tests for the output-guardrail TRANSFORM action and its write-back.

A transform substitutes the checked output wholesale instead of failing the run:
the executor re-validates the replacement (no agent re-run), and the write-back
rewrites both ``final_output`` and the trailing assistant message so the persisted
session events and memory extraction observe the masked text. The action is
opt-in (no ``on_transform`` → inert) and bounded (at most one transform per
guardrail, then the tripwire halts).
"""

from __future__ import annotations

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentGuardrails,
    AgentOutputGuardrail,
    AgentOutputGuardrailData,
)
from troopai.adk.exceptions import AgentOutputGuardrailTripwireTriggered
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.context import RunContext
from troopai.adk.run.guardrails_executor import run_output_guardrails
from troopai.adk.run.runner import apply_output_transform
from troopai.adk.types.guardrails import GuardrailAction
from troopai.adk.types.items.items import ItemHelpers, MessageOutputItem
from troopai.adk.types.responses.llm_response import LLMResponseText
from troopai.adk.types.run import RunResult


def _make_agent(output_guardrails: list[AgentOutputGuardrail[None]] | None = None) -> Agent:
    return Agent(
        name="test_agent",
        system_prompt="test",
        guardrails=AgentGuardrails(output=output_guardrails or []),
    )


def _make_context() -> RunContext:
    return RunContext(context=None)


def _make_hooks() -> RunHooks:
    return RunHooks()


class TestTransformBranch:
    async def test_transform_substitutes_then_revalidates_and_passes(self) -> None:
        transforms: list[str] = []
        remediate_calls = 0

        async def _redact(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            if "SECRET" in str(data.output):
                # Halt-with-fallback + a replacement: the executor substitutes it.
                return AgentGuardrailFunctionOutput(transformed_output="[MASKED]", tripwire_triggered=True)
            return AgentGuardrailFunctionOutput()

        guard: AgentOutputGuardrail[None] = AgentOutputGuardrail(guardrail_function=_redact, name="redact")

        async def on_transform(replacement: str) -> None:
            transforms.append(replacement)

        async def on_remediate(_msg: str) -> str:
            nonlocal remediate_calls
            remediate_calls += 1
            return "unused"

        ctx = _make_context()
        results = await run_output_guardrails(
            _make_agent([guard]),
            "this has SECRET in it",
            ctx,
            _make_hooks(),
            on_remediate=on_remediate,
            on_transform=on_transform,
        )

        # Transformed once; the masked output re-validated and passed; the agent
        # was never re-prompted (transform is cheaper than remediation).
        assert transforms == ["[MASKED]"]
        assert remediate_calls == 0
        assert len(results) == 1
        assert results[0].guardrail_output.resolved_action().value == "pass"

        # The transform is recorded once in the audit trail, with distinct hashes
        # for the original and the replacement (reviewable after the run).
        transform_records = [
            record for record in ctx.collect_guardrail_audit() if record.action is GuardrailAction.TRANSFORM
        ]
        assert len(transform_records) == 1
        assert transform_records[0].output_hash is not None
        assert transform_records[0].transformed_hash is not None
        assert transform_records[0].output_hash != transform_records[0].transformed_hash

    async def test_transform_mode_without_on_transform_halts_via_tripwire(self) -> None:
        async def _redact(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(transformed_output="[MASKED]", tripwire_triggered=True)

        guard: AgentOutputGuardrail[None] = AgentOutputGuardrail(guardrail_function=_redact, name="redact")

        # No on_transform wired: the verdict still halts via its tripwire fallback.
        with pytest.raises(AgentOutputGuardrailTripwireTriggered):
            await run_output_guardrails(_make_agent([guard]), "has SECRET", _make_context(), _make_hooks())

    async def test_plain_tripwire_does_not_invoke_on_transform(self) -> None:
        transforms: list[str] = []

        async def _trip(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(tripwire_triggered=True)

        guard: AgentOutputGuardrail[None] = AgentOutputGuardrail(guardrail_function=_trip, name="trip")

        async def on_transform(replacement: str) -> None:
            transforms.append(replacement)

        with pytest.raises(AgentOutputGuardrailTripwireTriggered):
            await run_output_guardrails(
                _make_agent([guard]),
                "x",
                _make_context(),
                _make_hooks(),
                on_transform=on_transform,
            )
        assert transforms == []

    async def test_always_transform_guardrail_is_bounded(self) -> None:
        transforms: list[str] = []

        async def _always(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
            return AgentGuardrailFunctionOutput(transformed_output="[X]", tripwire_triggered=True)

        guard: AgentOutputGuardrail[None] = AgentOutputGuardrail(guardrail_function=_always, name="always")

        async def on_transform(replacement: str) -> None:
            transforms.append(replacement)

        # One transform per guardrail; the re-flagged masked output then halts.
        with pytest.raises(AgentOutputGuardrailTripwireTriggered):
            await run_output_guardrails(
                _make_agent([guard]),
                "x",
                _make_context(),
                _make_hooks(),
                on_transform=on_transform,
            )
        assert transforms == ["[X]"]


class TestApplyOutputTransform:
    async def test_rewrites_final_output_and_trailing_message(self) -> None:
        message = MessageOutputItem(raw=[LLMResponseText(text="raw SECRET value")])
        result: RunResult[None] = RunResult(final_output="raw SECRET value", user_prompt="p", new_items=[message])

        await apply_output_transform(result, "[REDACTED]")

        assert result.final_output == "[REDACTED]"
        # The trailing message — and therefore its to_param() view used for session
        # events and memory extraction — carries the masked text, not the raw value.
        trailing = result.new_items[-1]
        assert isinstance(trailing, MessageOutputItem)
        assert ItemHelpers.text_message_output(trailing) == "[REDACTED]"
        param_text = str(trailing.to_param())
        assert "[REDACTED]" in param_text
        assert "SECRET" not in param_text

    async def test_no_trailing_message_only_updates_final_output(self) -> None:
        result: RunResult[None] = RunResult(final_output="raw value", user_prompt="p", new_items=[])

        await apply_output_transform(result, "[REDACTED]")

        assert result.final_output == "[REDACTED]"
