"""Tests for the guardrail audit side-car (record, accumulator, emit, drain)."""

from __future__ import annotations

from datetime import UTC, datetime

from troopai.adk.agents.agent import Agent
from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentGuardrails,
    AgentGuardrailSeverity,
    AgentOutputGuardrail,
    AgentOutputGuardrailData,
)
from troopai.adk.audit.event import hash_payload
from troopai.adk.hooks.hooks import RunHooks
from troopai.adk.run.context import RunContext
from troopai.adk.run.governance import emit_guardrail_audit
from troopai.adk.run.guardrails_executor import run_output_guardrails
from troopai.adk.types.guardrails import GuardrailAction, GuardrailSpan
from troopai.adk.types.run.guardrail_audit import GuardrailAuditRecord


def _record(**overrides: object) -> GuardrailAuditRecord:
    base: dict[str, object] = {
        "level": "agent_output",
        "guardrail_name": "g",
        "agent_name": "a",
        "action": GuardrailAction.PASS,
        "severity": None,
        "triggered": False,
        "output_hash": None,
        "transformed_hash": None,
        "timestamp": datetime(2026, 1, 1, tzinfo=UTC),
    }
    base.update(overrides)
    return GuardrailAuditRecord(**base)  # type: ignore[arg-type]


class TestGuardrailAuditRecord:
    def test_construction_and_defaults(self) -> None:
        record = _record()
        assert record.level == "agent_output"
        assert record.changed_spans == ()

    def test_carries_spans(self) -> None:
        record = _record(changed_spans=(GuardrailSpan(start=0, end=3, reason="pii"),))
        assert record.changed_spans[0].reason == "pii"


class TestRunContextAccumulator:
    def test_empty_by_default(self) -> None:
        assert RunContext(context=None).collect_guardrail_audit() == ()

    def test_record_then_collect_returns_immutable_tuple(self) -> None:
        ctx = RunContext(context=None)
        ctx.record_guardrail_audit(_record(guardrail_name="one"))
        ctx.record_guardrail_audit(_record(guardrail_name="two"))
        collected = ctx.collect_guardrail_audit()
        assert isinstance(collected, tuple)
        assert [r.guardrail_name for r in collected] == ["one", "two"]


class TestEmitGuardrailAudit:
    def test_hashes_the_checked_payload_never_raw(self) -> None:
        ctx = RunContext(context=None)
        secret = "alice@example.com"
        emit_guardrail_audit(
            ctx,
            level="agent_output",
            agent_name="a",
            guardrail_name="pii",
            action=GuardrailAction.RAISE,
            checked=secret,
        )
        record = ctx.collect_guardrail_audit()[0]
        assert record.output_hash == hash_payload(secret)
        assert secret not in (record.output_hash or "")
        assert record.triggered is True

    def test_pass_is_not_triggered(self) -> None:
        ctx = RunContext(context=None)
        emit_guardrail_audit(
            ctx,
            level="agent_input",
            agent_name="a",
            guardrail_name="g",
            action=GuardrailAction.PASS,
            checked="hello",
        )
        assert ctx.collect_guardrail_audit()[0].triggered is False

    def test_transformed_hash_only_when_supplied(self) -> None:
        ctx = RunContext(context=None)
        emit_guardrail_audit(
            ctx,
            level="agent_output",
            agent_name="a",
            guardrail_name="redactor",
            action=GuardrailAction.TRANSFORM,
            checked="raw secret",
            transformed="[REDACTED]",
        )
        record = ctx.collect_guardrail_audit()[0]
        assert record.transformed_hash == hash_payload("[REDACTED]")
        assert record.output_hash == hash_payload("raw secret")
        assert record.output_hash != record.transformed_hash


def _output_guard(severity: AgentGuardrailSeverity | None = None) -> AgentOutputGuardrail:
    async def _fn(_data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
        return AgentGuardrailFunctionOutput(tripwire_triggered=False, severity=severity)

    return AgentOutputGuardrail(guardrail_function=_fn, name="audited")


class TestExecutorEmitsAudit:
    async def test_run_output_guardrails_records_hashed_audit(self) -> None:
        agent = Agent(
            name="a",
            system_prompt="t",
            guardrails=AgentGuardrails(input=[], output=[_output_guard()]),
        )
        ctx = RunContext(context=None)
        await run_output_guardrails(agent, "the model output", ctx, RunHooks())
        audit = ctx.collect_guardrail_audit()
        assert len(audit) == 1
        assert audit[0].level == "agent_output"
        assert audit[0].action is GuardrailAction.PASS
        assert audit[0].output_hash == hash_payload("the model output")
        assert "the model output" not in (audit[0].output_hash or "")
