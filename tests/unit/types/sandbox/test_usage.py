"""Tests for ``troopai.adk.types.sandbox.usage``."""

from __future__ import annotations

import pytest

from troopai.adk.types.sandbox.usage import SandboxSingleExecUsage, SandboxUsage


class TestSandboxUsageDefaults:
    def test_empty_accumulator(self) -> None:
        u = SandboxUsage()
        assert u.exec_count == 0
        assert u.total_duration_ms == 0
        assert u.cpu_ms == 0
        assert u.memory_peak_mb == 0
        assert u.bytes_read == 0
        assert u.bytes_written == 0
        assert u.computed_cost_usd == 0.0
        assert u.billed_cost_usd is None
        assert u.executions == []


class TestSandboxUsageAddExec:
    def test_single_record(self) -> None:
        u = SandboxUsage()
        u.add_exec(
            SandboxSingleExecUsage(
                command="ls",
                exit_code=0,
                duration_ms=100,
                cpu_ms=50,
                memory_peak_mb=32,
                bytes_read=1024,
                bytes_written=256,
            )
        )
        assert u.exec_count == 1
        assert u.total_duration_ms == 100
        assert u.cpu_ms == 50
        assert u.memory_peak_mb == 32
        assert u.bytes_read == 1024
        assert u.bytes_written == 256
        assert len(u.executions) == 1

    def test_memory_peak_takes_max(self) -> None:
        u = SandboxUsage()
        u.add_exec(SandboxSingleExecUsage(command="a", exit_code=0, memory_peak_mb=10))
        u.add_exec(SandboxSingleExecUsage(command="b", exit_code=0, memory_peak_mb=50))
        u.add_exec(SandboxSingleExecUsage(command="c", exit_code=0, memory_peak_mb=20))
        assert u.memory_peak_mb == 50

    def test_other_counters_sum(self) -> None:
        u = SandboxUsage()
        u.add_exec(SandboxSingleExecUsage(command="a", exit_code=0, cpu_ms=10))
        u.add_exec(SandboxSingleExecUsage(command="b", exit_code=0, cpu_ms=20))
        assert u.cpu_ms == 30
        assert u.exec_count == 2


class TestSandboxUsageAdd:
    def test_add_empty_to_empty_is_empty(self) -> None:
        a = SandboxUsage()
        b = SandboxUsage()
        c = a + b
        assert c.exec_count == 0
        assert c.executions == []

    def test_add_is_associative(self) -> None:
        a = SandboxUsage(exec_count=1, total_duration_ms=10, cpu_ms=5)
        b = SandboxUsage(exec_count=2, total_duration_ms=20, cpu_ms=15)
        c = SandboxUsage(exec_count=3, total_duration_ms=30, cpu_ms=25)
        left = (a + b) + c
        right = a + (b + c)
        assert left.exec_count == right.exec_count
        assert left.total_duration_ms == right.total_duration_ms
        assert left.cpu_ms == right.cpu_ms

    def test_add_preserves_executions_order(self) -> None:
        a = SandboxUsage(executions=[SandboxSingleExecUsage(command="a", exit_code=0)])
        b = SandboxUsage(executions=[SandboxSingleExecUsage(command="b", exit_code=0)])
        c = a + b
        assert [r.command for r in c.executions] == ["a", "b"]

    def test_add_does_not_mutate_operands(self) -> None:
        a = SandboxUsage(exec_count=1)
        b = SandboxUsage(exec_count=2)
        _ = a + b
        assert a.exec_count == 1
        assert b.exec_count == 2

    def test_add_takes_max_of_memory_peak(self) -> None:
        a = SandboxUsage(memory_peak_mb=100)
        b = SandboxUsage(memory_peak_mb=50)
        c = a + b
        assert c.memory_peak_mb == 100


class TestSandboxUsageCost:
    def test_single_exec_carries_cost(self) -> None:
        rec = SandboxSingleExecUsage(command="ls", exit_code=0, duration_ms=60000, cost_usd=0.06)
        assert rec.cost_usd == 0.06

    def test_accumulator_sums_computed_cost(self) -> None:
        usage = SandboxUsage()
        usage.add_exec(SandboxSingleExecUsage(command="a", exit_code=0, duration_ms=60000, cost_usd=0.06))
        usage.add_exec(SandboxSingleExecUsage(command="b", exit_code=0, duration_ms=60000, cost_usd=0.04))
        assert usage.computed_cost_usd == pytest.approx(0.10)
        assert usage.exec_count == 2

    def test_add_exec_tolerates_none_cost(self) -> None:
        usage = SandboxUsage()
        usage.add_exec(SandboxSingleExecUsage(command="a", exit_code=0, cost_usd=None))
        assert usage.computed_cost_usd == 0.0

    def test_merge_combines_costs(self) -> None:
        left = SandboxUsage(computed_cost_usd=0.06, billed_cost_usd=0.10)
        right = SandboxUsage(computed_cost_usd=0.04, billed_cost_usd=0.20)
        merged = left + right
        assert merged.computed_cost_usd == pytest.approx(0.10)
        assert merged.billed_cost_usd == pytest.approx(0.30)

    def test_merge_billed_none_when_both_none(self) -> None:
        assert (SandboxUsage() + SandboxUsage()).billed_cost_usd is None

    def test_merge_billed_present_when_one_side_set(self) -> None:
        left = SandboxUsage(billed_cost_usd=0.10)
        right = SandboxUsage()
        assert (left + right).billed_cost_usd == pytest.approx(0.10)
        assert (right + left).billed_cost_usd == pytest.approx(0.10)
