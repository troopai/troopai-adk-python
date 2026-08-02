"""Tests for ``RunResult`` field defaults and invariants."""

from __future__ import annotations


def test_run_result_has_sandbox_usage_default_none() -> None:
    from troopai.adk.types.run.run_result import RunResult

    r = RunResult(final_output="x", user_prompt="p")
    assert r.sandbox_usage is None
