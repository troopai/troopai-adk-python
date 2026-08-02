from __future__ import annotations

from typing import Any

import pytest

from troopai.adk.agents import Agent
from troopai.adk.budgets import InMemoryCostLedger, TenantBudget
from troopai.adk.exceptions import UserError
from troopai.adk.run.config import RunConfig
from troopai.adk.run.cost import validate_budget_config
from troopai.adk.run.runner import Runner
from troopai.adk.swarms.policy import RoundRobinPolicy
from troopai.adk.swarms.swarm import Swarm
from troopai.adk.swarms.termination import MaxTurnsTermination


def _minimal_swarm() -> Swarm[Any]:
    m = Agent(name="m", system_prompt="x")
    return Swarm(
        members=(m,),
        entry=m,
        policy=RoundRobinPolicy(),
        termination=MaxTurnsTermination(1),
    )


def test_config_defaults_off() -> None:
    cfg = RunConfig()
    assert cfg.tenant_budget is None
    assert cfg.cost_ledger is None


def test_profile_sets_budget_and_ledger() -> None:
    ledger = InMemoryCostLedger()
    profile = Runner.configure().tenant_budget(TenantBudget(dollars_per_run=1.0)).cost_ledger(ledger)
    cfg = profile.run_config
    assert cfg.tenant_budget is not None
    assert cfg.cost_ledger is ledger


def test_validate_rejects_period_without_ledger() -> None:
    with pytest.raises(UserError, match="cost_ledger"):
        validate_budget_config(TenantBudget(dollars_per_period=5.0), None)


def test_validate_allows_per_run_without_ledger() -> None:
    validate_budget_config(TenantBudget(dollars_per_run=1.0), None)


def test_validate_allows_none_budget() -> None:
    validate_budget_config(None, None)


async def test_period_budget_without_ledger_fails_fast() -> None:
    cfg = RunConfig(tenant_budget=TenantBudget(dollars_per_period=5.0), tenant_id="t1")
    with pytest.raises(UserError, match="cost_ledger"):
        await Runner.arun(Agent(name="A", system_prompt="test"), "hi", run_config=cfg)


def test_swarm_runner_sets_budget_and_ledger() -> None:
    ledger = InMemoryCostLedger()
    runner = (
        Runner.configure().tenant_budget(TenantBudget(dollars_per_run=1.0)).cost_ledger(ledger).swarm(_minimal_swarm())
    )
    cfg = runner.run_config
    assert cfg.tenant_budget is not None
    assert cfg.cost_ledger is ledger
