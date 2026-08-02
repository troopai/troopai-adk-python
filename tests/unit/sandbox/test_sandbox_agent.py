"""Tests for ``SandboxAgent`` dataclass (P33)."""

from __future__ import annotations

import pytest

from troopai.adk.sandbox.agent import SandboxAgent
from troopai.adk.sandbox.capabilities import CompactionCapability
from troopai.adk.sandbox.runner_integration.concurrency_guard import (
    SandboxConcurrencyGuard,
)
from troopai.adk.types.sandbox.manifest import Manifest
from troopai.adk.types.sandbox.permissions import User


class TestConstruction:
    def test_minimal_construction(self) -> None:
        agent = SandboxAgent(name="coder")
        assert agent.name == "coder"
        assert agent.default_manifest is None
        assert agent.run_as is None
        # Default capabilities = [CompactionCapability()].
        assert len(agent.capabilities) == 1
        assert isinstance(agent.capabilities[0], CompactionCapability)

    def test_with_system_prompt(self) -> None:
        agent = SandboxAgent(name="coder", system_prompt="A coding assistant.")
        assert agent.system_prompt == "A coding assistant."

    def test_with_manifest(self) -> None:
        manifest = Manifest(root="/workspace")
        agent = SandboxAgent(name="coder", default_manifest=manifest)
        assert agent.default_manifest is manifest

    def test_no_system_prompt_uses_placeholder(self) -> None:
        # SandboxAgent without system_prompt construction succeeds —
        # the framework injects a sentinel that the Runner-side
        # composer substitutes at turn-time.
        agent = SandboxAgent(name="coder")
        assert agent.system_prompt is not None


class TestRunAsValidation:
    def test_run_as_user_object_accepted(self) -> None:
        user = User(name="alice")
        agent = SandboxAgent(name="coder", run_as=user)
        assert agent.run_as is user

    def test_run_as_str_accepted(self) -> None:
        agent = SandboxAgent(name="coder", run_as="alice")
        assert agent.run_as == "alice"

    def test_run_as_empty_str_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty string or User"):
            SandboxAgent(name="coder", run_as="")

    def test_run_as_wrong_type_rejected(self) -> None:
        with pytest.raises(TypeError, match="must be a User or non-empty str"):
            SandboxAgent(name="coder", run_as=42)  # type: ignore[arg-type]


class TestConcurrencyGuard:
    def test_guard_lazy_allocated(self) -> None:
        agent = SandboxAgent(name="coder")
        # Pre-access: stored as None.
        assert object.__getattribute__(agent, "_concurrency_guard") is None
        # First access constructs.
        guard = agent.get_concurrency_guard()
        assert isinstance(guard, SandboxConcurrencyGuard)
        # Second access returns the same instance.
        assert agent.get_concurrency_guard() is guard


class TestIsAlsoAnAgent:
    def test_isinstance_agent(self) -> None:
        from troopai.adk.agents.agent import Agent

        agent = SandboxAgent(name="coder")
        assert isinstance(agent, Agent)
