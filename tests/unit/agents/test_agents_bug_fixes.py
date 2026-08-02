"""Regression tests for agents/ module bug fixes.

Covers:
- SystemPrompt(role="") blank guard in Agent.__post_init__
- as_tool extractor non-str return raises UserError
- AgentOutputGuardrail.__post_init__ max_retries validation
- AgentGuardrailResults in agents/__init__.__all__
- Dead TContext_co TypeVar removed from agent_guardrails
- PEP-604 union in AgentToolOutputExtractor
"""

from __future__ import annotations

import ast
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from troopai.adk.agents import Agent
from troopai.adk.agents.agent_guardrails import (
    AgentGuardrailFunctionOutput,
    AgentGuardrailResults,
    AgentOutputGuardrail,
    AgentOutputGuardrailData,
)
from troopai.adk.exceptions import UserError
from troopai.adk.prompts import SystemPrompt

# ── Helpers ───────────────────────────────────────────────────────────


def _passing_output_guardrail_fn(data: AgentOutputGuardrailData) -> AgentGuardrailFunctionOutput:
    return AgentGuardrailFunctionOutput(tripwire_triggered=False)


def _agent(name: str = "TestAgent") -> Agent:
    return Agent(name=name, system_prompt="You are a test agent.")


def _mock_result(final_output: str = "Done") -> MagicMock:
    return MagicMock(requires_action=False, final_output=final_output)


async def _invoke(tool: Any, raw_input: str) -> str:
    assert tool.on_invoke is not None
    ctx = MagicMock()
    ctx.context = None
    ctx.run_config = None
    return await tool.on_invoke(ctx, raw_input)


# ── Fix 1: SystemPrompt blank-role guard ─────────────────────────────


class TestSystemPromptBlankRoleGuard:
    """Agent.__post_init__ must reject SystemPrompt(role='')."""

    def test_empty_string_still_rejected(self) -> None:
        """Existing guard: plain empty-string system_prompt raises."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Agent(name="a", system_prompt="")

    def test_none_still_rejected(self) -> None:
        """Existing guard: None system_prompt raises."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Agent(name="a", system_prompt=None)

    def test_system_prompt_empty_role_raises(self) -> None:
        """NEW: SystemPrompt(role='') must raise ValueError at construction."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Agent(name="a", system_prompt=SystemPrompt(role=""))

    def test_system_prompt_non_empty_role_accepted(self) -> None:
        """SystemPrompt with a real role is accepted."""
        agent = Agent(name="a", system_prompt=SystemPrompt(role="You help users."))
        assert agent.name == "a"


# ── Fix 2: as_tool extractor non-str runtime check ───────────────────


class TestAsToolExtractorStrContract:
    """as_tool extractor returning non-str must raise UserError."""

    @pytest.mark.asyncio
    async def test_extractor_returning_none_raises_user_error(self) -> None:
        """Extractor returning None must raise UserError."""
        agent = _agent("Extractor")
        tool = agent.as_tool(extractor=lambda _result: None)  # type: ignore[arg-type]

        with (
            patch(
                "troopai.adk.run.Runner.arun",
                new_callable=AsyncMock,
                return_value=_mock_result(),
            ),
            pytest.raises(UserError, match="extractor must return str"),
        ):
            await _invoke(tool, '{"input": "task"}')

    @pytest.mark.asyncio
    async def test_extractor_returning_int_raises_user_error(self) -> None:
        """Extractor returning int must raise UserError."""
        agent = _agent("Extractor")
        tool = agent.as_tool(extractor=lambda _result: 42)  # type: ignore[arg-type]

        with (
            patch(
                "troopai.adk.run.Runner.arun",
                new_callable=AsyncMock,
                return_value=_mock_result(),
            ),
            pytest.raises(UserError, match="extractor must return str"),
        ):
            await _invoke(tool, '{"input": "task"}')

    @pytest.mark.asyncio
    async def test_extractor_returning_str_succeeds(self) -> None:
        """Extractor returning str passes through correctly."""
        agent = _agent("Extractor")
        tool = agent.as_tool(extractor=lambda _result: "extracted!")

        with patch(
            "troopai.adk.run.Runner.arun",
            new_callable=AsyncMock,
            return_value=_mock_result(),
        ):
            result = await _invoke(tool, '{"input": "task"}')

        assert result == "extracted!"

    @pytest.mark.asyncio
    async def test_async_extractor_returning_str_succeeds(self) -> None:
        """Async extractor returning str passes through correctly."""
        agent = _agent("AsyncExtractor")

        async def async_extractor(_result: Any) -> str:
            return "async extracted!"

        tool = agent.as_tool(extractor=async_extractor)

        with patch(
            "troopai.adk.run.Runner.arun",
            new_callable=AsyncMock,
            return_value=_mock_result(),
        ):
            result = await _invoke(tool, '{"input": "task"}')

        assert result == "async extracted!"

    @pytest.mark.asyncio
    async def test_async_extractor_returning_none_raises_user_error(self) -> None:
        """Async extractor returning None must raise UserError."""
        agent = _agent("AsyncExtractorBad")

        async def bad_async_extractor(_result: Any) -> None:
            return None

        tool = agent.as_tool(extractor=bad_async_extractor)  # type: ignore[arg-type]

        with (
            patch(
                "troopai.adk.run.Runner.arun",
                new_callable=AsyncMock,
                return_value=_mock_result(),
            ),
            pytest.raises(UserError, match="extractor must return str"),
        ):
            await _invoke(tool, '{"input": "task"}')


# ── Fix 3: AgentOutputGuardrail max_retries validation ───────────────


class TestAgentOutputGuardrailMaxRetriesValidation:
    """AgentOutputGuardrail.__post_init__ must validate max_retries."""

    def test_valid_default_max_retries_accepted(self) -> None:
        """Default max_retries=1 with no remediation is valid."""
        g = AgentOutputGuardrail(guardrail_function=_passing_output_guardrail_fn)
        assert g.max_retries == 1

    def test_max_retries_zero_no_remediation_accepted(self) -> None:
        """max_retries=0 without remediation is valid (remediation not set)."""
        g = AgentOutputGuardrail(
            guardrail_function=_passing_output_guardrail_fn,
            max_retries=0,
        )
        assert g.max_retries == 0

    def test_negative_max_retries_raises(self) -> None:
        """Negative max_retries must raise ValueError regardless of remediation."""
        with pytest.raises(ValueError, match="non-negative"):
            AgentOutputGuardrail(
                guardrail_function=_passing_output_guardrail_fn,
                max_retries=-1,
            )

    def test_remediation_with_max_retries_zero_accepted_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """remediation + max_retries=0 is legal but inert: accepted, with a warning.

        ``max_retries=0`` is a valid "no retries" value, so construction must not
        raise. The remediation loop gates on ``count < max_retries`` (``0 < 0`` is
        False), so the message can never fire — the runner warns about the inert
        combination instead of silently dropping it.
        """
        with caplog.at_level(logging.WARNING):
            g = AgentOutputGuardrail(
                guardrail_function=_passing_output_guardrail_fn,
                remediation="Please fix your output.",
                max_retries=0,
            )
        assert g.max_retries == 0
        assert g.remediation == "Please fix your output."
        assert any("max_retries=0" in record.message for record in caplog.records)

    def test_remediation_with_valid_max_retries_accepted(self) -> None:
        """remediation + max_retries >= 1 is valid."""
        g = AgentOutputGuardrail(
            guardrail_function=_passing_output_guardrail_fn,
            remediation="Fix it.",
            max_retries=2,
        )
        assert g.max_retries == 2
        assert g.remediation == "Fix it."


# ── Fix 4: AgentGuardrailResults in __all__ ──────────────────────────


class TestAgentGuardrailResultsExported:
    """AgentGuardrailResults must be importable from troopai.adk.agents."""

    def test_importable_from_agents(self) -> None:
        """from troopai.adk.agents import AgentGuardrailResults must work."""
        import troopai.adk.agents as agents_mod

        assert agents_mod.AgentGuardrailResults is AgentGuardrailResults

    def test_in_all(self) -> None:
        """AgentGuardrailResults must appear in agents.__all__."""
        import troopai.adk.agents as agents_mod

        assert "AgentGuardrailResults" in agents_mod.__all__

    def test_can_instantiate(self) -> None:
        """AgentGuardrailResults imported from agents can be instantiated."""
        import troopai.adk.agents as agents_mod

        results = agents_mod.AgentGuardrailResults()
        assert results.input == ()
        assert results.output == ()


# ── Fix 5: Dead TContext_co TypeVar removed ───────────────────────────


class TestDeadTypeVarRemoved:
    """Module-level TContext_co TypeVar must be gone from agent_guardrails."""

    def test_module_level_tcontext_co_not_present(self) -> None:
        """TContext_co should not exist as a module-level attribute."""
        import troopai.adk.agents.agent_guardrails as mod

        assert not hasattr(mod, "TContext_co"), "TContext_co TypeVar was dead code and should have been removed"

    def test_typing_extensions_typevar_not_imported(self) -> None:
        """typing_extensions.TypeVar import was only for dead TContext_co — removed."""
        with open("src/troopai/adk/agents/agent_guardrails.py") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "typing_extensions":
                imported_names = [alias.name for alias in node.names]
                assert "TypeVar" not in imported_names, "TypeVar from typing_extensions should have been removed"


# ── Fix 6: PEP-604 union in AgentToolOutputExtractor ─────────────────


class TestAgentToolOutputExtractorPEP604:
    """AgentToolOutputExtractor must use PEP-604 union syntax."""

    def test_union_not_used_in_source(self) -> None:
        """Union[] from typing should not appear in agent_as_tool_types.py."""
        with open("src/troopai/adk/types/agents/agent_as_tool_types.py") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "typing":
                imported_names = [alias.name for alias in node.names]
                assert "Union" not in imported_names, "Union from typing should be replaced with PEP-604 | syntax"

    def test_extractor_type_alias_still_importable(self) -> None:
        """AgentToolOutputExtractor must still be importable after the change."""
        from troopai.adk.types.agents.agent_as_tool_types import AgentToolOutputExtractor

        assert AgentToolOutputExtractor is not None
