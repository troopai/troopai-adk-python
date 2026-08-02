"""Tests for assembling an Agent from a validated AgentConfig.

The assembler translates each config field into an explicit ``Agent(...)``
keyword argument. Absent optional fields must stay at the Agent's own
defaults — the loader never injects a value the config did not declare.
"""

from __future__ import annotations

import pytest

from troopai.adk.agents.agent import Agent
from troopai.adk.config import build_agent
from troopai.adk.config.assembler import build_llm
from troopai.adk.config.providers import PROVIDER_REGISTRY
from troopai.adk.exceptions import ConfigResolutionError
from troopai.adk.llms.anthropic.anthropic_config import AnthropicConfig
from troopai.adk.llms.anthropic.anthropic_model import AnthropicLLM
from troopai.adk.llms.llm import LLM
from troopai.adk.llms.llm_config import LLMConfig
from troopai.adk.prompts.system_prompt import SystemPrompt
from troopai.adk.skills.activation import SkillActivation
from troopai.adk.types.config import AgentConfig
from troopai.adk.types.config.llm_config import LLMProviderConfig
from troopai.adk.types.tools.tool_use_behavior import StopAtTools
from troopai.adk.verbose.config import VerboseConfig


def _build(data: dict[str, object]) -> Agent:
    return build_agent(AgentConfig.model_validate(data))


class TestStaticAssembly:
    def test_minimal_agent(self) -> None:
        agent = _build({"name": "triage", "system_prompt": "You triage."})
        assert isinstance(agent, Agent)
        assert agent.name == "triage"
        assert agent.system_prompt == "You triage."

    def test_description_mapped(self) -> None:
        agent = _build({"name": "a", "system_prompt": "p", "description": "does things"})
        assert agent.description == "does things"

    def test_description_absent_stays_none(self) -> None:
        agent = _build({"name": "a", "system_prompt": "p"})
        assert agent.description is None

    def test_structured_system_prompt_passthrough(self) -> None:
        agent = _build({"name": "a", "system_prompt": {"role": "You are a reviewer."}})
        assert isinstance(agent.system_prompt, SystemPrompt)
        assert agent.system_prompt.role == "You are a reviewer."

    def test_skill_activation_becomes_enum(self) -> None:
        agent = _build({"name": "a", "system_prompt": "p", "skill_activation": "eager"})
        assert agent.skill_activation is SkillActivation.EAGER

    def test_tool_use_behavior_stop_at_tools(self) -> None:
        agent = _build({"name": "a", "system_prompt": "p", "tool_use_behavior": {"stop_at_tool_names": ["done"]}})
        assert isinstance(agent.tool_use_behavior, StopAtTools)
        assert agent.tool_use_behavior.stop_at_tool_names == ["done"]


class TestVerboseAssembly:
    def test_verbose_absent_stays_none(self) -> None:
        agent = _build({"name": "a", "system_prompt": "p"})
        assert agent.verbose is None

    def test_verbose_mapped(self) -> None:
        agent = _build(
            {"name": "a", "system_prompt": "p", "verbose": {"enabled": True, "mode": "line", "use_color": False}}
        )
        assert isinstance(agent.verbose, VerboseConfig)
        assert agent.verbose.enabled is True
        assert agent.verbose.mode == "line"
        assert agent.verbose.use_color is False


class TestLLMAssembly:
    def test_string_llm_unchanged(self) -> None:
        agent = _build({"name": "a", "system_prompt": "p", "llm": "gpt-4o"})
        assert agent.llm == "gpt-4o"
        assert agent.llm_config is None

    def test_string_llm_with_agnostic_config(self) -> None:
        agent = _build({"name": "a", "system_prompt": "p", "llm": "gpt-4o", "llm_config": {"temperature": 0.5}})
        assert agent.llm == "gpt-4o"
        assert isinstance(agent.llm_config, LLMConfig)
        assert agent.llm_config.temperature == 0.5

    def test_provider_block_builds_llm_instance(self) -> None:
        agent = _build(
            {
                "name": "a",
                "system_prompt": "p",
                "llm": {"provider": "anthropic", "model": "claude-sonnet-4-5", "config": {"temperature": 0.5}},
            }
        )
        assert isinstance(agent.llm, AnthropicLLM)
        assert agent.llm.model == "claude-sonnet-4-5"
        assert isinstance(agent.llm_config, AnthropicConfig)
        assert agent.llm_config.temperature == 0.5

    def test_llm_absent_stays_none(self) -> None:
        agent = _build({"name": "a", "system_prompt": "p"})
        assert agent.llm is None
        assert agent.llm_config is None

    def test_provider_block_without_config_leaves_llm_config_none(self) -> None:
        agent = _build(
            {"name": "a", "system_prompt": "p", "llm": {"provider": "anthropic", "model": "claude-sonnet-4-5"}}
        )
        assert isinstance(agent.llm, AnthropicLLM)
        assert agent.llm_config is None


class TestBuildLLMErrors:
    def test_unregistered_provider_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delitem(PROVIDER_REGISTRY, "anthropic")
        config = AgentConfig.model_validate(
            {"name": "a", "system_prompt": "p", "llm": {"provider": "anthropic", "model": "m"}}
        )
        with pytest.raises(ConfigResolutionError, match="No factory registered"):
            build_llm(config)

    def test_missing_provider_dependency_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _import_failing(block: LLMProviderConfig) -> tuple[LLM, LLMConfig | None]:
            raise ImportError("No module named 'anthropic'")

        monkeypatch.setitem(PROVIDER_REGISTRY, "anthropic", _import_failing)
        config = AgentConfig.model_validate(
            {"name": "a", "system_prompt": "p", "llm": {"provider": "anthropic", "model": "m"}}
        )
        with pytest.raises(ConfigResolutionError, match="optional dependency"):
            build_llm(config)


class TestToolAssembly:
    def test_hosted_tool_built(self) -> None:
        from troopai.adk.tools.hosted.web_search_tool import WebSearchTool

        agent = _build({"name": "a", "system_prompt": "p", "tools": [{"type": "web_search", "args": {"max_uses": 3}}]})
        assert len(agent.tools) == 1
        tool = agent.tools[0]
        assert isinstance(tool, WebSearchTool)
        assert tool.max_uses == 3

    def test_mixed_string_and_hosted(self) -> None:
        agent = _build(
            {
                "name": "a",
                "system_prompt": "p",
                "tools": ["tests.unit.config.sample_symbols:sample_tool", {"type": "url_context"}],
            }
        )
        assert len(agent.tools) == 2


class TestGuardrailAssembly:
    def test_guardrails_built(self) -> None:
        agent = _build(
            {
                "name": "a",
                "system_prompt": "p",
                "guardrails": {
                    "input": [
                        {"ref": "tests.unit.config.sample_symbols:my_input_guard"},
                        {"ref": "tests.unit.config.sample_symbols:my_input_guard"},
                    ],
                    "output": [{"ref": "tests.unit.config.sample_symbols:my_output_guard"}],
                },
            }
        )
        assert len(agent.guardrails.input) == 2
        assert len(agent.guardrails.output) == 1

    def test_guardrails_absent_stays_default(self) -> None:
        agent = _build({"name": "a", "system_prompt": "p"})
        assert agent.guardrails.input == []
        assert agent.guardrails.output == []

    def test_dynamic_system_prompt_resolved(self) -> None:
        agent = _build({"name": "a", "system_prompt": {"dynamic": "tests.unit.config.sample_symbols:build_prompt"}})
        assert callable(agent.system_prompt)

    def test_non_callable_dynamic_prompt_ref_raises(self) -> None:
        with pytest.raises(ConfigResolutionError, match="non-callable"):
            _build({"name": "a", "system_prompt": {"dynamic": "tests.unit.config.sample_symbols:NOT_A_TOOL"}})
