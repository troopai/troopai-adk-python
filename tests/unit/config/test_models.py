"""Tests for the declarative agent-config schema models.

These pin the strict-validation contract: which keys are required, which
are rejected, how the optional ``$schema`` pointer is tolerated, and how a
code-only key produces a guiding error rather than a cryptic one. They
exercise the Pydantic models directly, so they assert ``ValidationError``
(the loader wraps that into ``ConfigParseError`` — covered separately).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from troopai.adk.prompts.system_prompt import SystemPrompt
from troopai.adk.types.config import AgentConfig
from troopai.adk.types.tools.tool_use_behavior import StopAtTools


class TestAgentConfigValidation:
    def test_minimal_valid_config(self) -> None:
        config = AgentConfig.model_validate({"name": "triage", "system_prompt": "You triage."})
        assert config.name == "triage"
        assert config.system_prompt == "You triage."

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig.model_validate({"system_prompt": "hi"})

    def test_missing_system_prompt_raises(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig.model_validate({"name": "triage"})

    def test_unknown_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig.model_validate({"name": "a", "system_prompt": "p", "temprature": 0.7})

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig.model_validate({"name": "", "system_prompt": "p"})


class TestSchemaPointer:
    def test_schema_pointer_tolerated(self) -> None:
        # An optional $schema pointer must NOT trip strict validation.
        config = AgentConfig.model_validate(
            {"$schema": "https://example.com/agent.schema.json", "name": "a", "system_prompt": "p"}
        )
        assert config.schema_ref == "https://example.com/agent.schema.json"

    def test_schema_pointer_absent_is_fine(self) -> None:
        config = AgentConfig.model_validate({"name": "a", "system_prompt": "p"})
        assert config.schema_ref is None


class TestSystemPromptForms:
    def test_plain_string(self) -> None:
        config = AgentConfig.model_validate({"name": "a", "system_prompt": "be helpful"})
        assert isinstance(config.system_prompt, str)

    def test_structured_prompt(self) -> None:
        config = AgentConfig.model_validate(
            {"name": "a", "system_prompt": {"role": "You are a reviewer.", "tone": "technical"}}
        )
        assert isinstance(config.system_prompt, SystemPrompt)
        assert config.system_prompt.role == "You are a reviewer."

    def test_typo_in_system_prompt_field_rejected(self) -> None:
        """A misspelled SystemPrompt key must raise ValidationError.

        Before the fix, SystemPrompt lacked ``extra='forbid'``, so a
        typo like ``guidelins`` (for ``guidelines``) was silently
        swallowed and ``guidelines`` was left as None, producing
        broken agent behavior with no validation error.
        """
        with pytest.raises(ValidationError):
            AgentConfig.model_validate(
                {
                    "name": "a",
                    "system_prompt": {
                        "role": "You are a helpful assistant.",
                        "guidelins": ["Be concise"],  # typo: should be 'guidelines'
                    },
                }
            )

    def test_valid_system_prompt_structured_fields_accepted(self) -> None:
        """All valid SystemPrompt fields pass validation."""
        config = AgentConfig.model_validate(
            {
                "name": "a",
                "system_prompt": {
                    "role": "You are a reviewer.",
                    "guidelines": ["Be concise", "Be accurate"],
                    "tone": "technical",
                    "constraints": ["Never reveal PII"],
                },
            }
        )
        assert isinstance(config.system_prompt, SystemPrompt)
        assert config.system_prompt.guidelines == ["Be concise", "Be accurate"]


class TestStaticFields:
    def test_skill_activation_defaults_lazy(self) -> None:
        config = AgentConfig.model_validate({"name": "a", "system_prompt": "p"})
        assert config.skill_activation == "lazy"

    def test_skill_activation_eager(self) -> None:
        config = AgentConfig.model_validate({"name": "a", "system_prompt": "p", "skill_activation": "eager"})
        assert config.skill_activation == "eager"

    def test_skill_activation_invalid_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig.model_validate({"name": "a", "system_prompt": "p", "skill_activation": "sometimes"})

    def test_tool_use_behavior_default(self) -> None:
        config = AgentConfig.model_validate({"name": "a", "system_prompt": "p"})
        assert config.tool_use_behavior == "run_llm_again"

    def test_tool_use_behavior_stop_on_first(self) -> None:
        config = AgentConfig.model_validate(
            {"name": "a", "system_prompt": "p", "tool_use_behavior": "stop_on_first_tool"}
        )
        assert config.tool_use_behavior == "stop_on_first_tool"

    def test_tool_use_behavior_invalid_literal_rejected(self) -> None:
        with pytest.raises(ValidationError):
            AgentConfig.model_validate({"name": "a", "system_prompt": "p", "tool_use_behavior": "always_stop"})

    def test_tool_use_behavior_stop_at_tools(self) -> None:
        config = AgentConfig.model_validate(
            {"name": "a", "system_prompt": "p", "tool_use_behavior": {"stop_at_tool_names": ["finish"]}}
        )
        assert isinstance(config.tool_use_behavior, StopAtTools)
        assert config.tool_use_behavior.stop_at_tool_names == ["finish"]


class TestCodeOnlyKeys:
    @pytest.mark.parametrize("key", ["hooks", "middleware", "skills"])
    def test_code_only_key_gives_guiding_error(self, key: str) -> None:
        with pytest.raises(ValidationError) as exc_info:
            AgentConfig.model_validate({"name": "a", "system_prompt": "p", key: "anything"})
        # The message should point the user to Python, not just say "extra field".
        assert "Python" in str(exc_info.value)


def test_empty_tool_ref_rejected() -> None:
    with pytest.raises(ValidationError):
        AgentConfig.model_validate({"name": "a", "system_prompt": "p", "tools": [""]})
