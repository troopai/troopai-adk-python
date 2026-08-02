"""Tests for guardrail/dynamic-prompt config models and assembly."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from troopai.adk.agents.agent_guardrails import AgentGuardrails
from troopai.adk.config.guardrails import build_guardrails
from troopai.adk.exceptions import ConfigResolutionError
from troopai.adk.types.config.guardrail_config import (
    DottedGuardrailRef,
    GuardrailsConfig,
)
from troopai.adk.types.config.prompt_config import DynamicPromptRef

_GUARDRAIL_ADAPTER: TypeAdapter[object] = TypeAdapter(DottedGuardrailRef)


class TestGuardrailModels:
    def test_dotted_ref(self) -> None:
        ref = _GUARDRAIL_ADAPTER.validate_python({"ref": "my_pkg:guard"})
        assert isinstance(ref, DottedGuardrailRef)
        assert ref.ref == "my_pkg:guard"

    def test_builtin_form_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _GUARDRAIL_ADAPTER.validate_python({"builtin": "pii"})

    def test_empty_ref_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _GUARDRAIL_ADAPTER.validate_python({"ref": ""})

    def test_extra_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _GUARDRAIL_ADAPTER.validate_python({"ref": "my_pkg:guard", "extra": 1})

    def test_guardrails_config(self) -> None:
        cfg = GuardrailsConfig.model_validate({"input": [{"ref": "my_pkg:in"}], "output": [{"ref": "my_pkg:out"}]})
        assert len(cfg.input) == 1
        assert len(cfg.output) == 1

    def test_guardrails_config_defaults_empty(self) -> None:
        cfg = GuardrailsConfig.model_validate({})
        assert cfg.input == []
        assert cfg.output == []

    def test_builtin_in_input_rejected(self) -> None:
        with pytest.raises(ValidationError):
            GuardrailsConfig.model_validate({"input": [{"builtin": "pii"}]})


class TestDynamicPromptRef:
    def test_valid(self) -> None:
        ref = DynamicPromptRef.model_validate({"dynamic": "my_pkg.prompts:build"})
        assert ref.dynamic == "my_pkg.prompts:build"

    def test_empty_rejected(self) -> None:
        with pytest.raises(ValidationError):
            DynamicPromptRef.model_validate({"dynamic": ""})


class TestBuildGuardrails:
    def test_build_guardrails_dotted_ref(self) -> None:
        cfg = GuardrailsConfig.model_validate(
            {
                "input": [{"ref": "tests.unit.config.sample_symbols:my_input_guard"}],
                "output": [{"ref": "tests.unit.config.sample_symbols:my_output_guard"}],
            }
        )
        guardrails = build_guardrails(cfg)
        assert isinstance(guardrails, AgentGuardrails)
        assert len(guardrails.input) == 1
        assert len(guardrails.output) == 1

    def test_input_ref_resolving_to_output_guard_rejected(self) -> None:
        cfg = GuardrailsConfig.model_validate({"input": [{"ref": "tests.unit.config.sample_symbols:my_output_guard"}]})
        with pytest.raises(ConfigResolutionError, match="AgentInputGuardrail"):
            build_guardrails(cfg)

    def test_output_ref_resolving_to_input_guard_rejected(self) -> None:
        cfg = GuardrailsConfig.model_validate({"output": [{"ref": "tests.unit.config.sample_symbols:my_input_guard"}]})
        with pytest.raises(ConfigResolutionError, match="AgentOutputGuardrail"):
            build_guardrails(cfg)
