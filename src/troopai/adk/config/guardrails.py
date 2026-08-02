"""Guardrail assembly for declarative configs.

A guardrail entry is a dotted ``ref`` resolved via the resolver to a runtime
guardrail object. ``build_guardrails`` turns a :class:`GuardrailsConfig` into
the runtime :class:`AgentGuardrails`; each ``ref`` is resolved to the
guardrail type matching the list it appears in (input or output).
"""

from __future__ import annotations

import logging
from typing import Any

from troopai.adk.agents.agent_guardrails import (
    AgentGuardrails,
    AgentInputGuardrail,
    AgentOutputGuardrail,
)
from troopai.adk.config.resolver import resolve_input_guardrail, resolve_output_guardrail
from troopai.adk.types.config.guardrail_config import GuardrailsConfig

logger = logging.getLogger(__name__)


def build_guardrails(config: GuardrailsConfig) -> AgentGuardrails:
    """Assemble runtime :class:`AgentGuardrails` from a guardrails config.

    Args:
        config: The validated guardrails config (input/output entry lists,
            each a dotted ``ref``).

    Returns:
        ``AgentGuardrails`` with resolved input and output guardrails.

    Raises:
        ConfigResolutionError: If a ``ref`` resolves to the wrong guardrail
            type for the list it appears in.
    """
    inputs: list[AgentInputGuardrail[Any]] = [resolve_input_guardrail(entry.ref) for entry in config.input]
    outputs: list[AgentOutputGuardrail[Any]] = [resolve_output_guardrail(entry.ref) for entry in config.output]

    logger.debug("Assembled %d input and %d output guardrails", len(inputs), len(outputs))
    return AgentGuardrails(input=inputs, output=outputs)
