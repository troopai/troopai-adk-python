"""Schema models for declarative agent guardrails.

A guardrail entry is a dotted ``ref`` to a user-defined guardrail object.
Phase is structural — entries live under the ``input`` or ``output`` list of
:class:`GuardrailsConfig`, and each ``ref`` is resolved to the guardrail type
matching the list it appears in.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DottedGuardrailRef(BaseModel):
    """A dotted-path reference to a user-defined guardrail object.

    Attributes:
        ref: Dotted-path reference resolving to an ``AgentInputGuardrail`` or
            ``AgentOutputGuardrail`` (per the list it appears in).
    """

    model_config = ConfigDict(extra="forbid")

    ref: str = Field(min_length=1)
    """Dotted-path reference to a guardrail object."""


class GuardrailsConfig(BaseModel):
    """Input and output guardrail lists for an agent.

    Attributes:
        input: Input-phase guardrail entries.
        output: Output-phase guardrail entries.
    """

    model_config = ConfigDict(extra="forbid")

    input: list[DottedGuardrailRef] = Field(default_factory=list)
    """Input-phase guardrail entries (dotted refs)."""

    output: list[DottedGuardrailRef] = Field(default_factory=list)
    """Output-phase guardrail entries (dotted refs)."""
