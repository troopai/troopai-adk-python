"""Schema models for declarative swarm configuration.

A ``SwarmRef`` names the swarm's members and entry (agent names in the
topology), its routing policy, a (possibly composed) termination condition,
and budget config. ``TerminationRef`` is recursive: ``or``/``and`` carry
nested ``conditions`` that the assembler folds with the runtime ``|``/``&``
operators.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class PolicyRef(BaseModel):
    """Swarm routing policy selector.

    Attributes:
        type: Which built-in policy to use.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["llm_handoff", "round_robin"] = "llm_handoff"
    """Which built-in policy to use."""


class MaxTurnsTerminationRef(BaseModel):
    """Stop after a fixed number of turns.

    Attributes:
        type: Discriminator literal ``"max_turns"``.
        limit: Positive turn limit.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["max_turns"]
    """Discriminator."""

    limit: int = Field(gt=0)
    """Turn limit (must be positive)."""


class ExplicitDoneTerminationRef(BaseModel):
    """Stop when an agent signals explicit completion.

    Attributes:
        type: Discriminator literal ``"explicit_done"``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["explicit_done"]
    """Discriminator."""


class HandoffToTerminationRef(BaseModel):
    """Stop when control hands off to a named agent.

    Attributes:
        type: Discriminator literal ``"handoff_to"``.
        target: Local name of the agent whose entry ends the swarm.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["handoff_to"]
    """Discriminator."""

    target: str = Field(min_length=1)
    """Local name of the handoff target."""


class OrTerminationRef(BaseModel):
    """Stop when ANY child condition fires.

    Attributes:
        type: Discriminator literal ``"or"``.
        conditions: Child conditions (at least two), folded with OR.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["or"]
    """Discriminator."""

    conditions: list[TerminationRef] = Field(min_length=2)
    """Child conditions; folded with OR (at least two)."""


class AndTerminationRef(BaseModel):
    """Stop when ALL child conditions fire.

    Attributes:
        type: Discriminator literal ``"and"``.
        conditions: Child conditions (at least two), folded with AND.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["and"]
    """Discriminator."""

    conditions: list[TerminationRef] = Field(min_length=2)
    """Child conditions; folded with AND (at least two)."""


TerminationRef = Annotated[
    MaxTurnsTerminationRef
    | ExplicitDoneTerminationRef
    | HandoffToTerminationRef
    | OrTerminationRef
    | AndTerminationRef,
    Field(discriminator="type"),
]
"""A termination condition, discriminated on ``type``; ``or``/``and`` recurse."""

OrTerminationRef.model_rebuild()
AndTerminationRef.model_rebuild()


class SwarmConfigRef(BaseModel):
    """JSON-friendly subset of ``SwarmConfig`` budgets.

    Attributes:
        max_handoffs: Swarm-wide cap on agent switches.
        max_total_tokens: Swarm-wide cumulative token cap.
    """

    model_config = ConfigDict(extra="forbid")

    max_handoffs: int = Field(default=20, gt=0)
    """Swarm-wide cap on agent switches (must be positive)."""

    max_total_tokens: int | None = Field(default=None, gt=0)
    """Swarm-wide cumulative token cap (positive when set)."""


class SwarmRef(BaseModel):
    """Declarative swarm: members, entry, policy, termination, budgets.

    Attributes:
        members: Local agent names participating in the swarm.
        entry: Local name of the agent that takes the first turn.
        policy: Routing policy (defaults to LLM handoff).
        termination: The (possibly composed) stop condition.
        config: Optional budget config.
    """

    model_config = ConfigDict(extra="forbid")

    members: list[Annotated[str, Field(min_length=1)]] = Field(min_length=1)
    """Local agent names participating in the swarm (at least one, each non-empty)."""

    entry: str = Field(min_length=1)
    """Local name of the agent that takes the first turn."""

    policy: PolicyRef = Field(default_factory=PolicyRef)
    """Routing policy (defaults to LLM handoff)."""

    termination: TerminationRef
    """The (possibly composed) stop condition."""

    config: SwarmConfigRef | None = None
    """Optional budget config."""
