"""Schema models for multi-agent topology configuration.

A topology declares several named agents and how they reference each other.
``AgentNodeConfig`` is an ``AgentConfig`` that additionally allows
``handoffs`` (meaningful only when a name registry exists). ``TopologyConfig``
is the root: an ``agents`` map plus an optional ``entry`` name.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from troopai.adk.types.config.agent_config import AgentConfig
from troopai.adk.types.config.graph_config import GraphRef
from troopai.adk.types.config.references import AgentFileRef, HandoffRef
from troopai.adk.types.config.swarm_config import SwarmRef


class AgentNodeConfig(AgentConfig):
    """An agent within a topology — an ``AgentConfig`` that may hand off.

    Attributes:
        handoffs: Delegation targets, each a bare local agent name or a
            ``HandoffRef`` (name plus optional description). Targets resolve
            against the topology's ``agents`` map.
    """

    handoffs: list[str | HandoffRef] = Field(default_factory=list)
    """Delegation targets by local name (bare string or ``HandoffRef``)."""


class TopologyConfig(BaseModel):
    """Root model for a multi-agent topology file.

    A topology may declare both ``swarm`` and ``graph``; they are independent
    views over the same ``agents`` map and the caller chooses which to run.

    Attributes:
        agents: Map of local agent name to its node configuration — an inline
            ``AgentNodeConfig`` or an ``AgentFileRef`` pointing at a standalone
            agent file. The keys are the handles handoff targets resolve
            against.
        entry: Optional name of the entry agent (a key in ``agents``).
        swarm: Optional swarm built over the declared agents.
        graph: Optional graph built over the declared agents.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_ref: str | None = Field(default=None, alias="$schema")
    """Optional ``$schema`` pointer; ignored at build time."""

    agents: dict[str, AgentNodeConfig | AgentFileRef] = Field(min_length=1)
    """Map of local agent name to its node configuration (at least one) — an
    inline ``AgentNodeConfig`` or an ``AgentFileRef`` (``{config_path}``)
    pointing at a standalone agent file resolved relative to this topology."""

    entry: str | None = None
    """Optional name of the entry agent."""

    swarm: SwarmRef | None = None
    """Optional swarm built over the declared agents."""

    graph: GraphRef | None = None
    """Optional graph built over the declared agents."""
