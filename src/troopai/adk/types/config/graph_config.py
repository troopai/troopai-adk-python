"""Schema models for declarative graph configuration.

A ``GraphRef`` declares nodes (each hosting a local agent), edges, an entry,
and terminals — mapping directly onto ``GraphBuilder``. Edge ``from`` is
aliased because it is a Python keyword.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class GraphNodeRef(BaseModel):
    """A graph node hosting a local agent.

    Attributes:
        agent: Local name of the agent this node runs.
        merge: Optional built-in fan-in merge name (``concat_text``,
            ``last_wins``, ``extend_items``, ``first_wins``).
        join: Optional join semantics for fan-in (``and`` default, ``or``).
    """

    model_config = ConfigDict(extra="forbid")

    agent: str
    """Local name of the agent this node runs."""

    merge: Literal["concat_text", "last_wins", "extend_items", "first_wins"] | None = None
    """Optional built-in fan-in merge name."""

    join: Literal["and", "or"] | None = None
    """Optional join semantics for fan-in."""


class GraphEdgeRef(BaseModel):
    """A directed edge between two nodes, optionally conditional.

    Attributes:
        from_: Source node id (the JSON key is ``from``).
        to: Destination node id.
        when: Optional dotted reference to an edge-condition predicate
            ``(NodeResult) -> bool``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    from_: str = Field(alias="from")
    """Source node id (JSON key ``from``)."""

    to: str
    """Destination node id."""

    when: str | None = None
    """Optional dotted reference to an edge-condition predicate."""


class GraphRef(BaseModel):
    """Declarative graph: nodes, edges, entry, terminals.

    Attributes:
        id: Graph identifier.
        nodes: Map of node id to node config.
        edges: Directed edges between node ids.
        entry: The entry node id.
        terminals: Terminal node ids (the run exits when any fires).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    """Graph identifier."""

    nodes: dict[str, GraphNodeRef] = Field(min_length=1)
    """Map of node id to node config (at least one)."""

    edges: list[GraphEdgeRef] = Field(default_factory=list)
    """Directed edges between node ids."""

    entry: str
    """The entry node id."""

    terminals: list[str] = Field(min_length=1)
    """Terminal node ids (at least one)."""
