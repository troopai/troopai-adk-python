"""Assemble a multi-agent topology from configuration.

Each ``agents`` map entry is first resolved to an effective node: an inline
config is itself, while a ``config_path`` pointer is read from its own file
(relative to the topology file) and validated as an agent node — so a member
may live in its own file yet still declare handoffs by name. Handoff loading
is then two-pass to support cycles: pass 1 builds every agent as a stub (no
handoffs), pass 2 looks each handoff target up in the name registry and
assigns it. ``Agent`` is mutable and validates no handoff targets at
construction, so A<->B cycles wire without proxies or ordering tricks. After
agents are wired, optional ``Swarm`` and ``Graph`` objects are assembled over
them (members/nodes resolved by name from the same registry).
"""

from __future__ import annotations

import functools
import logging
import operator
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from troopai.adk.agents.agent import Agent
from troopai.adk.config.assembler import build_agent
from troopai.adk.config.loader import read_config_document
from troopai.adk.config.resolver import importable_dir, resolve_dotted_spec
from troopai.adk.exceptions import ConfigParseError, ConfigResolutionError
from troopai.adk.graphs import Graph, JoinSemantics, Merge, MergeFn
from troopai.adk.handoffs import Handoff
from troopai.adk.swarms import (
    ExplicitDoneTermination,
    HandoffToTermination,
    LLMHandoffPolicy,
    MaxTurnsTermination,
    RoundRobinPolicy,
    Swarm,
    SwarmConfig,
    SwarmPolicy,
)
from troopai.adk.swarms.termination import TerminationCondition
from troopai.adk.types.config.graph_config import GraphRef
from troopai.adk.types.config.references import AgentFileRef, HandoffRef
from troopai.adk.types.config.swarm_config import (
    ExplicitDoneTerminationRef,
    HandoffToTerminationRef,
    MaxTurnsTerminationRef,
    OrTerminationRef,
    PolicyRef,
    SwarmRef,
    TerminationRef,
)
from troopai.adk.types.config.topology_config import AgentNodeConfig, TopologyConfig

_MERGE_BY_NAME: dict[str, MergeFn] = {
    "concat_text": Merge.concat_text,
    "last_wins": Merge.last_wins,
    "extend_items": Merge.extend_items,
    "first_wins": Merge.first_wins,
}
_JOIN_BY_NAME: dict[str, JoinSemantics] = {"and": JoinSemantics.AND, "or": JoinSemantics.OR}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentTopology:
    """A built multi-agent topology.

    Attributes:
        agents: Map of local name to the constructed ``Agent``, with handoffs
            wired.
        entry: Name of the entry agent, if the config declared one (always a
            key in ``agents`` when set).
        swarm: The built ``Swarm``, if the config declared one.
        graph: The built ``Graph``, if the config declared one.
    """

    agents: dict[str, Agent] = field(default_factory=dict)
    """Map of local name to the constructed ``Agent``."""

    entry: str | None = None
    """Name of the entry agent, if declared."""

    swarm: Swarm | None = None
    """The built ``Swarm``, if declared."""

    graph: Graph | None = None
    """The built ``Graph``, if declared."""


def _wire_handoffs(
    node: AgentNodeConfig,
    source_name: str,
    registry: dict[str, Agent],
) -> list[Agent | Handoff]:
    """Resolve a node's handoff entries against the agent registry.

    Args:
        node: The node config whose handoffs to resolve.
        source_name: The local name of the agent declaring the handoffs.
        registry: Map of local name to built agent.

    Returns:
        A list of handoff targets — a bare ``Agent`` for a name string, a
        ``Handoff`` for an object form with extra fields.

    Raises:
        ConfigResolutionError: If a handoff names a target absent from the
            topology.
    """
    resolved: list[Agent | Handoff] = []
    for entry in node.handoffs:
        target_name = entry if isinstance(entry, str) else entry.target
        if target_name not in registry:
            raise ConfigResolutionError(
                f"Agent {source_name!r} hands off to unknown agent {target_name!r}. Known agents: {sorted(registry)}."
            )
        target = registry[target_name]
        if isinstance(entry, HandoffRef):
            resolved.append(Handoff(target=target, description=entry.description))
        else:
            resolved.append(target)
    return resolved


def _build_policy(ref: PolicyRef) -> SwarmPolicy:
    """Build a swarm policy from its selector.

    Args:
        ref: The policy selector (``round_robin`` or the default
            ``llm_handoff``).

    Returns:
        The matching ``SwarmPolicy`` instance.
    """
    if ref.type == "round_robin":
        return RoundRobinPolicy()
    return LLMHandoffPolicy()


def _build_termination(ref: TerminationRef) -> TerminationCondition:
    """Build a (possibly composed) termination condition.

    Args:
        ref: The termination selector — a discriminated-union variant whose
            per-kind required fields are guaranteed by validation.

    Returns:
        The built condition; ``or``/``and`` fold children with ``|``/``&``.
    """
    if isinstance(ref, MaxTurnsTerminationRef):
        return MaxTurnsTermination(ref.limit)
    if isinstance(ref, ExplicitDoneTerminationRef):
        return ExplicitDoneTermination()
    if isinstance(ref, HandoffToTerminationRef):
        return HandoffToTermination(ref.target)
    children = [_build_termination(child) for child in ref.conditions]
    fold = operator.or_ if isinstance(ref, OrTerminationRef) else operator.and_
    return functools.reduce(fold, children)


def _build_swarm(ref: SwarmRef, registry: dict[str, Agent]) -> Swarm:
    """Build a ``Swarm`` from its config, resolving members by name.

    Args:
        ref: The swarm selector.
        registry: Map of local name to built agent.

    Returns:
        The constructed ``Swarm``.

    Raises:
        ConfigResolutionError: If a member or the entry names an agent absent
            from the topology, or the entry is not among the members.
    """
    for name in [*ref.members, ref.entry]:
        if name not in registry:
            raise ConfigResolutionError(f"Swarm references unknown agent {name!r}. Known agents: {sorted(registry)}.")
    members = tuple(registry[name] for name in ref.members)
    entry = registry[ref.entry]
    if entry not in members:
        raise ConfigResolutionError(f"Swarm entry {ref.entry!r} is not among its members {ref.members}.")

    config = (
        SwarmConfig()
        if ref.config is None
        else SwarmConfig(
            max_handoffs=ref.config.max_handoffs,
            max_total_tokens=ref.config.max_total_tokens,
        )
    )
    return Swarm(
        members=members,
        entry=entry,
        policy=_build_policy(ref.policy),
        termination=_build_termination(ref.termination),
        config=config,
    )


def _build_graph(ref: GraphRef, registry: dict[str, Agent]) -> Graph:
    """Build a ``Graph`` from its config, resolving node agents by name.

    Args:
        ref: The graph selector.
        registry: Map of local name to built agent.

    Returns:
        The compiled ``Graph``.

    Raises:
        ConfigResolutionError: If a node names an unknown agent, an edge or
            the entry/terminals name an unknown node, or an edge condition
            reference is unresolvable or not callable.
        ConfigParseError: If the graph fails structural validation at compile
            time (e.g. the entry has an incoming edge).
    """
    builder = Graph.new(ref.id)
    for node_id, node in ref.nodes.items():
        if node.agent not in registry:
            raise ConfigResolutionError(
                f"Graph node {node_id!r} references unknown agent {node.agent!r}. Known agents: {sorted(registry)}."
            )
        merge = None if node.merge is None else _MERGE_BY_NAME[node.merge]
        join = None if node.join is None else _JOIN_BY_NAME[node.join]
        builder.node(node_id, registry[node.agent], merge=merge, join=join)

    for edge in ref.edges:
        for endpoint in (edge.from_, edge.to):
            if endpoint not in ref.nodes:
                raise ConfigResolutionError(
                    f"Graph edge references unknown node {endpoint!r}. Known nodes: {sorted(ref.nodes)}."
                )
        when = None
        if edge.when is not None:
            when = resolve_dotted_spec(edge.when)
            if not callable(when):
                raise ConfigResolutionError(
                    f"Graph edge condition {edge.when!r} resolved to {type(when).__name__}, "
                    "expected a callable (NodeResult) -> bool."
                )
        builder.edge(edge.from_, edge.to, when=when)

    # entry / terminals must name declared nodes (reference resolution)...
    for node_id in [ref.entry, *ref.terminals]:
        if node_id not in ref.nodes:
            raise ConfigResolutionError(
                f"Graph entry/terminal references unknown node {node_id!r}. Known nodes: {sorted(ref.nodes)}."
            )

    # ...and the wired graph must be structurally valid (compile-time).
    try:
        builder.entry(ref.entry)
        builder.terminal(*ref.terminals)
        return builder.compile()
    except ValueError as exc:
        raise ConfigParseError(f"Invalid graph {ref.id!r}: {exc}") from exc


@dataclass(frozen=True)
class _ResolvedNode:
    """An ``agents`` map entry resolved to its effective node and source dir.

    Attributes:
        node: The effective ``AgentNodeConfig`` — an inline entry as-is, or the
            agent loaded from a ``config_path`` file.
        source_dir: The directory the node's dotted references resolve against
            (the config file's own directory), or ``None`` when unknown (an
            in-memory build with no originating file).
    """

    node: AgentNodeConfig
    """The effective node config."""

    source_dir: Path | None
    """Directory the node's dotted refs resolve against, or ``None``."""


def _resolve_node(entry: AgentNodeConfig | AgentFileRef, name: str, base_dir: Path | None) -> _ResolvedNode:
    """Resolve one ``agents`` map entry to its effective node and source dir.

    An inline ``AgentNodeConfig`` resolves to itself (its refs resolve against
    ``base_dir``). An ``AgentFileRef`` is read from its ``config_path`` (relative
    to ``base_dir``, or absolute) and validated as an ``AgentNodeConfig``; its
    source dir is the file's own directory so its sibling tool modules resolve.

    Args:
        entry: The map entry — inline node or a ``config_path`` pointer.
        name: The local agent name (map key), quoted in error messages.
        base_dir: Directory of the topology file, or ``None`` when the topology
            was built from an in-memory model rather than a file.

    Returns:
        The resolved node paired with the directory its dotted references
        resolve against.

    Raises:
        ConfigResolutionError: If ``entry`` is a ``config_path`` pointer but no
            ``base_dir`` is available to resolve it against.
        ConfigParseError: If the referenced file is not a valid agent file.
    """
    if isinstance(entry, AgentNodeConfig):
        return _ResolvedNode(entry, base_dir)
    if base_dir is None:
        raise ConfigResolutionError(
            f"Agent {name!r} uses config_path {entry.config_path!r}, which can only be resolved when the "
            "topology is loaded from a file. Use load_topology(path), or inline the agent."
        )
    target = Path(entry.config_path)
    target = (target if target.is_absolute() else base_dir / target).resolve()
    data = read_config_document(target)
    try:
        node = AgentNodeConfig.model_validate(data)
    except ValidationError as exc:
        logger.debug("Agent %r: config_path %r failed validation: %s", name, str(target), exc)
        raise ConfigParseError(f"Agent {name!r} config_path {str(target)!r} is not a valid agent file: {exc}") from exc
    return _ResolvedNode(node, target.parent)


def _build_node(node: AgentNodeConfig, source_dir: Path | None) -> Agent:
    """Build an agent (without handoffs) with its source dir temporarily importable.

    The source directory on ``sys.path`` lets dotted tool references in ``node``
    resolve against the config file's own directory. ``None`` skips that (an
    in-memory build with no originating file).
    """
    if source_dir is None:
        return build_agent(node)
    with importable_dir(source_dir):
        return build_agent(node)


def build_topology(config: TopologyConfig, base_dir: Path | None = None) -> AgentTopology:
    """Build an :class:`AgentTopology` from a validated config.

    Args:
        config: A validated :class:`TopologyConfig`.
        base_dir: Directory the topology file was loaded from, used to resolve
            ``config_path`` agent entries (and to make each agent's sibling tool
            modules importable). ``None`` when building from an in-memory model;
            a ``config_path`` entry then raises, since it has nothing to resolve
            against.

    Returns:
        The built topology with handoffs wired and any swarm or graph
        assembled.

    Raises:
        ConfigResolutionError: If a handoff, swarm, graph, or the topology
            ``entry`` names an unknown agent/node, a referenced
            tool/output-schema/predicate cannot be resolved, or a
            ``config_path`` entry appears without a ``base_dir``.
        ConfigParseError: If a ``config_path`` file is not a valid agent file,
            or a declared graph is structurally invalid.
    """
    # Keep the topology file's own directory importable for the whole build, so
    # inline agents' tool refs AND graph edge ``when`` predicates resolve against
    # it. config_path sub-agents in other directories get their own dir layered
    # on top per agent (see _build_node). nullcontext covers the in-memory build.
    outer = importable_dir(base_dir) if base_dir is not None else nullcontext()
    with outer:
        # Resolve each map entry to its effective node (inline, or loaded from a
        # config_path file) plus the directory its dotted references resolve against.
        nodes: dict[str, _ResolvedNode] = {
            name: _resolve_node(entry, name, base_dir) for name, entry in config.agents.items()
        }

        # Pass 1: build every agent as a stub (no handoffs).
        registry: dict[str, Agent] = {name: _build_node(rn.node, rn.source_dir) for name, rn in nodes.items()}

        # Pass 2: wire handoffs by name (cycles are fine — all stubs exist).
        for name, rn in nodes.items():
            if len(rn.node.handoffs) > 0:
                registry[name].handoffs = _wire_handoffs(rn.node, name, registry)

        if config.entry is not None and config.entry not in registry:
            raise ConfigResolutionError(
                f"Topology entry {config.entry!r} is not a declared agent. Known agents: {sorted(registry)}."
            )

        swarm = None if config.swarm is None else _build_swarm(config.swarm, registry)
        graph = None if config.graph is None else _build_graph(config.graph, registry)

    logger.info(
        "Built topology: %d agent(s), entry=%r, swarm=%s, graph=%s",
        len(registry),
        config.entry,
        swarm is not None,
        graph is not None,
    )
    return AgentTopology(agents=registry, entry=config.entry, swarm=swarm, graph=graph)


def load_topology(path: str | Path, *, document: dict[str, Any] | None = None) -> AgentTopology:
    """Load a multi-agent topology from a JSON or YAML config file.

    Args:
        path: Path to the ``.json`` / ``.yaml`` / ``.yml`` topology file.
        document: Pre-parsed root mapping of ``path``. When provided, the
            file is not re-read — callers that already parsed it (e.g. to
            detect the config kind) avoid a second disk read and the window
            where the file changes between the two reads.

    Returns:
        The built :class:`AgentTopology`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ConfigParseError: If the JSON/YAML is invalid, the extension is
            unsupported, a YAML file needs an uninstalled ``pyyaml``, the
            document fails schema validation, or a declared graph is
            structurally invalid.
        ConfigResolutionError: If a reference cannot be resolved.
    """
    path = Path(path)
    data = read_config_document(path) if document is None else document

    try:
        config = TopologyConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigParseError(f"Invalid topology config in {str(path)!r}: {exc}") from exc

    # Pass the file's directory as base_dir; build_topology keeps it importable
    # for the whole build (so sibling tool modules and graph predicates resolve)
    # and uses it to locate config_path sub-agent files.
    return build_topology(config, base_dir=path.resolve().parent)
