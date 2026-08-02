"""Mermaid ``flowchart`` emitters for :class:`Flow` and :class:`Graph`.

Pure functions translating immutable topology data into Mermaid
diagram strings. The output is a complete ``flowchart`` block ready
to paste into GitHub Markdown, Mermaid Live Editor, or any renderer
that consumes the Mermaid syntax.

Node shapes:

- ``@flow_start`` and entry graph nodes → rounded ``(name)``.
- ``@flow_listen`` and intermediate graph nodes → rectangle ``[name]``.
- ``@flow_router`` → diamond ``{name}``.
- AND-gates → small node ``((AND))``; OR-gates → ``((OR))``.

Edge shapes:

- Direct trigger → solid ``-->``.
- Conditional edge (Graph ``when=`` predicate) → dashed ``-.->``.
- Router route label → solid ``-->|"label"|``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from troopai.adk.flows.registry import build_transition_table
from troopai.adk.visualization.helpers import (
    assert_no_collision,
    build_step_lookup,
    escape_mermaid_label,
    gate_node_id,
    node_label_from_desc,
    safe,
)

if TYPE_CHECKING:
    from troopai.adk.flows.definition import FlowDefinition
    from troopai.adk.flows.flow import Flow
    from troopai.adk.flows.flow_wrappers import FlowStep
    from troopai.adk.flows.registry import FlowTransitionTable
    from troopai.adk.graphs.graph import Graph
    from troopai.adk.graphs.node import GraphEdge


MermaidDirection = Literal["LR", "TD", "TB", "RL", "BT"]
"""Layout direction accepted by Mermaid ``flowchart``.

``LR`` (left-to-right, default) and ``TD`` / ``TB`` (top-down) are the
common choices; ``RL`` and ``BT`` are accepted for completeness.
"""

_VALID_DIRECTIONS: frozenset[str] = frozenset({"LR", "TD", "TB", "RL", "BT"})
"""Runtime-validated direction values matching :data:`MermaidDirection`."""


def flow_to_mermaid(flow: Flow, *, direction: MermaidDirection = "LR") -> str:
    """Emit a Mermaid ``flowchart`` describing the Flow's topology.

    Walks the immutable :class:`FlowStepRegistry` of the flow's class and
    its derived :class:`FlowTransitionTable` to produce a complete
    Mermaid block. Pure function: no I/O, no side effects, idempotent.

    Args:
        flow: A constructed :class:`Flow` instance. The instance is
            only used to access the class-level registry — no run is
            triggered.
        direction: Mermaid layout direction. ``"LR"`` (default) reads
            left-to-right; ``"TD"`` / ``"TB"`` reads top-down.

    Returns:
        A Mermaid ``flowchart`` string starting with
        ``flowchart <direction>\\n`` and ending with one or more node
        / edge declarations.

    Raises:
        ValueError: When ``direction`` is not a Mermaid direction
            literal, or when two distinct identifiers collapse to the
            same sanitised node id.
    """
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {sorted(_VALID_DIRECTIONS)}; got {direction!r}.",
        )
    registry = flow.get_registry()
    table = build_transition_table(registry)
    lines: list[str] = [f"flowchart {direction}"]
    declared: set[str] = set()
    sanitised: dict[str, str] = {}
    inverse: dict[str, str] = {}
    step_lookup = build_step_lookup(flow)

    for name in sorted(registry.starts):
        assert_no_collision(sanitised, name, safe(name), inverse)
        lines.append(f"    {_node_decl(name, 'start', step_lookup)}")
        declared.add(name)
    for name in sorted(registry.listeners.keys()):
        assert_no_collision(sanitised, name, safe(name), inverse)
        lines.append(f"    {_node_decl(name, 'listen', step_lookup)}")
        declared.add(name)
    for name in sorted(registry.routers.keys()):
        assert_no_collision(sanitised, name, safe(name), inverse)
        lines.append(f"    {_node_decl(name, 'router', step_lookup)}")
        declared.add(name)

    lines.extend(_flow_direct_edges(table, sanitised, inverse))
    lines.extend(_flow_router_edges(table, sanitised, inverse))
    lines.extend(_flow_gate_edges(table, declared, sanitised, inverse))

    return "\n".join(lines)


def definition_to_mermaid(defn: FlowDefinition, *, direction: MermaidDirection = "LR") -> str:
    """Emit a Mermaid ``flowchart`` from a :class:`FlowDefinition`.

    Produces the same diagram as :func:`flow_to_mermaid` but accepts a
    :class:`~troopai.adk.flows.definition.FlowDefinition` directly, so
    visualisation is possible without constructing or running a
    :class:`~troopai.adk.flows.flow.Flow` instance.

    Pure function: no I/O, no side effects. Descriptions are read from
    :attr:`~troopai.adk.flows.definition.StepInfo.description` on each
    step.

    Args:
        defn: A frozen :class:`FlowDefinition` produced by
            :meth:`~troopai.adk.flows.flow.Flow.get_definition` or
            :func:`~troopai.adk.flows.definition.build_flow_definition`.
        direction: Mermaid layout direction. ``"LR"`` (default).

    Returns:
        A complete Mermaid ``flowchart`` block.

    Raises:
        ValueError: When ``direction`` is invalid or when two distinct
            identifiers collapse to the same sanitised node id.
    """
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {sorted(_VALID_DIRECTIONS)}; got {direction!r}.",
        )

    desc_lookup: dict[str, str | None] = {s.name: s.description for s in defn.steps}
    lines: list[str] = [f"flowchart {direction}"]
    declared: set[str] = set()
    sanitised: dict[str, str] = {}
    inverse: dict[str, str] = {}

    for step in defn.steps:
        assert_no_collision(sanitised, step.name, safe(step.name), inverse)
        lines.append(f"    {_mermaid_node_decl(step.name, step.role, desc_lookup)}")
        declared.add(step.name)

    lines.extend(_defn_direct_edges_mermaid(defn, sanitised, inverse))
    lines.extend(_defn_router_edges_mermaid(defn, sanitised, inverse))
    lines.extend(_defn_gate_edges_mermaid(defn, declared, sanitised, inverse))

    return "\n".join(lines)


def _mermaid_node_decl(name: str, role: str, desc_lookup: dict[str, str | None]) -> str:
    """Build a Mermaid node declaration using a pre-built description lookup.

    Args:
        name: Step method name (node id).
        role: ``"start"`` / ``"listen"`` / ``"router"``.
        desc_lookup: Mapping from step name to optional description string.
            Passed to :func:`~troopai.adk.visualization.helpers.node_label_from_desc`
            for explicit ``is not None`` label resolution.

    Returns:
        Mermaid node declaration string.
    """
    label = node_label_from_desc(name, desc_lookup)
    if role == "start":
        return f'{safe(name)}(("{escape_mermaid_label(label)}"))'
    if role == "router":
        return f'{safe(name)}{{"{escape_mermaid_label(label)}"}}'
    return f'{safe(name)}["{escape_mermaid_label(label)}"]'


def _defn_direct_edges_mermaid(
    defn: FlowDefinition,
    sanitised: dict[str, str],
    inverse: dict[str, str],
) -> list[str]:
    """Emit Mermaid direct-trigger edges from a :class:`FlowDefinition`.

    Reverses the ``listener → triggers`` mapping stored in
    :attr:`FlowDefinition.direct_triggers` to produce one edge per
    ``trigger → listener`` pair.

    Args:
        defn: The compiled flow definition.
        sanitised: Collision accumulator shared with the caller.
        inverse: Inverse mapping for O(1) collision detection.

    Returns:
        Sorted list of Mermaid edge declarations.
    """
    trigger_to_listeners: dict[str, list[str]] = {}
    for listener_name, triggers in defn.direct_triggers.items():
        for trigger in triggers:
            trigger_to_listeners.setdefault(trigger, []).append(listener_name)
    edges: list[str] = []
    for trigger in sorted(trigger_to_listeners.keys()):
        assert_no_collision(sanitised, trigger, safe(trigger), inverse)
        for listener in sorted(trigger_to_listeners[trigger]):
            assert_no_collision(sanitised, listener, safe(listener), inverse)
            edges.append(f"    {safe(trigger)} --> {safe(listener)}")
    return edges


def _defn_router_edges_mermaid(
    defn: FlowDefinition,
    sanitised: dict[str, str],
    inverse: dict[str, str],
) -> list[str]:
    """Emit Mermaid router-trigger edges from a :class:`FlowDefinition`.

    Args:
        defn: The compiled flow definition.
        sanitised: Collision accumulator shared with the caller.
        inverse: Inverse mapping for O(1) collision detection.

    Returns:
        Sorted list of Mermaid edge declarations.
    """
    trigger_to_routers: dict[str, list[str]] = {}
    for router_name, triggers in defn.router_triggers.items():
        for trigger in triggers:
            trigger_to_routers.setdefault(trigger, []).append(router_name)
    edges: list[str] = []
    for trigger in sorted(trigger_to_routers.keys()):
        assert_no_collision(sanitised, trigger, safe(trigger), inverse)
        for router in sorted(trigger_to_routers[trigger]):
            assert_no_collision(sanitised, router, safe(router), inverse)
            edges.append(f"    {safe(trigger)} --> {safe(router)}")
    return edges


def _defn_gate_edges_mermaid(
    defn: FlowDefinition,
    declared: set[str],
    sanitised: dict[str, str],
    inverse: dict[str, str],
) -> list[str]:
    """Emit Mermaid gate nodes and their edges from a :class:`FlowDefinition`.

    Args:
        defn: The compiled flow definition.
        declared: Mutable set of already-declared node ids.
        sanitised: Collision accumulator shared with the caller.
        inverse: Inverse mapping for O(1) collision detection.

    Returns:
        Lines for gate node declarations and wired edges.
    """
    lines: list[str] = []
    for gate in defn.gates:
        node_id = gate_node_id(gate.gate_id)
        assert_no_collision(sanitised, gate.gate_id, node_id, inverse)
        if node_id not in declared:
            label = "AND" if gate.kind == "and" else "OR"
            lines.append(f"    {node_id}(({label}))")
            declared.add(node_id)
        for trigger in sorted(gate.triggers):
            lines.append(f"    {safe(trigger)} --> {node_id}")
        lines.append(f"    {node_id} --> {safe(gate.listener_name)}")
    return lines


def _flow_direct_edges(table: FlowTransitionTable, sanitised: dict[str, str], inverse: dict[str, str]) -> list[str]:
    """Emit Mermaid edges for direct (non-gated, non-routed) triggers.

    Each trigger and listener is run through :func:`assert_no_collision`
    so a route label that would collide with an existing step name
    (or vice versa) raises ``ValueError`` instead of silently
    retargeting an edge.

    Args:
        table: The compiled transition table.
        sanitised: Accumulator threaded from the caller; mutated in
            place so every emitter pass shares one collision view.
        inverse: Inverse mapping (sanitised → original) maintained for
            O(1) collision detection.

    Returns:
        Sorted list of edge declarations (one per ``--> ``).
    """
    edges: list[str] = []
    for trigger in sorted(table.direct_listeners.keys()):
        assert_no_collision(sanitised, trigger, safe(trigger), inverse)
        for listener in sorted(table.direct_listeners[trigger]):
            assert_no_collision(sanitised, listener, safe(listener), inverse)
            edges.append(f"    {safe(trigger)} --> {safe(listener)}")
    return edges


def _flow_router_edges(table: FlowTransitionTable, sanitised: dict[str, str], inverse: dict[str, str]) -> list[str]:
    """Emit Mermaid edges from triggers into routers.

    Args:
        table: The compiled transition table.
        sanitised: Collision accumulator shared with the caller.
        inverse: Inverse mapping (sanitised → original) maintained for
            O(1) collision detection.

    Returns:
        Sorted list of edge declarations.
    """
    edges: list[str] = []
    for trigger in sorted(table.routers_for.keys()):
        assert_no_collision(sanitised, trigger, safe(trigger), inverse)
        for router in sorted(table.routers_for[trigger]):
            assert_no_collision(sanitised, router, safe(router), inverse)
            edges.append(f"    {safe(trigger)} --> {safe(router)}")
    return edges


def _flow_gate_edges(
    table: FlowTransitionTable,
    declared: set[str],
    sanitised: dict[str, str],
    inverse: dict[str, str],
) -> list[str]:
    """Emit Mermaid synthesised gate nodes and their incoming / outgoing edges.

    Gate node ids are derived from :attr:`GateSpec.gate_id` via
    :func:`gate_node_id`. The id is also registered in ``sanitised``
    so a step name that happens to slug to the same string raises a
    typed collision error instead of overlapping diagram nodes.

    Args:
        table: The compiled transition table.
        declared: Names of nodes already declared (mutated in place
            so the same gate is not declared twice).
        sanitised: Collision accumulator shared with the caller.
        inverse: Inverse mapping (sanitised → original) maintained for
            O(1) collision detection.

    Returns:
        Lines for gate node declarations and the wired edges.
    """
    lines: list[str] = []
    for gid in sorted(table.and_gates.keys()):
        spec = table.and_gates[gid]
        node_id = gate_node_id(gid)
        assert_no_collision(sanitised, gid, node_id, inverse)
        if node_id not in declared:
            lines.append(f"    {node_id}((AND))")
            declared.add(node_id)
        for trigger in sorted(spec.triggers):
            lines.append(f"    {safe(trigger)} --> {node_id}")
        lines.append(f"    {node_id} --> {safe(spec.listener_name)}")
    for gid in sorted(table.or_gates.keys()):
        spec = table.or_gates[gid]
        node_id = gate_node_id(gid)
        assert_no_collision(sanitised, gid, node_id, inverse)
        if node_id not in declared:
            lines.append(f"    {node_id}((OR))")
            declared.add(node_id)
        for trigger in sorted(spec.triggers):
            lines.append(f"    {safe(trigger)} --> {node_id}")
        lines.append(f"    {node_id} --> {safe(spec.listener_name)}")
    return lines


def graph_to_mermaid(graph: Graph, *, direction: MermaidDirection = "LR") -> str:
    """Emit a Mermaid ``flowchart`` describing the Graph's topology.

    Walks ``graph.nodes`` and ``graph.edges`` to produce a complete
    Mermaid block. Pure function: no I/O.

    Args:
        graph: A compiled :class:`Graph` instance.
        direction: Mermaid layout direction.

    Returns:
        A Mermaid ``flowchart`` string.

    Raises:
        ValueError: When ``direction`` is invalid or when two distinct
            node ids collapse to the same sanitised slug.
    """
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {sorted(_VALID_DIRECTIONS)}; got {direction!r}.",
        )
    lines: list[str] = [f"flowchart {direction}"]
    terminal_ids = set(graph.terminals)
    entry_id = graph.entry
    sanitised: dict[str, str] = {}
    inverse: dict[str, str] = {}
    for node in sorted(graph.nodes, key=lambda n: n.id):
        assert_no_collision(sanitised, node.id, safe(node.id), inverse)
        label = node.description if node.description is not None else node.id
        shape_open, shape_close = _graph_shape(node.id, entry_id, terminal_ids)
        lines.append(f'    {safe(node.id)}{shape_open}"{escape_mermaid_label(label)}"{shape_close}')
    for edge in sorted(graph.edges, key=lambda e: (e.source, e.target, e.label or "", e.priority)):
        lines.append(f"    {_render_graph_edge(edge)}")
    return "\n".join(lines)


def _render_graph_edge(edge: GraphEdge) -> str:
    """Format a single :class:`GraphEdge` as a Mermaid edge.

    Args:
        edge: A :class:`GraphEdge`.

    Returns:
        A Mermaid edge declaration string.
    """
    arrow = "-.->" if edge.when is not None else "-->"
    if edge.label is not None and len(edge.label) > 0:
        return f'{safe(edge.source)} {arrow}|"{escape_mermaid_label(edge.label)}"| {safe(edge.target)}'
    if edge.when is not None:
        return f'{safe(edge.source)} {arrow}|"when"| {safe(edge.target)}'
    return f"{safe(edge.source)} {arrow} {safe(edge.target)}"


def _graph_shape(node_id: str, entry: str, terminals: set[str]) -> tuple[str, str]:
    """Return the Mermaid shape brackets for a graph node id.

    Entry nodes render as rounded ``(...)``; terminal nodes as stadium
    ``([...])`` so the start / end of the run is visually obvious;
    other nodes as rectangle ``[...]``.

    Args:
        node_id: Node id under test.
        entry: The graph's entry node id.
        terminals: Set of terminal node ids.

    Returns:
        Tuple ``(open_bracket, close_bracket)``.
    """
    if node_id == entry:
        return "(", ")"
    if node_id in terminals:
        return "([", "])"
    return "[", "]"


def _node_decl(name: str, role: str, lookup: dict[str, FlowStep]) -> str:
    """Build a Mermaid node declaration for a Flow step.

    Args:
        name: The step's method name (also the node id).
        role: One of ``"start"`` / ``"listen"`` / ``"router"``.
        lookup: Mapping from name to bound :class:`FlowStep` for
            description lookup.

    Returns:
        A Mermaid node declaration string with shape brackets matching
        the role and label derived from the step's description.
    """
    step = lookup.get(name)
    label = step.description if step is not None and step.description is not None else name
    if role == "start":
        return f'{safe(name)}(("{escape_mermaid_label(label)}"))'
    if role == "router":
        return f'{safe(name)}{{"{escape_mermaid_label(label)}"}}'
    return f'{safe(name)}["{escape_mermaid_label(label)}"]'
