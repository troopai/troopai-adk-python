# Visualization Module

Pure-function emitters that translate the immutable topology data of
`Flow` and `Graph` into Mermaid or Graphviz DOT diagram strings. No
side effects, no I/O — callers print, paste, save, or embed the
returned string themselves.

## Files

| File | Purpose |
|---|---|
| `mermaid.py` | `flow_to_mermaid`, `definition_to_mermaid`, `graph_to_mermaid` — emitters producing Mermaid `flowchart` syntax |
| `dot.py`     | `flow_to_dot`, `definition_to_dot`, `graph_to_dot` — emitters producing Graphviz DOT syntax |

## Key Architectural Decisions

| # | Decision | Rationale |
|---|---|---|
| 1 | **Emitters are pure functions** | No I/O, no side effects, no class state. Same input → same output. Composable: callers pipeline the output through any sink. |
| 2 | **Read frozen topology data, not class metadata via introspection** | `flow_to_*` walks `flow.get_registry()` → `build_transition_table(...)`. `graph_to_*` walks `graph.nodes` + `graph.edges`. No source-code parsing (CrewAI's `get_possible_return_constants` antipattern). |
| 3 | **Node labels prefer `description`, fall back to method/node name** | `FlowStep.description` and `GraphNode.description` are the polymorphic-config attribute the codebase already uses on tools and graph nodes. When `None`, the method / node id is the label — today's implicit behaviour. |
| 4 | **AND / OR gates are emitted as synthesised gate nodes** | The transition table represents a gate as a `GateSpec`. Diagrams display it as a small `(( AND ))` / `(( OR ))` node so the visual matches the runtime: triggers flow in, the gate fires when conditions are met, the gated listener runs. |
| 5 | **Router edges carry their route label** | Mermaid `-->|"route_label"|` and DOT `[label="route_label"]`. Makes branching readable in PR diffs. |
| 6 | **`Flow.to_mermaid` / `Graph.to_mermaid` delegate to this module** | Thin instance methods on `Flow` and `Graph` are the ergonomic developer surface; the emitter functions are the testable seam. |
| 7 | **No external deps** | Both emitters produce strings using `f""` formatting only. No `mermaid` / `graphviz` Python package required. Consumers (Mermaid Live Editor, `dot` CLI, GitHub Markdown renderers, mermaid-cli) parse the string downstream. |

## Public Surface

Six pure-function emitters re-exported from `__init__.py`:
`flow_to_mermaid`, `flow_to_dot`, `graph_to_mermaid`, `graph_to_dot`,
`definition_to_mermaid`, `definition_to_dot`.

The `Flow` / `Graph` classes also expose ergonomic instance methods
(`to_mermaid` / `to_dot`) that delegate here. `definition_to_mermaid` /
`definition_to_dot` accept a
:class:`~troopai.adk.flows.definition.FlowDefinition` directly and render
the same topology without requiring a `Flow` instance or any run.

`helpers.py` is module-private (omitted from `__all__`).

See `docs/visualization/visualization.md` for usage and
`examples/flows/flow_diagram*.py` / `examples/graphs/graph_diagram*.py`
for runnable examples.
