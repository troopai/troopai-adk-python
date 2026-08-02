# Visualisation

TroopAI Agents ADK ships pure-function emitters that translate a
constructed `Flow` or a compiled `Graph` into a Mermaid `flowchart`
string or a Graphviz DOT digraph. The emitters read the immutable
topology data the framework already keeps (`FlowTransitionTable`,
`Graph.nodes` / `Graph.edges`) — no source-code parsing, no class-body
introspection, no run required.

## Surface

| Function | Argument | Returns |
|---|---|---|
| `flow_to_mermaid(flow, *, direction="LR")` | A `Flow` instance | Mermaid `flowchart` string |
| `flow_to_dot(flow, *, rankdir="LR")` | A `Flow` instance | Graphviz DOT string |
| `graph_to_mermaid(graph, *, direction="LR")` | A `Graph` instance | Mermaid `flowchart` string |
| `graph_to_dot(graph, *, rankdir="LR")` | A `Graph` instance | Graphviz DOT string |
| `Flow.to_mermaid(direction="LR")` | (instance method) | Same as `flow_to_mermaid(self)` |
| `Flow.to_dot(rankdir="LR")` | (instance method) | Same as `flow_to_dot(self)` |
| `Graph.to_mermaid(direction="LR")` | (instance method) | Same as `graph_to_mermaid(self)` |
| `Graph.to_dot(rankdir="LR")` | (instance method) | Same as `graph_to_dot(self)` |

Import either set:

```python
from troopai.adk.visualization import (
    flow_to_mermaid,
    flow_to_dot,
    graph_to_mermaid,
    graph_to_dot,
)
```

Or use the instance methods directly on a `Flow` / `Graph`.

## Flow shapes

| Decorator | Mermaid shape | DOT shape |
|---|---|---|
| `@flow_start` | `((label))` — rounded | `shape=oval` |
| `@flow_listen` | `[label]` — rectangle | `shape=box` |
| `@flow_router` | `{label}` — diamond | `shape=diamond` |
| AND-gate (synthesised) | `((AND))` — circle | `shape=circle`, label `"AND"` |
| OR-gate (synthesised) | `((OR))` — circle | `shape=circle`, label `"OR"` |

Each gate fan-in is rendered as a separate synthesised node so the
visual matches the runtime: each trigger flows into the gate; the
gate fires once when its condition is satisfied; the gated listener
runs.

Use the `description=` decorator keyword to set a richer label:

```python
@flow_listen("intake", description="Fact-check the article")
async def fact_check(self) -> None: ...
```

When `description` is `None` (the default), the method name is used.

## Graph shapes

| Position | Mermaid shape | DOT shape |
|---|---|---|
| Entry node | `(label)` — rounded | `shape=oval` |
| Intermediate node | `[label]` — rectangle | `shape=box` |
| Terminal node | `([label])` — stadium | `shape=doublecircle` |
| Conditional edge (`when=...`) | `-.->` (dashed) | `style=dashed` |
| Labelled edge | `-->|"label"|` | `[label="..."]` |

Use the `description=` argument on `GraphBuilder.node` for richer
labels:

```python
graph = Graph.new("review").node("fact_check", checker_agent, description="Fact-check").compile()
```

## Quick example: Flow

```python
from pydantic import BaseModel
from troopai.adk.flows import Flow, flow_listen, flow_router, flow_start


class State(BaseModel):
    article: str = ""


class ReviewFlow(Flow[State]):
    @flow_start(description="Receive article")
    async def receive(self) -> None: ...

    @flow_listen("receive", description="Fact-check")
    async def fact_check(self) -> None: ...

    @flow_listen("receive", description="Style-check")
    async def style_check(self) -> None: ...

    @flow_listen(fact_check & style_check, description="Merge")
    async def merge(self) -> None: ...

    @flow_router("merge")
    async def route(self) -> str:
        return "approve"

    @flow_listen("approve", description="Publish")
    async def publish(self) -> None: ...


flow = ReviewFlow(State)
print(flow.to_mermaid())
```

A runnable version of the above lives at `examples/flows/flow_diagram.py`.

## Quick example: Graph

```python
from troopai.adk.graphs.graph import Graph

graph = (
    Graph.new("review")
    .node("intake", intake_agent, description="Receive draft")
    .node("publish", publish_agent, description="Publish")
    .edge("intake", "publish", label="approved")
    .entry("intake")
    .terminal("publish")
    .compile()
)
print(graph.to_mermaid())
```

A runnable version with conditional edges lives at
`examples/graphs/graph_diagram.py`.

## Rendering the output

The emitters return **strings**. You can render them yourself in three
ways:

### 1. Manual / external tools (zero extra deps)

- **Mermaid**: paste into a GitHub Markdown ```` ```mermaid ```` block
  or the [Mermaid Live Editor](https://mermaid.live).
- **DOT**: pipe through the Graphviz `dot` CLI:
  ```bash
  python my_flow.py | dot -Tsvg -o flow.svg
  ```

### 2. Python `viz` extra — render DOT to SVG/PNG/PDF

```bash
pip install 'troopai-adk-python[viz]'
```

Adds the `graphviz` Python package, which shells out to the local
`dot` CLI (install Graphviz separately: `apt install graphviz` /
`brew install graphviz`). Then:

```python
from graphviz import Source
Source(flow.to_dot()).render(filename="flow", format="svg", cleanup=True)
```

If the `dot` CLI is missing, `graphviz.ExecutableNotFound` is raised —
save the raw `.dot` string and render later.

### 3. Python `mermaid` extra — render Mermaid via Mermaid Live

```bash
pip install 'troopai-adk-python[mermaid]'
```

Adds `mermaid-py`, which renders Mermaid strings through the Mermaid
Live online API:

```python
from mermaid import Mermaid
Mermaid(flow.to_mermaid()).to_png("flow.png")
```

Requires network access; on failure the runnable examples fall back
to saving raw `.mmd` you can paste at https://mermaid.live.

> **Neither extra is required.** `flow.to_mermaid()` and
> `flow.to_dot()` produce plain strings using only the Python standard
> library. The extras are convenience for in-process rendering.

Both runnable examples
(`examples/flows/flow_diagram.py` and
`examples/graphs/graph_diagram.py`) demonstrate the extras with
graceful fallback to raw files when the CLI / renderer is unavailable.
Agent-based variants (`examples/flows/flow_diagram_with_agents.py`,
`examples/graphs/graph_diagram_with_agents.py`) show that the
topology is captured statically — the diagram is identical whether or
not the flow has run.

## Determinism

Both emitters are pure functions of the topology: same inputs
produce byte-identical outputs. Diff-friendly. Suitable for storing
generated diagrams in source control alongside the code they describe.

## What's not in scope

- Per-run diagrams (which steps fired, current state) — emitters
  describe the **topology**, not a particular execution. For run
  observability, consume the streaming events (`flow.events`, `graph.events`).
- Embedding source-code snippets, return labels of routers, or runtime
  state in node labels — diagrams stay topology-only on purpose
  (mirrors the project's "no source-code introspection of routers"
  decision).
