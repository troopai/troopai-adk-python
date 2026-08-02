(references/api/index)=

# API Reference

Auto-generated reference for the public surface of the framework's core
modules, grouped by theme. Usage walkthroughs live under
[Guides](../../guides/index.md) and the per-topic sections linked from
each page.

## Core

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} `Agent`
:link: agent
:link-type: doc

Configuration-only object. Name, instructions, tools, handoffs,
guardrails.
:::

:::{grid-item-card} `Runner`
:link: runner
:link-type: doc

Execution entry-points: `arun`, `arun_streamed`, `arun_graph`,
`arun_swarm`.
:::

:::{grid-item-card} `LLM`
:link: llm
:link-type: doc

Framework-owned LLM abstract base class.
:::

:::{grid-item-card} `FunctionTool`
:link: tool
:link-type: doc

The Tool ABC + decorator + types.
:::

::::

## Orchestration

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Swarms
:link: swarms
:link-type: doc

Iterative multi-agent collaboration with explicit termination and
pluggable routing.
:::

:::{grid-item-card} Graphs
:link: graphs
:link-type: doc

State-machine orchestration with checkpointing, interrupts, and
streaming events.
:::

:::{grid-item-card} Flows
:link: flows
:link-type: doc

Decorator-driven multi-step orchestration over typed shared state.
:::

:::{grid-item-card} Tasks
:link: tasks
:link-type: doc

Declarative units of work composed into pipelines and groups.
:::

::::

## State and persistence

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Memory
:link: memory
:link-type: doc

Extracted, searchable knowledge carried across sessions.
:::

:::{grid-item-card} Session
:link: session
:link-type: doc

Conversation persistence for agent runs.
:::

::::

## Safety and protocols

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Guardrails
:link: guardrails
:link-type: doc

Built-in PII, prompt-injection, and wrong-language guardrails.
:::

:::{grid-item-card} MCP
:link: mcp
:link-type: doc

Model Context Protocol servers, lifecycle, and tool filters.
:::

:::{grid-item-card} A2A
:link: a2a
:link-type: doc

Agent-to-Agent protocol client and server surfaces.
:::

::::

## Foundations

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Exceptions
:link: exceptions
:link-type: doc

The framework exception hierarchy rooted at `TroopAIError`.
:::

:::{grid-item-card} Types
:link: types
:link-type: doc

Provider-agnostic wire and history types.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1
:caption: Core

agent
runner
llm
tool
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Orchestration

swarms
graphs
flows
tasks
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: State and persistence

memory
session
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Safety and protocols

guardrails
mcp
a2a
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Foundations

exceptions
types
```
