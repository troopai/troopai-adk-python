(architecture/index)=

# Architecture

> How the ADK fits together: the pipeline, the three type layers, the
> Runner loop, the LLM ABC, and the multi-agent composition primitives.

```{figure} ../_static/images/architecture/overview.svg
:alt: Big-picture pipeline — Input → input guardrails → Agent loop → output guardrails → result.
:width: 90%
:class: themed
:align: center

The big-picture pipeline. Every `Runner.arun(...)` call traverses this shape.
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} Overview
:link: overview
:link-type: doc

The five-stage pipeline and where each subsystem plugs in.
:::

:::{grid-item-card} Type layers
:link: type-layers
:link-type: doc

`LLMInputContentItem` (Layer 1), `ChatCompletion*` (Layer 2 wire),
`RunItem` (Layer 3 developer-facing).
:::

:::{grid-item-card} Runner
:link: runner
:link-type: doc

The agent loop, `max_turns`, retries, streaming.
:::

:::{grid-item-card} LLM ABC
:link: llm-abc
:link-type: doc

Framework-owned `LLM`, not OpenAI's `Model`. One conversion per
direction inside each provider.
:::

:::{grid-item-card} Handoffs & Swarms
:link: handoffs-and-swarms
:link-type: doc

Routing and iterative collaboration.
:::

:::{grid-item-card} Graphs
:link: graphs
:link-type: doc

State-machine orchestration with checkpointers and HITL.
:::

:::{grid-item-card} Governance
:link: governance
:link-type: doc

Tenant routing, allowlists, audit, cost ledger.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

overview
type-layers
runner
llm-abc
handoffs-and-swarms
graphs
governance
```
