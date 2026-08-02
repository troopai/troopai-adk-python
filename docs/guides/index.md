(guides/index)=

# Guides

How-to pages for the ADK's developer surface. Each guide is currently
a short pointer to the module-level docs under `docs/<module>/`. Full
migration into `docs/guides/` lands in a follow-up phase.

::::{grid} 1 2 3 3
:gutter: 3
:class-container: sd-text-center

:::{grid-item-card} Agents
:link: agents
:link-type: doc

`Agent` configuration: name, instructions, tools, handoffs,
guardrails.
:::

:::{grid-item-card} Tools
:link: tools
:link-type: doc

Function tools, hosted tools, MCP tools, tool guardrails.
:::

:::{grid-item-card} Handoffs
:link: handoffs
:link-type: doc

LLM-orchestrated and code-orchestrated routing.
:::

:::{grid-item-card} Guardrails
:link: guardrails
:link-type: doc

User-authored input and output safety gates. Decorator-based and config `ref` patterns.
:::

:::{grid-item-card} Memory
:link: memory
:link-type: doc

Episodic + semantic memory; vector stores; embedders.
:::

:::{grid-item-card} Skills
:link: skills
:link-type: doc

Reusable capability bundles (instructions + tools + governance).
:::

:::{grid-item-card} Tracing
:link: tracing
:link-type: doc

OpenInference / OpenTelemetry; Arize, Phoenix, Langfuse exporters.
:::

:::{grid-item-card} Cost
:link: cost
:link-type: doc

CostEstimator, CostLedger, LLMRouter (CheapestFirst, LatencyFirst).
:::

:::{grid-item-card} Sandbox
:link: sandbox
:link-type: doc

Sandbox-isolated tool execution; Docker / K8s / hosted-bridge
clients.
:::

:::{grid-item-card} A2A
:link: a2a
:link-type: doc

Agent-to-Agent protocol.
:::

:::{grid-item-card} MCP
:link: mcp
:link-type: doc

Model Context Protocol client + server.
:::

::::

```{toctree}
:hidden:
:maxdepth: 1

agents
tools
handoffs
guardrails
memory
skills
tracing
cost
sandbox
a2a
mcp
```
