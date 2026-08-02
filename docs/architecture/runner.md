(architecture/runner)=

# 🔁 The Runner

The `Runner` executes an `Agent` configuration. Agent is **config**;
Runner is **execution**. There is no `agent.arun()` — every run path
goes through `Runner.arun(...)`, `Runner.arun_graph(...)`,
`Runner.arun_swarm(...)`, or their streaming variants.

```{figure} ../_static/images/architecture/runner-loop.svg
:alt: The Runner agent loop with explicit budgets.
:width: 80%
:class: themed
:align: center

The agent loop with the `max_turns` / handoffs / retry boundaries
made visible.
```

## The loop

```{mermaid}
flowchart TB
  start([Runner.arun]) --> ig[input guardrails]
  ig --> step{LLM step}
  step -->|text only| ogr[output guardrails]
  step -->|tool calls| tools[execute tools]
  tools --> step
  step -->|handoff| route[route to next agent]
  route --> step
  ogr --> done([RunResult])
  step -. max_turns .-> done
```

## Budgets (every one of them is opt-in and cost-conservative)

| Budget                       | Lives on                  | What it bounds                           |
| ---------------------------- | ------------------------- | ---------------------------------------- |
| `max_turns`                  | `Runner.arun(...)`        | Total LLM steps per single-agent run.    |
| `SwarmConfig.max_handoffs`   | `SwarmConfig`             | Total handoffs allowed inside a swarm.   |
| `FunctionTool.max_retries`   | per-tool `FunctionTool`   | LLM retries before the tool is disabled. |
| `FunctionTool.max_result_tokens` | per-tool `FunctionTool` | Bound on a single tool result's tokens.  |
| `LLMConfig.num_retries`      | `LLMConfig`               | SDK-level transient retries.             |
| `LLMConfig.retry_policy`     | `LLMConfig`               | Framework backoff loop (`LLMRetryPolicy`).|
| `*_budget` (cost)            | `BudgetConfig`            | Per-run cost ceiling.                    |

The Runner does not assume the loop self-terminates. See
{ref}`Foundations: Halting Problem <halting-problem>`.

## Streaming

The Runner exposes streamed variants per execution shape:

- `Runner.arun_task_streamed(...)` — single-agent task with event stream.
- `Runner.arun_graph_streamed(...)` — graph orchestration with per-transition events.
- `Runner.arun_swarm_streamed(...)` — swarm with `SwarmTurnInterruptEvent` and per-turn events.

Each emits an async iterator of typed events. Consumers pick what they
care about; nothing is auto-printed.

## Retry semantics

- **Model retries**: SDK-level transient retries are bounded by
  `LLMConfig.num_retries`; framework-level classified-error backoff is
  configured via `LLMConfig.retry_policy` (`LLMRetryPolicy`). Both
  default to off / conservative.
- **Tool retries**: bounded by `FunctionTool.max_retries` per tool;
  default `None` (no auto-retry); set to an int to allow that many
  failed attempts before the tool is disabled for the run.
- **Handoff retries**: never auto-retried. A failed handoff surfaces
  as an exception or a `HandoffFailedItem` depending on
  `HandoffConfig`.

## What the Runner does NOT do

- It does not auto-inject system prompts, tool descriptions, or
  framework instructions. Every token the system adds is opt-in.
- It does not silently extend `max_turns` if the loop is "almost
  done". The boundary is hard.
- It does not call provider SDKs directly. All provider traffic
  goes through the `LLM` ABC (see [LLM ABC](llm-abc.md)).
