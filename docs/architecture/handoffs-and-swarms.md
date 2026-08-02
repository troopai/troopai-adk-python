(architecture/handoffs-and-swarms)=

# 🔀 Handoffs & Swarms

Two of the three multi-agent composition axes. Use **handoffs** for
directed routing; use **swarms** for iterative collaboration.

## Handoffs (directed routing)

```{mermaid}
flowchart LR
  triage[Triage agent] -->|handoff| sales[Sales agent]
  triage -->|handoff| support[Support agent]
  triage -->|handoff| billing[Billing agent]
```

A handoff is a special tool call the Runner intercepts. The current
agent's last assistant message names a `next_agent`; the Runner routes
execution to that agent and continues the loop with shared or sliced
history (see `HandoffInputData`).

| Aspect            | Mechanism                                          |
| ----------------- | -------------------------------------------------- |
| Routing decision  | LLM-orchestrated (the model decides) OR code-orchestrated (a function decides). |
| History slicing   | `Handoff.input_filter` projects what the next agent sees. |
| Budget            | No auto-retry — a failed handoff surfaces, doesn't loop. |
| Replay            | Each handoff produces a `HandoffCallItem` (the tool call) and a `HandoffOutputItem` (what the next agent receives) in `new_items`. |

### LLM-orchestrated vs. code-orchestrated

- **LLM-orchestrated**: the agent's tools include `transfer_to_<name>`
  tool calls. The model picks the next agent by emitting one of them.
- **Code-orchestrated**: a Python function inspects the current
  `RunResult` and decides which agent to invoke next. No model
  involvement in the routing decision.

Code-orchestrated handoffs are deterministic and cheap; LLM-orchestrated
handoffs are flexible at the cost of model judgement variance. Most
production setups mix both.

## Swarms (iterative collaboration)

```{mermaid}
flowchart LR
  user[user prompt] --> coord
  subgraph swarm
    coord[coordinator] --> a[agent A]
    a --> b[agent B]
    b --> c[agent C]
    c --> coord
  end
  coord -->|done| out[result]
```

A swarm is a cycle: specialised agents iterate, each one consuming the
previous turn's output. The cycle terminates on:

- A member calling the `swarm_done` tool (surfaces as `SwarmDoneEvent`
  on the streaming channel).
- A termination condition firing (`MaxTurnsTermination`,
  `TokenBudgetTermination`, `HandoffToTermination`,
  `TextMentionTermination`, or your own — composable with `|` / `&`).
- A hard guard tripping (`SwarmConfig.max_handoffs`,
  `SwarmConfig.max_total_tokens`) or the absolute
  `RunConfig.max_total_turns` net.

Use a swarm when the work shape is "iterate-and-refine" rather than
"route-and-execute".

| Aspect              | Mechanism                                                    |
| ------------------- | ------------------------------------------------------------ |
| Cycle entry         | `Runner.arun_swarm(swarm, user_prompt, ...)`                 |
| Termination signal  | Any member calls `swarm_done` → `SwarmDoneEvent` is emitted.  |
| Streaming events    | `SwarmStartEvent`, `SwarmTurnStartEvent`, `SwarmHandoffEvent`, `SwarmTurnEndEvent`, `SwarmTurnInterruptEvent`, `SwarmDoneEvent`. |
| HITL                | `SwarmTurnInterruptEvent` carries the interrupt request.     |
| Resume              | Deep resume via swarm checkpointer (`arun_swarm_from_checkpoint`). |
| Tracing             | Per-turn OpenTelemetry spans.                                |

## When to use which

| Pattern                                    | Use this              |
| ------------------------------------------ | --------------------- |
| One agent decides "who handles this next"  | Handoff               |
| Several agents refine an answer together   | Swarm                 |
| Long-running workflow with shared state    | Graph (next page)     |
| Tool-only delegation (no peer agent)       | Tools, not handoffs   |

See [Graphs](graphs.md) for the third axis.
