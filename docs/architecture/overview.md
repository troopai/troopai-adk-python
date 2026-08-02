(architecture/overview)=

# 🔭 Big-Picture Pipeline

Every call to `Runner.arun(agent, prompt)` traverses the same shape:

```
Input → Input guardrails → Agent loop (LLM ↔ tools ↔ handoffs) → Output guardrails → Result
```

```{mermaid}
flowchart LR
  inp([prompt + context]) --> igr[input guardrails]
  igr --> loop{agent loop}
  loop -->|tool calls| tools[tools]
  tools --> loop
  loop -->|handoffs| agentN[next agent]
  agentN --> loop
  loop -->|final| ogr[output guardrails]
  ogr --> out([final result])
```

## The five stages

### 1. Input

A prompt, plus optional `context` (free-form, developer-owned). The
prompt is a single user message or a structured list of `Layer 1`
content items (`LLMInputContentItem`).

### 2. Input guardrails

User-authored guardrail functions run before the loop opens. They can
short-circuit with a rejection (content policy, length checks, custom
application logic). They are pure: no LLM calls.

### 3. Agent loop

The Runner alternates LLM steps and tool execution until one of three
things happens:

- The model emits a "final" reply (no tool calls, no handoff).
- A handoff routes execution to another agent (which re-enters the loop).
- A `max_turns` / `max_handoffs` / `*_budget` boundary trips.

The model never sees the next agent's identity directly — handoffs are
modelled as tool calls that the Runner intercepts.

### 4. Output guardrails

User-authored guardrail functions that run once the loop produces its
final reply. Same shape as input guardrails; same purity rule. They run
on the final assistant message before it leaves the Runner.

### 5. Result

A `RunResult` carrying:

- `final_output` — the assistant's final reply text or structured output.
- `new_items` — Layer 3 `RunItem`s emitted during the run (one per
  message, tool call, tool result, handoff).
- `conversation_history` — the running record (also Layer 3).
- Telemetry & cost ledger entries.

## Where each subsystem plugs in

| Subsystem      | Stage                                | Module                  |
| -------------- | ------------------------------------ | ----------------------- |
| Guardrails     | Stages 2 + 4                         | `src/troopai/adk/agents/agent_guardrails.py` |
| Tools          | Stage 3 (inside the loop)            | `src/troopai/adk/tools/` |
| Handoffs       | Stage 3 (re-entry into the loop)     | `src/troopai/adk/handoffs/` |
| Memory         | Stage 3 (context provider)           | `src/troopai/adk/memory/` |
| Skills         | Stage 3 (instructions + tools + governance bundle) | `src/troopai/adk/skills/` |
| MCP            | Stage 3 (tool source)                | `src/troopai/adk/mcp/` |
| Tracing        | All stages                           | `src/troopai/adk/tracing/` |
| Cost           | Stage 3 (per LLM call)               | `src/troopai/adk/run/cost.py`, `src/troopai/adk/budgets/` |
| Governance     | Stages 2–5 (audit, allowlists)       | `src/troopai/adk/run/governance.py` |

## Multi-agent composition

The single-agent loop is the atom. Composition primitives stitch atoms
together along three axes:

- **Handoffs** — directed routing (one agent passes execution to a
  named next agent).
- **Swarms** — undirected iteration (specialised agents cycle until a
  termination condition fires).
- **Graphs** — state-machine orchestration with explicit transitions,
  checkpointers, HITL.

See [Handoffs & Swarms](handoffs-and-swarms.md) and [Graphs](graphs.md).

## Why this shape

The pipeline is deliberately the *smallest* container that supports the
three engineering responses spelled out under
[Foundations](../foundations/three-mathematical-limits.md):

- **Bounded loops.** The loop has explicit budgets at every step
  (Halting Problem).
- **Empirical evaluation.** The result + history shape feeds directly
  into the eval harness (Rice's Theorem).
- **Specialised composition.** Handoffs / swarms / graphs slot in at
  Stage 3 without bloating the single-agent atom (No Free Lunch).
